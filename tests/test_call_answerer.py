"""Tests for answering a call from Home Assistant as a real WebRTC peer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.bticino_intercom import call_answerer
from custom_components.bticino_intercom.call_answerer import (
    ANSWERER_KEY,
    EosCallAnswerer,
    async_answer_call,
    async_hangup_call,
    current_answerer,
    split_candidates,
    stun_servers,
)
from custom_components.bticino_intercom.const import DOMAIN

from .conftest import BRIDGE_MAC, EXTERNAL_UNIT_ID, SESSION_ID

aiortc = pytest.importorskip("aiortc")

# The loopback calls below open real UDP sockets (ICE), which the HA test
# harness blocks by default and allows towards 127.0.0.1 only.
media = pytest.mark.usefixtures("socket_enabled", "loopback_ice")


@pytest.fixture
def loopback_ice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gather host candidates on 127.0.0.1 only, the one host the harness allows."""
    import aioice.ice

    monkeypatch.setattr(aioice.ice, "get_host_addresses", lambda use_ipv4, use_ipv6: ["127.0.0.1"])


def test_stun_servers_keeps_stun_only() -> None:
    """TURN URLs and credentials are dropped; servers with no STUN URL vanish."""
    servers = stun_servers(
        [
            {"urls": ["stun:turn.netatmo.net:3478", "turn:turn.netatmo.net:3478"], "username": "u", "credential": "c"},
            {"urls": "turns:turn.netatmo.net:443", "username": "u", "credential": "c"},
            {"url": "stun:stun.example:19302"},
        ]
    )
    assert [s.urls for s in servers] == [["stun:turn.netatmo.net:3478"], ["stun:stun.example:19302"]]
    assert all(s.username is None and s.credential is None for s in servers)


def test_split_candidates_by_m_line() -> None:
    sdp = (
        "v=0\r\nm=video 9 X 96\r\na=candidate:1 1 udp 1 10.0.0.1 5000 typ host\r\n"
        "m=audio 9 X 111\r\na=candidate:2 1 udp 1 10.0.0.1 5001 typ host\r\na=end-of-candidates\r\n"
    )
    stripped, candidates = split_candidates(sdp)
    assert "a=candidate" not in stripped
    assert "end-of-candidates" not in stripped
    assert candidates == [
        (0, "candidate:1 1 udp 1 10.0.0.1 5000 typ host"),
        (1, "candidate:2 1 udp 1 10.0.0.1 5001 typ host"),
    ]


class FakeDevice:
    """A local aiortc peer that rings like the intercom: it offers, we answer.

    Also plays the signaling server: the answer and candidates the answerer
    sends are applied to this peer, as the cloud relays them to the device.
    """

    def __init__(self) -> None:
        self.pc = aiortc.RTCPeerConnection()
        self.audio_frames = 0
        self.readers: list[asyncio.Future[None]] = []
        self.terminated: list[str] = []
        self.answers: list[str] = []
        self.raw: list[dict[str, Any]] = []
        self._on_answer = None
        self._on_candidate = None
        self._on_event = None
        self.sessions: dict[str, Any] = {}

    async def make_offer(self) -> str:
        from aiortc.contrib.media import MediaPlayer  # noqa: F401 - aiortc media is importable

        self.pc.addTransceiver("video", direction="sendonly")
        self.pc.addTransceiver("audio", direction="sendrecv")

        @self.pc.on("track")
        def _on_track(track: Any) -> None:
            async def count() -> None:
                from aiortc.mediastreams import MediaStreamError

                try:
                    while True:
                        await track.recv()
                        self.audio_frames += 1
                except MediaStreamError:
                    return

            self.readers.append(asyncio.ensure_future(count()))

        await self.pc.setLocalDescription(await self.pc.createOffer())
        return self.pc.localDescription.sdp

    # --- the SignalingClient surface the answerer uses ---
    # Single-session, as the pybticino build with FCM: every send goes to the
    # current session, which set_session_from_push selects.

    session_id: str | None = None

    def set_session_from_push(self, session_id: str, tag_id: str, correlation_id: str, device_id: str) -> None:
        self.sessions[session_id] = (tag_id, device_id)
        self.session_id = session_id

    async def send_answer(self, sdp: str) -> None:
        assert self.session_id == SESSION_ID
        self.answers.append(sdp)
        await self.pc.setRemoteDescription(aiortc.RTCSessionDescription(sdp=sdp, type="answer"))

    async def send_candidate(self, candidate: str, sdp_m_line_index: int) -> None:
        from aiortc.sdp import candidate_from_sdp

        assert self.session_id == SESSION_ID
        parsed = candidate_from_sdp(candidate.removeprefix("candidate:"))
        parsed.sdpMLineIndex = sdp_m_line_index
        parsed.sdpMid = self.pc.getTransceivers()[sdp_m_line_index].mid
        await self.pc.addIceCandidate(parsed)

    async def send_terminate(self) -> None:
        self.terminated.append(self.session_id)
        self.session_id = None

    async def _send(self, message: dict[str, Any]) -> None:
        """Raw rtc messages: the answerer sends the app's "player" message this way."""
        # Only once the media is up, as the app sends it on the green handset.
        assert self.pc.connectionState == "connected"
        self.raw.append(message)

    async def disconnect(self) -> None:
        pass

    async def connect(self) -> None:
        pass

    async def ensure_connected(self) -> None:
        pass

    async def resubscribe(self) -> None:
        pass


def _call(sdp: str | None) -> dict[str, Any]:
    return {
        "session_id": SESSION_ID,
        "tag_id": "tag-1",
        "correlation_id": "42",
        "device_id": BRIDGE_MAC,
        "module_id": EXTERNAL_UNIT_ID,
        "sdp": sdp,
    }


@pytest.fixture
def account() -> SimpleNamespace:
    return SimpleNamespace(async_get_turn_servers=AsyncMock(return_value=[]))


@media
async def test_answer_connects_media_and_hangs_up(hass: HomeAssistant, account: SimpleNamespace) -> None:
    """A real loopback call: the answer connects, audio reaches the device, hang-up terminates."""
    device = FakeDevice()
    offer = await device.make_offer()
    answerer = EosCallAnswerer(hass, account, device, _call(offer))

    await answerer.async_answer()
    assert answerer.state == "connected"
    assert answerer.active
    assert device.pc.connectionState == "connected"
    # Our answer carries no candidates (they are trickled), and no forced stereo
    # OPUS: the app's answer has none.
    assert "a=candidate" not in device.answers[0]
    assert "stereo=1" not in device.answers[0]
    # The green handset: one "player unmute" on the call's session, as the app sends it.
    assert len(device.raw) == 1
    unmute = device.raw[0]
    assert unmute["action"] == "rtc"
    assert unmute["data"] == {"type": "player", "value": "unmute"}
    assert unmute["session_id"] == SESSION_ID
    assert unmute["tag_id"] == "tag-1"
    assert unmute["device_id"] == BRIDGE_MAC
    assert unmute["correlation_id"].isdigit()

    for _ in range(50):
        if device.audio_frames >= 5:
            break
        await asyncio.sleep(0.05)
    assert device.audio_frames >= 5, "the device received no audio from Home Assistant"

    ended = []
    answerer.async_on_end(lambda: ended.append(True))
    await answerer.async_hangup()
    assert device.terminated == [SESSION_ID]
    assert answerer.state == "ended"
    assert ended == [True]
    assert device._on_candidate is None  # signaling handed back
    await device.pc.close()


@media
async def test_device_terminate_ends_call_without_terminating_back(
    hass: HomeAssistant, account: SimpleNamespace
) -> None:
    device = FakeDevice()
    answerer = EosCallAnswerer(hass, account, device, _call(await device.make_offer()))
    await answerer.async_answer()

    await device._on_event(SESSION_ID, "terminate", {"data": {"type": "terminate"}})
    for _ in range(40):
        if answerer.state == "ended":
            break
        await asyncio.sleep(0.05)
    assert answerer.state == "ended"
    assert device.terminated == []
    await device.pc.close()


@media
async def test_other_session_events_are_ignored(hass: HomeAssistant, account: SimpleNamespace) -> None:
    device = FakeDevice()
    answerer = EosCallAnswerer(hass, account, device, _call(await device.make_offer()))
    await answerer.async_answer()

    await device._on_event("another-session", "terminate", {})
    await asyncio.sleep(0.1)
    assert answerer.state == "connected"
    await answerer.async_hangup()
    await device.pc.close()


@media
async def test_no_media_terminates_and_raises(
    hass: HomeAssistant, account: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the media never connects the session is terminated, not left ringing."""
    monkeypatch.setattr(call_answerer, "CONNECT_TIMEOUT", 0.5)
    device = FakeDevice()
    offer = await device.make_offer()

    async def drop_candidates(*_args: Any, **_kwargs: Any) -> None:
        return

    async def keep_answer(sdp: str, session_id: str | None = None) -> None:
        device.answers.append(sdp)  # never applied: no media can connect

    device.send_candidate = drop_candidates
    device.send_answer = keep_answer
    answerer = EosCallAnswerer(hass, account, device, _call(offer))
    with pytest.raises(call_answerer.CallAnswerError):
        await answerer.async_answer()
    assert device.terminated == [SESSION_ID]
    assert answerer.state == "ended"
    await device.pc.close()


async def test_answer_without_device_offer_is_refused(hass: HomeAssistant, account: SimpleNamespace) -> None:
    answerer = EosCallAnswerer(hass, account, FakeDevice(), _call(None))
    with pytest.raises(call_answerer.CallAnswerError, match="offer has not arrived"):
        await answerer.async_answer()


@media
async def test_service_helpers(hass: HomeAssistant, account: SimpleNamespace) -> None:
    """async_answer_call checks for a call; the answerer is tracked and hung up by entry."""
    device = FakeDevice()
    coordinator = SimpleNamespace(active_call=None)
    hass.data[DOMAIN] = {
        "entry1": {"coordinator": coordinator, "account": account, "signaling_client": device},
    }

    with pytest.raises(HomeAssistantError, match="No incoming call"):
        await async_answer_call(hass, "entry1")

    coordinator.active_call = {**_call(None), "status_only": True}
    with pytest.raises(HomeAssistantError, match="has not reached"):
        await async_answer_call(hass, "entry1")

    coordinator.active_call = _call(await device.make_offer())
    await async_answer_call(hass, "entry1", max_duration=30)
    assert current_answerer(hass, "entry1") is hass.data[DOMAIN]["entry1"][ANSWERER_KEY]

    with pytest.raises(HomeAssistantError, match="already answered"):
        await async_answer_call(hass, "entry1")

    await async_hangup_call(hass, "entry1")
    assert current_answerer(hass, "entry1") is None
    assert device.terminated == [SESSION_ID]
    await device.pc.close()
