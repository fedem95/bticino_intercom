"""Answer an incoming Classe 300 EOS call with a real WebRTC media session.

A Classe 300 EOS only counts a call as answered (and stops ringing on the
indoor monitor and on every other phone) once a real media session is up with
it, audio included. There is no separate "accept" message: the official app
answers the device offer, adds its microphone track, and the device takes the
call when that media connects. The device also sends its own audio only after
it receives audio from the peer.

The camera's answer mode bridges SDP between the device and the browser and
leaves the media to the browser; when the browser does not complete the audio,
the device sees no answer and the call ends as missed. Here Home Assistant is
the peer itself (aiortc), as the official app is:

- it answers the device offer received through FCM with its own SDP,
- it sends real audio at once (a soft tone for now, the user's microphone in a
  later step), so the device takes the call and opens its own audio,
- it receives the device audio and video.

Device quirks this works around, all found by answering real calls:

- the device presents an RSA DTLS certificate, while aiortc offers only
  ECDHE-ECDSA cipher suites: the cipher list is widened (_patch_dtls_ciphers);
- a relay candidate of our own makes the device nominate a pair the Netatmo
  TURN server refuses (403 Forbidden IP): only STUN servers are used;
- the device offers stereo OPUS and the app's answer mirrors it: so does ours.
"""

from __future__ import annotations

import array
import asyncio
import fractions
import logging
import math
import re
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN

if TYPE_CHECKING:
    from pybticino import AsyncAccount, SignalingClient

_LOGGER = logging.getLogger(__name__)

# The device rings for about 30 s; an answer later than that finds no call.
CONNECT_TIMEOUT = 20  # seconds from our answer to a connected media session
DEFAULT_MAX_DURATION = 120  # seconds an answered call stays open at most
ANSWER_SETTLE = 0.5  # seconds between our answer and our first candidate

# Soft 440 Hz tone: real audio the device accepts as a peer talking, quiet
# enough not to startle whoever is at the door. Silence is not enough: with a
# silent track the device never opens its own audio.
TONE_FREQUENCY = 440.0
TONE_AMPLITUDE = 0.08
TONE_SAMPLE_RATE = 8000
TONE_PTIME = 0.020  # seconds of audio per frame, as aiortc's own tracks

# ECDHE-RSA and plain RSA suites as a browser offers them, ECDSA first so a
# peer with an ECDSA certificate negotiates as before.
_DTLS_CIPHERS = (
    b"ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
    b"ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:"
    b"ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:"
    b"ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES128-SHA:"
    b"ECDHE-ECDSA-AES256-SHA:ECDHE-RSA-AES256-SHA:"
    b"AES128-GCM-SHA256:AES256-GCM-SHA384:AES128-SHA:AES256-SHA"
)
_dtls_patched = False


class CallAnswerError(Exception):
    """The call could not be answered."""


def _patch_dtls_ciphers() -> None:
    """Let aiortc negotiate DTLS with a peer that has an RSA certificate.

    Wraps aiortc's SSL context factory and widens the cipher list it sets;
    everything else in the context stays as aiortc builds it. Done once per
    process. Without it the handshake fails at once ("DTLS handshake failed").
    """
    global _dtls_patched
    if _dtls_patched:
        return
    from aiortc.rtcdtlstransport import RTCCertificate

    original = RTCCertificate._create_ssl_context

    def _create_ssl_context(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = original(self, *args, **kwargs)
        ctx.set_cipher_list(_DTLS_CIPHERS)
        return ctx

    RTCCertificate._create_ssl_context = _create_ssl_context
    _dtls_patched = True


def stun_servers(raw_servers: list[dict[str, Any]]) -> list[Any]:
    """Return aiortc ICE servers with the STUN URLs only, no TURN.

    The device offers its own relay candidates anyway. With a relay candidate
    on our side, the device nominated a pair that the Netatmo TURN server
    refused to relay (403 Forbidden IP on channel bind) and the call failed.
    """
    from aiortc import RTCIceServer

    servers = []
    for raw in raw_servers:
        urls = raw.get("urls", raw.get("url"))
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list):
            continue
        stun = [url for url in urls if isinstance(url, str) and url.startswith("stun:")]
        if stun:
            servers.append(RTCIceServer(urls=stun))
    return servers


def mirror_opus_fmtp(answer_sdp: str, offer_sdp: str) -> str:
    """Make the answer's OPUS parameters mirror the device offer.

    The device offers OPUS with ``sprop-stereo=1``. libwebrtc (the official
    app) answers with ``stereo=1;sprop-stereo=1`` and the device's capture and
    bitrate parameters; aiortc leaves them out. Add them to the answer.
    """
    opus_pt = None
    for line in offer_sdp.splitlines():
        if match := re.match(r"a=rtpmap:(\d+) opus/", line, re.IGNORECASE):
            opus_pt = match.group(1)
    if opus_pt is None:
        return answer_sdp

    prefix = f"a=fmtp:{opus_pt} "
    offer_fmtp = next((line[len(prefix) :] for line in offer_sdp.splitlines() if line.startswith(prefix)), "")
    wanted = ["stereo=1", "sprop-stereo=1"]
    for part in (p.strip() for p in offer_fmtp.split(";")):
        if part.split("=")[0] in ("sprop-maxcapturerate", "maxplaybackrate", "maxaveragebitrate"):
            wanted.append(part)

    lines = answer_sdp.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            params = [p.strip() for p in line[len(prefix) :].split(";") if p.strip()]
            keys = {p.split("=")[0] for p in params}
            params += [w for w in wanted if w.split("=")[0] not in keys]
            out.append(prefix + ";".join(params))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        rebuilt: list[str] = []
        for line in out:
            rebuilt.append(line)
            if re.match(rf"a=rtpmap:{opus_pt} opus/", line, re.IGNORECASE):
                rebuilt.append(prefix + ";".join(wanted))
        out = rebuilt
    return "\r\n".join(out) + ("\r\n" if answer_sdp.endswith(("\r\n", "\n")) else "")


def split_candidates(sdp: str) -> tuple[str, list[tuple[int, str]]]:
    """Split a gathered SDP into the SDP without candidates and its candidates.

    The candidates are trickled one by one after the answer, as the app does;
    the answer itself carries none. Returns (sdp, [(m-line index, candidate)]).
    """
    kept: list[str] = []
    candidates: list[tuple[int, str]] = []
    index = -1
    for line in sdp.splitlines():
        if line.startswith("m="):
            index += 1
        if line.startswith("a=candidate:") and index >= 0:
            candidates.append((index, line.removeprefix("a=")))
            continue
        if line == "a=end-of-candidates":
            continue
        kept.append(line)
    return "\r\n".join(kept) + "\r\n", candidates


def _tone_track() -> Any:
    """Return an aiortc audio track sending a soft continuous tone."""
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame

    class ToneTrack(AudioStreamTrack):
        """Real audio for the device: a quiet sine tone in 20 ms frames."""

        def __init__(self) -> None:
            super().__init__()
            self._samples = int(TONE_PTIME * TONE_SAMPLE_RATE)
            self._timestamp = 0
            self._start: float | None = None
            omega = 2.0 * math.pi * TONE_FREQUENCY / TONE_SAMPLE_RATE
            # One second of tone, a whole number of periods, looped seamlessly;
            # twice over so any 20 ms window is one contiguous slice.
            wave = array.array(
                "h", (int(TONE_AMPLITUDE * 32767 * math.sin(omega * n)) for n in range(TONE_SAMPLE_RATE))
            )
            self._wave = (wave + wave).tobytes()

        async def recv(self) -> Any:
            if self.readyState != "live":
                raise MediaStreamError
            if self._start is None:
                self._start = time.time()
            else:
                self._timestamp += self._samples
                wait = self._start + self._timestamp / TONE_SAMPLE_RATE - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            start = (self._timestamp % TONE_SAMPLE_RATE) * 2  # bytes: 16-bit samples
            frame = AudioFrame(format="s16", layout="mono", samples=self._samples)
            frame.planes[0].update(self._wave[start : start + self._samples * 2])
            frame.pts = self._timestamp
            frame.sample_rate = TONE_SAMPLE_RATE
            frame.time_base = fractions.Fraction(1, TONE_SAMPLE_RATE)
            return frame

    return ToneTrack()


ANSWERER_KEY = "call_answerer"


def current_answerer(hass: HomeAssistant, entry_id: str) -> EosCallAnswerer | None:
    """Return the call this entry is answering or holding, if any."""
    answerer = hass.data.get(DOMAIN, {}).get(entry_id, {}).get(ANSWERER_KEY)
    return answerer if answerer is not None and answerer.active else None


async def async_answer_call(hass: HomeAssistant, entry_id: str, max_duration: float = DEFAULT_MAX_DURATION) -> None:
    """Answer the entry's ringing call from Home Assistant. Raises HomeAssistantError."""
    entry_data = hass.data.get(DOMAIN, {}).get(entry_id)
    if entry_data is None:
        raise HomeAssistantError("BTicino intercom not loaded")
    if current_answerer(hass, entry_id):
        raise HomeAssistantError("A call is already answered from Home Assistant")
    call = entry_data["coordinator"].active_call
    if not call or not call.get("session_id"):
        raise HomeAssistantError("No incoming call to answer")
    if not call.get("sdp"):
        raise HomeAssistantError(
            "The call offer has not reached Home Assistant (for a Classe 300 EOS it comes through FCM)"
        )

    answerer = EosCallAnswerer(
        hass, entry_data["account"], entry_data["signaling_client"], call, max_duration=max_duration
    )
    entry_data[ANSWERER_KEY] = answerer
    _LOGGER.info("Answering call %s from Home Assistant", answerer.session_id)
    try:
        await answerer.async_answer()
    except CallAnswerError as err:
        raise HomeAssistantError(str(err)) from err


async def async_hangup_call(hass: HomeAssistant, entry_id: str) -> None:
    """Hang up the call answered from Home Assistant, if there is one."""
    if answerer := current_answerer(hass, entry_id):
        await answerer.async_hangup()


class EosCallAnswerer:
    """One incoming call answered by Home Assistant as a WebRTC peer.

    Owns the shared signaling client's callbacks while the call lasts, so no
    live view may run alongside it (the device would refuse a second peer
    anyway). Create one per call; it cannot be reused.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        account: AsyncAccount,
        signaling: SignalingClient,
        call: dict[str, Any],
        max_duration: float = DEFAULT_MAX_DURATION,
    ) -> None:
        """Prepare to answer `call` (the coordinator's active call, with the device SDP)."""
        self.hass = hass
        self._account = account
        self._signaling = signaling
        self.call = dict(call)
        self.session_id: str = str(call["session_id"])
        self.module_id: str | None = call.get("module_id")
        self._max_duration = max_duration
        self.state = "idle"  # idle, answering, connected, ended
        self._peer: Any = None
        self._remote_candidates: list[dict[str, Any]] = []
        self._remote_described = False
        self._connected = asyncio.Event()
        self._ended = asyncio.Event()
        self._remote_ended = False
        self._sinks: list[asyncio.Task[None]] = []
        self._lifetime_task: asyncio.Task[None] | None = None
        self._closed = False
        self._on_end_callbacks: list[Any] = []

    @property
    def active(self) -> bool:
        """Return whether the call is being answered or in progress."""
        return self.state in ("answering", "connected")

    @callback
    def async_on_end(self, func: Any) -> None:
        """Call `func()` once the call has ended."""
        self._on_end_callbacks.append(func)

    async def async_answer(self) -> None:
        """Answer the device offer and wait until the media session is up.

        Raises CallAnswerError when the call cannot be answered or the media
        does not connect in time; the session is terminated in that case.
        """
        if self.state != "idle":
            raise CallAnswerError("This call has already been answered")
        device_sdp = self.call.get("sdp")
        if not device_sdp:
            raise CallAnswerError("The device offer has not arrived: the call cannot be answered")
        self.state = "answering"
        try:
            await self._async_negotiate(device_sdp)
            try:
                await asyncio.wait_for(self._wait_connected(), timeout=CONNECT_TIMEOUT)
            except TimeoutError as err:
                raise CallAnswerError(f"The media session did not connect within {CONNECT_TIMEOUT}s") from err
        except Exception as err:
            await self.async_hangup(reason="answer failed")
            if isinstance(err, CallAnswerError):
                raise
            raise CallAnswerError(f"Could not answer the call: {err}") from err

        if self._closed:  # hung up while the media was connecting
            raise CallAnswerError("The call was hung up while answering")
        self.state = "connected"
        _LOGGER.info("Call %s answered from Home Assistant: media connected", self.session_id)
        self._lifetime_task = self.hass.async_create_background_task(
            self._async_lifetime(), f"bticino_intercom call {self.session_id}"
        )

    async def _async_negotiate(self, device_sdp: str) -> None:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        from .camera import _async_prepare_signaling

        _patch_dtls_ciphers()
        try:
            ice_servers = stun_servers(await self._account.async_get_turn_servers())
        except Exception as err:
            _LOGGER.warning("Could not load STUN servers (%s); answering with host candidates only", err)
            ice_servers = []

        await _async_prepare_signaling(self._signaling, self.call)
        self._take_signaling()

        peer = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        self._peer = peer
        peer.addTrack(_tone_track())

        @peer.on("track")
        def _on_track(track: Any) -> None:
            _LOGGER.debug("Call %s: receiving %s from the device", self.session_id, track.kind)
            self._sinks.append(self.hass.async_create_background_task(self._drain(track), "bticino drain"))

        @peer.on("connectionstatechange")
        async def _on_state() -> None:
            state = peer.connectionState
            _LOGGER.debug("Call %s: media %s", self.session_id, state)
            if state == "connected":
                self._connected.set()
            elif state in ("failed", "closed"):
                self._ended.set()

        # Netatmo labels the device SDP "call"; for WebRTC it is an offer.
        await peer.setRemoteDescription(RTCSessionDescription(sdp=device_sdp, type="offer"))
        self._remote_described = True
        await self._apply_remote_candidates()

        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)  # gathers our candidates
        answer_sdp, candidates = split_candidates(peer.localDescription.sdp)
        answer_sdp = mirror_opus_fmtp(answer_sdp, device_sdp)

        self._register_session()
        await self._signaling.send_answer(answer_sdp, session_id=self.session_id)
        # The server refuses candidates that reach it before it has processed
        # the answer ("Received ICE candidate before answer"); send_answer does
        # not wait for the server's ack, so leave it a moment.
        await asyncio.sleep(ANSWER_SETTLE)
        for m_line, candidate in candidates:
            await self._signaling.send_candidate(candidate, m_line, session_id=self.session_id)
        _LOGGER.debug(
            "Call %s: answer sent with %d local candidates (%d STUN servers)",
            self.session_id,
            len(candidates),
            len(ice_servers),
        )

    async def _wait_connected(self) -> None:
        connected = asyncio.ensure_future(self._connected.wait())
        ended = asyncio.ensure_future(self._ended.wait())
        try:
            await asyncio.wait({connected, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (connected, ended):
                task.cancel()
        if not self._connected.is_set():
            raise CallAnswerError("The call ended before the media session connected")

    async def _async_lifetime(self) -> None:
        """Keep the call until it ends on either side or reaches its maximum length."""
        try:
            await asyncio.wait_for(self._ended.wait(), timeout=self._max_duration)
        except TimeoutError:
            await self.async_hangup(reason="maximum duration reached")
            return
        if self._remote_ended:
            await self.async_hangup(reason="ended by the device", terminate=False)
        else:
            # The media dropped without a word from the device: free its slot.
            await self.async_hangup(reason="media lost")

    async def async_hangup(self, reason: str = "hung up", terminate: bool = True) -> None:
        """End the call: terminate the session on the device and close the media."""
        if self._closed:
            return
        self._closed = True
        self.state = "ended"
        self._ended.set()
        if self._lifetime_task and self._lifetime_task is not asyncio.current_task():
            self._lifetime_task.cancel()
        await self._log_media_stats()
        if terminate and not self._remote_ended:
            self._register_session()
            try:
                await self._signaling.send_terminate(session_id=self.session_id)
            except Exception as err:
                _LOGGER.debug("Call %s: terminate not sent: %s", self.session_id, err)
        for sink in self._sinks:
            sink.cancel()
        if self._peer is not None:
            with suppress(Exception):
                await self._peer.close()
        self._release_signaling()
        _LOGGER.info("Call %s ended (%s)", self.session_id, reason)
        for func in self._on_end_callbacks:
            with suppress(Exception):
                func()

    # --- signaling ---

    def _register_session(self) -> None:
        """Make sure the shared signaling client knows this call's session.

        A reconnect in between (idle connection, a server error) forgets it.
        """
        self._signaling.set_session_from_push(
            session_id=self.session_id,
            tag_id=self.call.get("tag_id") or "",
            correlation_id=str(self.call.get("correlation_id") or ""),
            device_id=self.call.get("device_id") or "",
        )

    def _take_signaling(self) -> None:
        """Route the signaling client's callbacks here; live views set their own again."""
        self._signaling._on_answer = self._on_signaling_answer
        self._signaling._on_candidate = self._on_signaling_candidate
        self._signaling._on_event = self._on_signaling_event

    def _release_signaling(self) -> None:
        if self._signaling._on_candidate == self._on_signaling_candidate:
            self._signaling._on_answer = None
            self._signaling._on_candidate = None
            self._signaling._on_event = None

    async def _on_signaling_answer(self, session_id: str, _sdp: str) -> None:
        _LOGGER.debug("Ignoring an SDP answer for session %s while answering a call", session_id)

    async def _on_signaling_candidate(self, session_id: str, candidate: dict[str, Any]) -> None:
        if session_id != self.session_id or self._closed:
            return
        self._remote_candidates.append(candidate)
        await self._apply_remote_candidates()

    async def _on_signaling_event(self, session_id: str, event_type: str, message: dict[str, Any]) -> None:
        if session_id != self.session_id:
            return
        error = (message.get("data") or {}).get("error") or {}
        _LOGGER.info(
            "Call %s: %s from the device%s",
            self.session_id,
            event_type,
            f" ({error.get('message')})" if error else "",
        )
        if event_type in ("terminate", "rescind"):
            self._remote_ended = True
            self._ended.set()

    async def _apply_remote_candidates(self) -> None:
        if not self._remote_described or self._peer is None:
            return
        from aiortc.sdp import candidate_from_sdp

        while self._remote_candidates:
            payload = self._remote_candidates.pop(0)
            raw = payload.get("candidate")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                candidate = candidate_from_sdp(raw.removeprefix("a=").removeprefix("candidate:"))
            except (ValueError, IndexError):
                _LOGGER.debug("Call %s: unreadable device candidate %r", self.session_id, raw)
                continue
            candidate.sdpMid = payload.get("sdp_mid") or payload.get("sdpMid")
            m_line = payload.get("sdp_m_line_index", payload.get("sdpMLineIndex", 0))
            candidate.sdpMLineIndex = int(m_line) if m_line is not None else 0
            await self._peer.addIceCandidate(candidate)

    # --- media ---

    async def _drain(self, track: Any) -> None:
        """Consume a device track (to be played to the user in a later step)."""
        from aiortc.mediastreams import MediaStreamError

        with suppress(MediaStreamError, asyncio.CancelledError):
            while True:
                await track.recv()

    async def _log_media_stats(self) -> None:
        """Log how much media flowed each way: proof the call was a real one."""
        if self._peer is None:
            return
        try:
            stats = await self._peer.getStats()
        except Exception:
            return
        flows = []
        for item in stats.values():
            kind = getattr(item, "kind", "?")
            if getattr(item, "type", "") == "inbound-rtp":
                flows.append(f"{kind} in={getattr(item, 'packetsReceived', '?')}")
            elif getattr(item, "type", "") == "outbound-rtp":
                flows.append(f"{kind} out={getattr(item, 'packetsSent', '?')}")
        if flows:
            _LOGGER.info("Call %s media packets: %s", self.session_id, ", ".join(sorted(flows)))
