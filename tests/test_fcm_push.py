"""Tests for Classe 300 EOS call offers through FCM."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.bticino_intercom.const import DOMAIN
from custom_components.bticino_intercom.fcm_push import FCM_CONFIG_FILE, async_start_fcm_listener, fcm_store

from .conftest import BRIDGE_MAC, EXTERNAL_UNIT_ID, HOME_ID, SESSION_ID

pytest.importorskip("pybticino.fcm")

BRIDGE_ID = "00:03:50:aa:bb:cc"
# Placeholders only: the real values belong to the Netatmo app and never go in the repo.
APP_CONFIG = {
    "api_key": "test-api-key",
    "app_id": "1:123456789:android:abcdef0123456789",
    "project_id": "test-project",
    "cert_sha1": "ABCDEF",
}


def _coordinator(bridge_type: str = "BNCX") -> SimpleNamespace:
    return SimpleNamespace(
        main_device_id=BRIDGE_ID,
        data={"modules": {BRIDGE_ID: {"id": BRIDGE_ID, "type": bridge_type}}},
        _handle_websocket_message=AsyncMock(),
    )


def _entry() -> SimpleNamespace:
    return SimpleNamespace(entry_id="entry1")


def _write_config(hass: HomeAssistant, content: str) -> None:
    with open(hass.config.path(FCM_CONFIG_FILE), "w", encoding="utf-8") as fh:
        fh.write(content)


@pytest.fixture(autouse=True)
def _remove_config_file(hass: HomeAssistant):
    """The config dir outlives the test: a file left behind would start real FCM elsewhere."""
    yield
    Path(hass.config.path(FCM_CONFIG_FILE)).unlink(missing_ok=True)


@pytest.fixture
def listener_cls():
    """Patch the pybticino listener; the instance records what it was built with."""
    instance = MagicMock()
    instance.async_start = AsyncMock()
    with patch("pybticino.fcm.FcmCallListener", return_value=instance) as cls:
        cls.instance = instance
        yield cls


async def test_not_started_for_other_bridges(hass: HomeAssistant, listener_cls) -> None:
    """A bridge that sends offers on the WebSocket needs no FCM."""
    _write_config(hass, json.dumps(APP_CONFIG))
    assert await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator("BNC1")) is None
    listener_cls.assert_not_called()


async def test_not_started_without_config_file(hass: HomeAssistant, listener_cls) -> None:
    """Without the user's file the integration runs as before."""
    assert await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator()) is None
    listener_cls.assert_not_called()


async def test_not_started_with_an_invalid_config_file(hass: HomeAssistant, listener_cls) -> None:
    _write_config(hass, json.dumps({"api_key": "only-this"}))
    assert await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator()) is None
    _write_config(hass, "{not json")
    assert await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator()) is None
    listener_cls.assert_not_called()


async def test_start_failure_does_not_raise(hass: HomeAssistant, listener_cls) -> None:
    _write_config(hass, json.dumps(APP_CONFIG))
    listener_cls.instance.async_start.side_effect = RuntimeError("PHONE_REGISTRATION_ERROR")
    assert await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator()) is None


async def test_started_with_saved_identity(hass: HomeAssistant, listener_cls) -> None:
    _write_config(hass, json.dumps(APP_CONFIG))
    await fcm_store(hass, "entry1").async_save({"android": True})
    account = MagicMock()

    listener = await async_start_fcm_listener(hass, _entry(), account, _coordinator())

    assert listener is listener_cls.instance
    args, kwargs = listener_cls.call_args
    assert args[0] is account
    assert args[1].app_id == APP_CONFIG["app_id"]
    assert kwargs["credentials"] == {"android": True}


async def test_only_rtc_pushes_reach_the_coordinator(hass: HomeAssistant, listener_cls) -> None:
    """The offer goes to the coordinator; status pushes already come from the WebSocket."""
    _write_config(hass, json.dumps(APP_CONFIG))
    coordinator = _coordinator()
    await async_start_fcm_listener(hass, _entry(), MagicMock(), coordinator)
    on_push = listener_cls.call_args[0][2]

    offer = {"push_type": "BNCX-rtc", "extra_params": {"session_id": "s1", "data": {"type": "offer"}}}
    on_push(offer)
    on_push({"push_type": "BNCX-incoming_call", "extra_params": {"event_type": "incoming_call"}})
    await hass.async_block_till_done()

    coordinator._handle_websocket_message.assert_awaited_once_with(offer)


async def test_eos_offer_from_fcm_enables_answer_mode(hass: HomeAssistant, mock_setup_entry_eos) -> None:
    """An FCM BNCX-rtc offer leaves the device SDP where the camera's answer mode looks for it."""
    from pybticino.fcm import parse_push

    coordinator = hass.data[DOMAIN][mock_setup_entry_eos.entry_id]["coordinator"]
    # FCM delivers extra_params as JSON text.
    push = parse_push(
        {
            "push_type": "BNCX-rtc",
            "extra_params": json.dumps(
                {
                    "session_id": SESSION_ID,
                    "tag_id": "tag-1",
                    "device_id": BRIDGE_MAC,
                    "home_id": HOME_ID,
                    "correlation_id": "65686571418260619",
                    "data": {
                        "type": "offer",
                        "session_description": {"type": "call", "sdp": "v=0\r\n", "module_id": EXTERNAL_UNIT_ID},
                    },
                }
            ),
        }
    )

    await coordinator._handle_websocket_message(push)

    call = coordinator.active_call
    assert call["sdp"] == "v=0\r\n"
    assert call["module_id"] == EXTERNAL_UNIT_ID
    assert call["tag_id"] == "tag-1"
    assert not call.get("status_only")


async def test_new_identity_is_saved(hass: HomeAssistant, listener_cls) -> None:
    _write_config(hass, json.dumps(APP_CONFIG))
    await async_start_fcm_listener(hass, _entry(), MagicMock(), _coordinator())
    on_credentials = listener_cls.call_args.kwargs["on_credentials_updated"]

    on_credentials({"android": True, "fcm": {"registration": {"token": "tok"}}})
    await hass.async_block_till_done()

    assert await fcm_store(hass, "entry1").async_load() == {"android": True, "fcm": {"registration": {"token": "tok"}}}
