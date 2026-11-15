"""Classe 300 EOS call offers through Firebase Cloud Messaging.

A Classe 300 EOS (bridge type BNCX) never sends its call offer on the push
WebSocket: the ``BNCX-rtc`` push with the device SDP only reaches FCM push
contexts registered as the Netatmo Android app. Without that offer the
camera cannot answer a ring (answer mode in camera.py needs the device SDP).

The app's Firebase identity is not shipped with this integration. It is read
from ``bticino_intercom_fcm.json`` in the Home Assistant config directory,
written by the user (see pybticino's ``FcmAppConfig.from_apk``). Without that
file the integration works as before: status pushes only, no answering.

Only ``-rtc`` pushes are forwarded to the coordinator: the status pushes
(incoming_call, missed_call...) already arrive on the WebSocket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from pybticino import AsyncAccount
from pybticino.exceptions import PyBticinoException

from .const import DOMAIN

if TYPE_CHECKING:
    from pybticino.fcm import FcmCallListener

    from .coordinator import BticinoIntercomCoordinator

_LOGGER = logging.getLogger(__name__)

FCM_CONFIG_FILE = "bticino_intercom_fcm.json"
FCM_STORAGE_VERSION = 1
FCM_BRIDGE_TYPES = ("BNCX",)


def fcm_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Return the store holding this entry's FCM identity (private: it receives the pushes)."""
    return Store(hass, FCM_STORAGE_VERSION, f"{DOMAIN}.fcm.{entry_id}", private=True)


def _bridge_type(coordinator: BticinoIntercomCoordinator) -> str | None:
    bridge_id = coordinator.main_device_id
    modules = (coordinator.data or {}).get("modules", {})
    return modules.get(bridge_id, {}).get("type") if bridge_id else None


def _read_app_config(path: Path) -> dict[str, Any] | None:
    """Read the user's Firebase config file; None when there is none (blocking I/O)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


async def async_start_fcm_listener(
    hass: HomeAssistant,
    entry: ConfigEntry,
    account: AsyncAccount,
    coordinator: BticinoIntercomCoordinator,
) -> FcmCallListener | None:
    """Start receiving call offers through FCM, when this bridge needs it and it is configured.

    Never raises: without FCM the integration keeps working on status pushes.
    """
    bridge_type = _bridge_type(coordinator)
    if bridge_type not in FCM_BRIDGE_TYPES:
        return None

    path = Path(hass.config.path(FCM_CONFIG_FILE))
    try:
        raw_config = await hass.async_add_executor_job(_read_app_config, path)
    except (OSError, ValueError) as err:
        _LOGGER.warning("Could not read %s: %s; incoming calls cannot be answered", path, err)
        return None
    if raw_config is None:
        _LOGGER.info(
            "Bridge %s sends call offers through FCM only; add %s to answer incoming calls from Home Assistant",
            bridge_type,
            FCM_CONFIG_FILE,
        )
        return None

    try:
        from pybticino.fcm import FcmAppConfig, FcmCallListener
    except ImportError as err:
        _LOGGER.warning("FCM support is not installed (%s); incoming calls cannot be answered", err)
        return None

    try:
        app_config = FcmAppConfig.from_dict(raw_config)
    except PyBticinoException as err:
        _LOGGER.warning("%s is not valid: %s", FCM_CONFIG_FILE, err)
        return None

    store = fcm_store(hass, entry.entry_id)
    credentials = await store.async_load()

    @callback
    def _on_push(push: dict[str, Any]) -> None:
        if not str(push.get("push_type", "")).endswith("-rtc"):
            return
        rtc_type = push.get("extra_params", {}).get("data", {}).get("type")
        _LOGGER.debug("FCM %s push (%s)", push.get("push_type"), rtc_type)
        hass.async_create_task(
            coordinator._handle_websocket_message(push),
            f"{DOMAIN} FCM push - {entry.entry_id}",
        )

    @callback
    def _on_credentials(new_credentials: dict[str, Any]) -> None:
        hass.async_create_task(store.async_save(new_credentials), f"{DOMAIN} FCM credentials save")

    listener = FcmCallListener(
        account,
        app_config,
        _on_push,
        credentials=credentials,
        on_credentials_updated=_on_credentials,
    )
    try:
        await listener.async_start()
    except Exception as err:
        _LOGGER.warning("Could not start FCM call offers: %s; incoming calls cannot be answered", err)
        return None
    _LOGGER.info("Receiving call offers through FCM for bridge %s", coordinator.main_device_id)
    return listener
