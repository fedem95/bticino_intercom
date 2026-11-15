"""Buttons to answer and hang up a call from Home Assistant."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .call_answerer import async_answer_call, async_hangup_call
from .const import DOMAIN
from .coordinator import BticinoIntercomCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Add the answer and hang-up buttons to the bridge."""
    coordinator: BticinoIntercomCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    if not coordinator.main_device_id:
        return
    async_add_entities([BticinoAnswerCallButton(coordinator), BticinoHangupCallButton(coordinator)])


class _BticinoCallButton(ButtonEntity):
    """A call control on the bridge device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: BticinoIntercomCoordinator, key: str) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{key}"

    @property
    def device_info(self) -> DeviceInfo:
        """Return the bridge device."""
        return DeviceInfo(identifiers={(DOMAIN, self._coordinator.main_device_id)})


class BticinoAnswerCallButton(_BticinoCallButton):
    """Answer the ringing call: Home Assistant joins it as a real peer."""

    _attr_name = "Answer call"
    _attr_icon = "mdi:phone-in-talk"

    def __init__(self, coordinator: BticinoIntercomCoordinator) -> None:
        super().__init__(coordinator, "answer_call")

    async def async_press(self) -> None:
        """Answer the call."""
        await async_answer_call(self.hass, self._coordinator.entry.entry_id)


class BticinoHangupCallButton(_BticinoCallButton):
    """Hang up the call answered from Home Assistant."""

    _attr_name = "Hang up call"
    _attr_icon = "mdi:phone-hangup"

    def __init__(self, coordinator: BticinoIntercomCoordinator) -> None:
        super().__init__(coordinator, "hangup_call")

    async def async_press(self) -> None:
        """Hang up."""
        await async_hangup_call(self.hass, self._coordinator.entry.entry_id)
