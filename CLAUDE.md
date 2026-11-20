# CLAUDE.md

Home Assistant custom integration for BTicino Classe 100X/300X, Classe 300 EOS and EasyKit intercoms on the Netatmo cloud (Home + Security app). The cloud protocol lives in [pybticino](https://github.com/k-the-hidden-hero/pybticino), checked out next to this repo at `../pybticino`.

## Where code goes

Protocol knowledge belongs in pybticino: payloads, message parsing, firmware quirks, what an error means. This repo holds the Home Assistant side: entities, the coordinator, storage, the config flow and the lifecycle. Protocol code that still lives here is moving in the steps of k-the-hidden-hero/pybticino#31. Write new protocol work in pybticino, as new API first; the integration switches to it afterwards.

## Commands

- Tests: `.venv/bin/python -m pytest` (about 2.5 minutes for the full suite). The local `.venv` installs pybticino in editable mode from `../pybticino`, so library changes are live in the integration's tests.
- Lint and format: `uvx ruff@0.16.7 check .` and `uvx ruff@0.16.7 format --check .`. CI pins ruff 0.16.7 on purpose (see the comment in `.github/workflows/ci.yaml`); bump it in its own commit together with the reformat it causes.
- HA validation: hassfest and HACS run in CI. Locally: `docker run --rm -v "$PWD":/github/workspace ghcr.io/home-assistant/hassfest`.
- Dev instance: `docker-compose.dev.yml` runs HA stable with the component mounted from this checkout.

`pytest-homeassistant-custom-component` still pins HA 2026.10, so the tests that need 2026.11 APIs skip themselves (`tests/test_webrtc_two_way_audio.py`).

## Architecture

All code is in `custom_components/bticino_intercom/`.

- `__init__.py` sets up the entry: `AuthHandler`, `AsyncAccount`, the push `WebsocketClient` (`/ws/`), the `SignalingClient` for WebRTC (`/appws/`), and the coordinator. It runs its own push reconnect loop with backoff, and registers the services (`reject_call`, and `inject_test_event` when the logger is at debug).
- `coordinator.py` polls every 5 minutes (topology, home status, events) and handles pushes: RTC `call`/`rescind`/`terminate` and the `incoming_call`/`missed_call`/`accepted_call` status pushes. It fires `bticino_intercom_call` bus events and dispatcher signals.
- `auth.py`: a rejected token is dropped and the password login is retried once; rejected credentials raise `ConfigEntryAuthFailed`, which starts reauth.
- `camera.py`: one WebRTC camera per external unit (`BNEU`), plus Call Home, which sends the same offer to the bridge's own module id and rings the indoor monitor. Offer mode for live view, answer mode for an incoming call's offer.
- Entities: `lock` (`BNDL`), call `binary_sensor` and doorbell `event` (`BNEU`), `light` (`BNSL`/`BNVL`), `sensor`, and `select`/`switch`/`number` for `BNC3` settings. Entities carry a `module_id` attribute; the camera and doorbell of one unit share it, and cards pair them on it.
- `history.py` and `media_source.py` keep the call history on disk; `websocket_api.py` serves it to cards as `bticino_intercom/history`.
- `fcm_push.py` receives the Classe 300 EOS call offers through FCM, with the user's local `bticino_intercom_fcm.json` (the Netatmo app's Firebase identity: never commit it or put it in code).
- `call_answerer.py` answers Classe 300 EOS (`ANSWER_BRIDGE_TYPES`) calls: services `answer_call`/`hangup_call` and the Answer call / Hang up call buttons. With no view open, Home Assistant is the media peer (aiortc; the answer is prepared when the offer arrives, nothing is sent until `answer_call`). When a card's view has answered the ring (the camera's answer mode, `async_bridge_started`), that is only the preview: the call is not marked answered and keeps ringing. The card's explicit answer, `bticino_intercom/call/unmute` (or `answer_call` with `from_view`), sends the unmute on that session, once; `webrtc_connected` takes nothing.

## Device facts

What the config and the code don't tell you:

- The module subtype comes from `variant`, or from `type` when `variant` is missing (EasyKit, EOS `BNCX`). The 344742 staircase light reports `BNVL`. Use `utils.get_module_subtype`.
- Netatmo sends one copy of each push per device registered on the account, so pushes are deduplicated (#106). The push socket belongs to the account, so pushes are filtered by home (#95).
- Snapshot and vignette URLs expire within minutes, so the history downloads the images as soon as `incoming_call` arrives.
- The bridge has very few peer slots. Every WebRTC session ends with `send_terminate(session_id)` naming its own session; without an id, pybticino terminates the most recent session, which may belong to another camera.
- BNC1 firmware reboots when an offer's SDP grows past about 8 KB, so the offer is trimmed to H264. The filter removes payload types and keeps their numbers, so the device's answer stays valid.
- The device uses Chrome's payload type numbers (111 Opus, 109 H264) whatever the SDP says, so live video works in Chrome/Chromium only. Analysis: `docs/firefox-webrtc-investigation.md`.
- HA 2026.11 native two-way audio: the talk button closes the stream and sends a new offer with the mic track, so closing a session must free its peer slot before the next offer goes out.
- How the device microphone is switched on is unsettled: pybticino `docs/webrtc-audio.md` and `camera.py` disagree (pybticino#31, step 4).
- EasyKit `BDIY`: an incoming call's offer arrives only as an FCM push, and the call becomes accepted after one `{"type":"player","value":"unmute"}` sent once the peer connection is `connected`. `unmute` toggles, so a second one mutes the call. Work in progress in #72.
- Classe 300 EOS `BNCX` (traced from the Android app 4.1.1.3): the offer arrives only through FCM. An SDP answer to it is only the preview: the cloud tells the other phones someone answered, but the indoor monitor keeps ringing and the call ends as missed after ~27 s. One `player`/`unmute` on the call's session takes it: the device stops ringing and reports `accepted_call`. The EOS offers video first (mids `video0`/`audio1`), so each side gets a real answer to its own offer (`answer_for_device`/`answer_for_browser`) instead of the other side's offer. aiortc needs a wider DTLS cipher list for the device's RSA certificate.

## Workflow

- Conventional commits (`fix:`, `feat:`, `chore:`, `test:`, `docs:`, `ci:`). PRs merge with a merge commit.
- Release: bump `manifest.json` (`2.0.5-rc1`) and `pyproject.toml` (`2.0.5rc1`) in a `chore: bump version to X` commit on main, then push a signed `vX` tag. `release.yaml` builds the zip and marks `-rc` tags as prereleases. Replace the generated notes with hand-written ones (`gh release edit --notes-file`). RC tags stay on GitHub.
- When the integration needs a new pybticino release, raise the requirement in `manifest.json` (`pybticino>=…`) in the same change.
