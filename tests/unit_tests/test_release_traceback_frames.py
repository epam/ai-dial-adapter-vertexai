import gc
import weakref

import pytest

from aidial_adapter_vertexai.dial_api.exceptions import release_traceback_frames


class _Payload:
    pass


def _fail(payload_ref: list[weakref.ref]) -> None:
    payload = _Payload()
    payload_ref.append(weakref.ref(payload))
    state = {}
    try:
        raise ValueError("upstream error")
    except ValueError as e:
        # frame -> state -> exception -> traceback -> frame,
        # like the tenacity retry state does
        state["exception"] = e
        raise


@pytest.mark.parametrize("release", [True, False])
def test_release_traceback_frames(release: bool):
    payload_ref: list[weakref.ref] = []

    gc.disable()
    try:
        try:
            _fail(payload_ref)
        except ValueError as e:
            if release:
                release_traceback_frames(e)

        is_released = payload_ref[0]() is None
    finally:
        gc.enable()
        gc.collect()

    assert is_released == release
