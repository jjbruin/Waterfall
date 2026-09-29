"""Is freezing switched on? One answer, asked by every path that can freeze.

WHY THIS EXISTS. On Sep 29 2026 the all-investors batch was run against
production as a SINGLE request covering ~145 investors, and the app was
unavailable for about 35 minutes. The freeze itself is correct — the rows were
right and were cleanly unfrozen afterwards — but one request that assembles
every investor's report is the wrong shape for this job. Freezing stays OFF
until it runs as a background job.

DEFAULT OFF, and that direction is the point. A flag that defaults ON protects
nobody: the day somebody deploys without setting it, the buttons are live again
with nothing saying so. ``FREEZE_ENABLED`` must be set explicitly, to a value
that plainly means yes, before anything can freeze.

ENFORCED AT THE CORE, NOT ONLY AT THE DOORS. ``freeze_part`` itself raises, so
every caller is covered — the two batch buttons, the published-overlay freeze,
re-freeze, the Portfolio Snapshot approval chain and the One Pager approval —
including any entry point added after this was written. The endpoints check it
too, but only so the refusal is a clean 503 with a sentence a person can read,
rather than a 500 from an exception escaping.

WHAT IT DOES NOT TOUCH. Unfreeze, reads of already-frozen quarters, and the
approval chain itself. An approval still completes and still advances the
workflow; it simply does not freeze on the way through, which the approval path
already tolerates (it wraps the freeze precisely so a failure cannot cost
somebody their approval).
"""
from __future__ import annotations

import os

#: The one sentence every refusal uses, so the screen, the API and the logs all
#: say the same thing.
FREEZE_DISABLED_MESSAGE = (
    "Freezing is temporarily disabled. It is switched back on once the freeze "
    "runs as a background job rather than a single request."
)

#: Values that mean yes. Anything else — including unset, '', '0', 'false' and
#: any typo — means no, because this fails CLOSED.
_TRUE = frozenset({"1", "true", "yes", "on"})


class FreezeDisabled(RuntimeError):
    """Raised by the freeze core when freezing is switched off."""

    def __init__(self, message: str = FREEZE_DISABLED_MESSAGE):
        super().__init__(message)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUE


def freeze_enabled() -> bool:
    """True only when freezing has been switched on explicitly.

    Prefers the Flask config so the flag can be flipped per environment, and
    falls back to the environment variable so the guardrails and the offline
    scripts — which have no application context — get the same answer rather
    than crashing or silently defaulting the other way.
    """
    try:
        from flask import current_app
        if current_app:
            return _truthy(current_app.config.get("FREEZE_ENABLED", False))
    except Exception:
        pass
    return _truthy(os.environ.get("FREEZE_ENABLED"))


def require_freeze_enabled() -> None:
    """Raise :class:`FreezeDisabled` unless freezing is switched on."""
    if not freeze_enabled():
        raise FreezeDisabled()
