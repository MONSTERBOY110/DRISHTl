"""
v2 package entry point: ``register_v2(app)``.

Why a single hook: the baseline ``app.py`` mounts ``Frontend/`` at "/" as its
last statement, and any router included after that mount is unreachable
(TRD §2 quirk 1). ``app.py`` therefore calls ``register_v2(app)`` once, right
before the mount, and this function pulls in every v2 router, so no teammate
ever needs to touch ``app.py`` again.

Routers owned by M2/M4 are optional while they are being written: a file that
does not exist yet is skipped quietly, but a file that exists and fails to
import is logged as an error with the traceback, so a bug never hides behind a
silent 404.
"""

import importlib
import logging
import threading

from fastapi import FastAPI

from . import config

logger = logging.getLogger(__name__)

# (module, attribute, owner/task) — registered in this order when present.
_OPTIONAL_ROUTERS = (
    ("src.backend.v2.routes_cases", "router", "M4 D2"),
    ("src.backend.v2.routes_alerts", "router", "M4 D3"),
    ("src.backend.v2.routes_intake", "router", "M2 B7"),
)
_WATCHLIST = ("src.backend.v2.watchlist", "start", "M4 D4")


def _import_optional(module_name: str, owner: str):
    """Import a teammate's module, or None. Only a *missing file* is silent."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            logger.info(f"v2: {module_name} not present yet ({owner}); skipping")
            return None
        # The module exists but imports something that is not installed.
        logger.error(f"v2: {module_name} ({owner}) needs missing module '{e.name}'", exc_info=True)
        return None
    except Exception:
        logger.error(f"v2: {module_name} ({owner}) failed to import", exc_info=True)
        return None


def _include_optional_router(app: FastAPI, module_name: str, attr: str, owner: str) -> bool:
    mod = _import_optional(module_name, owner)
    if mod is None:
        return False
    router = getattr(mod, attr, None)
    if router is None:
        logger.error(f"v2: {module_name} ({owner}) has no '{attr}'")
        return False
    app.include_router(router)
    logger.info(f"v2: registered {module_name} ({owner})")
    return True


def _start_optional_watchlist() -> None:
    module_name, attr, owner = _WATCHLIST
    mod = _import_optional(module_name, owner)
    if mod is None:
        return
    start = getattr(mod, attr, None)
    if start is None:
        logger.error(f"v2: {module_name} ({owner}) has no '{attr}()'")
        return
    try:
        start()
        logger.info(f"v2: watchlist started ({owner})")
    except Exception:
        logger.error(f"v2: watchlist.start() failed ({owner})", exc_info=True)


def _warm_face_model() -> None:
    """Load InsightFace and run one dummy detection in a daemon thread, so the first real
    search doesn't pay the ~2 s model load plus cuDNN's first-inference autotune."""
    def _run():
        try:
            import numpy as np
            from src.backend.vision import detector
            detector.extract_faces(np.zeros((320, 320, 3), dtype=np.uint8))
            logger.info("v2: face model warm")
        except Exception as e:  # CPU-only box without insightface: searches still work by description
            logger.warning(f"v2: face model warm-up skipped: {e}")

    threading.Thread(target=_run, name="v2-face-warmup", daemon=True).start()


def register_v2(app: FastAPI) -> None:
    """Include every v2 router and start background services. Call BEFORE the static mount."""
    for d in (config.DATA_DIR, config.CROPS_DIR, config.FRAMES_DIR, config.CASES_DIR, config.UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    from . import store  # M1: creates data/v2/drishti_v2.db with the TRD §5 schema
    store.get_store().init_db()

    from .routes_core import router as core_router  # M1, always present
    app.include_router(core_router)

    for module_name, attr, owner in _OPTIONAL_ROUTERS:
        _include_optional_router(app, module_name, attr, owner)

    _start_optional_watchlist()
    if config.WARM_FACE_MODEL:
        _warm_face_model()
