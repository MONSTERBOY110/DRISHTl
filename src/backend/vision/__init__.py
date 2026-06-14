"""
Drishti vision package.

Modular face-search pipeline that replaces the legacy Haar Cascade + ResNet18
recognizer with an open-set InsightFace pipeline:

    detect -> quality gate -> aligned ArcFace embedding -> IoU tracking ->
    track-level open-set decision (strong / possible / no reliable match)

All tunable thresholds live in `config.py`. The recognizer/detector backends
are selectable so AdaFace / SCRFD can be added later without touching callers.
"""
