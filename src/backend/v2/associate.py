"""
Face → person association.

Rule (TRD §4): a face belongs to a person box when the face centre lies inside
the upper 45 % of that box; if several person boxes qualify, the largest IoU
between the face box and the person box wins. Each person gets at most one
face (the one with the highest det_score), so a track never stores a
bystander's face just because the boxes overlapped.
"""

from typing import Any, Dict, List, Tuple

from src.backend.vision.tracker import iou

Box = Tuple[int, int, int, int]


def _centre_in_upper(face_box: Box, person_box: Box, upper_frac: float = 0.45) -> bool:
    fx = (face_box[0] + face_box[2]) / 2.0
    fy = (face_box[1] + face_box[3]) / 2.0
    px1, py1, px2, py2 = person_box
    return px1 <= fx <= px2 and py1 <= fy <= py1 + (py2 - py1) * upper_frac


def assign_faces(faces: List[Dict[str, Any]],
                 persons: List[Tuple[int, Box, float]]) -> Dict[int, Dict[str, Any]]:
    """{person_track_id: face_dict} for faces that sit in a person's head region."""
    out: Dict[int, Dict[str, Any]] = {}
    for face in faces:
        fbox = tuple(int(v) for v in face["bbox"])
        best_id, best_iou = None, -1.0
        for track_id, pbox, _conf in persons:
            if not _centre_in_upper(fbox, pbox):
                continue
            score = iou(fbox, pbox)
            if score > best_iou:
                best_id, best_iou = track_id, score
        if best_id is None:
            continue
        current = out.get(best_id)
        if current is None or face.get("det_score", 0.0) > current.get("det_score", 0.0):
            out[best_id] = face
    return out
