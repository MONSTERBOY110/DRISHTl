"""
Unit tests for face-to-person association (B6).

Pure logic tests verifying TRD §4 rules:
1. Face center must lie in the upper 45% of the person bounding box.
2. If multiple person boxes qualify, the highest IoU wins.
3. Each person gets at most one face (highest det_score wins).
"""

from src.backend.v2.associate import assign_faces, _centre_in_upper


def test_centre_in_upper():
    person_box = (100, 100, 200, 300)  # w=100, h=200, upper 45% is y: 100..190

    # Face center at (150, 140) -> inside upper 45%
    face_valid = (140, 130, 160, 150)
    assert _centre_in_upper(face_valid, person_box, 0.45) is True

    # Face center at (150, 220) -> lower region (below upper 45%)
    face_low = (140, 210, 160, 230)
    assert _centre_in_upper(face_low, person_box, 0.45) is False

    # Face center outside horizontal bounds
    face_out_x = (80, 130, 95, 150)
    assert _centre_in_upper(face_out_x, person_box, 0.45) is False


def test_assign_faces_single_person():
    person = (1, (100, 100, 200, 300), 0.9)
    face = {"bbox": [130, 110, 170, 150], "det_score": 0.85, "name": "face1"}

    res = assign_faces([face], [person])
    assert 1 in res
    assert res[1]["name"] == "face1"


def test_assign_faces_multiple_faces_highest_det_score():
    person = (1, (100, 100, 200, 300), 0.9)
    face_low_score = {"bbox": [130, 110, 170, 150], "det_score": 0.65, "id": "low"}
    face_high_score = {"bbox": [135, 115, 165, 145], "det_score": 0.92, "id": "high"}

    res = assign_faces([face_low_score, face_high_score], [person])
    assert len(res) == 1
    assert res[1]["id"] == "high"


def test_assign_faces_multiple_persons_best_iou():
    # Two overlapping person boxes where face falls in upper region of both
    p1 = (1, (100, 100, 200, 300), 0.9)  # larger overlap
    p2 = (2, (120, 100, 300, 300), 0.9)

    # Face positioned closer to center of p1
    face = {"bbox": [110, 110, 150, 150], "det_score": 0.88}

    res = assign_faces([face], [p1, p2])
    assert len(res) == 1
    assert 1 in res


def test_assign_faces_no_match():
    # Person box far away from face
    person = (1, (500, 500, 600, 700), 0.9)
    face = {"bbox": [100, 100, 150, 150], "det_score": 0.90}

    res = assign_faces([face], [person])
    assert res == {}

