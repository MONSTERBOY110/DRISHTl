"""
Tests for appearance.py — embeddings, OpenCLIP integration, and color attributes.
"""

import numpy as np
from src.backend.v2 import appearance


def test_embed_text_shape_and_norm():
    vec = appearance.embed_text("yellow top, blue jeans, glasses, beard")
    assert vec.shape == (appearance.DIM,)
    assert vec.dtype == np.float32
    norm = np.linalg.norm(vec)
    if norm > 0:
        assert np.isclose(norm, 1.0, atol=1e-4)


def test_embed_images_shape_and_norm():
    crop1 = np.full((100, 50, 3), (0, 255, 255), dtype=np.uint8)  # yellow
    crop2 = np.full((100, 50, 3), (255, 0, 0), dtype=np.uint8)    # blue
    mat = appearance.embed_images([crop1, crop2])
    assert mat.shape == (2, appearance.DIM)
    assert mat.dtype == np.float32
    for row in mat:
        n = np.linalg.norm(row)
        if n > 0:
            assert np.isclose(n, 1.0, atol=1e-4)


def test_color_attributes():
    crop = np.full((100, 50, 3), (0, 255, 255), dtype=np.uint8)  # yellow
    attrs = appearance.color_attributes(crop)
    assert "upper" in attrs and "lower" in attrs
    assert isinstance(attrs["upper"], str)
    assert isinstance(attrs["lower"], str)

