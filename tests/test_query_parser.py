"""
Tests for query_parser.py — free-text description parser covering clothing colours,
outfit types, facial & head features (beard, mustache, glasses, hair style/color, cap, mask),
physical features (age, gender, build), and English/Hindi/Bengali terms.
"""

from src.backend.v2 import query_parser


def test_parse_english_clothing_and_features():
    desc = "child, 6 years old, yellow top, blue jeans, black cap, short black hair, glasses"
    p = query_parser.parse(desc)
    assert "yellow" in p["upper"]
    assert "blue" in p["lower"]
    assert "glasses" in p["facial_features"]
    assert "short black hair" in p["facial_features"]
    assert "cap" in p["facial_features"]
    assert "child" in p["physical_features"]
    assert p["age_hint"] == "6 years"
    assert "black" not in p["upper"]
    assert "yellow top" in p["clip_text"]
    assert "blue jeans" in p["clip_text"]


def test_parse_romanised_hindi():
    desc = "chota baccha, 8 saal, laal kurta, neela pant, chashma, mooch, dhadhi"
    p = query_parser.parse(desc)
    assert "red" in p["upper"]
    assert "blue" in p["lower"]
    assert "glasses" in p["facial_features"]
    assert "beard" in p["facial_features"]
    assert "mustache" in p["facial_features"]
    assert "child" in p["physical_features"]
    assert p["age_hint"] == "8 years"


def test_parse_romanised_bengali():
    desc = "khokhababu, holud frock, kalo chul, choshma, dadhi, gof"
    p = query_parser.parse(desc)
    assert "yellow" in p["upper"]
    assert "glasses" in p["facial_features"]
    assert "beard" in p["facial_features"]
    assert "mustache" in p["facial_features"]
    assert "black hair" in p["facial_features"]
    assert "child" in p["physical_features"]
    assert "black" not in p["upper"]


def test_parse_colour_after_garment_word():
    p = query_parser.parse("shirt yellow, jeans blue, cap black")
    assert "yellow" in p["upper"]
    assert "blue" in p["lower"]
    assert "black cap" in p["keywords"]


def test_parse_empty_and_none():
    p1 = query_parser.parse("")
    assert p1["upper"] == []
    assert p1["lower"] == []
    assert p1["keywords"] == []
    assert p1["facial_features"] == []
    assert p1["physical_features"] == []
    assert p1["clip_text"] == "a person"

    p2 = query_parser.parse(None)
    assert p2["clip_text"] == "a person"

