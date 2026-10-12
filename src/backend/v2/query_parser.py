"""
Free-text description → structured query parser.

parse(text) -> {
    "upper": [colour...],
    "lower": [colour...],
    "keywords": [...],
    "facial_features": [...],
    "physical_features": [...],
    "age_hint": str | None,
    "clip_text": str
}

Understands English, romanised Hindi (laal, neela, chashma, mooch, dhadhi, etc.),
and romanised Bengali (holud, sada, choshma, dadhi, gof, etc.). Supports parsing
clothing colours, outfit types, facial & head features (hair color/style, beard,
mustache, glasses, mask, cap), and physical attributes (age, gender, build).
"""

import re
from typing import Any, Dict, List, Optional

# canonical colour -> surface forms (English, romanised Hindi, romanised Bengali)
COLOR_WORDS: Dict[str, List[str]] = {
    "red": ["red", "laal", "lal", "maroon"],
    "blue": ["blue", "neela", "nila", "nil", "navy"],
    "yellow": ["yellow", "peela", "pila", "holud", "halud"],
    "green": ["green", "hara", "sabuj", "sobuj"],
    "black": ["black", "kala", "kaala", "kalo"],
    "white": ["white", "safed", "sada", "shada"],
    "pink": ["pink", "gulabi", "golapi"],
    "orange": ["orange", "narangi", "komola", "saffron", "kesari"],
    "purple": ["purple", "violet", "baingani", "beguni"],
    "brown": ["brown", "bhura", "khoyeri", "khaki"],
    "gray": ["gray", "grey", "silver", "dhusar"],
}

UPPER_WORDS = [
    "shirt", "t-shirt", "tshirt", "tee", "top", "kurta", "kurti", "frock", "dress", "jacket",
    "hoodie", "sweater", "sweatshirt", "saree", "sari", "blouse", "kameez", "punjabi", "coat", "jersey", "suit"
]

LOWER_WORDS = [
    "jeans", "jean", "pant", "pants", "trousers", "trouser", "shorts", "skirt", "pyjama", "pajama",
    "salwar", "dhoti", "lungi", "leggings", "capri", "bottom"
]

ACCESSORY_WORDS = [
    "cap", "hat", "bag", "backpack", "glasses", "spectacles", "chashma", "choshma", "goggles", "specs", "mask", "clip",
    "umbrella", "watch", "scarf", "dupatta", "shoes", "sandals", "balloon", "topi", "tupi", "jhola", "turban", "pagdi"
]
HAIR_WORDS = ["hair", "baal", "chul"]

FACIAL_FEATURE_MAP: Dict[str, List[str]] = {
    "glasses": ["glasses", "spectacles", "chashma", "choshma", "goggles", "specs", "goggle", "sunglasses", "chasma"],
    "beard": ["beard", "bearded", "dhadhi", "dadi", "dadhi"],
    "mustache": ["mustache", "mustaches", "moustaches", "mooch", "mouch", "gof", "goph"],
    "mask": ["mask", "facemask", "face mask"],
    "short black hair": ["short black hair", "short-black-hair"],
    "long black hair": ["long black hair", "long-black-hair"],
    "black hair": ["black hair", "dark hair", "black-hair", "dark-hair", "kalabaal", "kala baal", "kalo-chul", "kalo chul"],
    "white hair": ["white hair", "gray hair", "grey hair", "white-hair", "gray-hair", "grey-hair", "safed-baal", "safed baal", "sada-chul", "sada chul"],
    "long hair": ["long hair", "long-hair", "lamba-baal", "lamba baal", "lomba-chul", "lomba chul"],
    "short hair": ["short hair", "short-hair", "chota-baal", "chota baal", "choto-chul", "choto chul"],
    "bald": ["bald", "ganja", "tak"],
    "cap": ["cap", "hat", "topi", "tupi", "turban", "pagri", "pagdi"]
}

PHYSICAL_FEATURE_MAP: Dict[str, List[str]] = {
    "child": ["child", "kid", "boy", "girl", "baby", "toddler", "bachcha", "baccha", "bachha", "chhele", "meye", "ladka", "ladki", "khokhababu"],
    "elderly": ["elderly", "old", "grandfather", "grandmother", "dada", "dadi", "dadu", "thakuma", "didima", "nana", "nani", "buzurg", "uncle", "aunt", "aunty"],
    "tall": ["tall", "lamba", "denga"],
    "short": ["short", "naata", "chota"],
    "slim": ["thin", "slim", "patla", "skinny"],
    "heavy": ["fat", "heavy", "mota"]
}

_COLOR_LOOKUP = {w: c for c, ws in COLOR_WORDS.items() for w in ws}

_AGE_WORDS = {
    "child": ["child", "kid", "boy", "girl", "baby", "toddler", "bachcha", "baccha", "bachha", "chhele", "meye", "ladka", "ladki", "khokhababu"],
    "elderly": ["elderly", "old", "grandfather", "grandmother", "dada", "dadi", "dadu", "thakuma", "didima", "nana", "nani", "buzurg"],
}
_AGE_LOOKUP = {w: a for a, ws in _AGE_WORDS.items() for w in ws}
_AGE_NUM = re.compile(r"\b(\d{1,2})\s*(?:years?|yrs?|y/?o|saal|bochor|bachar)?\b")
_WINDOW = 3


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)?|\d+", text.lower())


def parse(text: Optional[str]) -> Dict[str, Any]:
    text = (text or "").strip()
    lower_text = text.lower()
    toks = _tokens(text)
    upper: List[str] = []
    lower: List[str] = []
    keywords: List[str] = []
    facial_features: List[str] = []
    physical_features: List[str] = []
    unassigned: List[str] = []
    accessory_colours: Dict[int, List[str]] = {}

    # Clean text of age expressions (e.g. '6 years old') and hair descriptions (e.g. 'short black hair')
    # so they do not falsely trigger general physical traits like 'old' (elderly) or 'short'
    cleaned_physical = re.sub(r"\b\d{1,2}\s*(?:years?|yrs?|y/?o|saal|bochor|bachar)\s*(?:old)?\b", "", lower_text)
    cleaned_physical = re.sub(r"\b(?:short|long|black|white|dark)\s+(?:black\s+)?(?:hair|baal|chul)\b", "", cleaned_physical)

    # Parse facial features directly from lower_text
    for feature, words in FACIAL_FEATURE_MAP.items():
        if any(w in lower_text for w in words):
            if feature not in facial_features:
                facial_features.append(feature)
            if feature not in keywords:
                keywords.append(feature)

    # Parse physical features using cleaned_physical
    for feature, words in PHYSICAL_FEATURE_MAP.items():
        if any(w in cleaned_physical for w in words):
            if feature not in physical_features:
                physical_features.append(feature)

    # Determine default phrasing orientation:
    # "yellow top, blue jeans" -> color before garment
    # "shirt yellow, jeans blue" -> garment before color (post_garment_mode)
    first_col_idx = next((idx for idx, t in enumerate(toks) if _COLOR_LOOKUP.get(t)), None)
    first_garm_idx = next((idx for idx, t in enumerate(toks) if t in UPPER_WORDS or t in LOWER_WORDS), None)
    post_garment_mode = (first_garm_idx is not None and first_col_idx is not None and first_garm_idx < first_col_idx)

    for i, tok in enumerate(toks):
        colour = _COLOR_LOOKUP.get(tok)
        if colour is None:
            continue

        prev_tok = toks[i - 1] if i > 0 else ""
        next_tok = toks[i + 1] if i + 1 < len(toks) else ""

        def _kind(w: str) -> Optional[str]:
            if w in UPPER_WORDS:
                return "upper"
            if w in LOWER_WORDS:
                return "lower"
            if w in ACCESSORY_WORDS:
                return "accessory"
            if w in HAIR_WORDS:
                return "feature"
            return None

        if post_garment_mode:
            target_word = prev_tok if _kind(prev_tok) else next_tok
        else:
            target_word = next_tok if _kind(next_tok) else prev_tok

        kind = _kind(target_word)
        if kind == "upper":
            target = upper
        elif kind == "lower":
            target = lower
        elif kind == "accessory":
            target = keywords
            colour = f"{colour} {target_word}"
        elif kind == "feature":
            continue
        else:
            target = unassigned

        if colour not in target:
            target.append(colour)

    for colour in unassigned:
        if not upper:
            upper.append(colour)
        elif not lower:
            lower.append(colour)

    age_hint: Optional[str] = None
    m = _AGE_NUM.search(lower_text)
    if m and m.group(0).strip() != m.group(1):
        age_hint = f"{m.group(1)} years"
    else:
        plain_age = _AGE_PLAIN.search(text)
        if plain_age and len(toks) <= 6:
            age_hint = f"{plain_age.group(1)} years"
        for tok in toks:
            if age_hint is None and tok in _AGE_LOOKUP:
                age_hint = _AGE_LOOKUP[tok]
                break
        if age_hint is None and m and len(toks) <= 6:
            age_hint = f"{m.group(1)} years"

    parts = []
    if upper:
        garment = next((w for w in toks if w in UPPER_WORDS), "top")
        parts.append(f"{' '.join(upper)} {garment}")
    if lower:
        garment = next((w for w in toks if w in LOWER_WORDS), "trousers")
        parts.append(f"{' '.join(lower)} {garment}")
    acc = [k for k in keywords if k not in UPPER_WORDS and k not in LOWER_WORDS and k not in facial_features]
    if acc:
        parts.append(", ".join(acc))

    who_parts = []
    if age_hint:
        who_parts.append(age_hint)
    for phys in physical_features:
        if phys not in who_parts:
            who_parts.append(phys)
    who = " ".join(who_parts) if who_parts else "person"

    facial_str = f" with {', '.join(facial_features)}" if facial_features else ""
    wearing_str = f" wearing {' and '.join(parts)}" if parts else ""

    clip_text = f"a photo of a {who}{facial_str}{wearing_str}".strip() if (parts or facial_features or who != "person") else (text or "a person")

    return {
        "upper": upper,
        "lower": lower,
        "keywords": keywords,
        "facial_features": facial_features,
        "physical_features": physical_features,
        "age_hint": age_hint,
        "clip_text": clip_text,
    }
