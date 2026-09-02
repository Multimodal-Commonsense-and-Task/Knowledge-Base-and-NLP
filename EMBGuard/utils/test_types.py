"""Shared EMBGuardTest type aliases and labels."""
from typing import List, Optional


TEST_TYPE_CODES = ("HR", "HNR", "MHR", "NHR")

TEST_TYPE_CODE_TO_LABEL = {
    "HR": "Causal Risky",
    "NHR": "Absent Benign",
    "HNR": "Decoupled Benign",
    "MHR": "Selective Risky",
}

TEST_TYPE_CODE_TO_SLUG = {
    "HR": "causal_risky",
    "NHR": "absent_benign",
    "HNR": "decoupled_benign",
    "MHR": "selective_risky",
}


def _alias_key(value: object) -> str:
    text = str(value).strip().lower()
    return " ".join(text.replace("_", " ").replace("-", " ").split())


TEST_TYPE_ALIASES = {
    "hr": "HR",
    "hazard risk": "HR",
    "causal risky": "HR",
    "causal risk": "HR",
    "nhr": "NHR",
    "no hazard risk": "NHR",
    "absent benign": "NHR",
    "absent": "NHR",
    "hnr": "HNR",
    "hazard no risk": "HNR",
    "decoupled benign": "HNR",
    "decoupled": "HNR",
    "mhr": "MHR",
    "multi hazard risk": "MHR",
    "multiple hazard risk": "MHR",
    "selective risky": "MHR",
    "selective risk": "MHR",
}


def normalize_test_type_code(
    value: object,
    default: Optional[str] = None,
    strict: bool = False,
) -> Optional[str]:
    """Return the canonical EMBGuardTest code for a code, label, or slug alias."""
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    code = TEST_TYPE_ALIASES.get(_alias_key(text))
    if code:
        return code

    if strict:
        valid = ", ".join(sorted(TEST_TYPE_ALIASES))
        raise ValueError(f"Unknown EMBGuardTest type '{value}'. Valid aliases include: {valid}")

    return default if default is not None else text


def normalize_test_type_key(value: object, strict: bool = False) -> Optional[str]:
    """Return the lowercase config/test-set key for an EMBGuardTest type alias."""
    code = normalize_test_type_code(value, strict=strict)
    return code.lower() if code in TEST_TYPE_CODES else code


def test_type_label(value: object) -> str:
    code = normalize_test_type_code(value)
    return TEST_TYPE_CODE_TO_LABEL.get(code, str(value))


def test_type_slug(value: object) -> str:
    code = normalize_test_type_code(value)
    return TEST_TYPE_CODE_TO_SLUG.get(code, str(value).strip().lower().replace(" ", "_"))


def test_type_safety_type(value: object) -> str:
    """Return the dataset safety class for an EMBGuardTest scenario type."""
    code = normalize_test_type_code(value)
    if code in {"HR", "MHR"}:
        return "unsafe"
    if code in {"HNR", "NHR"}:
        return "safe"
    return ""


def hf_split_candidates(value: object) -> List[str]:
    """Return preferred HF split names for a requested EMBGuardTest type."""
    code = normalize_test_type_code(value)
    if code in TEST_TYPE_CODES:
        requested = str(value).strip()
        slug = TEST_TYPE_CODE_TO_SLUG[code]
        candidates = []
        for candidate in (slug, code, requested):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    return [str(value).strip()] if value is not None and str(value).strip() else []


def is_risky_test_type(value: object) -> bool:
    return normalize_test_type_code(value) in {"HR", "MHR"}


def is_benign_test_type(value: object) -> bool:
    return normalize_test_type_code(value) in {"HNR", "NHR"}


def extract_test_type_code_from_text(text: object) -> Optional[str]:
    """Extract a canonical type code from filenames or other free-form text."""
    if text is None:
        return None

    normalized = _alias_key(text)
    padded = f" {normalized} "
    for alias, code in sorted(TEST_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if f" {alias} " in padded:
            return code

    return None
