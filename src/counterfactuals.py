"""Counterfactual text utilities for name-swap bias checks."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NamePair:
    female: str
    male: str


DEFAULT_NAME_PAIRS = [
    NamePair(female="Anna", male="Jan"),
    NamePair(female="Kasia", male="Piotr"),
    NamePair(female="Agnieszka", male="Marek"),
]


def _preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source and source[0].isupper():
        return target.capitalize()
    return target.lower()


def swap_name(text: str, source: str, target: str) -> str:
    pattern = rf"\b{re.escape(source)}\b"
    return re.sub(pattern, lambda m: _preserve_case(m.group(0), target), text)


def gender_counterfactual_pairs(text: str, name_pairs: list[NamePair] | None = None) -> list[tuple[str, str]]:
    """Return (original, swapped) text pairs for every gendered name occurrence."""
    pairs = name_pairs or DEFAULT_NAME_PAIRS
    outputs: list[tuple[str, str]] = []
    for pair in pairs:
        if pair.female in text:
            swapped = swap_name(text, pair.female, pair.male)
            if swapped != text:
                outputs.append((text, swapped))
        if pair.male in text:
            swapped = swap_name(text, pair.male, pair.female)
            if swapped != text:
                outputs.append((text, swapped))
    return outputs
