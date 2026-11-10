"""Last safety check on an Answered reply: the assistant never states a dose that is not written
in the cited documents. If the answer contains a dose-like amount (a number with a mass, volume,
unit or piece word, such as "4 g", "500 mg", "2 tablets", "3 حبات", or a frequency such as "three
times a day", "كل 8 ساعات") that does not occur in the text of the cited Sections, the reply is
replaced by the fixed Dose escalation. Strength figures that ARE in the cited text (Paracetamol
500 mg) pass. Numbers are compared as (value, canonical unit); Arabic-Indic digits and the usual
Arabic unit words are understood. A false alarm only costs a fixed "check the leaflet" reply."""
from __future__ import annotations

import re

from .textnorm import normalize

_WORDS = {"half": "0.5", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "nine": "9", "ten": "10", "once": "1", "twice": "2", "thrice": "3",
          "نص": "0.5", "نصف": "0.5", "واحد": "1", "اتنين": "2", "اثنين": "2", "تلات": "3", "ثلاث": "3",
          "ثلاثه": "3", "اربع": "4", "اربعه": "4", "خمس": "5", "خمسه": "5"}
_NUM = r"(\d+(?:[.,]\d+)*|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + ")"

_UNIT_CLASSES = {
    "mg": r"mg|mgs|milligrams?|ملجم|مجم|ملغ|ملغم|ملي ?جرام|مللي ?جرام",
    "mcg": r"mcg|µg|μg|micrograms?|ميكروجرام|ميكروغرام|مكج",
    "g": r"g|gm|gms|grams?|جم|جرام|جرامات|غرام",
    "ml": r"ml|mls|milliliters?|millilitres?|cc|مل|ملي|مللي|سم3",
    "iu": r"iu|i\.u\.|وحده دوليه|وحدات دوليه",
    "unit": r"units?|وحده|وحدات",
    "piece": (r"tablets?|tabs?|pills?|capsules?|caps|puffs?|drops?|sachets?|ampoules?|injections?|sprays?|"
              r"حبايه|حبه|حبات|حبايات|حبوب|قرص|اقراص|كبسول\w*|بخه|بخات|نقطه|نقط|قطره|قطرات|كيس|اكياس|امبول\w*"),
}
_UNIT_RX = [(name, re.compile(rf"^(?:{rx})$")) for name, rx in _UNIT_CLASSES.items()]
_ALL_UNITS = "|".join(f"(?:{rx})" for rx in _UNIT_CLASSES.values())
_AMOUNT = re.compile(rf"(?<![\w.]){_NUM}\s*({_ALL_UNITS})(?![a-z؀-ۿ])")

_FREQ = [
    re.compile(rf"\b{_NUM}\s*times\s*(?:a|per|each|every)\s*(day|week|hour|night)\b"),
    re.compile(r"\b(once|twice|thrice)\s*(?:a|per|each)?\s*(day|daily|week|weekly|night)\b"),
    re.compile(r"\bevery\s*(\d+(?:\s*(?:-|to)\s*\d+)?)\s*hours?\b"),
    re.compile(rf"(?<!\w){_NUM}\s*(?:مره|مرات|مرتين)\s*(?:في|ب|كل)?\s*(?:ال)?(يوم|اسبوع|ليل|ساع)"),
    re.compile(r"(مره|مرتين)\s*(?:في|ب|كل)?\s*(?:ال)?(يوم|اسبوع|ليل)"),
    re.compile(r"كل\s*(\d+(?:\s*(?:-|الي|الى)\s*\d+)?)\s*ساع"),
]
_PERIOD = {"daily": "day", "weekly": "week", "يوم": "day", "اسبوع": "week", "ليل": "night", "ساع": "hour"}
_AR_COUNT = {"مره": "1", "مرتين": "2"}


def _value(raw: str) -> str:
    raw = _AR_COUNT.get(raw, _WORDS.get(raw, raw))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
        raw = raw.replace(",", "")
    raw = raw.replace(",", ".")
    if "." in raw:
        raw = raw.rstrip("0").rstrip(".") or "0"
    return raw


def _unit(raw: str) -> str:
    raw = raw.strip()
    for name, rx in _UNIT_RX:
        if rx.match(raw):
            return name
    return raw


def dose_tokens(text: str) -> set[str]:
    """The dose-like amounts and frequencies written in `text`, as comparable tokens."""
    t = normalize(text)
    out: set[str] = set()
    for m in _AMOUNT.finditer(t):
        out.add(f"{_value(m.group(1))}{_unit(m.group(2))}")
    for rx in _FREQ:
        for m in rx.finditer(t):
            groups = [g for g in m.groups() if g]
            if len(groups) == 1:  # "every 6-8 hours": the interval is the amount
                out.add("freq:q" + re.sub(r"\s*(to|الي|الى)\s*", "-", groups[0]).replace(" ", "") + "h")
            else:
                out.add(f"freq:{_value(groups[0])}/{_PERIOD.get(groups[1], groups[1])}")
    return out


def unsupported_amounts(answer: str, cited_text: str) -> set[str]:
    """Dose-like amounts of the answer that the cited text does not contain."""
    return dose_tokens(answer) - dose_tokens(cited_text)
