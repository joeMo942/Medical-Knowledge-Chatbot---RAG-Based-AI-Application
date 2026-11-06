"""Answer Language, simple heuristic until the full rules land in #7."""
import re

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def detect_language(message: str) -> str:
    """'ar' when the message contains Arabic script, else 'en'."""
    return "ar" if _ARABIC.search(message) else "en"
