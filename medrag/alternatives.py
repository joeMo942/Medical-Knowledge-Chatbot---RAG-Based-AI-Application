"""The words that ask for Alternatives (other Trade Names of the same Active Ingredient), one list for everyone.

hybrid.py reads it to give such a question every Section that lists the Trade Names; support.py reads it so the same
words count as the question's frame and aspect (the KB answers "alternatives of Brufen" by the products table, never by
the word), not as an unknown topic that would make the reply Out of Scope. English, Arabic, dialect and Franco-Arabic.
"""
from __future__ import annotations

from .textnorm import normalize

ALT_WORDS = frozenset(normalize(w) for w in """
alternative alternatives alternate substitute substitutes substitution equivalent equivalents
replace replaces replacing replacement replacements
بديل بدائل البدائل بدايل بدايله بديله البديل بدائلها بدايلها
bedail bedayel badail badayel bdail bdayel bdayl badil bedil bdil bidail bidayel
""".split())

# "trade names of <ingredient>" asks the same thing when it names an ingredient rather than a brand.
BRAND_WORDS = frozenset(normalize(w) for w in (
    "trade", "brand", "brands", "التجارية", "تجارية", "تجاري", "tegareya", "tegarya", "tigareya"))
