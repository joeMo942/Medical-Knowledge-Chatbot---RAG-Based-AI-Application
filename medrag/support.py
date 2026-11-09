"""Support check: does any cited Section support what the question asks?

An Answered reply is only as good as its Citations. Matching drug names is not enough: the
hallucination trap is "Which drugs are used to treat malaria?", where retrieval surfaces
Plaquenil / Doxycycline sections although the KB never mentions malaria, and a model that
fills the gap from its own knowledge cites a real row for a claim the row does not make.

The check is deterministic and works on the QUESTION, not on the answer text (partial answers
deliberately contain words the KB lacks, "price" for example, and a model may translate):

  topic terms  = the question's words minus frame words ("which", "drugs", "treat", "الأدوية"),
                 numbers, and KB drug names (a drug name only says which section to look in)
  aspect words = what the KB expresses by structure rather than by the word itself: ingredient,
                 class, alternatives / trade names, side effects, contraindications, ...

The reply is supported when
  - the question has no topic term (nothing to deny: "What is the active ingredient of Controloc?"), or
  - a topic term is found in the cited Sections' text and no topic term is unknown to the whole KB, or
  - it asks an aspect of a drug that the cited Sections cover ("ingredient of Controloc and its price"):
    that is a partly covered question and the covered part is Answered.
Otherwise nothing that was asked is covered and the reply becomes Out of Scope.

Words are compared after Arabic folding (hamza, ta marbuta, harakat), with the Arabic clitics
(ال, و, ب, ل ...) and common English suffixes stripped, so "Antihypertensive" matches
"hypertension", "tremors" matches "Tremor" and "الحمل" matches "الحمل،". A few bilingual
pairs of the KB's own vocabulary (pregnancy / الحمل, alcohol / الكحول ...) are aliases.
Known limits: a question about a topic that is only a generic word the cited text happens to
contain (an Augmentin question about "children") passes this check and is left to the model;
an unknown Arabic word next to an aspect word (a transliterated brand) is not held against it.
"""
from __future__ import annotations

import re

from .models import Section
from .rules import normalize

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ARABIC = re.compile(r"[؀-ۿ]")
_NUMBERISH = re.compile(r"^\d+[a-z]*$")

_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف", "ك")
_AR_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "هم", "ه", "ي")
_EN_SUFFIXES = ("ations", "ation", "ions", "ion", "ives", "ive", "ings", "ing", "ies", "ed", "es", "al", "ic", "ly", "s")

# Words that say how to ask, not what is asked. English, Arabic (Modern + Egyptian) and Franco-Arabic.
_FRAME_EN = """
a an the of in on at to for from with by and or but not no yes if than then as it its this that these those
there here any some all each other another same also very more most much many one about into over under
between during after before using use used uses usage what which who whom whose how when where why is are was
were do does did can could should would will may might be been being am i me my we you your he she they them
tell give list name names kind type information info know say get take
drug drugs medicine medicines medication medications treat treats treated treating treatment treatments
therapy therapies safe safety suitable recommended common commonly mainly main usual usually please want need
patient patients people person anything something everything thing things
cause causes caused causing help helps helping lower lowers reduce reduces increase increases work works
stop start long short high low possible likely taken taking
year years old age aged month months week weeks day days
knowledge base kb approved there
ingredient ingredients active substance substances contain contains containing composition content contents
strength strengths concentration class classes classification category group groups type
alternative alternatives alternate substitute substitutes equivalent equivalents trade trades brand brands
available market egypt egyptian
side effect effects adverse reaction reactions contraindication contraindications contraindicated warning
warnings avoid interact interaction interactions interactions
otc prescription counter
"""
_FRAME_AR = """
ما ماذا هو هي هذا هذه ذلك التي الذي اللي ايه ايش اي ازاي امتي امتي ليه كام فين مين هل في فيه علي علي من الي الي عن مع او و
ده دي دا بتاع بتاعه بتاعت بتاعة بتاعته بتاعتها يعني كده برضو كمان لو لما عند عندي
دواء ادويه دوا دويه علاج علاجات يعالج بيعالج استخدام استعمال بيستخدم يستخدم يستعمل بيتصنف يتصنف
امن امان آمن مسموح مسموحه ينفع
مريض مرضي
مادة مادته ماده الفعاله فعاله مكونات مكون تركيب تركيز تركيبه محتوي
مجموعه مجموعة فئه تصنيف تصنيفه نوع انواع قسم
بديل بدائل البدائل بدايل التجاريه تجاري تجاريه اسماء اسم تجارية مصر المصري المصريه سوق متاح متاحه
عرض اعراض الاعراض جانبي جانبيه الجانبيه مضاعفات اثار
مانع موانع ممنوع ممنوعه تحذير تحذيرات تحذيره تداخل تداخلات
روشته وصفه
سنه سنين سنوات شهر شهور اسبوع اسابيع يوم ايام عمر عمره سن
"""
_FRAME_FRANCO = """
el ela l fe fi le li ma mn min w we wi eh ehh eih ee ey emta imta kam ya da dah di deh dee dy bta3 bta3t
beta3 beta3t beta3to bta3to bta3ty ba2a ba2 maada mada el3ada fa3ala fa3ela a3rad gaanbeya ganbeya
bedail bedayel tegareya tegarya mamnoo3 mamno3 momken mawad3 mawane3 3ayez 3ayza 3ayzen ana enta
"""
# Aspect words: what the KB answers by structure (a table column, a heading, a list), so a drug's
# Section answers them without ever containing the word.
_ASPECT = """
ingredient ingredients active substance contain contains containing composition content contents strength
concentration class classes classification category group trade brand alternative alternatives substitute
side effect effects adverse contraindication contraindications contraindicated warning warnings
interact interaction interactions avoid otc prescription
مادة مادته ماده الفعاله فعاله مكونات مكون تركيب تركيز مجموعه فئه تصنيف تصنيفه
بديل بدائل البدائل التجاريه تجاري تجاريه اسماء
عرض اعراض الاعراض جانبي جانبيه الجانبيه مضاعفات اثار
مانع موانع ممنوع ممنوعه تحذير تحذيرات تداخل تداخلات
maada fa3ala a3rad gaanbeya ganbeya bedail bedayel tegareya mamnoo3 mamno3 mawane3
"""
# KB vocabulary that appears in both languages: either side grounds the other.
_ALIASES = [
    {"pregnancy", "pregnant", "الحمل", "حمل", "حامل", "الحوامل"},
    {"breastfeeding", "lactation", "الرضاعه", "رضاعه", "الرضاعة"},
    {"alcohol", "الكحول", "كحول"},
    {"children", "child", "kids", "pediatric", "paediatric", "الاطفال", "اطفال", "طفل", "الاطفال"},
    {"elderly", "كبار", "المسنين"},
    {"kidney", "renal", "الكلي", "كلي", "الكلوي"},
    {"liver", "hepatic", "الكبد", "كبد"},
    {"asthma", "الربو", "ربو"},
    {"pneumonia", "رئوي"},
    {"hypertension", "antihypertensive", "الضغط", "ضغط"},
    {"diabetes", "السكر", "سكر", "antidiabetic"},
    {"cough", "كحه", "السعال", "سعال"},
    {"pain", "الم", "الالم", "ألم"},
]


def _words(block: str) -> set[str]:
    return {w for w in normalize(block).split() if w}


FRAME = _words(_FRAME_EN) | _words(_FRAME_AR) | _words(_FRAME_FRANCO)
ASPECT = _words(_ASPECT)


def stem(word: str) -> str:
    """A crude, language-aware stem; only ever used to compare two words."""
    w = word
    if _ARABIC.search(w):
        for p in _AR_PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 3:
                w = w[len(p):]
                break
        for s in _AR_SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 3:
                w = w[: -len(s)]
                break
        return w
    for s in _EN_SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 4:
            return w[: -len(s)]
    return w


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


_ALIAS_GROUPS = [{normalize(w) for w in g} for g in _ALIASES]
_ALIAS_STEMS = [{stem(w) for w in g} for g in _ALIAS_GROUPS]


def _aliases_of(w: str) -> set[str]:
    out = {w}
    for group, stems in zip(_ALIAS_GROUPS, _ALIAS_STEMS):
        if w in group or stem(w) in stems:
            out |= group
    return out


class _Text:
    """Searchable stems of some text."""

    def __init__(self, text: str):
        self.stems = {stem(t) for t in tokens(text)}

    def has(self, word: str) -> bool:
        for alias in _aliases_of(word):
            s = stem(alias)
            if len(s) >= 4:
                if any(s in t for t in self.stems):  # "hypertens" inside "antihypertens"
                    return True
            elif s in self.stems:
                return True
        return False


class SupportCheck:
    """Built once from the Knowledge Base (its drug-name vocabulary and its full text)."""

    def __init__(self, sections: list[Section] | tuple[Section, ...]):
        names: set[str] = set()
        for sec in sections:
            for table in sec.tables:
                if table.kind != "products":
                    continue
                cols = [c for c in (table.trade_name_col, table.active_ingredient_col) if c is not None]
                for row in table.rows:
                    for c in cols:
                        if c < len(row.cells):
                            names.update(t for t in tokens(row.cells[c]) if len(t) >= 3)
        self.drug_names = names
        self._kb = _Text("\n".join(s.render() for s in sections))

    def topic_terms(self, question: str) -> list[str]:
        out = []
        for t in tokens(question):
            if t in FRAME or t in self.drug_names or len(t) < 3 or _NUMBERISH.match(t):
                continue
            out.append(t)
        return out

    def supported(self, question: str, cited: list[Section]) -> bool:
        """True when the cited Sections cover (part of) what the question asks."""
        text = _Text("\n".join(s.render() for s in cited))
        qtokens = tokens(question)
        topic = self.topic_terms(question)
        if not topic:
            return True
        unknown = [t for t in topic if not self._kb.has(t)]
        if any(text.has(t) for t in topic) and not unknown:
            return True
        # A partly covered question: an aspect the KB answers by structure, asked of a drug
        # (or a brand written in Arabic) the cited Sections cover.
        asks_aspect = any(t in ASPECT for t in qtokens)
        drugs = [t for t in qtokens if t in self.drug_names and t not in FRAME]
        drug_covered = any(text.has(d) for d in drugs)
        arabic_only_unknown = bool(unknown) and all(_ARABIC.search(t) for t in unknown)
        return asks_aspect and (drug_covered or arabic_only_unknown)
