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

from .alternatives import ALT_WORDS
from .drugs import product_cells, tokens
from .models import Section
from .translit import same_word, skeleton
from .textnorm import normalize

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
انا انت انتي احنا هو اخد اخذ ياخد خد بخد عايز عاوز عايزه عاوزه محتاج محتاجه ينفع لازم ممكن
"""
_FRAME_FRANCO = """
el ela l fe fi le li ma mn min w we wi eh ehh eih ee ey emta imta kam ya da dah di deh dee dy bta3 bta3t
beta3 beta3t beta3to bta3to bta3ty ba2a ba2 bta3et beta3et bta3at beta3at bet3a bta3 maada mada el3ada fa3ala fa3ela a3rad gaanbeya ganbeya
bedail bedayel tegareya tegarya mamnoo3 mamno3 momken mawad3 mawane3 3ayez 3ayza 3ayzen ana enta
3andy 3andi 3ndy 3ndi a5od a5d a5ood ya5od dawa dawaa edy ezay ezzay leh lel lil fel fil bdayel bdail
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
# KB vocabulary that appears in both languages (and everyday dialect words next to the KB's formal ones): either side
# grounds the other, for the support check and, as extra query words, for search (hybrid.py: `alias_words`).
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
    {"hypertension", "antihypertensive", "pressure", "الضغط", "ضغط"},
    {"fever", "feverish", "antipyretic", "antipyretics", "الحراره", "حراره", "سخونه", "سخونيه", "السخونه", "الحمي", "حمي"},
    {"diabetes", "السكر", "سكر", "antidiabetic"},
    {"cough", "كحه", "السعال", "سعال"},
    {"pain", "الم", "الالم", "ألم"},
]


# Attributes the KB never carries: asking one makes a question partly covered, not out of scope.
ABSENT = {normalize(w) for w in """price prices pricing cost costs cheaper expensive manufacturer manufacturers manufacture
manufactures company companies maker producer discovered discover invented invent affect affects
سعر اسعار سعره سعرها بكام ثمن بيتباع تكلفه تكلفته شركه الشركه المصنعه مصنع اكتشف اكتشاف اخترع""".split()}
_CLAUSES = re.compile(r"\s+(?:and|but|also|و|وكمان|كمان)\s+|[,،;؛؟?]+", re.IGNORECASE)


def _words(block: str) -> set[str]:
    return {w for w in normalize(block).split() if w}


FRAME = _words(_FRAME_EN) | _words(_FRAME_AR) | _words(_FRAME_FRANCO) | ALT_WORDS
ASPECT = _words(_ASPECT) | ALT_WORDS
# A Franco-Arabic word that is the transliteration of an Arabic frame word ("3andy" ~ عندي, "emta" ~ امتي, "ezay" ~ ازاي)
# is frame too: a rewrite into Arabic letters may drop or keep it (rewrite_ok).
_FRAME_SKELETONS = {sk for sk in (skeleton(w) for w in _words(_FRAME_AR)) if len(sk) >= 2}
_LATIN = re.compile(r"[a-z0-9]+")


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


# Words that introduce the condition a drug is for: the word after one is a disease, not a brand.
INDICATION = {stem(normalize(w)) for w in "دواء ادويه دوا علاج لعلاج يعالج بيعالج تعالج علاجات مرض امراض".split()}


_ALIAS_GROUPS = [{normalize(w) for w in g} for g in _ALIASES]
_ALIAS_STEMS = [{stem(w) for w in g} for g in _ALIAS_GROUPS]


def _aliases_of(w: str) -> set[str]:
    out = {w}
    for group, stems in zip(_ALIAS_GROUPS, _ALIAS_STEMS):
        if w in group or stem(w) in stems:
            out |= group
    return out


def alias_words(question: str) -> list[str]:
    """The other members of every alias group that a word of the question belongs to ("سخونية" -> fever, الحرارة ...),
    for search: an everyday or dialect word the KB writes differently still finds its Sections. Never a word that is
    already in the question."""
    toks = tokens(question)
    seen = {stem(t) for t in toks}  # one word per stem: a repeated stem would count twice in BM25
    out: list[str] = []
    for t in toks:
        if t in FRAME:
            continue
        for group, stems in zip(_ALIAS_GROUPS, _ALIAS_STEMS):
            if t in group or stem(t) in stems:
                for w in sorted(group):
                    if stem(w) not in seen:
                        seen.add(stem(w))
                        out.append(w)
    return out


class _Text:
    """Searchable stems of some text."""

    def __init__(self, text: str):
        self.stems = {stem(t) for t in tokens(text)}

    def has(self, word: str) -> bool:
        for alias in _aliases_of(word):
            s = stem(alias)
            if len(s) >= 7:
                if any(s in t for t in self.stems):  # "hypertens" inside "antihypertens"
                    return True
            elif len(s) >= 4:
                if any(t.startswith(s) for t in self.stems):  # not "cost" inside "corticosteroid"
                    return True
            elif s in self.stems:
                return True
        return False


class SupportCheck:
    """Built once from the Knowledge Base (its drug-name vocabulary and its full text)."""

    def __init__(self, sections: list[Section] | tuple[Section, ...]):
        names: set[str] = set()
        for _sec, _table, _cells, trade, active in product_cells(sections):
            for cell in (trade, active):
                names.update(t for t in tokens(cell) if len(t) >= 3)
        self.drug_names = names
        self.section_ids = frozenset(sec.section_id for sec in sections)  # every Section of the KB (answering.py)
        # Drug Class names (the headings of the class Sections): "Which drug class is it in?" is rewritten
        # to name the class or the drug, so both are words a rewrite may bring in from the history.
        self.class_names = {t for sec in sections if sec.kind == "class" for t in tokens(sec.section) if len(t) >= 3}
        self._kb = _Text("\n".join(s.render() for s in sections))

    def rewrite_ok(self, message: str, rewrite: str, *, from_history: bool = True, translate: bool = False) -> bool:
        """True when a standalone rewrite of `message` adds nothing it may not.

        Every number of the rewrite is in the message. Every content word of the rewrite that the message does not
        have (frame and aspect words aside) must be one the use allows:
          - `from_history` (a Session): a KB drug name (Trade Name or Active Ingredient) or a Drug Class name, which is
            what resolves "it" / "its"; any other word (a price, a symptom, a claim from an earlier reply) is refused,
            so the earlier turns can resolve a reference but never add a fact;
          - `translate` (a Franco-Arabic message rewritten into Arabic letters for search and for this check): a word that
            is the SAME word as one of the message's in another script (translit.same_word: "so5oneya" -> "سخونية",
            "daghT" -> "ضغط"). A translation into a different word ("so5oneya" -> "ملاريا", or -> "حمى") is refused,
            so a rewrite cannot bring in a condition the user did not name. A first-turn rewrite (not `from_history`)
            gets no drug-name exemption either: its drug and class names must be the message's own.
        With `translate`, the rewrite must also keep every content word of the message (a word dropped from "dawa el
        malaria eh" would turn it into a question that has no topic to be denied)."""
        mtoks = tokens(message)
        have = {stem(t) for t in mtoks}
        rtoks = tokens(rewrite)

        def same(t: str, others: list[str]) -> bool:
            return translate and any(same_word(stem(t), stem(o)) for o in others)

        for t in rtoks:
            if _NUMBERISH.match(t):
                if t not in mtoks:
                    return False
                continue
            if len(t) < 3 or t in FRAME or t in ASPECT or stem(t) in have:
                continue
            if from_history and (t in self.drug_names or t in self.class_names):
                continue
            if same(t, mtoks):
                continue
            return False
        if translate:
            kept = {stem(t) for t in rtoks}
            for m in mtoks:
                if _NUMBERISH.match(m) or len(m) < 3 or m in FRAME or m in ASPECT or stem(m) in kept:
                    continue
                if same(m, rtoks) or (_LATIN.fullmatch(m) is not None and skeleton(m) in _FRAME_SKELETONS):
                    continue
                return False
        return True

    def topic_terms(self, question: str) -> list[str]:
        return self._topic(tokens(question))

    def _topic(self, toks: list[str]) -> list[str]:
        return [t for t in toks
                if not (t in FRAME or t in ABSENT or t in self.drug_names or len(t) < 3 or _NUMBERISH.match(t))]

    def _brand_like(self, toks: list[str], i: int) -> bool:
        """An unknown Arabic word that may be a brand written in Arabic letters (بروفين, لكتافلام):
        not a definite noun (ال..., لل...) and not the object of an indication frame
        (دواء / علاج / يعالج + X), which is how a disease is named."""
        w = toks[i]
        if not _ARABIC.search(w) or w.startswith(("ال", "لل", "بال", "وال")):
            return False
        return not any(stem(p) in INDICATION for p in toks[max(0, i - 2):i])

    def supported(self, question: str, cited: list[Section]) -> bool:
        """True when the cited Sections cover (part of) what the question asks. Out of Scope only when
        nothing asked is covered: a covered half keeps the reply Answered, whatever the other half is."""
        text = _Text("\n".join(s.render() for s in cited))
        qtokens = tokens(question)
        drugs = [t for t in qtokens if t in self.drug_names and t not in FRAME]
        if drugs and not any(text.has(d) for d in drugs):
            return False  # the Sections cited say nothing about the drug that was asked about
        if not self._topic(qtokens):
            return True
        # A partly covered question: an aspect the KB answers by structure, asked of a drug
        # (or a brand written in Arabic letters) the cited Sections cover.
        asks_aspect = any(t in ASPECT for t in qtokens)
        if asks_aspect and drugs:
            return True
        # Otherwise some part (clause) of the question must be covered by itself: it has topic terms,
        # none is unknown to the whole KB (unless it may be an Arabic brand), and one is in the cited text.
        for clause in _CLAUSES.split(question):
            toks = tokens(clause)
            topic = self._topic(toks)
            if not topic or not any(text.has(w) for w in topic):
                continue
            unknown = [w for w in topic if not self._kb.has(w)]
            if all(self._brand_like(toks, toks.index(w)) for w in unknown):
                return True
        # Only Arabic brand-like unknown words next to an aspect word: a brand the KB lists under its Latin name.
        if asks_aspect:
            toks = qtokens
            unknown = [w for w in self._topic(toks) if not self._kb.has(w)]
            return bool(unknown) and all(self._brand_like(toks, toks.index(w)) for w in unknown) and not drugs
        return False
