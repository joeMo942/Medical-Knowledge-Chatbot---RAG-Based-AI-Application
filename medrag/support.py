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
from .models import Line, Section
from .language import NOT_FRANCO, _is_franco_digit_word
from .translit import same_word, skeleton, strict_same, strict_skeleton
from .textnorm import normalize

_ARABIC = re.compile(r"[؀-ۿ]")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?([a-z%]*)$")


def is_number(t: str) -> bool:
    """A number: digits, alone or with a unit or ordinal suffix ("500", "5mg", "2nd"). A Franco-Arabic word that starts with
    a digit-letter ("7amel", "3ayan", "2alb", "7asasiya") is a word, never a number: it must be kept like any other."""
    m = _NUMBER.match(t)
    return bool(m) and not _is_franco_digit_word(t) and (m.group(1) == "" or m.group(1) in NOT_FRANCO)

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
3andy 3andi 3ndy 3ndi a5od a5d a5ood ya5od dawa dawaa edy ezay ezzay leh lih lel lil fel fil amaan aman bdayel bdail meen men feen fein emta
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
    {"hypertension", "antihypertensive", "الضغط", "ضغط"},
    {"fever", "feverish", "antipyretic", "antipyretics", "الحراره", "حراره", "سخونه", "سخونيه", "السخونه"},
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
_FRAME_AR_WORDS = [w for w in _words(_FRAME_AR) if len(strict_skeleton(w)) >= 3 and _ARABIC.search(w)]
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


_ASPECT_STEMS = frozenset(stem(w) for w in ASPECT)
_FRAME_STEMS = frozenset(stem(w) for w in FRAME if len(stem(w)) >= 3)


def is_aspect(t: str) -> bool:
    """An aspect word, with its article or clitic ("المادة" for "مادة")."""
    return t in ASPECT or stem(t) in _ASPECT_STEMS


# Words that introduce the condition a drug is for: the word after one is a disease, not a brand.
INDICATION = {stem(normalize(w)) for w in "دواء ادويه دوا علاج لعلاج يعالج بيعالج تعالج علاجات مرض امراض".split()}


# A question that asks what treats a condition (or what is used for one). Its condition term must be grounded where the KB
# says what a class is FOR, not where it says when NOT to use it or what it does wrong (`indication_text`).
_INDICATION_EN = {"treat", "treats", "treated", "treating", "treatment", "treatments", "therapy", "dawa", "dawaa", "3elag", "3lag"}
_DRUG_WORDS = {"drug", "drugs", "medicine", "medicines", "medication", "medications", "meds"}
_INDICATION_PAIRS = {("help", "with"), ("helps", "with"), ("helping", "with"), ("used", "for"), ("used", "in"), ("use", "for"),
                     ("anything", "for")}
# What a question asks about a drug's risks, or what a drug causes: these are not indication questions even with no drug named.
_RISK_ASPECT = """
side effect effects adverse contraindication contraindications contraindicated warning warnings avoid interact interaction
interactions cause causes caused causing
عرض اعراض الاعراض جانبي جانبيه الجانبيه مضاعفات اثار مانع موانع ممنوع ممنوعه تحذير تحذيرات تداخل تداخلات
تسبب بتسبب يسبب بيسبب بتعمل بيعمل a3rad gaanbeya ganbeya mamnoo3 mamno3 mawane3
"""
_RISK_ASPECT_WORDS = _words(_RISK_ASPECT)
_RISK_ASPECT_STEMS = frozenset(stem(w) for w in _RISK_ASPECT_WORDS)
_PATIENT = {"patient", "patients", "مريض", "مرضي", "مريضه", "عيان", "عيانه"}
_TAKE = {"take", "takes", "taking", "ياخد", "اخد", "بياخد", "ياخدوا", "بياخدوا", "يستخدم", "بيستخدم", "يستعمل", "a5od", "ya5od"}
_QWORDS = {"what", "which", "ايه", "ايش", "اي", "ماذا", "ما", "eh", "eih", "ee"}
_YESNO_FIRST = {"can", "could", "may", "might", "is", "are", "do", "does", "should", "would", "will", "هل", "ممكن", "ينفع"}
# Words of the indication frame that are not the condition ("the suitable treatment for cancer").
_INDICATION_FILLER = {stem(normalize(w)) for w in "مناسب مناسبه افضل احسن suitable appropriate best good right".split()}


def _is_risk_aspect(t: str) -> bool:
    return t in _RISK_ASPECT_WORDS or stem(t) in _RISK_ASPECT_STEMS


def asks_indication(question: str, named_drug: bool = False) -> bool:
    """Indication wording and no risk aspect, no drug named: "what treats / is used for / helps with a condition" (دواء /
    علاج / treat / "drugs for" ...; a drug word anywhere counts, as ادوية does). Questions about a drug's side effects,
    contraindications, interactions or what causes a symptom, and any question that names a drug, keep the plain check;
    class, group, ingredient, trade-name and alternatives words switch the rule off only when a drug is named.
    "What does a cancer patient take" asks the same thing: a patient word, a take word followed by a question word or
    nothing, and a question that is not a yes/no one."""
    if named_drug:
        return False
    toks = tokens(question)
    if not toks or toks[0] in _YESNO_FIRST or any(_is_risk_aspect(t) for t in toks):
        return False  # a yes/no question ("هل اكتوس بيعالج ...") is about one drug, which may be written in Arabic letters
    if any(t in _INDICATION_EN or t in _DRUG_WORDS or stem(t) in INDICATION for t in toks):
        return True
    if any(pair in _INDICATION_PAIRS for pair in zip(toks, toks[1:])):
        return True
    # "ايه اللي ينفع لـ X": what is good for X (a yes/no "ينفع" with a drug named never gets here)
    if any(t == "ينفع" and (i >= 1 and toks[i - 1] in _QWORDS or i >= 2 and toks[i - 1] == "اللي" and toks[i - 2] in _QWORDS)
           for i, t in enumerate(toks)):
        return True
    if any(t in _PATIENT for t in toks):
        for i, t in enumerate(toks):
            if t in _TAKE and (i + 1 == len(toks) or toks[i + 1] in _QWORDS):
                return True
    return False


# Population words: who the drug is for, which the KB states in cautions as much as in usage lines (pregnancy, children ...).
_POPULATION_STEMS = [stems for group, stems in zip(
    [{normalize(w) for w in g} for g in _ALIASES], [{stem(normalize(w)) for w in g} for g in _ALIASES])
    if group & {"pregnancy", "breastfeeding", "children", "elderly", "kidney", "liver"}]


def _is_population(w: str) -> bool:
    return any(w in group or stem(w) in group for group in _POPULATION_STEMS)


# The label lines that open a contraindication / caution / side-effect block, and what ends one.
_RISK_LABEL = re.compile(r"^(?:common\s+)?(?:contraindications?|side\s+effects?|adverse|warnings?|precautions?|serious\s+risks?|"
                         r"important\s+(?:drug\s+)?interactions?|موانع|تحذيرات|الاعراض|اعراض|مضاعفات)")
_RISK_LINE = re.compile(r"^(?:الحذر|لا\s+يستخدم)")  # a caution on its own line
_USAGE_LINE = re.compile(r"^(?:الاستخدام|الاستعمال|استخدام|uses?|indications?)\s*(?::|$)")  # the label, not a sentence that starts with it
_SUB_LABEL = re.compile(r"^(?:[a-z]\.\s+\S|اولا|ثانيا|ثالثا|رابعا)")
_NON_RISK_LABEL = re.compile(r"^(?:trade\s+names?|الادويه|otc|prescription|common\s+(?:active\s+)?ingredients?|combination)")
_CLAUSE_BREAK = re.compile("[,،;؛]")


def _is_label(text: str) -> bool:
    t = text.strip()
    return len(t.split()) <= 7 and not _CLAUSE_BREAK.search(t) and (t.endswith((":", "؟", "?")) or not t.endswith("."))


def indication_text(section: Section) -> str:
    """The text of a Section a condition may be grounded in for an indication question: its heading (section name and
    disease title), usage / indication lines and every other body line, but NOT a table of side effects or a line under a
    contraindication / caution / side-effect label. The block such a label opens (with the sub-labels inside it:
    "Intranasal:", "Serious Risks:") runs to the next usage line, products table, lettered sub-heading or non-risk label
    (Trade Names, الأدوية, OTC, Common Ingredients). Read from the Section's structure, never from KB sentences."""
    out = [f"{section.title} {section.section}" if section.heading_present else section.title]
    risky = False
    for item in section.items:
        if not isinstance(item, Line):
            if item.kind == "side_effects":
                risky = True
            elif item.kind in ("products", "otc"):
                risky = False
            if not risky:
                out.extend(r.text for r in item.rows)
            continue
        t = normalize(item.text)
        if _RISK_LABEL.match(t) and _is_label(item.text):
            risky = True
        elif _USAGE_LINE.match(t):
            risky = False
            out.append(item.text)
        elif risky:
            if _SUB_LABEL.match(t) or _NON_RISK_LABEL.match(t):
                risky = False
        elif not _RISK_LINE.match(t):
            out.append(item.text)
    return "\n".join(out)


_ALIAS_GROUPS = [{normalize(w) for w in g} for g in _ALIASES]
_ALIAS_STEMS = [{stem(w) for w in g} for g in _ALIAS_GROUPS]


def _aliases_of(w: str) -> set[str]:
    out = {w}
    for group, stems in zip(_ALIAS_GROUPS, _ALIAS_STEMS):
        if w in group or stem(w) in stems:
            out |= group
    return out


# Phrases (not bare words) that name a KB topic in another language: "pressure" alone is eye or intracranial pressure or a
# pressure ulcer, so only the whole phrase counts as the KB's hypertension.
_PHRASES = tuple((normalize(p), w) for p, w in (
    ("high blood pressure", "hypertension"), ("blood pressure", "hypertension"), ("ضغط الدم", "hypertension"),
    ("الضغط العالي", "hypertension"), ("ضغط عالي", "hypertension")))


def with_phrases(text: str) -> str:
    """`text` normalised, with each alias PHRASE replaced by the one KB word that stands for it."""
    t = normalize(text)
    for phrase, word in _PHRASES:
        t = t.replace(phrase, word)
    return t


def alias_words(question: str) -> list[str]:
    """The other members of every alias group that a word of the question belongs to ("سخونية" -> fever, الحرارة ...),
    for search: an everyday or dialect word the KB writes differently still finds its Sections. Never a word that is
    already in the question."""
    toks = tokens(with_phrases(question))
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
        malaria eh" would turn it into a question that has no topic to be denied) and what is asked: an aspect word
        (side effects, alternatives, contraindications ...) is translated like any other word, never swapped for
        another ("a3rad gaanbeya" -> "بدائل" is refused)."""
        mtoks = tokens(message)
        have = {stem(t) for t in mtoks}
        rtoks = tokens(rewrite)

        def same(t: str, others: list[str]) -> bool:
            # with and without the stem: the stemmer strips a letter that is part of the word as often as a clitic (بدايل)
            # The aspect words are a closed list (ingredient, side effects, alternatives ...): between two of them the
            # lenient comparison is enough ("maada" ~ "المادة"); every other word needs the strict one.
            return translate and any(
                (same_word if (is_aspect(t) and is_aspect(o)) else strict_same)(a, b)
                for o in others for a in {t, stem(t)} for b in {o, stem(o)})

        for t in rtoks:
            if is_number(t):
                if t not in mtoks:
                    return False
                continue
            if len(t) < 2 or stem(t) in have:  # a single letter is a clitic's remnant; two letters (سل, دم, مش) are words
                continue
            if is_aspect(t):  # what is asked (side effects, alternatives ...): a translation may not change it
                if translate and not same(t, mtoks):
                    return False
                continue
            if t in FRAME:
                continue
            if from_history and (t in self.drug_names or t in self.class_names):
                continue
            if same(t, mtoks):
                continue
            return False
        if translate:
            kept = {stem(t) for t in rtoks}
            for m in mtoks:
                if is_number(m) or len(m) < 3 or stem(m) in kept or (m in FRAME and not is_aspect(m)):
                    continue
                if same(m, rtoks) or (_LATIN.fullmatch(m) is not None and any(strict_same(m, w) for w in _FRAME_AR_WORDS)):
                    continue
                return False
        return True

    def translated(self, message: str, rewrite: str, *, from_history: bool = False) -> str | None:
        """A Franco-Arabic message's rewrite into Arabic letters, cleaned, or None when it cannot be used.

        Every word of the rewrite that is not the message's own word in Arabic letters is DELETED, not accepted: a gloss
        the model adds ("ضغط الدم" for "daghT", "الحلق" after "eltehab") could bring in a condition the user did not
        name, and deleting a word can never add a fact. Frame words and the aspect the user asked (side effects,
        alternatives ...) stay when they are the message's own (rewrite_ok decides); in a Session the KB drug and Drug
        Class names of the history stay too. The cleaned text must then pass `rewrite_ok(translate=True)`: every number
        and content word of the message is kept, and what is asked has not changed. So a rewrite that swaps the user's
        word for another is refused (the word it would keep is missing), one that only decorates it is trimmed."""
        mtoks = tokens(message)
        have = {stem(t) for t in mtoks}
        kept = []
        for m in re.finditer(r"[^\W_]+", rewrite):
            t = normalize(m.group())
            drop = (len(t) >= 2 and not is_number(t) and stem(t) not in have and not is_aspect(t) and t not in FRAME
                    and not (from_history and (t in self.drug_names or t in self.class_names))
                    and not any(strict_same(a, b) for o in mtoks for a in {t, stem(t)} for b in {o, stem(o)}))
            if not drop:
                kept.append(m.group())
        cleaned = " ".join(kept)
        return cleaned if cleaned and self.rewrite_ok(message, cleaned, from_history=from_history, translate=True) else None

    def topic_terms(self, question: str) -> list[str]:
        return self._topic(tokens(question))

    def _topic(self, toks: list[str]) -> list[str]:
        return [t for t in toks
                if not (t in FRAME or t in ABSENT or t in self.drug_names or len(t) < 3 or is_number(t))]

    def _condition_terms(self, toks: list[str]) -> list[str]:
        """The topic terms of an indication question that are not the frame itself: "الادويه", "بتعالج", "العلاج" (article
        and verb forms of the indication and frame words) and "المناسب" say what is asked, not which condition."""
        return [t for t in self._topic(toks) if stem(t) not in _FRAME_STEMS and stem(t) not in INDICATION
                and stem(t) not in _INDICATION_FILLER]

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
        question = with_phrases(question)
        text = _Text("\n".join(s.render() for s in cited))
        qtokens = tokens(question)
        drugs = [t for t in qtokens if t in self.drug_names and t not in FRAME]
        if drugs and not any(text.has(d) for d in drugs):
            return False  # the Sections cited say nothing about the drug that was asked about
        # An indication question grounds its condition only in what the Sections say a class is FOR.
        indication = asks_indication(question, bool(drugs))
        grounded = _Text("\n".join(indication_text(s) for s in cited)) if indication else text
        if not self._topic(qtokens) or (indication and not self._condition_terms(qtokens)):
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
            topic = self._condition_terms(toks) if indication else self._topic(toks)
            if not topic:
                continue
            if indication:
                # who it is for (pregnancy, children, kidney ...) is stated in cautions as much as in usage lines: grounded
                # in the whole text; any other term only in what the Sections say the class is for
                covered = any((text if _is_population(w) else grounded).has(w) for w in topic)
            else:
                covered = any(grounded.has(w) for w in topic)
            if not covered:
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
