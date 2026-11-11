"""Self-care questions and lookups (ADR 0003): what the pipeline does with a first-person "what can I take / use" question.

Two different questions look alike:

  - a LOOKUP asks what is used FOR a condition ("Which drugs can I use for asthma?", "What can I take for migraine?", "What can
    a pregnant woman take for her heartburn?"). The users are pharmacists and doctors; it is a normal drug-information question:
    it is Answered (or Out of Scope when the KB lacks the condition), never Escalated;
  - a SELF-CARE QUESTION is a first-person statement of the user's OWN symptom with an ask for what to take ("I have a headache,
    what can I take?", "عندي صداع اخد ايه", "3andy sora3 a5od eh"). With no risk qualifier it is Answered with general KB
    information and a fixed pharmacist note; with a qualifier it is an Escalated Treatment Decision.

Risk qualifiers (each is about the PERSON asking, so a group phrasing stays a lookup): pregnancy or breastfeeding, a child or an
age under 18, an elderly person, kidney or liver disease or another chronic condition THE USER SAYS THEY HAVE (not the condition
being asked about: "for asthma" is a lookup, "I have asthma and a headache" is not), other medicines being taken, an allergy, a
severity or red-flag word, how long it has lasted, a request for a Dose. Another specific person (my wife, my son, my patient, a
pronoun with no generic noun before it) is never self-care. Migraine, arthritis, IBS and heartburn are everyday indications of
the KB, not chronic qualifiers.

`judge` returns what the pipeline does. The invariant: a first-person strict "what can I take" ask that states the user's own
symptom or condition is EITHER self-care (with the note) OR escalated, never a plain answer; the only status ever lowered is the
router's `treatment_decision` (never emergency, dose or diagnosis, which `rules.check` and the router decide and nothing lowers);
a qualifier found anywhere (the message, its typo-corrected form, the router's rewrite, the earlier turns of the Session) escalates.
Misspelled qualifier words ("pregnent", "sever", "childern") are matched within one edit of a short keyword list.

English, Arabic, Egyptian dialect and Franco-Arabic, on normalised text (textnorm.normalize).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .rules import check
from .textnorm import normalize
from .typos import damerau

# Appended server-side to an answered self-care reply, in the reply's language (Arabic for `ar` and `mixed`).
PHARMACIST_NOTE: dict[str, str] = {
    "en": "A pharmacist can tell you which of these suits you.",
    "ar": "يمكن للصيدلي أن يخبرك أيها يناسبك.",
}


def pharmacist_note(language: str) -> str:
    return PHARMACIST_NOTE["en" if language == "en" else "ar"]


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts))


_DUR_COUNT = r"(\d+|a few|a couple of|a couple|several|many|two|three|four|five|six|seven|eight|nine|ten|some)"
_AR_DUR = r"(يوم|يومين|ايام|اسبوع|اسبوعين|اسابيع|شهر|شهرين|شهور|سنه|سنتين|سنين|سنوات|فتره|مده)"

# ------------------------------------------------------------------------- misspelled qualifier words

# Short keyword list: a Latin word of five letters or more within ONE edit of exactly one of these is read as it. This is separate
# from the KB's typo corrector (typos.py) and works on the qualifier words only, so a misspelled risk is still a risk.
_FUZZ_WORDS = ("pregnant", "pregnancy", "breastfeeding", "severe", "severely", "children", "elderly", "diabetic", "diabetes",
                "kidney", "allergic", "alcohol", "hypertension", "asthma", "weakness", "dizziness", "unbearable")
_WORD = re.compile(r"[a-z]{5,}")


def _fix_keywords(t: str) -> str:
    def fix(m: re.Match[str]) -> str:
        w = m.group()
        if w in _FUZZ_WORDS:
            return w
        near = [k for k in _FUZZ_WORDS if abs(len(k) - len(w)) <= 1 and damerau(w, k, 1) <= 1]
        return near[0] if len(near) == 1 else w

    return _WORD.sub(fix, t)


def _prep(text: str) -> str:
    return _fix_keywords(normalize(text))


# ---------------------------------------------------------------------------------------- other person

# Possessive relatives and "my patient" are a specific person. A pronoun is one only with no generic noun before it in the
# sentence: "If a patient has asthma, what can he take?" and "What can a pregnant woman take for her heartburn?" are about a group.
_RELATIVE = _rx(
    r"\b(my|his|her|their|our)\s+(wife|husband|son|daughter|mother|mom|mum|mommy|father|dad|daddy|brother|sister|friend|baby|"
    r"child|children|kid|kids|boy|girl|patients?|grand\w+|cousin|uncle|aunt|neighbou?r|colleague|boyfriend|girlfriend|partner|"
    r"fianc\w*|relatives?|parents?|family|students?|customers?|clients?|employee|nephew|niece|\w+-in-law)\b",
    r"\b(my|his|her)\s+\d+[- ]?(year|yr|month|week)s?[- ]?old\b",
)
_PRONOUN = re.compile(r"\b(he|she|him|himself|herself|hers|his|her|someone|somebody|we|us|our|ourselves)\b")
_GENERIC_NOUN = re.compile(
    r"\b(a|an|the|any|every|each|some)\s+(\w+\s+){0,3}(patient|woman|man|person|adult|child|kid|lady|girl|boy|people|one|case|customer|client)\b"
    r"|\b(patients|women|men|people|adults|children|kids)\b")
_AR_RELATIVE = re.compile(
    r"\b[لوبف]?(ماما|بابا|امي|ابويا|ابوي|والدي|والدتي|ابني|ابنتي|بنتي|ولدي|زوجي|زوجتي|جوزي|مراتي|اخويا|اخي|اختي|صاحبي|صديقي|"
    r"صديقتي|جاري|جارتي|جدي|جدتي|حماتي|عمي|خالي|عمتي|خالتي)\b")
_AR_PRONOUN = re.compile(r"\b(عنده|عندها|عندهم|عندنا|بتاعه|بتاعها|احنا)\b")
_AR_GENERIC = re.compile(r"\b(مريض|مريضه|المريض|المريضه|عيان|عيانه|حد|واحد|شخص|ست|راجل|طفل|طفله)\b")
_FR_RELATIVE = re.compile(
    r"\b(mama|baba|mami|ommy|omy|abouya|abooya|abouy|ebny|ebni|ebn|bnty|binty|gozy|gozi|merati|mrati|a5oya|a5ouya|o5ty|2o5ty|"
    r"sa7by|so7by|tefly|walady)\b")
_FR_PRONOUN = re.compile(r"\b(3ando|3andaha|3andoh|3andahom|3andena|ya5od|ta5od|yetnawel|ahna|e7na)\b")
_FR_GENERIC = re.compile(r"\b(mareed|mareeda|3ayan|3ayana|tefl|walad|bent)\b")


def specific_other(text: str) -> bool:
    """The text is about another SPECIFIC person (a relative, "my patient", a pronoun with no generic noun before it)."""
    t = normalize(text)
    for s in _sentences(t):
        if _RELATIVE.search(s) or _AR_RELATIVE.search(s) or _FR_RELATIVE.search(s):
            return True
        for rx, gen in ((_PRONOUN, _GENERIC_NOUN), (_AR_PRONOUN, _AR_GENERIC), (_FR_PRONOUN, _FR_GENERIC)):
            for m in rx.finditer(s):
                if not gen.search(s[: m.start()]):
                    return True
    return False


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"[.?!؟\n]+", t) if s.strip()]


# --------------------------------------------------------------------------------------- stated by the user

_OWN_EN = re.compile(r"\b(i have|i had|i got|i am|i was|i suffer|i'm|im|i've|ive|my|suffer\w* from|diagnosed with|history of)\b")
_ASKED_EN = re.compile(r"\b(for|treat\w*|against|about)\s+(my |the |a |an |some |any )?(\w+ )?$")
_OWN_AR = re.compile(r"(عندي|عندى|انا|وانا|بعاني|مصاب|مصابه)")
_ASKED_AR = re.compile(r"(?:^|\s)(?:ل|لل|علاج|دوا|دواء|ادويه)(?:ال)?$")
_OWN_FR = re.compile(r"\b(3andy|3andi|3ndy|3ndi|3andii|ana)\b")
_ASKED_FR = re.compile(r"\b(le|lel|li|lil|dawa|3elag)\s+(el\s+)?$")


def _stated(t: str, term: re.Pattern[str]) -> bool:
    """`term` is a condition the user says THEY have: an own-marker ("I have", "I'm", "my", "عندي", "3andy") shortly before it in
    the same sentence, and it is not the condition being asked about ("for asthma", "treat hypertension", "لل ضغط")."""
    for s in _sentences(t):
        for m in term.finditer(s):
            before = s[: m.start()]
            tail = before[-45:]
            own = _OWN_EN.search(tail) or _OWN_AR.search(before[-40:]) or _OWN_FR.search(before[-40:])
            if not own:
                continue
            if _ASKED_EN.search(before) or _ASKED_AR.search(before) or _ASKED_FR.search(before):
                continue
            return True
    return False


# ------------------------------------------------------------------------------------------- qualifiers

_AGE_YEARS = re.compile(
    r"\b(\d{1,3})\s*[- ]?\s*(?:year|yr|y)s?\s*[- /]?\s*(?:old|o)\b"
    r"|\b(?:i'?m|i am|im|aged?|age is|age)\s*:?\s*(\d{1,3})\b(?!\s*(?:mg|mcg|ml|g|kg|%|days?|weeks?|months?|hours?|times))"
    r"|\b(\d{1,3})\s*(?:yo|y/o)\b"
    r"|(?:عندي|عمري|عمره|عمرها|سني)\s*(\d{1,3})\b"
    r"|\b(\d{1,3})\s*(?:سنه|سنين|سنوات|عام)(?!\s*(?:ونص|و نص))"
    r"|\b(?:3omry|3amri|3omri)\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:sana|sanaa|sanawat|sena)\b"
)
_AGE_SMALL = re.compile(r"\b\d+\s*[- ]?\s*(?:month|week|day)s?\s*[- ]?\s*old\b|\b\d+\s*(?:شهور|شهر|اسابيع|اسبوع)\s*(?:عمره|عمرها)")
MAX_CHILD_AGE = 17
MIN_ELDERLY_AGE = 65

# Stated about oneself ("I'm pregnant", "انا حامل"); "in pregnancy", "a pregnant woman", "للحامل" are group phrasings and do not count.
_PREGNANT_EN = re.compile(
    r"\b(i'?m|i am|im|i was|am|i think i'?m|i have been|my)\b[^.?!]{0,25}\b(preg\w*|expecting|breast ?feed\w*|nursing|lactat\w*|trimester)\b"
    r"|\bi\b[^.?!]{0,20}\b(breast ?feed\w*|nursing)\b|\b(breast ?feeding|nursing) (my|a) (baby|child|son|daughter)\b"
    r"|\b(i|my|me)\b[^.?!]*?(?<!\bin )(?<!\bduring )(?<!\bfor )(?<!\ba )(?<!\bthe )\b(preg\w*|breast ?feed\w*|nursing)\b"
    r"(?! (woman|women|patient|patients|lady|ladies|female|mothers?|mums?))")
_PREGNANT_OTHER = _rx(
    r"(?<!\w)[وف]?(حامل|حامله|برضع|مرضع\w*|ارضع|رضعت)(?!\w)|(?<!\w)حمل(?!\w)|بالرضاعه|رضاعه طبيعيه",
    r"\b(7amel|7amla|berda3|b?erda3|merd3a|merda3a)\b",
)
_CHILD_SELF = _rx(
    r"\b(i'?m|i am|im)\s+(a |an )?(child|kid|teen\w*|minor|adolescent|toddler)\b|\bunder (18|eighteen)\b",
    r"انا (طفل|طفله|مراهق|مراهقه|صغير|صغيره)|\bana (tefl|tefla)\b",
)
_ELDERLY_SELF = _rx(
    r"\b(i'?m|i am|im)\s+(an? )?(elderly|old|senior|geriatric|retired|pensioner)\b|\bin my (6|7|8|9)0s\b|\b(elderly|senior|geriatric)\b",
    r"انا (مسن|مسنه|كبير في السن|عجوز)|\b(mosen|3agooz)\b",
)
_KIDNEY_LIVER = _rx(
    r"\b(kidneys?|renal|nephr\w*|dialysis|creatinine|liver|hepatic|hepat\w*|cirrho\w*|jaundice)\b",
    r"كلي|كلى|كلوي|غسيل كلو|كبد|كبدي|صفرا|تليف",
    r"\b(kelya|kolya|kelwy|kebd|ghaseel kelwy)\b",
)
# Migraine, arthritis, IBS and GERD / heartburn are everyday indications the KB covers, not chronic qualifiers. A word that says the
# condition is long-lasting ("chronic", "long-term") always qualifies.
_CHRONIC = _rx(
    r"\b(diabet\w*|sugar|insulin|hypertens\w*|blood pressure|bp|asthma\w*|copd|heart|cardiac|cardio\w*|angina|arrhythm\w*|"
    r"ulcers?|epilep\w*|thyroid|hypothyroid\w*|cancer|chemo\w*|tumou?rs?|hiv|aids|lupus|gout|crohn\w*|colitis|glaucoma|"
    r"cholesterol|stroke|clots?|anemi\w*|anaemi\w*|sickle|tuberculosis|tb|pneumonia|depress\w*|schizo\w*|bipolar|psychiatric|"
    r"transplant|immuno\w*|disease|disorder|syndrome)\b",
    r"سكر|ضغط|ربو|حساسيه الصدر|قلب|قلبي|ذبحه|جلطه|سكته|صرع|غده|درقيه|سرطان|ورم|كوليسترول|دهون|قرحه|نقرس|انيميا|فقر دم|\bسل\b|"
    r"التهاب رئوي|رئوي|اكتئاب|\bمرض\b|زرع",
    r"\b(sokkar|sukkar|sokar|daghT|daght|dght|rabw|rabu|2alb|sarataan|kolestrol|nokres|el2alb)\b",
)
_ALWAYS_CHRONIC = _rx(r"\b(chronic|long[- ]term)\b", r"مزمن\w*|\bmozmen\b|\bmezmen\b")

_OTHER_MEDICINES = _rx(
    r"\b(i|im|i'm|i am|i was|i have been|i've been|ive been)\s+(also |currently |already |still |regularly )?(taking|using)\b",
    r"\bi (already|also|currently|regularly|usually|daily|still|always)\s+(take|use)\b",
    r"\bi (take|use)\s+(?!for\b|if\b|when\b|to\b|at\b|in\b|with\b|and\b|or\b|it\b|that\b|this\b|them\b|something\b|anything\b|"
    r"what\b|which\b|now\b|today\b|tonight\b|tomorrow\b|first\b|here\b|again\b|so\b|as\b|on\b|from\b|about\b|after\b|before\b|"
    r"during\b|without\b|safely\b|just\b|while\b|then\b|instead\b|the best\b|the most\b|some\b|any\b)[a-z]",
    r"\b(my|current|regular|prescribed) (medic\w*|drugs?|meds?|pills|tablets|treatment|prescription\w*|therapy)\b",
    r"\bon (medication|medicine|treatment|meds|antibiotics?|steroids?)\b",
    r"\bi'?m on\s+(\w+\s+){0,2}(warfarin|clopidogrel|plavix|insulin|metformin|lithium|methotrexate|digoxin|statins?|"
    r"blood thinners?|anticoagulants?|antiplatelets?|steroids?|cortisone|prednisolone|antidepressants?|contraceptives?|"
    r"birth control|the pill|ssri|diuretics?|beta[- ]?blockers?|ace inhibitors?)\b",
    r"\b(along|together) with\b|\bin addition to\b|\bon top of\b|\bwhile (taking|on|using)\b|\bwith (my|other|another)\b",
    r"\b(instead of|(switch|replace|change|stop|start)\w*\s+(to |taking |using |my |the )(medic\w*|drugs?|meds?|pills?|tablets?|treatment|taking|using|\w+ing)\b)",
    r"\b(resistan\w*|not (working|helping)|didn'?t (work|help)|doesn'?t (work|help)|did not (work|help)|no (effect|relief))\b",
    r"باخد|بخد|بتناول|بستخدم|بستعمل|اخدت|استخدمت|علاج تاني|ادويه تانيه|دوا تاني|ادويتي|دوايا|علاجي|مع ادويه|مع دوا|"
    r"مع علاج|بدل\b|بدلا من|بدال|ابطل|اوقف|اغير|مفيش فايده|ما نفعش|منفعش|مش نافع|مش بيفرق",
    r"\b(ba5od|bakhod|ba5d|bastakhdem|dawaya|adwety|dawa tany|badal|abatal|a2af|aghayar|mesh bye?far2|ma nafa3sh)\b",
)
_ALLERGY = _rx(
    r"\ballerg(ic|y|ies)\s+(to|against)\b|\b(i'?m|i am|am|is|are)\s+allergic\b|\ballergic reaction\b|\bhypersensitiv\w*|"
    r"\banaphyla\w*|\bintoleran\w*",
    r"(عندي|عنده|عندها)?\s*حساسيه\s*(من|ضد|ل)\b|حساسيه (بنسلين|مضاد|دوا)|بيتحسس|بتتحسس|اتحسس|متحسس",
    r"\b(7asasiya|hasasya|hasasiyya)\s*(men|ded|le)\b|\bbyet7ases|\bbet7ases|\bmet7ases",
)
_SEVERITY = _rx(
    r"\b(worst|worse|worsen\w*|sudden\w*|sev[eai]+r\w*|severity|very (bad|strong|high|painful|intense|much)|really (bad|strong|painful)|"
    r"so (bad|painful)|terrible|horrible|awful|unbearable|intolerable|excruciating|agoni\w*|intense\w*|extreme\w*|"
    r"killing me|can'?t (stand|bear|take it|sleep|eat|move|walk|breathe|open)|cannot (stand|bear)|persistent|persisting|"
    r"recurr\w*|repeated\w*|keeps? (coming|returning|happening|getting)|again and again|won'?t (go|stop)|not going away|"
    r"doesn'?t go away|never (goes|ends|stops)|constant\w*|continu\w+|ongoing|getting worse|spreading|radiat\w*|throbbing|"
    r"stabbing|pounding|bleeding|blood|bloody|black stools?|tarry|blurr\w*|vision|numb\w*|tingl\w*|weak\w*|"
    r"dizz\w*|faint\w*|chest|palpitation\w*|confus\w*|swollen|swelling|lump|unusual|strange|first time|"
    r"never (had|felt|experienced)|head injury|hit my head|trauma|surgery|operation|post[- ]?op\w*|emergency|urgent\w*|"
    r"alcohol|drunk|hangover)\b",
    r"\bstiff\w*[^.?!]{0,20}\bneck|\bneck[^.?!]{0,20}\bstiff",
    r"\b(fever|temperature|temp)\b[^.?!]{0,15}\b(39|40|41|42|10[3-9])\b",
    r"\b(10[3-9]|39|4[0-2])(\.\d)?\s*(°|degrees?|f|c)\b",
    # for / since / over a stretch of days, weeks, months, years (hours and minutes are an ordinary headache)
    rf"\b(for|since|over|past|last)\s+(about |almost |nearly |around |more than |the |past |last )*({_DUR_COUNT}\s*(days?|weeks?|months?|years?)|"
    r"(a|one|an)\s+(week|month|year)|days|weeks|months|years|ages|long|a long time)\b",
    rf"\b{_DUR_COUNT}\s*(days?|weeks?|months?)\s+(now|ago|already)\b|\b(days|weeks|months) now\b",
    # Arabic
    r"\bجامد\w*|\bشديد\w*|\bرهيب\w*|\bفظيع\w*|مش طايق\w*|مش قادر\w*|مش متحمل\w*|لا يطاق|لا تحتمل|مش بيروح|مبيروحش|"
    r"مش راضي يروح|ما بيروحش|مش بيخف|مبيخفش|بيزيد|بتزيد|بيزداد|\bاسوا\b|اسوء|فجاه|مفاجئ\w*|مفاجا\w*|\bحاد\w*|\bقوي\b|\bقويه\b|"
    r"\bجدا\b|\bاوي\b|\bاووي+\b|\bوحش\w*|\bصعب\b|مستمر\w*|متكرر\w*|بيتكرر|\bدايم\w*|كل شويه|\bدم\b|نزيف|دوخه|دوار|زغلل\w*|"
    r"تنميل|خدر|تورم|\bورم\b|اول مره|عمري ما|ضربه|حادث\w*|طوارئ|اسعاف|صدري|صدر\b|خفقان|رعشه|تشنج|"
    r"ضعف|ضعيف\w*|كحول|خمره|سكران|"
    r"رقب\w*[^.؟?!]{0,20}(ناشف\w*|متخشب\w*|خشب\w*|يابس\w*)|(ناشف\w*|متخشب\w*)[^.؟?!]{0,20}رقب\w*",
    rf"\b(من|بقالي|بقالى|بقاله|بقالها|لمده|بقلي)\s+(\w+\s+){{0,2}}({_AR_DUR}|كتير|كثير)\b|\bمن (يوم|امبارح|ايام|فتره)\b",
    rf"\b\d+\s*{_AR_DUR}\b|\b(يومين|ايام|اسبوعين|اسابيع|شهرين|شهور|سنين)\b",
    # Franco-Arabic
    r"\b(shedeed|shadeed|shaded|gamed|gamda|gamd|gaamed|ra3ee?b|fazee?3|mesh (ta2e?t|tayye?2|taye2|me7tamel|mt7ml)|"
    r"mish (ta2et|tayyi2)|awy|awi|awii?|2awy|geddan|gedan|fag2a|fag2et|byzeed|bizeed|btzeed|aswa2|da2em|damman|nazif|dokha|"
    r"dawkha|zaghlala|tanmeel|ayam|yomen|yomeen|esbo3|esbo3een|shahr|shahren|sakran|khamra|d3f|do3f)\b",
    r"\b(men|ba2ali|ba2aly|ba2alo)\s+(\w+\s+)?(ayam|yom|youm|yomen|yomeen|esbo3|esbo3een|shahr|shahren|sena|sanawat)\b",
    r"\b\d+\s*(ayam|yom|esbo3|shahr|sana|sanawat)\b",
)
_DOSE = _rx(
    r"\bhow (much|many|often|long|frequently)\b|\b(dose|doses|dosage|dosing|posology)\b|\bmaximum\b|\bmax\b|\boverdos\w*|\btoo much\b|"
    r"\bsafe (amount|limit)\b|\bper (day|dose|hour|week)\b|\bdaily\b|\btimes a\b|\b(once|twice|thrice) (a|per|every)\b",
    r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|gm|grams?|iu|units?|tablets?|pills?|capsules?|drops?|sachets?)\b",
    r"\bكام\b|\bكم\b|جرع\w*|مجم|ملجم|ملغ|\bحب[ه]\b|\bحبايه\b|\bحبوب\b|\bقرص\b|اقراص|كبسول\w*|نقط\w*|\bمل\b|مره في|مرتين|\bمرات\b|"
    r"في اليوم|باليوم|يوميا|اقصي|اقصى|حد اقصي",
    r"\bka+m\b|\bgor3\w*|\bgar3\w*|\bdoza\b|\bmg\b|\b7ab(a|et|at)\b|\bhab(a|et|at)\b|\b2ors\b|\bmara\b|\bmarra\b|\bmarteen\b|\bfel yom\b",
)

# Families that are about the person by construction (they need an own-marker, a first-person form or a stated age) versus the
# symptom words (severity, duration), which need the user's own symptom statement to count.
OWN_FAMILIES = ("pregnancy", "child", "elderly", "kidney_liver", "chronic", "other_medicines", "allergy", "age")


def _age_flag(text: str) -> bool:
    """An age under 18 or from 65, or a baby counted in months or weeks."""
    if _AGE_SMALL.search(text):
        return True
    for m in _AGE_YEARS.finditer(text):
        for g in m.groups():
            if g is not None:
                age = int(g)
                if age <= MAX_CHILD_AGE or age >= MIN_ELDERLY_AGE:
                    return True
    return False


def qualifiers(message: str) -> list[str]:
    """The risk-qualifier families present in the message (empty when none); "other_person" for another specific person."""
    t = _prep(message)
    found: list[str] = []
    if _PREGNANT_EN.search(t) or _PREGNANT_OTHER.search(t):
        found.append("pregnancy")
    if _CHILD_SELF.search(t):
        found.append("child")
    if _ELDERLY_SELF.search(t):
        found.append("elderly")
    if _age_flag(t):
        found.append("age")
    if _stated(t, _KIDNEY_LIVER):
        found.append("kidney_liver")
    if _stated(t, _CHRONIC) or _ALWAYS_CHRONIC.search(t):
        found.append("chronic")
    for name, rx in (("other_medicines", _OTHER_MEDICINES), ("allergy", _ALLERGY), ("severity", _SEVERITY), ("dose", _DOSE)):
        if rx.search(t):
            found.append(name)
    if specific_other(t):
        found.append("other_person")
    return found


# ----------------------------------------------------------------------------------- what the user asks

# The strict ways of asking what to take (a verb of taking): the only asks that can RAISE a status.
_STRICT_ASK = _rx(
    r"\b(what|which)\b[^.?!]{0,40}\bi\b[^.?!]{0,12}\b(take|use|try|drink|get|buy)\b",
    r"\bwhat to (take|use|try|drink)\b",
    r"(?<!\w)(اخد|اخذ|اتناول|استخدم|استعمل|اشرب|اجرب|اجيب|اشتري)\s+(ايه|اي|ايش|ماذا|ما)\b",
    r"\b(ممكن|ينفع|اقدر)\s+(اخد|اخذ|اتناول)\s+(ايه|اي)\b",
    r"\b(a5od|akhod|a5d|akhd|a5ud|a5ood|akhud)\s+(eh|ehh|eih|ay|ey|e7|ayh|aye)\b",
)
_ASK = _rx(
    _STRICT_ASK.pattern,
    r"\b(what|which)\s+(is |are |'s )?(the )?(best|good|better|safe|safest|recommended|helpful|right)\b",
    r"\b(what|which)\s+(medicine|medication|drug|pill|tablet|painkiller|painkillers|remedy|treatment|meds)s?\b",
    r"\bany (medicine|medication|drug|pill|tablet|painkiller|remedy)s?\b|\banything (for|to|that)\b|\bwhat (helps|works)\b|"
    r"\bwhat (do|would) you (recommend|suggest)\b|\brecommend\w*|\bsuggest\w*",
    r"\b(ايه|اي|ايش|ماذا|ما)\s+(هو\s+)?(ال)?(دوا|دواء|علاج|مسكن|حباية|حبايه|ادويه|احسن|افضل|الانسب|المناسب|اللي ينفع|ينفع|يساعد|يفيد|يخفف)",
    r"\bايه اللي (اخده|اخدها|اشربه|اتناوله|ينفع)",
    r"\b(eh|eih|ay|ayh)\s+(el\s+)?(dawa|dawaa|3elag|mosaken|a7san|afdal|el a7san)\b|\bdawa\s+(le|lel|li|lil)\b",
)
# A first-person mark: needed because some ways of asking ("what is good for a headache?") have no subject.
_FIRST_PERSON = _rx(
    r"\b(i|i'm|im|i've|ive|i'd|my|me|myself)\b",
    r"\b(عندي|انا|حاسس|حاسه|بعاني|تعبان|تعبانه|اخد|اخذ|اتناول|استخدم|اشرب|اجرب|راسي|دماغي|بطني|زوري|حلقي|معدتي|ضهري|رجلي|ايدي|سناني|ضرسي)\b",
    r"\b(3andy|3andi|3ndy|3ndi|3andii|ana|a5od|akhod|a5d|akhd)\b",
)
# The user states their OWN symptom or condition: "I have", "I feel", "my head hurts", "عندي", "3andy".
_OWN_SYMPTOM = _rx(
    r"\bi (have|had|got|feel|get|suffer)\b|\bi'?m (having|suffering|feeling)\b|\bi am (having|suffering|feeling)\b|\bi've\b|\bive (got|had)\b",
    r"\bmy (head|stomach|throat|back|tooth|teeth|ear|ears|eye|eyes|chest|knee|leg|arm|neck|belly|tummy|nose)\b|\bmy \w*(ache|aches)\b",
    r"\bi'?m\b[^.?!]{0,30}\b(have|having|feel|feeling|suffer\w*)\b",
    r"(عندي|عندى|حاسس|حاسه|بعاني|تعبان|تعبانه|راسي|دماغي|بطني|زوري|حلقي|معدتي|ضهري|سناني|ضرسي)",
    r"\b(3andy|3andi|3ndy|3ndi|3andii)\b",
)


def asks_what_to_take(message: str) -> bool:
    """First-person wording that asks what to take (or what is good) for something."""
    t = normalize(message)
    return _ASK.search(t) is not None and _FIRST_PERSON.search(t) is not None


def asks_strictly(message: str) -> bool:
    """A first-person "what can I take / use", "اخد ايه", "a5od eh": the only asks that can raise a status."""
    return _STRICT_ASK.search(normalize(message)) is not None


def states_own_symptom(message: str) -> bool:
    return _OWN_SYMPTOM.search(normalize(message)) is not None


# -------------------------------------------------------------------------------------------- the verdict

@dataclass(frozen=True)
class Verdict:
    raise_to: str | None = None  # "treatment_decision" or "dose": a status the router may have missed
    lower: bool = False  # the router's treatment_decision is lowered to none (a lookup or a self-care question)
    self_care: bool = False  # the user's own symptom, no qualifier anywhere: the pharmacist note is added


# Another specific person asking what to take or be given ("my wife has a headache, what can I give her", "ماما عندها صداع تاخد ايه").
_OTHER_ASK = _rx(
    r"\bwhat\b[^.?!]{0,40}\b(take|use|give|giving|try)\b",
    r"(?<!\w)(ياخد|تاخد|ياخدوا|اخد|اديه|اديها|اديله|اديلها|اعطيه|اعطيها|اديلو)\s+(ايه|اي|ايش)\b",
    r"\b(ya5od|ta5od|a5od|edeeh|edeha|edeelo|edeela|edih|edi)\s+(eh|ay|ey)\b",
)


def judge(message: str, *also: str, context: tuple[str, ...] = (), rewrite: str | None = None) -> Verdict:
    """What the self-care policy says about a message, whose forms are `message` and `also` (its typo-corrected text), with the
    earlier user turns in `context` and the router's standalone rewrite as `rewrite`.

    A first-person strict ask ("what can I take / use ..."):
      - a Dose wording ("and how many?") raises dose; another specific person, a qualifier the user states about themselves (or
        a symptom word with their own symptom statement), or any qualifier or other person in the earlier turns or the rewrite
        raises treatment_decision;
      - otherwise it is a lookup (lowered, answered as a normal question), and with the user's own symptom statement it is
        self-care (the pharmacist note is added).
    No strict ask, but a group phrasing ("what can a pregnant woman take for her heartburn?", "if a patient has asthma, what can
    he take?") with no specific person: lowered. Everything else is left to the router."""
    texts = tuple(t for t in (message, *also) if t)
    ctx = tuple(t for t in (*context, rewrite) if t)
    q_msg = {f for t in texts for f in qualifiers(t)}
    q_ctx = {f for t in ctx for f in qualifiers(t)}
    own = any(states_own_symptom(t) for t in texts)
    strict = any(asks_strictly(t) for t in texts)
    broad = strict or any(asks_what_to_take(t) for t in texts)
    if not strict and "other_person" in q_msg and any(_OTHER_ASK.search(normalize(t)) for t in texts):
        return Verdict(raise_to="treatment_decision")
    if strict:
        if "dose" in q_msg:
            return Verdict(raise_to="dose")
        if "other_person" in q_msg or q_ctx or q_msg & set(OWN_FAMILIES) or ("severity" in q_msg and own):
            return Verdict(raise_to="treatment_decision")
        return Verdict(lower=True, self_care=own and not q_msg)
    if broad and own and not q_msg and not q_ctx:
        return Verdict(lower=True, self_care=True)
    if broad and q_ctx:
        return Verdict()
    if not q_ctx and "other_person" not in q_msg and any(_GENERIC_NOUN.search(normalize(t)) or _AR_GENERIC.search(normalize(t))
                                                          for t in texts) and not own:
        return Verdict(lower=True)
    return Verdict()


def is_self_care(message: str, *also: str, context: tuple[str, ...] = ()) -> bool:
    """True when the message is a self-care question (the note is added): the user's own symptom, asking what to take, no qualifier
    and no Emergency or Dose request (`rules.check`)."""
    if any(check(t) for t in (message, *also) if t):
        return False
    return judge(message, *also, context=context).self_care
