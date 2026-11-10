"""Self-care questions (ADR 0003): a first-person question about the user's OWN everyday symptom, asking what they
can take ("I have a headache, what can I take?", "عندي صداع اخد ايه", "3andy sora3 a5od eh").

Such a question is answered with general Knowledge Base information (the drugs and drug classes the KB lists for the
condition, with Citations) and a fixed note that a pharmacist can say which one suits the person. It is NOT a Treatment
Decision. It stays Escalated when ANY risk qualifier is present: pregnancy or breastfeeding, a child or an age under 18,
an elderly person, kidney or liver disease or another chronic condition, other medicines being taken, an allergy, a
severity or red-flag word, how long it has lasted, a request for a Dose, or an Emergency. A question about another
specific person (my wife, my son, my patient, he, she) is never self-care.

`is_self_care` is deterministic and only ever answers True for a message that is first-person, asks what to take and has
no qualifier. `pipeline.decide` uses it for exactly one thing: it may lower a router's `treatment_decision` to none. It
never lowers emergency, dose or diagnosis (those come from `rules.check` and are never lowered by anything), and a
qualifier found anywhere (the message, its typo-corrected form, the router's rewrite, the earlier turns of the Session)
keeps the escalation. The lists are deliberately broad: a word that is not a risk but sits on a list only escalates a
question that would otherwise have been answered, which is the safe direction.

English, Arabic, Egyptian dialect and Franco-Arabic, on normalised text (textnorm.normalize).
"""
from __future__ import annotations

import re

from .rules import check
from .textnorm import normalize

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

# ------------------------------------------------------------------------------------------- other person

_OTHER_PERSON = _rx(
    r"\b(my|his|her|their|our|your)\s+(wife|husband|son|daughter|mother|mom|mum|mommy|father|dad|daddy|brother|sister|friend|"
    r"baby|child|children|kid|kids|boy|girl|patients?|grand\w+|cousin|uncle|aunt|neighbou?r|colleague|boyfriend|girlfriend|"
    r"partner|fianc\w*|relatives?|parents?|family|students?|customers?|clients?|employee|nephew|niece|\w+-in-law)\b",
    r"\b(he|she|him|himself|herself|hers|they|them|their|someone|somebody|we|us|our|ourselves)\b",
    r"\b(his|her)\b",
    r"\b(a|the) (man|woman|boy|girl|patient|child|customer|client|lady|guy)\b",
    r"\bfor (a|the|my|this) (patient|customer|client|person|man|woman|lady|guy)\b",
    r"\bpatients?\b|\bcustomers?\b|\bclients?\b",
    # Arabic (normalised: alef, ya, ta marbuta folded)
    r"\b[لوبف]?(ماما|بابا|امي|ابويا|ابوي|والدي|والدتي|ابني|ابنتي|بنتي|ولدي|زوجي|زوجتي|جوزي|مراتي|اخويا|اخي|اختي|صاحبي|"
    r"صديقي|صديقتي|جاري|جارتي|جدي|جدتي|حماتي|حمايا|عمي|خالي|عمتي|خالتي|ابن|بنت|عيل|عيله|طفل|طفله|اطفال|رضيع|"
    r"مريض|مريضه|المريض|المريضه|عيان|عيانه|زبون|عميل|عنده|عندها|عندهم|عندنا|بتاعه|بتاعها|احنا|لها|له|لهم)\b",
    # Franco-Arabic
    r"\b(mama|baba|mami|ommy|omy|abouya|abooya|abouy|ebny|ebni|ebn|bnty|binty|bent|gozy|gozi|merati|mrati|a5oya|a5ouya|"
    r"o5ty|2o5ty|sa7by|so7by|tefl|tefly|walad|walady|3ayan|3ayana|mareed|mareeda|3ando|3andaha|3andoh|3andahom|3andena|"
    r"3ayal|3yal|3eyal|ya5od|ta5od|yetnawel|ahna|e7na)\b",
)

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

_FAMILIES: dict[str, re.Pattern[str]] = {
    "pregnancy": _rx(
        r"\b(pregnan\w*|expecting|breast ?feed\w*|nursing|lactat\w*|trimester|gestation\w*|prenatal|antenatal|conceiv\w*)\b",
        r"حامل|حمل\b|حملي|الحمل|رضاعه|بالرضاعه|برضع|مرضع\w*|ارضع|رضع|7amel|7aml|ra3a|rada3a|b?erda3|7awamel|7ml",
    ),
    "child": _rx(
        r"\b(child|children|kid|kids|infant|infants|baby|babies|toddler|toddlers|newborn|neonat\w*|teen|teens|teenager|"
        r"adolescent|pediatric|paediatric|minor|boy|girl|son|daughter|school)\b|\bunder (18|eighteen)\b",
        r"طفل|اطفال|رضيع|عيل|عيال|ولد|بنت|ابن|مراهق|صغير|صغيره|مدرسه",
        r"\b(tefl|atfal|rade3|ra?d[ei]3|walad|bent|eyal|3eyal|3yal)\b",
    ),
    "elderly": _rx(
        r"\b(elderly|senior|seniors|geriatric|old (man|woman|age|person|people|lady)|pensioner|retired|in my (6|7|8|9)0s)\b",
        r"مسن|مسنه|كبير في السن|كبره|عجوز|شيخ|كبار السن|معاش",
        r"\b(mosen|mosena|3agooz|kebeer fe el sen)\b",
    ),
    "kidney_liver": _rx(
        r"\b(kidneys?|renal|nephr\w*|dialysis|creatinine|liver|hepatic|hepat\w*|cirrho\w*|jaundice)\b",
        r"كلي|كلى|الكلي|كلوي|غسيل كلو|كبد|الكبد|كبدي|صفرا|تليف",
        r"\b(kelya|kolya|kelwy|kebd|el kebd|ghaseel kelwy)\b",
    ),
    "chronic": _rx(
        r"\b(diabet\w*|sugar|insulin|hypertens\w*|blood pressure|bp|asthma\w*|copd|heart|cardiac|cardio\w*|angina|arrhythm\w*|"
        r"ulcers?|epilep\w*|thyroid|hypothyroid\w*|cancer|chemo\w*|tumou?rs?|hiv|aids|lupus|arthritis|rheumat\w*|gout|ibs|"
        r"irritable|crohn\w*|colitis|glaucoma|cholesterol|stroke|clots?|anemi\w*|anaemi\w*|sickle|tuberculosis|tb|pneumonia|"
        r"depress\w*|schizo\w*|bipolar|psychiatric|transplant|immuno\w*|chronic|long[- ]term|condition|disease|disorder|"
        r"syndrome|migraines?)\b",
        r"سكر|ضغط|ربو|حساسيه الصدر|قلب|قلبي|ذبحه|جلطه|سكته|صرع|غده|درقيه|سرطان|ورم|كوليسترول|دهون|قرحه|قولون|"
        r"روماتيزم|روماتويد|نقرس|التهاب مفاصل|مفاصل|انيميا|فقر دم|سل\b|رئوي|التهاب رئوي|اكتئاب|مزمن\w*|امراض|مرض\b|زرع",
        r"\b(sokkar|sukkar|el sokar|daghT|daght|dght|rabw|rabu|2alb|alb|sarataan|kolestrol|nokres|ro2matizm|ro2matism|mozmen|mezmen|"
        r"ta7ses sadr|el2alb|el 2alb)\b",
    ),
    "other_medicines": _rx(
        r"\b(i|im|i'm|i am|i was|i have been|i've been|ive been)\s+(also |currently |already |still |regularly )?(taking|on|using)\b",
        r"\bi (already|also|currently|regularly|usually|daily|still|always)\s+(take|use)\b",
        r"\bi (take|use)\s+(?!for\b|if\b|when\b|to\b|at\b|in\b|with\b|and\b|or\b|it\b|that\b|this\b|them\b|something\b|anything\b|"
        r"what\b|which\b|now\b|today\b|tonight\b|tomorrow\b|first\b|here\b|again\b|so\b|as\b|on\b|from\b|about\b|after\b|before\b|"
        r"during\b|without\b|safely\b|just\b|while\b|then\b|instead\b|the best\b|the most\b|some\b|any\b)[a-z]",
        r"\b(other|another|current|regular|daily|usual|prescribed|my) (medic\w*|drugs?|meds?|pills|tablets|treatment|prescription\w*|therapy)\b",
        r"\bon (medication|medicine|treatment|meds|antibiotics?|steroids?)\b",
        r"\b(along|together) with\b|\bin addition to\b|\bon top of\b|\bwhile (taking|on|using)\b|\bwith (my|other|another)\b",
        r"\b(combine|combined|combination|mix|mixing|interact\w*|interfere\w*)\b",
        r"\b(warfarin|clopidogrel|plavix|insulin|metformin|lithium|methotrexate|digoxin|statins?|blood thinners?|anticoagulants?|"
        r"antiplatelets?|steroids?|cortisone|prednisolone|antidepressants?|contraceptives?|birth control|ssri|diuretics?|"
        r"beta[- ]?blockers?|ace inhibitors?)\b",
        r"\b(instead of|switch\w*|replac\w*|stop\w*|start\w*|change)\b",
        r"\b(resistan\w*|not (working|helping)|didn'?t (work|help)|doesn'?t (work|help)|did not (work|help)|no (effect|relief))\b",
        r"باخد|بخد|باخد|بتناول|بستخدم|بستعمل|اخدت|استخدمت|علاج تاني|ادويه تانيه|دوا تاني|ادويتي|دوايا|علاجي|مع ادويه|مع دوا|"
        r"مع علاج|جانب|بدل\b|بدلا من|بدال|ابطل|اوقف|اغير|مفيش فايده|ما نفعش|منفعش|مش نافع|مش بيفرق|خلصت",
        r"\b(ba5od|bakhod|bakhod|ba5d|bastakhdem|7at|dawaya|adwety|dawa tany|badal|abatal|a2af|aghayar|mesh bye?far2|ma nafa3sh)\b",
    ),
    "allergy": _rx(
        r"\ballerg(ic|y|ies)\s+(to|against)\b|\b(i'?m|i am|am|is|are)\s+allergic\b|\ballergic reaction\b|\bhypersensitiv\w*|"
        r"\banaphyla\w*|\bintoleran\w*",
        r"(عندي|عنده|عندها)?\s*حساسيه\s*(من|ضد|ل)\b|حساسيه (بنسلين|مضاد|دوا)|بيتحسس|بتتحسس|اتحسس|متحسس",
        r"\b(7asasiya|hasasya|hasasiyya)\s*(men|ded|le)\b|\bbyet7ases|\bbet7ases|\bmet7ases",
    ),
    "severity": _rx(
        r"\b(worst|worse|worsen\w*|sudden\w*|severe\w*|severity|very (bad|strong|high|painful|intense|much)|really (bad|strong|painful)|"
        r"so (bad|painful)|terrible|horrible|awful|unbearable|intolerable|excruciating|agoni\w*|intense\w*|extreme\w*|"
        r"killing me|can'?t (stand|bear|take it|sleep|eat|move|walk|breathe|open)|cannot (stand|bear)|persistent|persisting|"
        r"recurr\w*|repeated\w*|keeps? (coming|returning|happening|getting)|again and again|won'?t (go|stop)|not going away|"
        r"doesn'?t go away|never (goes|ends|stops)|constant\w*|continu\w+|ongoing|getting worse|spreading|radiat\w*|throbbing|"
        r"stabbing|pounding|bleeding|blood|bloody|black stools?|tarry|blurr\w*|vision|stiff neck|numb\w*|tingl\w*|weakness|"
        r"dizz\w*|faint\w*|chest|palpitation\w*|confus\w*|swollen|swelling|lump|unusual|strange|first time|"
        r"never (had|felt|experienced)|head injury|hit my head|trauma|surgery|operation|post[- ]?op\w*|emergency|urgent\w*)\b",
        r"\b(fever|temperature|temp)\b[^.?!]{0,15}\b(39|40|41|42|10[3-9])\b",
        # for / since / over a stretch of days, weeks, months, years (hours and minutes are an ordinary headache)
        rf"\b(for|since|over|past|last)\s+(about |almost |nearly |around |more than |the |past |last )*({_DUR_COUNT}\s*(days?|weeks?|months?|years?)|"
        r"(a|one|an)\s+(week|month|year)|days|weeks|months|years|ages|long|a long time)\b",
        rf"\b{_DUR_COUNT}\s*(days?|weeks?|months?)\s+(now|ago|already)\b|\b(days|weeks|months) now\b",
        # Arabic
        r"\bجامد\w*|\bشديد\w*|\bرهيب\w*|\bفظيع\w*|مش طايق\w*|مش قادر\w*|مش متحمل\w*|لا يطاق|لا تحتمل|مش بيروح|مبيروحش|"
        r"مش راضي يروح|ما بيروحش|مش بيخف|مبيخفش|بيزيد|بتزيد|بيزداد|\bاسوا\b|اسوء|فجاه|مفاجئ\w*|مفاجا\w*|\bحاد\w*|\bقوي\b|\bقويه\b|"
        r"\bجدا\b|\bاوي\b|\bاووي+\b|\bوحش\w*|\bصعب\b|مستمر\w*|متكرر\w*|بيتكرر|\bدايم\w*|كل شويه|\bدم\b|نزيف|دوخه|دوار|زغلل\w*|"
        r"تنميل|خدر|تورم|\bورم\b|اول مره|عمري ما|ضربه|حادث\w*|طوارئ|اسعاف|صدري|صدر\b|خفقان|رعشه|تشنج",
        rf"\b(من|بقالي|بقالى|بقاله|بقالها|لمده|بقلي)\s+(\w+\s+){{0,2}}({_AR_DUR}|كتير|كثير)\b|\bمن (يوم|امبارح|ايام|فتره)\b",
        rf"\b\d+\s*{_AR_DUR}\b|\b(يومين|ايام|اسبوعين|اسابيع|شهرين|شهور|سنين)\b",
        # Franco-Arabic
        r"\b(shedeed|shadeed|shaded|gamed|gamda|gamd|gaamed|ra3ee?b|fazee?3|mesh (ta2e?t|tayye?2|taye2|me7tamel|mt7ml)|"
        r"mish (ta2et|tayyi2)|awy|awi|awii?|2awy|geddan|gedan|fag2a|fag2et|byzeed|bizeed|btzeed|aswa2|da2em|damman|nazif|dokha|"
        r"dawkha|zaghlala|tanmeel|ayam|yomen|yomeen|esbo3|esbo3een|shahr|shahren)\b",
        r"\b(men|ba2ali|ba2aly|ba2alo)\s+(\w+\s+)?(ayam|yom|youm|yomen|yomeen|esbo3|esbo3een|shahr|shahren|sena|sanawat)\b",
        r"\b\d+\s*(ayam|yom|esbo3|shahr|sana|sanawat)\b",
    ),
    "dose": _rx(
        r"\bhow (much|many|often|long|frequently)\b|\b(dose|doses|dosage|dosing|posology)\b|\bmaximum\b|\bmax\b|\boverdos\w*|\btoo much\b|"
        r"\bsafe (amount|limit)\b|\bper (day|dose|hour|week)\b|\bdaily\b|\btimes a\b|\b(once|twice|thrice) (a|per|every)\b",
        r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|gm|grams?|iu|units?|tablets?|pills?|capsules?|drops?|sachets?)\b",
        r"\bكام\b|\bكم\b|جرع\w*|مجم|ملجم|ملغ|\bحب[ه]\b|\bحبايه\b|\bحبوب\b|\bقرص\b|اقراص|كبسول\w*|نقط\w*|\bمل\b|مره في|مرتين|\bمرات\b|"
        r"في اليوم|باليوم|يوميا|اقصي|اقصى|حد اقصي",
        r"\bka+m\b|\bgor3\w*|\bgar3\w*|\bdoza\b|\bmg\b|\b7ab(a|et|at)\b|\bhab(a|et|at)\b|\b2ors\b|\bmara\b|\bmarra\b|\bmarteen\b|\bfel yom\b",
    ),
}


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
    """The risk-qualifier families present in the message (empty when none). Includes "other_person" for a question
    about someone else, because that too keeps a question out of self-care."""
    t = normalize(message)
    found = [name for name, rx in _FAMILIES.items() if rx.search(t)]
    if _age_flag(t):
        found.append("age")
    if _OTHER_PERSON.search(t):
        found.append("other_person")
    return found


# ----------------------------------------------------------------------------------- what the user asks

_ASK = _rx(
    r"\b(what|which)\b[^.?!]{0,40}\bi\b[^.?!]{0,12}\b(take|use|try|drink|get|buy)\b",
    r"\bwhat to (take|use|try|drink)\b",
    r"\b(what|which)\s+(is |are |'s )?(the )?(best|good|better|safe|safest|recommended|helpful|right)\b",
    r"\b(what|which)\s+(medicine|medication|drug|pill|tablet|painkiller|painkillers|remedy|treatment|meds)s?\b",
    r"\bany (medicine|medication|drug|pill|tablet|painkiller|remedy)s?\b|\banything (for|to|that)\b|\bwhat (helps|works)\b|"
    r"\bwhat (do|would) you (recommend|suggest)\b|\brecommend\w*|\bsuggest\w*",
    # Arabic
    r"(?<!\w)(اخد|اخذ|اتناول|استخدم|استعمل|اشرب|اجرب|اجيب|اشتري)\s+(ايه|اي|ايش|ماذا|ما)\b",
    r"\b(ايه|اي|ايش|ماذا|ما)\s+(هو\s+)?(ال)?(دوا|دواء|علاج|مسكن|حباية|حبايه|ادويه|احسن|افضل|الانسب|المناسب|اللي ينفع|ينفع|يساعد|يفيد|يخفف)",
    r"\bايه اللي (اخده|اخدها|اشربه|اتناوله|ينفع)",
    r"\b(ممكن|ينفع|اقدر)\s+(اخد|اخذ|اتناول)\s+(ايه|اي)\b",
    # Franco-Arabic
    r"\b(a5od|akhod|a5d|akhd|a5ud|a5ood|akhud)\s+(eh|ehh|eih|ay|ey|e7|ayh|aye)\b",
    r"\b(eh|eih|ay|ayh)\s+(el\s+)?(dawa|dawaa|3elag|mosaken|a7san|afdal|el a7san)\b|\bdawa\s+(le|lel|li|lil)\b",
)
# A first-person mark: needed because some ways of asking ("what is good for a headache?") have no subject.
_FIRST_PERSON = _rx(
    r"\b(i|i'm|im|i've|ive|i'd|my|me|myself)\b",
    r"\b(عندي|انا|حاسس|حاسه|بعاني|تعبان|تعبانه|اخد|اخذ|اتناول|استخدم|اشرب|اجرب|راسي|دماغي|بطني|زوري|حلقي|معدتي|ضهري|رجلي|ايدي|سناني|ضرسي)\b",
    r"\b(3andy|3andi|3ndy|3ndi|3andii|ana|a5od|akhod|a5d|akhd)\b",
)


def asks_what_to_take(message: str) -> bool:
    """First-person wording that asks what to take (or what is good) for something."""
    t = normalize(message)
    return _ASK.search(t) is not None and _FIRST_PERSON.search(t) is not None


def is_self_care(message: str, *also: str, context: tuple[str, ...] = ()) -> bool:
    """True when the message is a self-care question: it (or one of `also`, its typo-corrected or standalone forms) asks
    in the first person what to take, and none of the texts (the message, `also`, and the earlier user turns in
    `context`) has a risk qualifier, names another person, or is an Emergency or a Dose request (`rules.check`)."""
    texts = (message, *also)
    if not any(asks_what_to_take(t) for t in texts):
        return False
    return not any(check(t) or qualifiers(t) for t in (*texts, *context) if t)



def escalates_with_qualifier(message: str, *also: str) -> bool:
    """True when the message asks in the first person what to take AND carries a risk qualifier or another person, in the message
    or in `also` (its typo-corrected form): "عندي صداع جامد اخد ايه", "I take warfarin, what can I take for a headache?".
    `pipeline.decide` then raises the status to a Treatment Decision even when the router said none: the policy is that a
    qualifier keeps such a question Escalated, and a router that missed it must not turn it into an answer. It can only raise.
    A message that does not ask what to take ("Is Brufen safe in pregnancy?") is left to the router, as before."""
    if not asks_what_to_take(message) and not any(asks_what_to_take(x) for x in also):
        return False
    return any(qualifiers(x) for x in (message, *also))
