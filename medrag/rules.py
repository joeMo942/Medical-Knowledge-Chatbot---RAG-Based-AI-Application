"""Fixed rule check for the two Escalated triggers that wording alone can reveal: Emergency
(red-flag symptoms) and Dose (how much, how often, to whom). English, Arabic, Egyptian dialect
and Franco-Arabic, on normalised text.

The check can only raise a status to Escalated, never lower one: a message it does not flag
is simply passed on to the routing call and the normal pipeline. It must not flag Strength
questions ("how much paracetamol is in Panadol Extra"): Strength is product composition as
written in the KB, Dose is how much to GIVE or TAKE. So a quantity word alone ("how much",
"كام مجم", "kam mg") is a Dose only together with an intake cue (take, give, per day, اخد, ...).
Diagnosis and Treatment Decision need judgement and are left to the routing call.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

RuleTrigger = Literal["emergency", "dose"]

_AR_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640]")  # harakat, dagger alef, tatweel
_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    **{chr(0x660 + i): str(i) for i in range(10)},  # Arabic-Indic digits
    **{chr(0x6F0 + i): str(i) for i in range(10)},  # Persian digits
})


def normalize(text: str) -> str:
    """Lower-case, fold Arabic letter variants, drop harakat and tatweel, unify quotes and spaces."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _AR_DIACRITICS.sub("", t).translate(_AR_MAP)
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", t).strip()


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts))


# ------------------------------------------------------------------------------- emergency

_EN_TOOK = r"(took|taken|swallowed|ate|eaten|drank|drunk|gulped|downed|ingested|consumed|popped|chewed)"
_EN_MANY = (r"\d{2,}|too many|too much|a whole|the whole|whole|half (a |an |the |of a |of the )?(bottle|box|strip|pack|packet)|"
            r"a lot of|lots of|handful|several|a bunch of|all (of )?(the|my|his|her)")
_EN_UNIT = (r"pills?|tablets?|tabs|capsules?|caps|bottles?|strips?|packs?|packets?|boxes|box|sachets?|ampoules?|"
            r"syrup|drops|puffs?|medicines?|medications?|meds|drugs?|dawa")

# Wording that is an emergency wherever it appears: a person who is acutely unwell right now.
_EMERGENCY_STRONG = _rx(
    # English
    r"\b(can'?t|cannot|can not|couldn'?t|unable to|not able to|struggling to|gasping for)\b.{0,12}\bbreath",
    r"\b(stopped|not|isn'?t|barely) breathing\b|\bbarely breathe\b|\bbarely breathing\b",
    r"\b(blue|bluish|purple|grey|gray) (lips?|face|skin|fingers?|tongue)\b|\b(lips?|face|skin) (is |are |turned |turning |went |going )?(blue|bluish)\b|\bturning blue\b",
    r"\b(unconscious|unresponsive|passed out|collapsed|not waking|won'?t wake|not responding|lost consciousness)\b",
    r"\b(having|had|started|is having|has had|just had) (a |an )?(seizure|convulsion|fit)\b|\b(is|was|started|keeps) (seizing|convulsing|fitting)\b",
    r"\boverdos\w*|\bswallowed (a|an|the|his|her|my|whole|too many|\d+)\b.{0,25}\b(box|bottle|strip|pack|packet|pills|tablets|capsules|syrup)\b",
    r"\b(took|taken|ate|drank) (a|an|the|too many|too much|whole|\d+)\b.{0,20}\b(whole )?(box|bottle|strip|pack|packet)\b",
    r"\b(vomit\w*|cough\w*|spit\w*|pass\w*) (up )?(blood|bloody)\b|\bblood in (his|her|my|the) (vomit|stool|urine)\b",
    r"\bkill (myself|himself|herself)\b|\bdon\'?t want to (live|be alive)\b|\bno reason to live\b|\bbetter off dead\b|\bend it all\b|\bend (my|his|her) life\b|\bwants? to die\b",
    r"\bambulance\b",
    # a reported intake of many pills / a bottle / "some of my pills": overdose or accidental ingestion
    rf"\b{_EN_TOOK}\b[^.?!]{{0,25}}\b({_EN_MANY})\b[^.?!]{{0,40}}\b({_EN_UNIT})\b",
    rf"\b{_EN_TOOK}\b[^.?!]{{0,12}}\b(some|any|a few|several|few|a bunch|lots|a lot) of (my|his|her|the|their|your|our)\b",
    rf"\b(child|kid|baby|toddler|infant|son|daughter|\d+[- ]?(year|yr|month)s?[- ]?old)\b[^.?!]{{0,40}}\b(swallowed|ate|eaten|drank|drunk|ingested|got into|chewed)\b[^.?!]{{0,40}}\b({_EN_UNIT}|meds|drug|drugs|panadol|medicine)\b",
    # Arabic (normalised: alef, ya, ta marbuta folded)
    r"(مش|ما|لا|مو)\s*(قادر\w*|عارف\w*|بيعرف|بتعرف|بقدر|بيقدر)\s*(ا|ي|ت|ن)?تنفس|(مش|ما)\s*(بيتنفس|بتتنفس)",
    r"نفس\w*\s*(مقطوع|بيقطع|واقف|محبوس|مخنوق)|(مقطوع|انقطاع|انحباس)\s*(ال)?نفس|بيختنق|مختنق",
    r"(شفايف|شفاه|شفه|شفت|شفتي|شفتيه|وش|وشه|وشها|اصابع|لسان)\w*\s+(ال)?(ازرق|زرقا|زرقاء|زرقه|بتزرق|بيزرق)|ازرقاق|زرقان|بيزرق|بتزرق",
    r"اغمي عليه?|مغمي|فاقد\w* (ال)?وعي|فقد (ال)?وعي|مش بيرد|مش بتفوق|مش بيفوق|اغماء",
    r"نوبه\s*(صرع|تشنج)|نوبات\s*(صرع|تشنج)|بيتشنج",
    r"جرعه\s*(زايده|زائده|زياده)|جرعات\s*(زايده|زائده)|اوفر\s*دوز|\boverdose\b",
    r"(بلع|ابتلع|شرب|اخد|اخدت|اخذ)\w*\s*(علبه|علب|شريط|زجاجه|ازازه)\w*\s*(كامل\w*|كلها|كله)?|بلع (كتير|كميه كبيره)",
    r"(بيرجع|بترجع|رجع|رجعت|بيتقيا|بتتقيا|تقيا|كحه|بيكح|بتكح)\w*\s*(منه|منها)?\s*دم\b|قيء دموي|قي دموي|دم في (ال)?(براز|بول|قي)",
    r"عايز\w* اموت|نفسي اموت|بفكر\w* (اموت|انتحر)|(اقتل|اذي)\s*نفس\w*|اسعاف",
    # Arabic: swallowed / drank pills or medicine (accidental ingestion), or took many of them
    r"(بلع|ابتلع|بلعت|ابتلعت|بلعوا|شرب|شربت|اكل|اكلت|لحس)\w*\s*(\w+\s*){0,2}(حبوب|حبات|اقراص|كبسولات|ادويه|دوا|الدوا|الادويه|شراب|ازازه|زجاجه|شريط|علبه|علب)",
    r"(اخد|اخدت|اخذ|خد|خدت|تناول|تناولت)\w*\s*(\w+\s*){0,2}(\d{2,}|كتير|كتيره|كميه كبيره|عدد كبير|كل)\s*(حب|قرص|اقراص|كبسول|حبوب|شريط|علبه)",
    r"(بلع|ابتلع|شرب|اكل)\w*\s*(\w+\s*){0,2}\d+\s*(حب|قرص|اقراص|كبسول|حبوب)",
    r"عرق\s*بارد",
    # Franco-Arabic (digits stand for Arabic letters: 3=ع 7=ح 2=ء 5=خ 9=ق 6=ط)
    r"\bd(ee?|i)?2\s*(el\s*)?nafa?s\w*|\bdee?q\s*(el\s*)?nafa?s",
    r"\bnaf(a|u|i)?s\w*\s*(ma?a?2?too?3|ma?2to3|maqto3|maktoo?3|ma2too3)",
    r"\bm[ae]?sh\s*(2|9|q|k|g)?[ae]d(er|ra|ar|a)h?\w*\s*(y|a|e|n|t)?[et]?nafa?s|\bm[ae]?sh\s*(3|a)?aref\s*(y|a|e|t|n)?[et]nafa?s",
    r"\b(shafa?yf|shefa?yf|shafaif|shafayef|shefay?f|shafteh|shaftoh|shafa|wesh|lesan)\w*\s*(el?\s*)?(zar2a|zar2|azra2|zr2a|zarqa|azraq)|\bzar2a\b|\bazra2\b",
    r"\b(ghayb|ghaib)\w*|\bfa2ed\w* (el )?wa3y|\bfa2d (el )?wa3y|\bmogma 3al|\bughma\w* 3al|\bmoghma\b|\bmesh (by|bi)?rod\b",
    r"\b(3ayez|3ayz|3ayzah|3ayza|3ayiz)\w* (amoot|a?mot)\b|\bhentahar\b|\bantahir\b|\bantahar\b",
    r"\bby?r[ei]?g[ae]?a?3\s*dam\b|\brega3\w* dam\b|\bbyrg3 dam\b",
    r"\b3ar2 bared\b|\b3ara2 bared\b",
    r"\b(by?)?takhod\w* (el )?(3elba|3elab|elba) (kolha|kolaha)\b|\b(ba3|bal3|bala3)\w* (el )?3elba\b",
    # Franco: swallowed / drank pills or medicine, or took many of them
    r"\b(bala3|bal3|bala3et|bala3t|bal3et|shereb|shereb|sharab|sherib|akal|akalet|akl)\w*\s*(\w+\s*){0,2}(7ab\w*|hab\w*|dawa|2ors|a2ras|capsul\w*|kapsul\w*|shreet|azaza|3elba|3elbet|syrup)",
    r"\b(a5ad|akhad|5ad|a5adt|akhadt|5adet|a5adet|5adt|akhdt)\w*\s*(\w+\s*){0,2}(\d{2,}|kteer|ketir|kiteer|kol)\s*(7ab\w*|hab\w*|2ors|a2ras|capsul\w*|shreet|3elba|3elbet)",
    r"\bover ?dose\b|\boverdoz\w*",
)

# Wording that is a red flag only when it describes a person right now: it is also a normal topic of
# a side-effect or contraindication question. Dropped when the message is framed that way.
_EMERGENCY_WEAK = _rx(
    r"\b(short(ness)? of breath|breathless\w*|difficulty breathing|trouble breathing|hard to breathe|can'?t catch (his|her|my) breath|wheez\w*|chest pain|chest tightness|chest pressure|tight chest|palpitations?|fast heart ?beat|heart attack|cardiac arrest)\b",
    r"\b(stroke|slurred speech|face (is )?drooping|drooping face|cannot move (his|her|my) (arm|leg)|can'?t move (his|her|my) (arm|leg)|paraly\w*|numb\w* on one side|severe headache|worst headache|confus\w+|disorient\w*|very drowsy|can'?t stay awake)\b",
    r"\b(seizures?|convuls\w+|fits?|anaphyla\w*|severe allergic|choking|suicid\w*|self[- ]harm|faint\w*)\b",
    r"\b(throat|tongue|face|lips?)\b.{0,15}\b(swell\w*|swollen|closing|closed)\b|\b(swollen|swelling of( the| his| her| my)?) (throat|tongue|face|lips?)\b",
    r"\b(bleed\w*|hemorrhag\w*|haemorrhag\w*) (heavily|a lot|badly|profusely|won'?t stop|that won'?t stop)\b|\bheavy bleeding\b|\bwon'?t stop bleeding\b",
    r"\b(severe|sudden|crushing|intense|unbearable) (pain|dizziness|weakness|bleeding|vomiting|diarrh\w+)\b",
    r"\b(dee?2|di2|dee?q|deeq)\s*(el\s*)?nafas|\bwaga3 (f[ei]? )?(el )?(sedr|sadr)\w*|\bwag3 (f[ei]? )?(el )?(sedr|sadr)\w*|\b(sedr|sadr)\w* (byewga3|byo?waga3|bi?wag3)\w*|\bkhafa2an\b|\bdawkha shedeed\b|\b(tashanog|tashannog|tashanug)\w*|\bnazif\w* (shaded|shadid|shedeed)\b",
)

_EMERGENCY_WEAK_AR = _rx(
    r"(ضيق|صعوبه|صعب)\s*(في)?\s*(ال)?(نفس|تنفس)|(الم|وجع|الام)\s*(في|ب)?\s*(ال)?(صدر|قلب)|(صدري|صدره|صدرها)\s*(بيوجع\w*|واجع\w*)|خفقان|سكته|جلطه|دوخه شديده|تنميل|تلعثم|تشنج|صرع|اغماء|حساسيه شديده|صدمه تحسسيه|انتحار|انتحر\w*|ازمه قلبيه|تورم\s*(في|ب)?\s*(ال)?(حلق|لسان|وش|وجه|شفايف|شفاه)|ورم\s*(في)?\s*(ال)?حلق|(ال)?(حلق|لسان)\w*\s*(بيورم|متورم|بيقفل|بيتقفل)|نزيف\s*(شديد|حاد|مش بيقف|مستمر|غزير)",
)

# A side-effect / contraindication / "does X cause ..." frame: a weak red flag in it is a topic.
_TOPIC_FRAME = _rx(
    r"\b(side effects?|adverse (effects?|reactions?)|contraindicat\w*|warnings?|precautions?|interact\w*)\b",
    r"\b(does|do|can|could|may|might|will|is it|are)\b[^.?!]{0,50}\b(cause|causes|causing|lead to|trigger)\b",
    r"\bwhat (are|is) the (common )?(symptoms?|signs?)\b|\bsymptoms? of\b",
    r"الاعراض الجانبيه|اعراض جانبيه|الاثار الجانبيه|اثار جانبيه|موانع|تحذير|تفاعل|يسبب|بيسبب|تسبب|بتسبب|هل\s+\S+\s+(يسبب|بيسبب|بتسبب)",
    r"\b(a3rad|a3ra?d|3awarad|el a3rad)\b|\bbyseb?eb\b|\b(by|bi)?sabbeb\b|\bmawane3\b|\bmamnoo?3\b",
)

# A person is described: a sign that a weak red flag is happening now.
_PERSON_NOW = _rx(
    r"\b(i|i'm|im|i am|i have|he|she|they|his|her|my|patient|someone|baby|child|kid|son|daughter|father|mother|wife|husband|dad|mom|mum|grand\w+|brother|sister|friend|man|woman|boy|girl)\b",
    r"عنده|عندها|عندي|عيان|المريض|المريضه|ابني|بنتي|ابنتي|بابا|ماما|امي|ابويا|ابوي|جوزي|مراتي|اخويا|اختي|صاحبي|حالته|حاله|بيعاني|بتعاني|مش قادر|حاسس|حاسه|فجاه|دلوقتي|دلوقت|حاليا",
    r"\b3and[iou]\w*|\b3ayan\w*|\b3ayana\b|\bel (mare?ed|3ay[ae]n)\b|\babn[iy]\b|\bbabaa?\b|\bmama\b|\bgozy\b|\bmerat[iy]\b|\bw?a5[oy]ya\b|\bda?lwa2ty\b|\bfag2a\b|\bfag2et\b|\b7ales\b",
)

# ------------------------------------------------------------------------------------ dose

# Cues that a quantity question is about how much to give or take (not about product composition).
_EN_INTAKE_CUE = (r"(take|taking|taken|takes|give|giving|given|gave|administer\w*|inject\w*|swallow\w*|"
                  r"drink\w*|use|using|used|dose|doses|per (day|night|dose|kg|week|hour)|a day|each day|every|daily|"
                  r"at (night|once|a time|bedtime)|once|twice|times|maximum|max|limit|too much|too many|at most)")

_DOSE_EN = _rx(
    rf"\bhow (many|much)\b[^.?!]{{0,70}}\b{_EN_INTAKE_CUE}\b",
    r"\bhow (often|frequently|long|many days|many weeks)\b[^.?!]{0,40}\b(take|taking|give|giving|use|using|inject\w*|used|taken|apply\w*|continue)\b",
    r"\b(mg|mcg|ml|milligrams?|grams?|tablets?|pills?|capsules?|puffs?|drops?|units?|sachets?|ampoules?|injections?|sprays?|tsp|teaspoons?)\b[^.?!]{0,25}\b(per|a|each|every)\s*(day|night|morning|hour|week|dose|kg|kilo|time)\b",
    r"\b\d+(\.\d+)?\s*(mg|mcg|ml|units?)\s*/\s*(kg|day|dose)\b|\bmg\s*/\s*kg\b|\bmg per kg\b",
    r"\b(once|twice|thrice|one time|two times|three times|\d+ times|every \d+ hours?)\b[^.?!]{0,15}\b(a|per|each|every|in a)\s*(day|daily|week|night|hours?)\b",
    r"\b(take|taking|give|giving|inject\w*|administer\w*)\s+(\d+(\.\d+)?|one|two|three|four|half|double|extra|another|more|less|an extra|a few|some)\s*(more |extra )?(mg|mcg|ml|tablets?|pills?|capsules?|puffs?|drops?|units?|sachets?|ampoules?|doses?)\b",
    r"\b(can|could|may|should) (i|we|he|she|they|you)\b[^.?!]{0,20}\b(double|halve|split|crush|skip|increase|reduce|decrease|take more|take extra|take less)\b[^.?!]{0,25}\b(dose|tablet|pill|capsule|puff|amount)s?\b",
    r"\b(missed|forgot|forgotten|skipped|skip|double|doubling|loading|maintenance|pediatric|paediatric|paed|children'?s|child'?s|adult|maximum|max|lethal|toxic|recommended|usual|normal|standard|safe|starting|initial|daily) (dose|dosage)\b",
    r"\b(maximum|max|recommended|usual|normal|safe|total) (daily |single )?(amount|quantity|intake)\b",
    r"\bhow (many|much) (should|can|do|could|may|would)\b",
    r"\bwhat (amount|quantity)\b[^.?!]{0,30}\b(take|give|use|safe|daily)\b",
    r"\bregimen\b[^.?!]{0,10}\bfor (a |an |my |his |her )?(child|adult|patient|baby|infant)\b",
)

_DOSE_AR_UNIT = (r"(حب[ه]|حبايه|حبايات|حبوب|حبه|قرص|اقراص|كبسول\w*|كبسوله|نقط[ه]?|نقطه|قطره|قطرات|بخ\w*|امبول\w*|حقن\w*|كيس|اكياس|"
                 r"مل|ملي|مللي|ملجم|مجم|ملغ|جرام|غرام|ملعق\w*|وحد\w*|ساش\w*|لبوس\w*|مره|مرات|مرتين)")
_AR_INTAKE_CUE = (r"(اخد|آخد|اخذ|ياخد|تاخد|ناخد|يتاخد|تتاخد|ياخدها|ياخده|اتناول|يتناول|اعطي|اعطيه|اعطيها|ادي|اديه|اديها|اديله|اديلها|"
                  r"يدي|ادّي|يعطي|يستخدم|استخدم|يستعمل|استعمل|احقن|يحقن|اشرب|يشرب|اخلي|يوميا|في اليوم|فى اليوم|باليوم|بالنهار|بالليل|في الاسبوع|"
                  r"كل\s*\d*\s*(ساع|يوم|ليل|صباح)|مره في|مرتين في|مرات في|ينفع|اقدر|اقصي|اقصى|الاقصي|"
                  r"للاطفال|لطفل|للكبار|لكبير|للحامل|للرضيع|لمريض|حد اقصي|زياده|اكتر|اقل)")

_DOSE_AR = _rx(
    rf"\b(كام|كم|قد ايه|قدر ايه|اد ايه|ايه كميه|كميه)\s*(ال)?\s*{_DOSE_AR_UNIT}\b[^.؟?!]{{0,60}}{_AR_INTAKE_CUE}|{_AR_INTAKE_CUE}[^.؟?!]{{0,40}}\b(كام|كم)\s*(ال)?\s*{_DOSE_AR_UNIT}",
    r"\b(كام|كم)\s*(ال)?\s*(حب[ه]|حبايه|حبايات|حبوب|قرص|اقراص|كبسول\w*|نقط\w*|قطر\w*|بخ\w*|امبول\w*|حقن\w*|كيس|اكياس|ملعق\w*|لبوس\w*)(?!\s*(في|فى|فيه|جوه|داخل)\s*(ال)?(علب|شريط|عبو|باكت|كرتون|ورق))",
    r"\b(كام|كم)\s*(مره|مرات|مرتين|يوم|ايام|اسبوع|اسابيع|ساعه|ساعات)\b[^.؟?!]{0,30}(اخد|ياخد|تاخد|يتاخد|اتناول|يستخدم|استخدم|اعطي|ادي|اديه|اديله|اديها|يدي|احقن|يحقن)",
    r"(اخد|ياخد|تاخد|يتاخد|يتناول|اتناول|اعطي|اعطيه|ادي|اديه|اديها|اديله|اديلها|يدي|احقن|يحقن|استخدم|يستخدم)\s*(ه|ها|ك)?\s*(كام|كم|قد ايه|اد ايه|كميه|نص|نصف|ضعف|اتنين|تلات|ثلاث|\d+)\s*(ال)?\s*(حب|قرص|اقراص|كبسول|نقط|قطر|بخ|امبول|ملعق|مجم|ملجم|مل|وحد)?",
    r"(كل\s*كام\s*(ساع|يوم)|كل قد ايه|(ال)?جرعه (كام|كم)|(كام|كم)\s*(في|ب)\s*(ال)?(يوم|اسبوع|ليل))",
    r"(مره|مرتين|مرات|\d+ مرات?|تلات مرات|ثلاث مرات)\s*(في|ب|كل)?\s*(ال)?(يوم|اسبوع|ليل|ساع)",
    r"(اضاعف|ضاعف|اضاعفها|اضاعفه|اقلل|اقسم|اكسر|اسحق|افوت|فوت|نسيت)\s*(ال)?\s*(جرعه|حبه|قرص|الحبايه|الحبه)",
    r"(اقصي|اقصى|الاقصي|الاقصى|اعلي|اعلى|حد اقصي|حد اقصى)\s*(كميه|حد|جرعه|عدد|ما)",
    r"(ال)?(حد|الحد) (ال)?(اقصي|اقصى|الاقصي)",
    r"(اخد|ياخد|اعطي|ادي|اديله|اديها|اديه)\s*(كام|كم)\b",
)

# Franco: kam + unit (mg only with an intake cue, since "kam mg fe Panadol" is a Strength question)
_FR_UNIT = (r"(ha+b[ab]\w*|7a+b[ab]\w*|7aba|hb[ab]\w*|a?2ors|a2ras|2ors|a2ra?s|ors|capsul\w*|kapsul\w*|kabsul\w*|kapsula|"
            r"no2ta|noo?2ta|no2at|no2a\w*|ta3ma|ba5a\w*|bakha\w*|ambool\w*|amboul\w*|amp\w*|kis\b|akyas|wa7da|wahda|units?|ml\b|"
            r"mara|marra|mra|marat|mrat|tablet\w*|pill\w*|puff\w*|spoon\w*|ma3la2\w*|ma3la3\w*|molsa2\w*)")
_FR_MG = r"(mg|mgm|mgs|milligram\w*|mellig\w*|mili\w*|gram\w*|g)\b"
_FR_CUE = (r"(a3mel|a3ml|a3ty|a3ti|e3ty|e3ti|a3tih\w*|a3te\w*|edy|edi|edih\w*|edeh\w*|ady|adi|adih\w*|adeeh\w*|aedy|aedih\w*|"
           r"a5od|a5d|akhod|akhud|a5ud|akhd|ya5od|yakhod|ya5d|yakhd|ye5od|ta5od|takhod|takhd|yetaa?5ed|yet?akhed|yenfa3|yenfa3?|"
           r"a7ot|a7ott|a7ut|ahot|fel\s*yom|fe\s*el\s*yom|fi\s*el\s*yom|bel\s*yom|el\s*yom|youm\w*|yawm\w*|f el yom|"
           r"fel\s*layl|belayl|kol\s*(\d+\s*)?(sa3|sa3a|yom|youm|layl)|yomeyan|yawmeyan|yawmiyan|marra fel|marteen fel|"
           r"e2dar|a2dar|a2dr|ye3ty)")

_FR_CUE_RX = re.compile(rf"\b{_FR_CUE}\b")

_DOSE_FR = _rx(
    r"\bgo?r3[ae]\w*|\bgar3[ae]\w*|\bgor3\w*|\bgor3a\b|\bgara3a\b|\bgor3et\b|\bgo?ra3a\b|\bg?[ou]r3[ae]t?\b|\bgor3a?t?\b|\bgore3a\b",
    rf"\bka+m\s*(el\s*)?{_FR_UNIT}(?!\s*(fe|fi|fih|feeh|gowa|gwa)\s*(el\s*)?(3elba|3elab|shreet|sherit|box|pack|karton))",
    rf"\bka+m\s*(el\s*)?{_FR_MG}[^.?!]{{0,60}}\b{_FR_CUE}\b|\b{_FR_CUE}\b[^.?!]{{0,40}}\bka+m\s*(el\s*)?{_FR_MG}",
    r"\b(a3mel|a3ml|a3ty|a3ti|e3ty|e3ti|a3tih\w*|edy|edi|edih\w*|edeh\w*|ady|adi|adih\w*|aedy|aedih\w*|a5od|akhod|akhud|a5ud|ya5od|yakhod|ye5od|ta5od|takhod|a7ot|a7ott|yetaa?5ed)\s+(el\s*)?(\w+\s+)?ka+m\b",
    r"\bkol\s*ka+m\s*(sa3|sa3a|sa3et|youm|yom|yawm)\w*|\bka+m\s*(marra|mara|mra|marat|mrat)\b|\bka+m\s*(youm|yom|iyam|ayam|esbo3|isbo3|sa3a|sa3at)\b",
    r"\b(marra|marteen|martein|marten|talat marat|talat mrat|\d+ marat|\d+ mrat|\d+ times)\s*(f(e|i)l?|bel|kol)?\s*(el\s*)?(yom|youm|yawm|esbo3|isbo3|layl|sa3a)\b",
    r"\b(a?da3|a?d3f|ad3af|adaa3ef|a3aadef|ada3ef|a3ad3ef|a?dd3f|a?da3af|a3add)\w*\s*(el\s*)?(gor3|jor3|dose|7aba|haba|2ors)\w*",
    r"\b(a2sa|a2sa|a2sa|aqsa|a2sa|akbar|a2sa)\s*(kamyya|kameya|kamiya|7ad|had|gor3a|3adad)\w*|\b(el\s*)?(7ad|had)\s*(el\s*)?(a2sa|aqsa|a2sa)\b",
    r"\b(nesit|nsit|nseet|nesyt|fatet|fatt)\w*\s*(el\s*)?(gor3a|7aba|haba|2ors|dose)\b",
    r"\bdoza\b|\bdoz\b|\bposology\b",
)


def _first(pattern: re.Pattern[str], text: str) -> bool:
    return pattern.search(text) is not None


# "kam mg" is a Dose unless the message is about what is IN a product ("kam mg fe Panadol Extra")
_FR_KAM_MG = re.compile(r"\bka+m\s*(mg|mgm|mgs|ml|mili\w*|gram\w*)\b")
_FR_STRENGTH_MARK = re.compile(
    r"\b(fe|fi|fih|feeh|feh|gowa|gwa|beyehtawy|byehtawy|yehtawy|tarkeez|tarkez|tarkib|composition|strength)\b")


def has_red_flag(message: str) -> bool:
    t = normalize(message)
    if _first(_EMERGENCY_STRONG, t):
        return True
    if _first(_TOPIC_FRAME, t):
        return False
    return _first(_EMERGENCY_WEAK_AR, t) or (_first(_EMERGENCY_WEAK, t) and _first(_PERSON_NOW, t))


# The bare word "dose" is a Dose request only with a cue that an amount, a person or a frequency
# is asked about, and never when the question is about what the KB says in general ("is the dose
# adjusted in kidney failure?", "what strengths and doses does X come in?": group-level, Answered).
_DOSE_WORD = re.compile(r"\b(dose|doses|dosage|dosages|dosing|posology|dosed|doza|doz)\b(?!\s*forms?\b)")
_DOSE_WORD_AR = re.compile(r"جرع[ه]|جرعات|جرعتي|جرعته|جرعتها")
_DOSE_CUE = re.compile(
    r"\b(what'?s|what is|what are|tell me|give me|state|the right|correct|recommended|usual|normal|standard|"
    r"safe|toxic|lethal|max\w*|min\w*|starting|initial|loading|maintenance|pediatric|paediatric|adult|adults|"
    r"child|children|kid|kids|infant|baby|my|his|her|our|per|daily|weekly|twice|once|kg|missed|forgot\w*|"
    r"skip\w*|doubl\w*|increase|decrease|reduce|how much|how many|how often|should i|can i|\d+)\b"
    r"|كام|كم|الجرع[ه] (ال)?(صحيح|مناسب|المناسب|الصحيح|القصوي|اليوميه)|للاطفال|لطفل|للكبار|للرضيع|ايه|ما هي|ماهي|قولي|اعطيني|اعرف|\bkam\b|\ba3mel\b")
_DOSE_GROUP_FRAME = re.compile(
    r"\b(adjust\w*|differ\w*|chang\w*|modif\w*|vary|varies|come|comes|available|forms?|strengths?|"
    r"require\w*|needed|necessary)\b|تتعدل|بتتعدل|تعديل|تختلف|بتختلف|تتغير|اشكال|تركيز|يتعدل|بيتعدل|هل يلزم|هل تحتاج")


def _asks_amount(t: str) -> bool:
    return _first(_DOSE_EN, t) or _first(_DOSE_AR, t) or _first(_DOSE_FR, t)


def has_dose_wording(message: str) -> bool:
    t = normalize(message)
    if _asks_amount(t):
        return True
    if _first(_FR_KAM_MG, t) and (_first(_FR_CUE_RX, t) or not _first(_FR_STRENGTH_MARK, t)):
        return True
    if _first(_DOSE_WORD, t) or _first(_DOSE_WORD_AR, t):
        return _first(_DOSE_CUE, t) and not _first(_DOSE_GROUP_FRAME, t)
    return False


def check(message: str) -> RuleTrigger | None:
    """The trigger the wording alone reveals: emergency before dose, None when nothing matched
    (which never means "safe": the routing call and the pipeline still decide). The rules only
    ever raise a status: they have no way to lower one."""
    if has_red_flag(message):
        return "emergency"
    if has_dose_wording(message):
        return "dose"
    return None
