"""Escalated replies: fixed server-written text per trigger, in the user's language. No drug
facts, no Citations, never model text. Arabic for `ar` and `mixed`, English for `en`."""
from __future__ import annotations

from .answering import ChatResponse

ESCALATION_TEXT: dict[str, dict[str, str]] = {
    "emergency": {
        "en": ("This may be an emergency. Call 123 (ambulance in Egypt) or go to the nearest emergency "
               "department now. Do not wait for medication advice from this assistant."),
        "ar": ("هذه حالة طارئة محتملة. اتصل بالإسعاف على 123 أو توجّه إلى أقرب قسم طوارئ الآن. "
               "لا تنتظر نصيحة دوائية من هذا المساعد."),
    },
    "dose": {
        "en": ("I cannot give doses. Please check the approved leaflet of the product, or ask the "
               "prescriber or a pharmacist."),
        "ar": ("لا أستطيع تحديد الجرعات. من فضلك راجع النشرة المعتمدة للمنتج، أو اسأل الطبيب المعالج "
               "أو الصيدلي."),
    },
    "diagnosis": {
        "en": ("I cannot say what condition a person has. This needs the treating physician: please "
               "see or call the doctor."),
        "ar": ("لا أستطيع تحديد ما هي حالة المريض المرضية. هذا الأمر يحتاج إلى الطبيب المعالج: "
               "من فضلك راجع الطبيب أو اتصل به."),
    },
    "treatment_decision": {
        "en": ("Choosing, changing, starting or stopping a treatment for a specific person needs the "
               "treating physician. Please ask the doctor."),
        "ar": ("اختيار علاج لمريض بعينه أو تغييره أو بدؤه أو إيقافه يحتاج إلى الطبيب المعالج. "
               "من فضلك اسأل الطبيب."),
    },
}


def escalated(trigger: str, language: str) -> ChatResponse:
    text = ESCALATION_TEXT[trigger]["en" if language == "en" else "ar"]
    return ChatResponse(answer=text, citations=[], status="escalated", language=language)
