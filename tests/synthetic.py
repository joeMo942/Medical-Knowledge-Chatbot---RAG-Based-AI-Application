"""Small synthetic Sections and a fake LLM shared by the API tests (no KB, no Ollama)."""
import asyncio

from medrag.llm import LLMResult
from medrag.models import Line, Row, Section, Table

PRODUCTS = Table(
    header=Row.from_cells(["Trade Name", "Active Ingredient"]),
    rows=(Row.from_cells(["Cataflam", "Diclofenac potassium"]), Row.from_cells(["Voltaren", "Diclofenac sodium"])),
    kind="products", trade_name_col=0, active_ingredient_col=1,
)
NSAIDS = Section("KB-04", "Pain & Arthritis", "NSAIDs", "class",
                 (Line("Used for pain and inflammation."), PRODUCTS))
PPI = Section("KB-02", "Peptic Ulcer & GERD", "Proton Pump Inhibitors (PPIs)", "class",
              (Table(Row.from_cells(["Trade Name", "Active Ingredient"]),
                     (Row.from_cells(["Controloc", "Pantoprazole"]),), "products", 0, 1),))
OVERVIEW = Section("KB-01", "Asthma", "Asthma", "overview", (Line("Asthma is a chronic airway disease."),),
                   heading_present=False)
SECTIONS = [NSAIDS, PPI, OVERVIEW]
ROW = "Cataflam – Diclofenac potassium"
HEADER_ROW = "Trade Name – Active Ingredient"


NO_TRIGGER = {"trigger": "none", "language": "en", "is_drug_question": True, "question": "q"}


def routed(trigger, language="en"):
    return {"trigger": trigger, "language": language, "is_drug_question": True, "question": "q"}


class FakeLLM:
    model = "fake-model"

    def __init__(self, reply=None, exc=None, delay=0.0, tokens=(None, None), route=None, route_exc=None, route_fn=None,
                 route_delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        # The routing call (before retrieval) is told apart by its schema; `calls` holds answer calls only.
        self.route = route if route is not None else NO_TRIGGER
        self.route_exc = route_exc
        self.route_fn = route_fn  # user prompt -> route reply (a router that can be fooled)
        self.route_delay = route_delay
        self.answer_schemas = []  # the schema of each answer call
        self.order = []  # "route" / "answer" in the order the calls started
        self.route_calls = []
        self.usage = tokens  # (prompt, completion) tokens each call reports
        self.calls = []
        self.cancelled = False
        self.caps = []  # max_tokens passed by the gate, per call

    async def complete_json(self, system, user, schema, max_tokens=None):
        is_route = "is_drug_question" in schema.get("properties", {})
        (self.route_calls if is_route else self.calls).append((system, user))
        self.order.append("route" if is_route else "answer")
        if not is_route:
            self.answer_schemas.append(schema)
        self.caps.append(max_tokens)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if is_route:
            if self.route_delay:
                await asyncio.sleep(self.route_delay)
            if self.route_exc:
                raise self.route_exc
            if self.route_fn:
                return LLMResult(data=self.route_fn(user), prompt_tokens=self.usage[0], completion_tokens=self.usage[1])
            return LLMResult(data=self.route, prompt_tokens=self.usage[0], completion_tokens=self.usage[1])
        if self.exc:
            raise self.exc
        return LLMResult(data=self.reply, prompt_tokens=self.usage[0], completion_tokens=self.usage[1])


def answered(snippet=ROW, section_id=NSAIDS.section_id):
    return {"answer": "Diclofenac.", "status": "answered",
            "citations": [{"section_id": section_id, "snippet": snippet}]}
