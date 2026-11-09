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


class FakeLLM:
    model = "fake-model"

    def __init__(self, reply=None, exc=None, delay=0.0, tokens=(None, None)):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.usage = tokens  # (prompt, completion) tokens each call reports
        self.calls = []
        self.cancelled = False
        self.caps = []  # max_tokens passed by the gate, per call

    async def complete_json(self, system, user, schema, max_tokens=None):
        self.calls.append((system, user))
        self.caps.append(max_tokens)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.exc:
            raise self.exc
        return LLMResult(data=self.reply, prompt_tokens=self.usage[0], completion_tokens=self.usage[1])


def answered(snippet=ROW, section_id=NSAIDS.section_id):
    return {"answer": "Diclofenac.", "status": "answered",
            "citations": [{"section_id": section_id, "snippet": snippet}]}
