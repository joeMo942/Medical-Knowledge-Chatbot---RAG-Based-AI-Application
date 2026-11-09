"""Trade Name <-> Active Ingredient index (medrag/drugs.py), on small synthetic Sections and on the real KB."""
import pytest

from medrag.drugs import DrugIndex, ingredient_set, trade_names
from medrag.models import Line, Row, Section, Table


def products(rows, header=("Trade Name", "Active Ingredient")):
    return Table(Row.from_cells(header), tuple(Row.from_cells(r) for r in rows), "products", 0, 1)


def section(kb, title, name, rows, lines=()):
    return Section(kb, title, name, "class", (*(Line(x) for x in lines), products(rows)))


PAIN = section("KB-03", "Pain", "NSAIDs", [("Brufen", "Ibuprofen"), ("Nurofen", "Ibuprofen"), ("Cataflam", "Diclofenac potassium")])
HEAD = section("KB-03", "Headache", "NSAIDs", [("Brufen", "Ibuprofen"), ("Flabu", "Ibuprofen"),
                                               ("Spididol", "Ibuprofen (as ibuprofen arginine)"),
                                               ("Cetafen Plus N", "Paracetamol 500 mg + Ibuprofen 200 mg + Caffeine 65 mg")])
THROAT = section("KB-04", "Throat", "NSAIDs", [("Brufen", "Ibuprofen"), ("Nurofen", "Ibuprofen")])
PPI = section("KB-02", "GERD", "PPIs", [("Controloc", "Pantoprazole"), ("Losec", "Omeprazole")])
BIO = Section("KB-09", "Arthritis", "Biologics", "class",
              (Table(Row.from_cells(["Drug Class / Generic", "Trade Names"]),
                     (Row.from_cells(["Adalimumab", "Humira, Amgevita, Hyrimoz"]),), "products", 1, 0),))
ALL = [PAIN, HEAD, THROAT, PPI, BIO]


def test_cell_normalisation():
    assert trade_names("Humira, Amgevita, Hyrimoz") == ["Humira", "Amgevita", "Hyrimoz"]
    assert trade_names("Constella / Linzess") == ["Constella", "Linzess"]
    assert trade_names("بعض مستحضرات Ibuprofen topical") == []
    assert ingredient_set("Ibuprofen (as ibuprofen arginine)") == frozenset({"ibuprofen"})
    assert ingredient_set("Paracetamol 500 mg + Caffeine 65 mg") == frozenset({"paracetamol", "caffeine"})
    assert ingredient_set("30% Soluble Insulin + 70% Isophane Insulin") == frozenset({"soluble insulin", "isophane insulin"})


def test_a_trade_name_gives_its_ingredient_and_back():
    d = DrugIndex(ALL)
    m = d.mentions("What is the active ingredient in Controloc?")
    assert m.trade == ("controloc",) and [d.ingredient_text(s) for s in m.ingredients] == ["pantoprazole"]
    m = d.mentions("Which drugs contain Pantoprazole")
    assert m.trade == () and [d.trade_names_of(s) for s in m.ingredients] == [["controloc"]]


def test_alternatives_are_the_same_ingredient_set_only():
    d = DrugIndex(ALL)
    ing = d.mentions("alternatives to Brufen").ingredients
    assert d.trade_names_of(ing[0]) == ["brufen", "nurofen", "flabu", "spididol"]  # not Cataflam, not Cetafen Plus N
    assert [d.display(k) for k in d.trade_names_of(d.mentions("Humira").ingredients[0])] == ["Humira", "Amgevita", "Hyrimoz"]


def test_alternative_views_cover_every_name_from_the_fewest_sections():
    d = DrugIndex(ALL)
    views = d.alternative_views(d.mentions("alternatives to Brufen").ingredients)
    assert [v.section_id for v in views] == [HEAD.section_id, PAIN.section_id]
    assert [r.text for r in views[0].rows] == ["Brufen – Ibuprofen", "Flabu – Ibuprofen", "Spididol – Ibuprofen (as ibuprofen arginine)"]
    assert [r.text for r in views[1].rows] == ["Brufen – Ibuprofen", "Nurofen – Ibuprofen"]  # Cataflam row left out
    assert all(v.lines == () for v in views)  # only product rows; the Section itself is not altered
    assert len(PAIN.rows) == 3


def test_a_combination_is_not_an_alternative_of_a_part():
    d = DrugIndex(ALL)
    combo = d.mentions("Cetafen Plus N").ingredients[0]
    assert d.trade_names_of(combo) == ["cetafen plus n"]


@pytest.mark.kb
def test_real_kb_ibuprofen_trade_names(real_kb):
    d = DrugIndex([s for doc in real_kb.values() for s in doc.sections])
    ing = d.mentions("Brufen").ingredients
    assert [d.display(k) for k in d.trade_names_of(ing[0])] == ["Brufen", "Nurofen", "Ibuflam", "Profinal", "Flabu", "Spididol"]
    views = d.alternative_views(ing)
    assert views[0].section_id.startswith("KB-03") and len(views[0].rows) == 6
