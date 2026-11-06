"""Snippet rules: one line or row, word boundaries, salvaging a multi-line model
snippet, and the repair tie-break. Synthetic Sections only (no KB, no LLM)."""
import pytest

from medrag.answering import build_citations, is_valid_snippet
from medrag.models import Line, Row, Section, Table

# Shaped like KB-03 › NSAIDs: a label line right before the products table.
LABELLED = Section(
    "KB-03", "Headache & Migraine – الصداع", "NSAIDs", "class",
    (
        Line("الاستخدام: علاج الألم والالتهاب، مُسَكِّن للألم."),
        Line("Trade Names في مصر:"),
        Table(Row.from_cells(["Trade Name", "Active Ingredient"]),
              (Row.from_cells(["Brufen", "Ibuprofen"]), Row.from_cells(["Nurofen", "Ibuprofen"])),
              "products", 0, 1),
    ),
)
BRUFEN, NUROFEN = "Brufen – Ibuprofen", "Nurofen – Ibuprofen"


def cite(snippet, hints="q"):
    (c,) = build_citations([{"section_id": LABELLED.section_id, "snippet": snippet}], [LABELLED], hints)
    return c.snippet


@pytest.mark.parametrize("snippet, valid", [
    (BRUFEN, True),               # a whole row
    ("  Ibuprofen ", True),       # whole word inside one row, 9 chars, outer spaces ignored
    ("Brufen", False),            # whole word but under 8 chars and not the whole row
    (f"{BRUFEN}\n{NUROFEN}", False),  # two rows
])
def test_one_row_and_minimum_length(snippet, valid):
    assert is_valid_snippet(snippet, LABELLED) is valid


@pytest.mark.parametrize("fragment", [
    "fen – Ibupro",               # starts and ends inside words
    "n – Ibuprofen",              # starts inside "Brufen"
    "Brufen – Ibupro",            # ends inside "Ibuprofen"
    "خدام: علاج الألم",            # Arabic: starts inside "الاستخدام"
    "علاج الألم والالت",           # Arabic: ends inside "والالتهاب"
    "مُسَكِّن للأل",               # Arabic with harakat: ends inside "للألم"
])
def test_fragments_cut_inside_a_word_are_rejected(fragment):
    assert not is_valid_snippet(fragment, LABELLED)


@pytest.mark.parametrize("words", ["علاج الألم والالتهاب", "مُسَكِّن للألم", "Trade Names في مصر:"])
def test_arabic_and_mixed_whole_word_spans_are_accepted(words):
    assert is_valid_snippet(words, LABELLED)


def test_multi_line_model_snippet_keeps_its_first_valid_line():
    pasted = f"Trade Name – Active Ingredient\n{NUROFEN}\n{BRUFEN}"
    assert cite(pasted) == NUROFEN


def test_repair_tie_prefers_a_table_row_over_a_label_line():
    # "trade" matches the label line, "brufen" matches the row: a 1-1 tie.
    assert cite("invented text", hints="trade brufen") == BRUFEN


def test_repair_still_picks_the_best_overlap():
    assert cite("invented", hints="nurofen ibuprofen") == NUROFEN
