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


def test_multi_line_model_snippet_keeps_one_of_its_own_valid_lines():
    pasted = f"Trade Name – Active Ingredient\n{NUROFEN}\n{BRUFEN}"
    assert cite(pasted) == NUROFEN  # no term overlap: the first of its rows
    assert cite(pasted, hints="alternatives to brufen") == BRUFEN  # the line the answer is about


# Shaped like KB-01 › SABA and KB-08 › ARBs: label lines followed by item lines.
SIDE_EFFECTS = Section(
    "KB-01", "Asthma", "SABA – Short-Acting Beta-2 Agonists", "class",
    (Line("Common side effects:"), Line("Tremors."), Line("Hypokalemia مع الجرعات العالية."),
     Line("Contraindications"), Line("Pregnancy: ممنوعة أثناء الحمل.")),
)


def cite_in(section, snippet, hints):
    (c,) = build_citations([{"section_id": section.section_id, "snippet": snippet}], [section], hints)
    return c.snippet


def test_a_label_line_from_a_pasted_block_is_never_the_citation():
    pasted = "Common side effects:\nTremors.\nHypokalemia مع الجرعات العالية."
    hints = "What are the common side effects of salbutamol? Common side effects include tremors and hypokalemia."
    assert cite_in(SIDE_EFFECTS, pasted, hints) == "Hypokalemia مع الجرعات العالية."


def test_a_bare_heading_word_loses_to_a_line_with_content():
    pasted = "Contraindications\nPregnancy: ممنوعة أثناء الحمل."
    hints = "What are the contraindications? Contraindications include pregnancy."
    assert cite_in(SIDE_EFFECTS, pasted, hints) == "Pregnancy: ممنوعة أثناء الحمل."


def test_repair_tie_prefers_a_table_row_over_a_label_line():
    # "trade" matches the label line, "brufen" matches the row: a 1-1 tie.
    assert cite("invented text", hints="trade brufen") == BRUFEN


def test_repair_still_picks_the_best_overlap():
    assert cite("invented", hints="nurofen ibuprofen") == NUROFEN
