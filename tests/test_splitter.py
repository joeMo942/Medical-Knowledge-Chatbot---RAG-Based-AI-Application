import pytest

from medrag.models import Line, Table
from medrag.tables import classify

from medrag.splitter import ParagraphBlock as P, SplitError, TableBlock as T, split_sections

H2, H3 = "Heading 2", "Heading 3"
DISEASES = ["1. Asthma", "2. COPD"]


def table(*rows):
    return T(rows=tuple(tuple(r) for r in rows))


def split(blocks, diseases=DISEASES):
    return split_sections("KB-XX", diseases, blocks)


def test_same_style_for_disease_and_class_uses_manifest_to_tell_them_apart():
    _, secs = split([
        P("1. Asthma", H2), P("1. SABA", H2), table(["Trade", "Active"], ["A", "B"]),
        P("2. COPD", H2), P("1. SABA", H2), table(["Trade", "Active"], ["C", "D"]),
    ])
    assert [s.path for s in secs] == ["Asthma › SABA", "COPD › SABA"]
    assert secs[0].rows[0].text == "A – B"
    assert secs[1].rows[0].text == "C – D"


def test_plain_paragraph_headings_are_recognised_by_numbering():
    _, secs = split([
        P("1. Asthma"), P("1. Paracetamol"), P("some line"),
        P("2. NSAIDs – Non-Steroidal"), P("another line"),
    ], ["1. Asthma"])
    assert [s.section for s in secs] == ["Paracetamol", "NSAIDs – Non-Steroidal"]
    assert secs[0].lines == ("some line",)


def test_subheadings_stay_inside_their_class():
    _, secs = split([
        P("1. Asthma", H2), P("5. Mucosal Protective Agents", H2),
        P("A. Sucralfate", H3), table(["T", "A"], ["x", "y"]),
        P("Side Effects", H3), P("الأدوية الموجودة في السوق المصري", H3),
        P("B. Bismuth", H3), table(["T", "A"], ["z", "w"]),
    ], ["1. Asthma"])
    assert len(secs) == 1
    assert secs[0].lines == (
        "A. Sucralfate", "Side Effects", "الأدوية الموجودة في السوق المصري", "B. Bismuth"
    )
    assert secs[0].row_count == 2 and len(secs[0].tables) == 2


def test_numbered_heading3_class_and_duplicate_numbers():
    _, secs = split([
        P("1. Asthma", H2), P("3. PCSK9", H2), P("4. Fibrates", H3), P("4. Omega-3", H2),
    ], ["1. Asthma"])
    assert [s.section for s in secs] == ["PCSK9", "Fibrates", "Omega-3"]


def test_repeated_disease_heading_does_not_split_or_duplicate():
    diseases, secs = split([
        P("1. Pneumonia", H2), P("overview line"), P("10. Pneumonia", "Heading 1"),
        P("1. Penicillins", H2),
    ], ["1. Pneumonia", "10. Pneumonia"])
    assert diseases == ("Pneumonia",)
    assert [(s.kind, s.section) for s in secs] == [
        ("overview", "Pneumonia"), ("class", "Penicillins")
    ]
    assert not secs[0].heading_present  # cites its disease title
    assert secs[0].lines == ("overview line",)


def test_empty_overview_is_dropped():
    _, secs = split([P("1. Asthma"), P("1. SABA")], ["1. Asthma"])
    assert [s.kind for s in secs] == ["class"]


def test_otc_heading_is_its_own_section():
    _, secs = split([
        P("1. Asthma", H2), P("1. SABA", H2), table(["T", "A"], ["x", "y"]),
        P("OTC ولا Prescription؟", H2),
        table(["Drug Class", "Classification"], ["SABA", "OTC"]),
    ], ["1. Asthma"])
    assert [(s.section, s.kind) for s in secs] == [
        ("SABA", "class"), ("OTC ولا Prescription؟", "otc")
    ]
    assert secs[1].heading_present


def test_otc_label_with_line_break_is_body_text():
    _, secs = split([
        P("1. Asthma"), P("1. Paracetamol"), P("OTC ولا Prescription؟\nغالبًا OTC"),
    ], ["1. Asthma"])
    assert len(secs) == 1 and secs[0].lines == ("OTC ولا Prescription؟\nغالبًا OTC",)


def test_headingless_drug_class_table_opens_unnamed_otc_section():
    _, secs = split([
        P("1. Asthma"), P("7. Antibiotics"), table(["T", "A"], ["x", "y"]),
        table(["Drug class", "التصنيف"], ["Amoxicillin", "Rx"]),
    ], ["1. Asthma"])
    # No heading of its own: it cites the disease title, nothing invented.
    assert [(s.section, s.kind, s.heading_present) for s in secs] == [
        ("Antibiotics", "class", True), ("Asthma", "otc", False)
    ]
    assert secs[1].path == "Asthma › [table with no heading in source]"


def test_drug_class_header_inside_a_class_is_not_an_otc_section():
    # KB-09's bDMARD table starts with "Drug Class / Generic".
    _, secs = split([
        P("1. Asthma"), P("4. Biologics"),
        table(["Drug Class / Generic", "Trade Names"], ["a", "b"]),
    ], ["1. Asthma"])
    assert len(secs) == 1


def test_table_rows_verbatim_with_header_empty_cells_and_en_dash():
    _, secs = split([
        P("1. Asthma"), P("1. PPIs"),
        table(
            ["الاسم التجاري", "Active ingredient"],
            ["Controloc", "Pantoprazole"],
            ["Strepsils Plus / Orofar", ""],
        ),
    ], ["1. Asthma"])
    (tbl,) = secs[0].tables
    assert tbl.header.cells == ("الاسم التجاري", "Active ingredient")
    assert tbl.header.text == "الاسم التجاري – Active ingredient"
    assert [r.text for r in tbl.rows] == ["Controloc – Pantoprazole", "Strepsils Plus / Orofar"]
    assert tbl.rows[1].cells == ("Strepsils Plus / Orofar", "")
    assert "–" in tbl.rows[0].text and "-" not in tbl.rows[0].text
    assert (tbl.kind, tbl.trade_name_col, tbl.active_ingredient_col) == ("products", 0, 1)


def test_body_lines_are_verbatim():
    _, secs = split([P("1. Asthma"), P("1. SABA"), P("  Tremors / رعشة.  ")], ["1. Asthma"])
    assert secs[0].lines == ("  Tremors / رعشة.  ",)


def test_content_before_first_disease_is_an_error():
    with pytest.raises(SplitError, match="before the first disease"):
        split([P("stray"), P("1. Asthma")], ["1. Asthma"])


def test_missing_disease_is_an_error():
    with pytest.raises(SplitError, match="manifest"):
        split([P("1. Asthma"), P("1. SABA")])


def test_duplicate_section_path_is_an_error():
    with pytest.raises(SplitError, match="duplicate"):
        split([P("1. Asthma"), P("1. SABA"), P("2. SABA")], ["1. Asthma"])


def test_section_ids_are_unique_and_carry_kb_id():
    _, secs = split([P("1. Asthma"), P("1. SABA"), P("2. COPD"), P("1. SABA")])
    assert [s.section_id for s in secs] == ["KB-XX › Asthma › SABA", "KB-XX › COPD › SABA"]


def test_items_keep_reading_order_so_tables_tie_to_their_sublabel():
    _, secs = split([
        P("1. Asthma"), P("5. Antibiotics"), P("intro"),
        P("A. Penicillins", H3), table(["T", "A"], ["Amoxil", "Amoxicillin"]),
        P("D. Lincosamides", H3), table(["T", "A"], ["Dalacin C", "Clindamycin"]),
        P("Common side effects:"),
    ], ["1. Asthma"])
    items = secs[0].items
    assert [type(i) for i in items] == [Line, Line, Table, Line, Table, Line]
    assert items[3].text == "D. Lincosamides"
    assert items[4].rows[0].cells == ("Dalacin C", "Clindamycin")
    assert secs[0].lines == ("intro", "A. Penicillins", "D. Lincosamides", "Common side effects:")
    assert [r.text for r in secs[0].rows] == ["Amoxil – Amoxicillin", "Dalacin C – Clindamycin"]
    assert secs[0].render().splitlines() == [
        "Asthma › Antibiotics", "intro", "A. Penicillins", "T – A", "Amoxil – Amoxicillin",
        "D. Lincosamides", "T – A", "Dalacin C – Clindamycin", "Common side effects:",
    ]


@pytest.mark.parametrize("header,expected", [
    (("Trade name", "Active ingredient"), ("products", 0, 1)),
    (("Trade Name", "Active Ingredients"), ("products", 0, 1)),
    (("Brand name", "Active ingredient"), ("products", 0, 1)),
    (("الاسم التجاري", "Active ingredient"), ("products", 0, 1)),
    (("الاسم التجاري في مصر", "Active ingredient"), ("products", 0, 1)),
    (("Trade name", "Active ingredient", "Main indication"), ("products", 0, 1)),
    (("Trade Name", "Active Ingredient", "الاستخدام"), ("products", 0, 1)),
    (("Generic Name", "Examples of International Trade Names"), ("products", 1, 0)),
    (("Drug Class / Generic", "Trade Names"), ("products", 1, 0)),
    (("Side effect", "التوضيح"), ("side_effects", None, None)),
    (("Drug", "Common / Important Side Effects"), ("side_effects", None, None)),
    (("Drug Class", "Classification"), ("otc", None, None)),
    (("Drug class / Product", "التصنيف"), ("otc", None, None)),
    (("Drug", "Active Ingredient"), ("other", None, None)),
    (("Foo", "Bar"), ("other", None, None)),
])
def test_table_classification_from_header(header, expected):
    assert classify(header) == expected


def test_otc_heading_that_is_heading3_stays_a_line_in_its_class():
    _, secs = split([
        P("1. Asthma"), P("4. Omega-3", H2), P("Contraindications", H3),
        P("OTC ولا Prescription؟", H3), P("Fish oil: OTC"),
        P("5. Bile Acid", H2), table(["T", "A"], ["x", "y"]),
    ], ["1. Asthma"])
    assert [(s.section, s.kind) for s in secs] == [("Omega-3", "class"), ("Bile Acid", "class")]
    assert secs[0].lines == ("Contraindications", "OTC ولا Prescription؟", "Fish oil: OTC")


def test_otc_heading_without_table_of_its_own_stays_a_line():
    _, secs = split([
        P("1. Asthma"), P("1. SABA", H2), P("OTC vs Prescription", H2), P("text"),
        P("2. LABA", H2), table(["T", "A"], ["x", "y"]),
    ], ["1. Asthma"])
    assert [s.section for s in secs] == ["SABA", "LABA"]
    assert secs[0].lines == ("OTC vs Prescription", "text")


@pytest.mark.parametrize("rows", [(), ((),), (("a", "b"), ())])
def test_malformed_tables_raise_split_error_naming_the_kb(rows):
    with pytest.raises(SplitError, match="KB-XX"):
        split([P("1. Asthma"), P("1. SABA"), T(rows=rows)], ["1. Asthma"])


def test_headingless_blocks_render_and_id_without_placeholder_labels():
    _, secs = split([
        P("1. Asthma"), P("overview line"), P("1. SABA"),
        table(["Drug class", "x"], ["a", "b"]),
    ], ["1. Asthma"])
    overview, saba, otc = secs
    assert overview.render().splitlines()[0] == "Asthma"
    assert otc.render().splitlines()[0] == "Asthma"
    assert saba.render().splitlines()[0] == "Asthma › SABA"
    for s in secs:
        assert "[" not in s.render() and "[" not in s.section_id
    assert len({s.section_id for s in secs}) == 3
    assert "[overview" in overview.path  # label stays internal (dump only)
