from synthetic import NSAIDS, PPI, SECTIONS

from medrag.language import detect_language
from medrag.retrieval import LexicalRetriever


def test_bm25_ranks_the_matching_section_first():
    r = LexicalRetriever(SECTIONS)
    assert r.search("active ingredient of Cataflam", 3)[0] is NSAIDS
    assert r.search("Controloc", 3) == [PPI]


def test_no_term_overlap_returns_nothing():
    assert LexicalRetriever(SECTIONS).search("zzzz", 3) == []
    assert LexicalRetriever([]).search("anything", 3) == []


def test_language_heuristic():
    assert detect_language("what is Cataflam") == "en"
    assert detect_language("ايه هو كتافلام") == "ar"
