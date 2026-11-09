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


def test_bm25_folds_arabic_and_ignores_frame_words_in_the_question():
    from medrag.models import Line, Section

    heartburn = Section("KB-02", "GERD", "PPIs", "class", (Line("علاج الحموضة وحرقان المعدة"),))
    noise = Section("KB-08", "BP", "Statins", "class", (Line("انا عندي اخد ايه دواء"),))
    r = LexicalRetriever([noise, heartburn])
    assert r.search("الحموضه", 2) == [heartburn]  # article and taa marbuta folded: ال + حموضة
    assert r.search("انا عندي علاج الحموضة اخد ايه", 2)[0] is heartburn  # انا/عندي/اخد/ايه do not outvote the topic
    assert LexicalRetriever([noise, heartburn], normalized=False).search("انا عندي علاج الحموضة اخد ايه", 2)[0] is noise
