import os

import pytest

from medrag.ingest import kb_dir, load_knowledge_base, missing_files
from medrag.manifest import load_manifest


@pytest.fixture(scope="session")
def real_kb():
    """The nine KB Documents from the real .docx files; skips when absent
    (data/ is not committed, so CI has no KB files)."""
    directory = kb_dir()
    missing = missing_files(load_manifest(), directory)
    if missing:
        if os.environ.get("MEDRAG_REQUIRE_KB") == "1":
            pytest.fail(f"MEDRAG_REQUIRE_KB=1 but KB files are missing from {directory}: {missing}")
        pytest.skip(f"KB .docx files not found in {directory} (set MEDRAG_KB_DIR)")
    return {d.kb_id: d for d in load_knowledge_base(directory)}
