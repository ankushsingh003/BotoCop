import os

TEST_DB_PATH = "./test_case_management.db"
os.environ["CASE_DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_case_db():
    """
    Runs once for the whole test session, before any test module has a
    chance to import backend.src.case.db under a different path. Fixes a
    real bug: with the env var set per-file, whichever test module
    imports the db module first wins (Python only imports once), and its
    teardown would delete the file the OTHER test file is still using.
    """
    from backend.src.case.db import init_db
    init_db()
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
