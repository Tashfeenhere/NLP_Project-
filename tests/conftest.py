"""Shared fixtures.

The spaCy pipeline is expensive to build, so it is session-scoped: all rule,
negation and offset tests reuse one instance.  API tests run against a
temporary SQLite file.
"""
from __future__ import annotations

import pytest

from nlp.pipeline import build_pipeline, get_pipeline


@pytest.fixture(scope="session")
def pipeline():
    # The API's singleton: one model load for the whole test session.
    return get_pipeline()


@pytest.fixture(scope="session")
def rules_pipeline():
    return build_pipeline(use_ner=False)
