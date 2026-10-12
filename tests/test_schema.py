"""Schema smoke tests: the typed contract must hold."""

import pytest
from pydantic import ValidationError

from memstrand.schema import Category, MemoryAtom, MemoryEdge, Provenance, RelationType


def test_thirteen_categories():
    assert len(Category) == 13


def test_atom_defaults():
    a = MemoryAtom(category=Category.fact, content="the sky is blue")
    assert a.provenance == Provenance.inferred
    assert 0.0 <= a.confidence <= 1.0
    assert a.expires_at is None
    assert a.id.startswith("mem:")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        MemoryAtom(category=Category.fact, content="x", confidence=1.5)


def test_contradicts_edge_is_additive():
    e = MemoryEdge(
        source_id="mem:a", target_id="mem:b",
        relation=RelationType.contradicts, reason="dates differ",
    )
    assert e.relation == RelationType.contradicts
    assert e.reason
