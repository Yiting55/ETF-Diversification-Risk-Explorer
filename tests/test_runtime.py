import importlib

from engine.runtime import load_engine


def test_unchanged_engine_keeps_module_classes_and_revision():
    package, revision = load_engine()
    analysis_class = package.Analysis
    again, again_revision = load_engine()
    assert again_revision == revision
    assert again.Analysis is analysis_class
    assert "correlation_prices" in analysis_class.__dataclass_fields__


def test_stale_engine_exports_are_refreshed():
    package, revision = load_engine()
    module = importlib.import_module("engine.analyze")
    original_class = module.Analysis
    # Simulate an already-running server retaining the old result schema.
    module.Analysis = type("LegacyAnalysis", (), {})
    package.Analysis = module.Analysis
    package._source_revision = "old-source"
    refreshed, new_revision = load_engine()
    assert new_revision == revision
    assert refreshed.Analysis is not original_class
    assert refreshed.Analysis is module.Analysis
    assert "correlation_prices" in refreshed.Analysis.__dataclass_fields__
    assert "correlations" in refreshed.Analysis.__dataclass_fields__
