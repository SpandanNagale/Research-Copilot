import pytest

from core.knowledge import Embedder, KnowledgeBase, _sentence_chunks, label_themes_tfidf
from core.models import Paper


@pytest.fixture
def tfidf_embedder(monkeypatch):
    monkeypatch.setattr(
        "core.knowledge._load_sentence_transformer",
        lambda: (_ for _ in ()).throw(RuntimeError("no weights")),
    )
    e = Embedder()
    assert e.active == "tfidf-fallback"
    return e


def test_sentence_chunks_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = _sentence_chunks(text, size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 260  # size + a little slack for the last appended sentence


def test_sentence_chunks_single_short_text():
    assert _sentence_chunks("Just one short sentence.") == ["Just one short sentence."]


def test_kb_build_cluster_search_on_tfidf_fallback(tfidf_embedder, sample_papers):
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build(sample_papers)

    assert kb.paper_vectors.shape[0] == len(sample_papers)

    labels = kb.cluster(k=2)
    assert len(labels) == len(sample_papers)
    assert set(labels.tolist()) == {0, 1}
    # cluster 0 must be the largest (or tied largest)
    counts = {c: (labels == c).sum() for c in set(labels.tolist())}
    assert counts[0] == max(counts.values())

    hits = kb.search("vision transformers", k=2)
    assert len(hits) <= 2
    assert all("paper" in h and "text" in h and "score" in h for h in hits)


def test_kb_cluster_clamps_k_to_data_size(tfidf_embedder, sample_papers):
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build(sample_papers)
    labels = kb.cluster(k=50)
    assert len(set(labels.tolist())) <= len(sample_papers)


def test_kb_search_respects_theme_filter(tfidf_embedder, sample_papers):
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build(sample_papers)
    kb.cluster(k=2)
    hits = kb.search("neural networks", k=10, theme=0)
    assert all(h["paper"].cluster == 0 for h in hits)


def test_kb_search_caps_chunks_per_paper(tfidf_embedder):
    p = Paper(
        title="Long paper",
        abstract="short",
        full_text="Sentence one. Sentence two. " * 200,
    )
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build([p])
    hits = kb.search("sentence", k=10, max_per_paper=2)
    assert len(hits) <= 2


def test_centroid_nearest_orders_by_distance_to_centroid(tfidf_embedder, sample_papers):
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build(sample_papers)
    kb.cluster(k=2)
    nearest = kb.centroid_nearest(0, n=1)
    assert len(nearest) == 1
    assert nearest[0].cluster == 0


def test_project_2d_returns_two_columns(tfidf_embedder, sample_papers):
    kb = KnowledgeBase(embedder=tfidf_embedder)
    kb.build(sample_papers)
    coords = kb.project_2d()
    assert coords.shape == (len(sample_papers), 2)


def test_label_themes_tfidf_returns_a_label_per_cluster(sample_papers):
    labels = label_themes_tfidf(sample_papers)
    assert set(labels.keys()) == {0, 1}
    assert all(isinstance(v, str) and v for v in labels.values())
