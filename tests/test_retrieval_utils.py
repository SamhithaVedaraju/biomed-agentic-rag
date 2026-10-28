import math

from biorag.eval.metrics import dedupe_docs, ndcg_at_k, recall_at_k, reciprocal_rank
from biorag.retrieval.fusion import reciprocal_rank_fusion
from biorag.retrieval.sparse import BM25Index, tokenize


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ids = [i for i, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids.index("c") > 1 and ids.index("d") > 1


def test_tokenize_keeps_hyphenated_terms():
    assert tokenize("The IL-6 levels of COVID-19 patients") == ["il-6", "levels", "covid-19",
                                                                "patients"]


def test_bm25_finds_matching_document():
    index = BM25Index(
        ["x", "y", "z"],
        ["metformin lowers glucose", "statins lower cholesterol", "aspirin and bleeding"],
    )
    assert index.search("does metformin reduce glucose", 2)[0] == "x"


def test_metrics():
    ranked = dedupe_docs(["d2", "d2", "d1", "d3"])
    assert ranked == ["d2", "d1", "d3"]
    gold = {"d1"}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 1.0
    assert reciprocal_rank(ranked, gold) == 0.5
    assert math.isclose(ndcg_at_k(ranked, gold, 10), 1 / math.log2(3))
