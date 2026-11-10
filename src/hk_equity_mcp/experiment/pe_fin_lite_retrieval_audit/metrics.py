"""Hit@K / Recall@K / MRR / complete-route metrics (evaluation-only)."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import DESCRIBE_N, TOP_K
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.gold_reference import (
    covered_node_count,
    node_covered,
)


def hit_at_k(ranked_ids: Sequence[str], acceptable: Set[str], k: int) -> float:
    top = [str(x) for x in ranked_ids[:k]]
    return 1.0 if any(x in acceptable for x in top) else 0.0


def recall_at_k(
    ranked_ids: Sequence[str],
    equivalents_by_node: Sequence[Set[str]],
    k: int,
) -> float:
    if not equivalents_by_node:
        return 0.0
    covered = covered_node_count(equivalents_by_node, ranked_ids[:k])
    return covered / float(len(equivalents_by_node))


def complete_route_at_k(
    ranked_ids: Sequence[str],
    equivalents_by_node: Sequence[Set[str]],
    k: int,
) -> float:
    if not equivalents_by_node:
        return 0.0
    return 1.0 if covered_node_count(equivalents_by_node, ranked_ids[:k]) == len(
        equivalents_by_node
    ) else 0.0


def first_relevant_rank(ranked_ids: Sequence[str], acceptable: Set[str]) -> Optional[int]:
    for i, oid in enumerate(ranked_ids, start=1):
        if str(oid) in acceptable:
            return i
    return None


def mrr_from_rank(rank: Optional[int]) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def rank4_to5_not_described(
    ranked_ids: Sequence[str],
    acceptable: Set[str],
    *,
    describe_n: int = DESCRIBE_N,
    top_k: int = TOP_K,
) -> bool:
    """True when first hit is in ranks describe_n+1..top_k (Binder-invisible)."""
    rank = first_relevant_rank(ranked_ids[:top_k], acceptable)
    if rank is None:
        return False
    return describe_n < rank <= top_k


def score_ranked_list(
    *,
    ranked_ids: Sequence[str],
    equivalents_by_node: Sequence[Set[str]],
    acceptable_union: Set[str],
    ks: Sequence[int] = (1, 3, 5),
    describe_n: int = DESCRIBE_N,
) -> Dict[str, Any]:
    eq = [set(s) for s in equivalents_by_node]
    acc = set(acceptable_union)
    ranks = []
    for node_eq in eq:
        ranks.append(first_relevant_rank(ranked_ids, node_eq))
    first = first_relevant_rank(ranked_ids, acc)
    out: Dict[str, Any] = {
        "hit": {f"@{k}": hit_at_k(ranked_ids, acc, k) for k in ks},
        "recall": {f"@{k}": recall_at_k(ranked_ids, eq, k) for k in ks},
        "complete_route": {
            f"@{k}": complete_route_at_k(ranked_ids, eq, k) for k in ks
        },
        "mrr": mrr_from_rank(first),
        "first_relevant_rank": first,
        "per_node_ranks": ranks,
        "rank4_to5_not_described": rank4_to5_not_described(
            ranked_ids, acc, describe_n=describe_n
        ),
        "exact_candidate_set_coverage": (
            1.0
            if acc
            and acc.issubset(set(str(x) for x in ranked_ids[: max(ks)]))
            else 0.0
        ),
    }
    return out


def score_multi_step_route(
    *,
    step_ranked_ids: Sequence[Sequence[str]],
    equivalents_by_node: Sequence[Set[str]],
    acceptable_union: Set[str],
    ks: Sequence[int] = (1, 3, 5),
    describe_n: int = DESCRIBE_N,
) -> Dict[str, Any]:
    """Union Top-K across retrieve steps, plus per-step Hit mean."""
    eq = [set(s) for s in equivalents_by_node]
    acc = set(acceptable_union)
    per_step_hits = []
    union_by_k: Dict[int, List[str]] = {k: [] for k in ks}
    seen_by_k: Dict[int, Set[str]] = {k: set() for k in ks}
    for ranked in step_ranked_ids:
        per_step_hits.append({f"@{k}": hit_at_k(ranked, acc, k) for k in ks})
        for k in ks:
            for oid in ranked[:k]:
                s = str(oid)
                if s not in seen_by_k[k]:
                    seen_by_k[k].add(s)
                    union_by_k[k].append(s)
    # Use max-k union list for MRR/rank diagnostics.
    max_k = max(ks)
    union_ranked = union_by_k[max_k]
    base = score_ranked_list(
        ranked_ids=union_ranked,
        equivalents_by_node=eq,
        acceptable_union=acc,
        ks=ks,
        describe_n=describe_n,
    )
    # Override complete-route/recall using each k's union.
    base["recall"] = {
        f"@{k}": recall_at_k(union_by_k[k], eq, len(union_by_k[k]) or k) for k in ks
    }
    base["complete_route"] = {
        f"@{k}": complete_route_at_k(union_by_k[k], eq, len(union_by_k[k]) or k)
        for k in ks
    }
    base["hit"] = {
        f"@{k}": (
            1.0
            if any(node_covered(e, union_by_k[k]) for e in eq) or hit_at_k(union_by_k[k], acc, len(union_by_k[k]) or k)
            else 0.0
        )
        for k in ks
    }
    # Macro Hit: mean of per-step hits (descriptive).
    base["per_step_hit_mean"] = {
        f"@{k}": (
            statistics.mean(h[f"@{k}"] for h in per_step_hits) if per_step_hits else 0.0
        )
        for k in ks
    }
    base["n_retrieve_steps"] = len(step_ranked_ids)
    return base


def cross_domain_route_coverage(
    *,
    ranked_ids: Sequence[str],
    domain_branches: Dict[str, Sequence[str]],
    k: int = 5,
) -> Dict[str, Any]:
    top = set(str(x) for x in ranked_ids[:k])
    covered = {}
    for dom, ops in domain_branches.items():
        opset = set(ops)
        covered[dom] = 1.0 if (not opset or opset & top) else 0.0
    complete = 1.0 if covered and all(v >= 1.0 for v in covered.values()) else 0.0
    return {"per_domain_branch": covered, "complete_cross_domain_route": complete}


def aggregate_metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-task metric dicts produced by score_ranked_list."""
    if not rows:
        return {"n": 0}

    def mean_key(path: Tuple[str, ...]) -> float:
        vals = []
        for r in rows:
            cur: Any = r
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    cur = None
                    break
                cur = cur[p]
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return statistics.mean(vals) if vals else 0.0

    ranks = [
        r.get("first_relevant_rank")
        for r in rows
        if isinstance(r.get("first_relevant_rank"), int)
    ]
    out = {
        "n": len(rows),
        "hit@1": mean_key(("hit", "@1")),
        "hit@3": mean_key(("hit", "@3")),
        "hit@5": mean_key(("hit", "@5")),
        "recall@1": mean_key(("recall", "@1")),
        "recall@3": mean_key(("recall", "@3")),
        "recall@5": mean_key(("recall", "@5")),
        "complete_route@3": mean_key(("complete_route", "@3")),
        "complete_route@5": mean_key(("complete_route", "@5")),
        "mrr": mean_key(("mrr",)),
        "exact_candidate_set_coverage": mean_key(("exact_candidate_set_coverage",)),
        "rank4_to5_not_described_count": sum(
            1 for r in rows if r.get("rank4_to5_not_described")
        ),
        "mean_target_rank": statistics.mean(ranks) if ranks else None,
        "median_target_rank": statistics.median(ranks) if ranks else None,
        "miss_count": sum(1 for r in rows if r.get("first_relevant_rank") is None),
    }
    return out


def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return n / d
