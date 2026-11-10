"""Catalog coverage audit: formal-gold / distractors / excluded leakage."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set


def analyze_catalog_coverage(
    *,
    formal_gold_ids: Sequence[str],
    distractor_ids: Sequence[str],
    excluded_ids: Sequence[str],
    per_query_rankings: Sequence[Dict[str, Any]],
    top_k_values: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """Aggregate operation coverage / distractor intrusion / blocked leakage.

    ``per_query_rankings`` items must include ``ranked_ids`` (list) and optionally
    ``task_id``. Evaluation-only; does not alter retrieval.
    """
    formal = set(formal_gold_ids)
    distractors = set(distractor_ids)
    excluded = set(excluded_ids)

    appear_counts: Counter = Counter()
    best_rank: Dict[str, int] = {}
    distractor_at: Dict[int, int] = {k: 0 for k in top_k_values}
    distractor_freq: Counter = Counter()
    gold_squeezed = 0
    blocked_leak_events = 0
    blocked_leak_ops: Counter = Counter()

    for row in per_query_rankings:
        ranked = [str(x) for x in (row.get("ranked_ids") or [])]
        for i, oid in enumerate(ranked, start=1):
            appear_counts[oid] += 1
            if oid not in best_rank or i < best_rank[oid]:
                best_rank[oid] = i
        leaked_here = set()
        for k in top_k_values:
            top = set(ranked[:k])
            dhit = top & distractors
            if dhit:
                distractor_at[k] += 1
                for d in dhit:
                    distractor_freq[d] += 1
            bleaks = top & excluded
            leaked_here |= bleaks
        if leaked_here:
            blocked_leak_events += 1
            for b in leaked_here:
                blocked_leak_ops[b] += 1
        # Squeeze heuristic: distractor in top-1 while any formal gold later in list.
        if ranked and ranked[0] in distractors and any(x in formal for x in ranked[1:]):
            gold_squeezed += 1

    orphans = sorted(oid for oid in formal if appear_counts.get(oid, 0) == 0)
    formal_rank_dist = {
        oid: best_rank.get(oid) for oid in sorted(formal) if oid in best_rank
    }
    return {
        "formal_gold": {
            "n": len(formal),
            "appeared_in_topk": len(formal) - len(orphans),
            "orphan_operations": orphans,
            "best_rank_by_operation": formal_rank_dist,
            "appearance_counts": {oid: appear_counts.get(oid, 0) for oid in sorted(formal)},
        },
        "distractors": {
            "n": len(distractors),
            "intrusion@1": distractor_at.get(1, 0),
            "intrusion@3": distractor_at.get(3, 0),
            "intrusion@5": distractor_at.get(5, 0),
            "intrusion_rate@1": distractor_at.get(1, 0) / max(1, len(per_query_rankings)),
            "intrusion_rate@3": distractor_at.get(3, 0) / max(1, len(per_query_rankings)),
            "intrusion_rate@5": distractor_at.get(5, 0) / max(1, len(per_query_rankings)),
            "frequency": dict(distractor_freq),
            "gold_squeezed_by_distractor_top1": gold_squeezed,
            "note": "Distractors are not required to be highly retrievable.",
        },
        "excluded": {
            "n": len(excluded),
            "candidate_leakage_events": blocked_leak_events,
            "leakage_by_operation": dict(blocked_leak_ops),
            "expected_leakage": 0,
            "ok": blocked_leak_events == 0,
        },
        "n_rankings": len(per_query_rankings),
    }
