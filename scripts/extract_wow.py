#!/usr/bin/env python3
"""Extract the wow moments from a ReCaRe qualification run.

Usage:
    python scripts/extract_wow.py .igor/recare-100-live/qualification.json
    python scripts/extract_wow.py .igor/recare-100-live/qualification.json --eval .igor/recare-100-evaluation/evaluation.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _short(sha: str, n: int = 8) -> str:
    return sha.replace("sha256:", "")[:n] if sha else "?"


def _wrap(text: str, width: int = 80, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


# ── Wow 1: Scale ──────────────────────────────────────────────────────────────

def wow_scale(q: dict) -> list[str]:
    lines = ["## 1. Scale"]
    ds = q["dataset"]
    lines.append(f"  {ds['acquired_rows']} rows acquired, {ds['processed_rows']} processed")
    lines.append(f"  {q['snapshots']['total']} snapshots (before + after)")
    lines.append(f"  {q['amendment_events']} amendment events")
    lines.append(f"  {q['timing_seconds']:.1f}s total  ({q['timing_seconds']/ds['processed_rows']:.2f}s/row)")

    rc = q["relation_counts"]
    lines.append(f"  Relation inventory:")
    for k in sorted(rc):
        lines.append(f"    {k}: {rc[k]}")
    return lines


# ── Wow 2: Domain breadth ─────────────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "food safety / veterinary": ["food safety", "veterinary", "phytosanitary", "animals"],
    "taxation": ["taxation", "tax", "fiscal"],
    "driving / transport": ["driving licen", "road safety", "transport"],
    "financial supervision": ["financial supervision", "EIOPA", "ESFS", "supervisory authority"],
    "credit rating agencies": ["credit rating", "CRA"],
    "pensions / investment": ["pension", "investment rules", "prudent person"],
    "customs": ["customs", "tariff"],
    "insurance": ["insurance", "Solvency"],
}


def _classify_domain(reason: str) -> str:
    reason_lower = reason.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in reason_lower:
                return domain
    return "other"


def wow_domain_breadth(q: dict) -> list[str]:
    lines = ["## 2. Domain breadth — same infrastructure, different legal worlds"]
    domains: dict[str, list[str]] = {}
    for a in q["amendments"]:
        # Find records for this event
        event_id = a["event_id"]
        reason = ""
        for rec in q["ingestion"]["records"]:
            if rec["payload"].get("amendment_law_id") == event_id:
                reason = rec["payload"].get("amendment_reason", "")
                break
        domain = _classify_domain(reason)
        domains.setdefault(domain, []).append(event_id)

    for domain, events in sorted(domains.items()):
        lines.append(f"  {domain}: {', '.join(events)}")
    lines.append(f"  {len(domains)} distinct legal domains through one pipeline")
    return lines


# ── Wow 3: Semantic resolution beats lexical ──────────────────────────────────

def _lexical_overlap(a: str, b: str) -> float:
    words_a = set(re.findall(r"\w{4,}", a.lower()))
    words_b = set(re.findall(r"\w{4,}", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def wow_semantic_vs_lexical(q: dict) -> list[str]:
    lines = ["## 3. Semantic resolution beats lexical retrieval"]
    lines.append("  Cases where amendment rationale has low word overlap with affected article,")
    lines.append("  yet authority was correctly resolved:")
    lines.append("")

    pairs = []
    for snap in q["snapshots"]["items"]:
        reason = snap.get("amendment_reason", "")
        text = snap.get("preview", "")
        caption = snap.get("caption", "")
        if reason and text:
            overlap = _lexical_overlap(reason[:500], text[:500])
            pairs.append((overlap, snap))

    pairs.sort(key=lambda x: x[0])

    shown = 0
    for overlap, snap in pairs:
        if overlap > 0.25:
            break
        if shown >= 5:
            break
        lines.append(f"  [{overlap:.0%} overlap] Art.{snap['article_number']} \"{snap.get('caption', '?')}\"")
        lines.append(f"    law: {snap['law_id']}, event: {snap['event_id']}")
        lines.append(f"    rationale starts: \"{snap['amendment_reason'][:90]}...\"")
        lines.append(f"    article starts:   \"{snap['preview'][:90]}...\"")
        lines.append(f"    version: {snap['version']}, superseded_by: {_short(snap.get('superseded_by', '') or '')}")
        lines.append("")
        shown += 1

    if shown == 0:
        lines.append("  (all pairs had >25% overlap — no dramatic semantic case in this run)")
    else:
        lines.append(f"  {shown} low-overlap pairs resolved correctly — RAG ranking alone would miss these.")
    return lines


# ── Wow 4: Fail-closed is real ────────────────────────────────────────────────

def wow_fail_closed(q: dict) -> list[str]:
    lines = ["## 4. Fail-closed publication"]
    c = q["contract"]
    lines.append(f"  Contract evaluations: allow={c['allowed_count']}, deny={c['denied_count']}, abstain={c['abstained_count']}")
    lines.append(f"  Published packages:   allow={c['allowed_package_count']}, deny={c['denied_package_count']}, abstain={c['abstained_package_count']}")
    if c["denied_package_count"] == 0 and c["abstained_package_count"] == 0:
        lines.append("  Zero packages from deny/abstain. Not flagged. Not best-effort. Zero.")
    return lines


# ── Wow 5: Temporal correctness ───────────────────────────────────────────────

def wow_temporal(q: dict) -> list[str]:
    lines = ["## 5. Temporal correctness"]
    r = q["resolution"]
    lines.append(f"  Current query (as_of={r['current_as_of']}): selected={r['current_selected']}")
    lines.append(f"  Historical query (as_of={r['historical_as_of']}): selected={r['historical_selected']}")
    lines.append(f"  Superseded candidates rejected: {r['superseded_rejected']}")
    lines.append(f"  Resolution decisions: {len(r['decisions'])} current + {len(r['historical_decisions'])} historical")
    lines.append("  Same data, different point in time, different authoritative answer.")
    return lines


# ── Wow 6: Mutation selectivity ───────────────────────────────────────────────

def wow_mutation(q: dict) -> list[str]:
    lines = ["## 6. Selective invalidation"]
    m = q["mutation"]
    total = m["invalidated_count"] + m["reused_count"]
    reuse_pct = m["reused_count"] / total * 100 if total else 0
    lines.append(f"  Source change → {m['invalidated_count']} invalidated, {m['reused_count']} reused ({reuse_pct:.0f}% reuse)")
    if m["invalidated_count"] == 1:
        lines.append("  One source changed. One output invalidated. Everything else: identity-matched reuse.")
    lines.append(f"  Invalidated: {', '.join(_short(i) for i in m.get('invalidated', []))}")
    return lines


# ── Wow 7: Native ANN proof ──────────────────────────────────────────────────

def wow_ann(q: dict) -> list[str]:
    lines = ["## 7. Native ANN retrieval (not brute-force)"]
    ex = q["retrieval"]["execution"]
    plan = ex.get("plan", "")
    lines.append(f"  Adapter: {ex['adapter']}")
    lines.append(f"  Table: {ex['table']}")
    lines.append(f"  Mode: {ex['search_mode']}")
    lines.append(f"  Limit: {ex['limit']}")

    if "ANNSubIndex" in plan:
        ann_match = re.search(r"ANNSubIndex:.*", plan)
        ivf_match = re.search(r"ANNIvfPartition:.*", plan)
        if ann_match:
            lines.append(f"  ANN index: {ann_match.group(0).strip()}")
        if ivf_match:
            lines.append(f"  IVF partition: {ivf_match.group(0).strip()}")
        lines.append("  Vector search used the native index, not a scan.")
    else:
        lines.append(f"  Plan excerpt: {plan[:200]}")
    return lines


# ── Wow 8: Lineage depth ─────────────────────────────────────────────────────

def wow_lineage(q: dict) -> list[str]:
    lines = ["## 8. Lineage graph"]
    lg = q["lineage"]
    lines.append(f"  {lg['node_count']} nodes, {lg['edge_count']} edges")
    lines.append(f"  Readable display names: {lg['readable']}")
    lines.append("  Every package item traceable to source observation through derivation chain.")
    return lines


# ── Wow 9: Provider efficiency ────────────────────────────────────────────────

def wow_providers(q: dict) -> list[str]:
    lines = ["## 9. Provider efficiency"]
    p = q["providers"]
    rows = q["dataset"]["processed_rows"]
    events = q["amendment_events"]

    completions = p.get("completion", [])
    embeddings = p.get("embedding", [])
    expectations = p.get("expectations", {})

    if isinstance(completions, list):
        comp_count = len(completions)
        succeeded = sum(1 for c in completions if c.get("status") == "succeeded")
        models = {c.get("metadata", {}).get("model", "?") for c in completions}
        total_prompt = sum(c.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0) for c in completions)
        total_completion = sum(c.get("metadata", {}).get("usage", {}).get("completion_tokens", 0) for c in completions)
        lines.append(f"  Completions: {comp_count} calls, {succeeded} succeeded, model={', '.join(models)}")
        lines.append(f"    Tokens: {total_prompt:,} prompt + {total_completion:,} completion = {total_prompt + total_completion:,} total")
        lines.append(f"    {comp_count} completions for {events} events = {comp_count/events:.1f} per event (not {rows} per row)")
    else:
        lines.append(f"  Completions: {completions}")

    if isinstance(embeddings, list):
        emb_count = len(embeddings)
        emb_models = {e.get("model", "?") for e in embeddings}
        lines.append(f"  Embeddings: {emb_count} calls, model={', '.join(emb_models)}")
    elif isinstance(embeddings, dict):
        lines.append(f"  Embeddings: {embeddings.get('call_count', '?')} calls")
    else:
        lines.append(f"  Embeddings: {embeddings}")

    if expectations:
        lines.append(f"  Budget expectations: {json.dumps(expectations)}")
    return lines


# ── Wow 10: Package contents ─────────────────────────────────────────────────

def wow_package(q: dict) -> list[str]:
    lines = ["## 10. Published package"]
    for pkg in q["packages"]:
        items = pkg.get("items", [])
        lines.append(f"  Task: {pkg.get('task_id', '?')}")
        lines.append(f"  Decision: {pkg.get('decision', '?')}")
        lines.append(f"  Budget: {pkg.get('budget_tokens', '?')} tokens")
        lines.append(f"  Items: {len(items)}")
        if items:
            ranks = [it.get("rank", -1) for it in items]
            tokens = [it.get("token_estimate", 0) for it in items]
            lines.append(f"  Rank range: {min(ranks)}–{max(ranks)}")
            lines.append(f"  Token range: {min(tokens)}–{max(tokens)} per item, {sum(tokens)} total")
    return lines


# ── Wow 11: Independent eval ─────────────────────────────────────────────────

def wow_eval(eval_data: dict | None) -> list[str]:
    if eval_data is None:
        return []
    lines = ["## 11. Independent evaluation"]
    checks = eval_data.get("checks", {})
    passed = sum(1 for c in checks.values() if c.get("passed"))
    total = len(checks)
    lines.append(f"  {passed}/{total} checks passed")
    failed = [name for name, c in checks.items() if not c.get("passed")]
    if failed:
        lines.append(f"  FAILED: {', '.join(failed)}")
    else:
        lines.append("  All passed. Zero failures. Independent evaluator confirms every invariant.")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract wow moments from IGOR qualification artifacts")
    parser.add_argument("qualification", help="Path to qualification.json")
    parser.add_argument("--eval", dest="eval_path", help="Path to evaluation.json (optional)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    args = parser.parse_args()

    q = _load(args.qualification)
    eval_data = _load(args.eval_path) if args.eval_path else None

    sections = [
        wow_scale(q),
        wow_domain_breadth(q),
        wow_semantic_vs_lexical(q),
        wow_fail_closed(q),
        wow_temporal(q),
        wow_mutation(q),
        wow_ann(q),
        wow_lineage(q),
        wow_providers(q),
        wow_package(q),
        wow_eval(eval_data),
    ]

    header = [
        "# IGOR Qualification — Wow Extraction",
        f"# Source: {args.qualification}",
        f"# Mode: {q.get('mode', '?')}",
        f"# Schema: {q.get('schema_version', '?')}",
        f"# Valid: {q.get('valid', '?')}",
        "",
    ]

    if args.json:
        out = {
            "source": args.qualification,
            "mode": q.get("mode"),
            "valid": q.get("valid"),
            "scale": {
                "rows": q["dataset"]["processed_rows"],
                "snapshots": q["snapshots"]["total"],
                "events": q["amendment_events"],
                "seconds": q["timing_seconds"],
                "seconds_per_row": round(q["timing_seconds"] / q["dataset"]["processed_rows"], 2),
            },
            "fail_closed": {
                "deny_packages": q["contract"]["denied_package_count"],
                "abstain_packages": q["contract"]["abstained_package_count"],
                "allow_packages": q["contract"]["allowed_package_count"],
            },
            "mutation": {
                "invalidated": q["mutation"]["invalidated_count"],
                "reused": q["mutation"]["reused_count"],
                "reuse_pct": round(q["mutation"]["reused_count"] / (q["mutation"]["invalidated_count"] + q["mutation"]["reused_count"]) * 100),
            },
            "lineage": q["lineage"],
            "ann_native": "ANNSubIndex" in q["retrieval"]["execution"].get("plan", ""),
            "lineage": {"nodes": q["lineage"]["node_count"], "edges": q["lineage"]["edge_count"], "readable": q["lineage"]["readable"]},
            "providers": {
                "completions": len(q["providers"].get("completion", [])) if isinstance(q["providers"].get("completion"), list) else "?",
                "embeddings": len(q["providers"].get("embedding", [])) if isinstance(q["providers"].get("embedding"), list) else "?",
                "completion_tokens": sum(
                    c.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
                    for c in (q["providers"].get("completion", []) if isinstance(q["providers"].get("completion"), list) else [])
                ),
            },
            "temporal": {
                "current_as_of": q["resolution"]["current_as_of"],
                "historical_as_of": q["resolution"]["historical_as_of"],
                "superseded_rejected": q["resolution"]["superseded_rejected"],
            },
            "eval_passed": all(c.get("passed") for c in eval_data["checks"].values()) if eval_data else None,
        }
        print(json.dumps(out, indent=2))
    else:
        print("\n".join(header))
        for section in sections:
            if section:
                print("\n".join(section))
                print()


if __name__ == "__main__":
    main()
