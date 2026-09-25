"""Batch canonicalization of extracted issue and feature request signals."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .config import ASPECT_CATEGORIES


NORMALIZATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"groups": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"canonical_name": {"type": "string", "minLength": 1},
                       "source_ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
        "required": ["canonical_name", "source_ids"],
    }}},
    "required": ["groups"],
}

CROSS_CATEGORY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"groups": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "canonical_name": {"type": "string", "minLength": 1},
            "category": {"type": "string", "enum": list(ASPECT_CATEGORIES)},
            "source_ids": {"type": "array", "items": {"type": "string", "minItems": 1}},
            "justification": {"type": "string"},
        },
        "required": ["canonical_name", "category", "source_ids", "justification"],
    }}},
    "required": ["groups"],
}


def _key(value: str) -> str:
    """Fold case/spacing and insignificant punctuation, retaining negation and numbers."""
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"[.,!?;:'\"`()\[\]{}]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _hash(value: Any, length: int = 20) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _signal_field(kind: str) -> str:
    return "issues" if kind == "issues" else "feature_requests"


def _signal_label(kind: str) -> str:
    return "issue" if kind == "issues" else "feature request"


def _collect(kind: str, analyses: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict], dict[tuple[str, str, str], str]]:
    field = _signal_field(kind)
    gathered: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rid, result in analyses.items():
        if not isinstance(result, Mapping) or str(result.get("status", "")).lower() != "success":
            continue
        for ai, aspect in enumerate(result.get("aspects", []) or []):
            if not isinstance(aspect, Mapping):
                continue
            if aspect.get("evidence_verified", True) is False:
                continue
            category = str(aspect.get("category", "other"))
            aspect_name = str(aspect.get("aspect", "other"))
            for si, signal in enumerate(aspect.get(field, []) or []):
                if not isinstance(signal, Mapping) or signal.get("evidence_verified", signal.get("verified", True)) is False:
                    continue
                for variation in [signal, *(signal.get("alternate_formulations") or [])]:
                    description = str(variation.get("description", "")).strip()
                    if not description:
                        continue
                    var_category = str(variation.get("category", category))
                    var_aspect = str(variation.get("aspect", aspect_name))
                    key = (var_category, _key(var_aspect), _key(description))
                    item = gathered.setdefault(key, {"category": var_category, "aspect": var_aspect,
                                                      "description": description, "evidence": set(), "review_ids": set()})
                    item.setdefault("descriptions", set()).add(description)
                    evidence = variation.get("evidence")
                    if isinstance(evidence, str) and evidence.strip():
                        item["evidence"].add(evidence.strip())
                    item["review_ids"].add(str(rid))
    sources: dict[str, dict] = {}
    key_to_id = {}
    for key, item in gathered.items():
        sid = "src_" + _hash([kind, *key], 24)
        # Stable collision check is performed at source creation.
        if sid in sources and (sources[sid]["category"], _key(sources[sid]["aspect"]), _key(sources[sid]["description"])) != key:
            raise ValueError("source ID hash collision")
        descriptions = sorted(item["descriptions"], key=lambda x: (_key(x), x))
        sources[sid] = {"id": sid, "category": item["category"], "aspect": item["aspect"],
                        "description": descriptions[0], "descriptions": descriptions,
                        "evidence": sorted(item["evidence"]), "review_ids": sorted(item["review_ids"])}
        key_to_id[key] = sid
    # Recompute locations to map each extracted signal to the deduplicated source id.
    loc_to_id = {}
    for rid, result in analyses.items():
        if not isinstance(result, Mapping) or str(result.get("status", "")).lower() != "success":
            continue
        for ai, aspect in enumerate(result.get("aspects", []) or []):
            if not isinstance(aspect, Mapping):
                continue
            if aspect.get("evidence_verified", True) is False:
                continue
            cat = str(aspect.get("category", "other")); asp = str(aspect.get("aspect", "other"))
            for si, signal in enumerate(aspect.get(field, []) or []):
                if not isinstance(signal, Mapping) or signal.get("evidence_verified", signal.get("verified", True)) is False:
                    continue
                desc = str(signal.get("description", "")).strip()
                key = (cat, _key(asp), _key(desc))
                if key in key_to_id:
                    loc_to_id[(str(rid), ai, si)] = key_to_id[key]
    return sources, loc_to_id


def _validate_groups(answer: Mapping[str, Any], ids: set[str], categories: Mapping[str, str]) -> list[dict]:
    groups = answer.get("groups")
    if not isinstance(groups, list):
        raise ValueError("normalization response has no groups array")
    seen: set[str] = set()
    names: set[tuple[str, str]] = set()
    valid = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("canonical_name"), str) or not group["canonical_name"].strip():
            raise ValueError("invalid canonical group")
        group_ids = group.get("source_ids")
        if not isinstance(group_ids, list) or not group_ids:
            raise ValueError("canonical group must contain source IDs")
        for sid in group_ids:
            if not isinstance(sid, str) or sid not in ids:
                raise ValueError(f"unknown source ID: {sid!r}")
            if sid in seen:
                raise ValueError(f"source ID occurs more than once: {sid}")
            seen.add(sid)
        group_cats = {categories[sid] for sid in group_ids}
        if len(group_cats) != 1:
            raise ValueError("a canonical group cannot combine different categories")
        name_key = (next(iter(group_cats)), _key(group["canonical_name"]))
        if name_key in names:
            raise ValueError("duplicate canonical name in separate groups")
        names.add(name_key)
        valid.append({"canonical_name": group["canonical_name"].strip(), "source_ids": list(group_ids),
                      "category": next(iter(group_cats))})
    missing = ids - seen
    if missing:
        raise ValueError(f"normalization omitted source IDs: {sorted(missing)}")
    return valid


_COMMON = {"the", "a", "an", "to", "for", "of", "on", "in", "per", "and", "with", "user", "app", "is", "are"}
_NEGATION = {"not", "no", "cannot", "unable", "without", "never"}


def _cross_candidate(a: dict, b: dict) -> bool:
    if a["category"] == b["category"]:
        return False
    left = set(_key(a["canonical_name"]).split()) - _COMMON
    right = set(_key(b["canonical_name"]).split()) - _COMMON
    if not left or not right:
        return False
    if {x for x in left if x.isdigit()} != {x for x in right if x.isdigit()}:
        return False
    if bool(left & _NEGATION) != bool(right & _NEGATION):
        return False
    overlap = len(left & right)
    return _key(a["canonical_name"]) == _key(b["canonical_name"]) or (
        overlap >= 2 and overlap / min(len(left), len(right)) >= 0.5)


async def _reconcile_cross_categories(kind: str, groups: list[dict], sources: dict[str, dict],
                                      client: Any, cache_dir: Path) -> list[dict]:
    """Ask the same LLM to reconcile only plausible cross-category duplicates.

    It must explain every cross-category merge; local lexical and qualifier
    guards reject broad or numerically inconsistent combinations.
    """
    candidates = {i for i, a in enumerate(groups) for j, b in enumerate(groups)
                  if i != j and _cross_candidate(a, b)}
    if not candidates:
        return groups
    selected = [groups[i] for i in sorted(candidates)]
    # Each record is a provisional canonical group, never a full review.
    records = []
    lookup = {}
    for group in selected:
        pid = "provisional_" + _hash([kind, group["category"], group["canonical_name"], sorted(group["source_ids"])], 24)
        if pid in lookup:
            raise ValueError("provisional ID collision")
        lookup[pid] = group
        records.append({"id": pid, "category": group["category"],
                        "canonical_name": group["canonical_name"],
                        "source_descriptions": [sources[sid]["description"] for sid in group["source_ids"]][:4]})
    # Keep the reconciliation request bounded. For very large candidate sets,
    # reconcile chunks; obvious exact-name repeats remain a candidate together
    # because groups are sorted by canonical name.
    records.sort(key=lambda r: (_key(r["canonical_name"]), r["id"]))
    output = [g for i, g in enumerate(groups) if i not in candidates]
    for start in range(0, len(records), 30):
        chunk = records[start:start + 30]
        path = cache_dir / f"{kind}_cross_{_hash(chunk, 64)}.json"
        if path.exists():
            answer = json.loads(path.read_text(encoding="utf-8"))
        else:
            answer = await client.json_completion(
                schema_name=f"{kind}_cross_category", schema=CROSS_CATEGORY_SCHEMA,
                system=("Review provisional canonical groups. Merge across categories ONLY if they are clearly "
                        "the same user-observed scenario. Preserve negation, numbers, and restrictions. "
                        "Keep uncertain groups separate. Give each input ID exactly once. Choose a category "
                        "from the merged inputs and explain any cross-category merge in justification. "
                        "Use an empty justification for singletons. Return valid JSON."),
                user=json.dumps({"groups": chunk}, ensure_ascii=False))
        expected = {r["id"] for r in chunk}
        seen: set[str] = set()
        merged: list[dict] = []
        try:
            for item in answer["groups"]:
                ids = item["source_ids"]
                if not ids or any(pid not in expected or pid in seen for pid in ids):
                    raise ValueError("invalid cross-category IDs")
                seen.update(ids)
                originals = [lookup[pid] for pid in ids]
                if item["category"] not in {g["category"] for g in originals}:
                    raise ValueError("cross-category output invented a category")
                if len(ids) > 1:
                    if len(item["justification"].strip()) < 12:
                        raise ValueError("cross-category merge lacks justification")
                    if any(not _cross_candidate(a, b) for i, a in enumerate(originals)
                           for b in originals[i + 1:]):
                        raise ValueError("cross-category merge lacks lexical support")
                merged.append({"canonical_name": item["canonical_name"],
                               "category": item["category"],
                               "source_ids": [sid for g in originals for sid in g["source_ids"]],
                               "category_reconciliation": item["justification"].strip() or None})
            if seen != expected:
                raise ValueError("cross-category reconciliation omitted groups")
        except (KeyError, TypeError, ValueError):
            output.extend(lookup[r["id"]] for r in chunk)
            continue
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
        output.extend(merged)
    return output


async def _cached_completion(client: Any, kind: str, records: list[dict], cache_dir: Path,
                             *, stage: str) -> list[dict]:
    fingerprint = _hash({"kind": kind, "stage": stage, "records": records}, 64)
    path = cache_dir / f"{kind}_{stage}_{fingerprint}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            categories = {x["id"]: x["category"] for x in records}
            return _validate_groups(cached, set(categories), categories)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    group_instructions = (
        "Merge descriptions only when they describe the same user-visible scenario. Keep uncertain pairs separate. "
        "Do not group merely because category or aspect matches. Preserve material distinctions such as negation, "
        "numbers, billing-after-cancellation, inability-to-cancel, and price. Names must be concise and specific; "
        "do not invent claims. Never combine records from different categories. Put every supplied ID into exactly one group."
    )
    if stage == "reconcile":
        group_instructions += " These input records are provisional groups from separate batches; combine equivalent groups only."
    system = f"Normalize { _signal_label(kind) } descriptions. {group_instructions} Return schema-valid JSON."
    user = json.dumps({"items": records}, ensure_ascii=False)
    answer = await client.json_completion(schema_name=f"{kind}_{stage}", schema=NORMALIZATION_SCHEMA,
                                          system=system, user=user)
    categories = {x["id"]: x["category"] for x in records}
    groups = _validate_groups(answer, set(categories), categories)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return groups


async def normalize_signals(kind: str, analyses: Mapping[str, Mapping[str, Any]], client: Any,
                            cache_dir: str | Path, max_batch_size: int = 30) -> tuple[dict, list[dict]]:
    """Normalize issue or feature-request signals, update analysis IDs, return canonical catalog.

    `client.json_completion` must accept schema_name, schema, system, and user keyword arguments.
    The returned catalog retains all unique original descriptions and review IDs.
    """
    if kind not in {"issues", "feature_requests"}:
        raise ValueError("kind must be 'issues' or 'feature_requests'")
    if not isinstance(max_batch_size, int) or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    updated = copy.deepcopy(dict(analyses))
    sources, location_ids = _collect(kind, updated)
    if not sources:
        return updated, []
    cache_path = Path(cache_dir)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(sources.values(), key=lambda x: (x["category"], _key(x["aspect"]), _key(x["description"]), x["id"])):
        by_category[item["category"]].append({"id": item["id"], "category": item["category"],
                                               "aspect": item["aspect"], "description": item["description"],
                                               "evidence": item["evidence"][:3]})
    final_groups: list[dict] = []
    if len(sources) <= max_batch_size:
        all_records = [record for records in by_category.values() for record in records]
        try:
            final_groups = await _cached_completion(client, kind, all_records, cache_path, stage="all")
        except ValueError:
            # A malformed semantic grouping (often a cross-category merge)
            # is retried in independent category batches, never trusted.
            pass
    if not final_groups:
        for category, records in sorted(by_category.items()):
            chunks = [records[i:i + max_batch_size] for i in range(0, len(records), max_batch_size)]
            partial_groups = []
            for chunk_index, chunk in enumerate(chunks):
                groups = await _cached_completion(client, kind, chunk, cache_path, stage=f"batch_{chunk_index}")
                partial_groups.extend(groups)
            if len(chunks) == 1:
                final_groups.extend(partial_groups)
            else:
                provisional = []
                for group in partial_groups:
                    pid = "grp_" + _hash([kind, category, group["canonical_name"], sorted(group["source_ids"])], 24)
                    source_data = [sources[sid] for sid in group["source_ids"]]
                    provisional.append({"id": pid, "category": category, "aspect": source_data[0]["aspect"],
                                        "description": group["canonical_name"], "evidence": []})
                reconciled = await _cached_completion(client, kind, provisional, cache_path, stage="reconcile_" + _hash(provisional, 12))
                id_to_group = {"grp_" + _hash([kind, category, g["canonical_name"], sorted(g["source_ids"])], 24): g
                               for g in partial_groups}
                for merged in reconciled:
                    source_ids = [sid for pid in merged["source_ids"] for sid in id_to_group[pid]["source_ids"]]
                    final_groups.append({"canonical_name": merged["canonical_name"], "source_ids": source_ids, "category": category})

    final_groups = await _reconcile_cross_categories(kind, final_groups, sources, client, cache_path)
    source_to_canonical: dict[str, str] = {}
    catalog_by_id: dict[str, dict] = {}
    for group in final_groups:
        cat = group["category"]
        canonical_key = _key(group["canonical_name"])
        cid = ("issue_" if kind == "issues" else "feature_") + _hash([kind, cat, canonical_key], 24)
        if cid in catalog_by_id and (catalog_by_id[cid]["category"], _key(catalog_by_id[cid]["canonical_name"])) != (cat, canonical_key):
            raise ValueError("canonical ID hash collision")
        entry = catalog_by_id.setdefault(cid, {"canonical_id": cid, "canonical_name": group["canonical_name"],
                                                "category": cat, "source_issues": [], "source_requests": [], "review_ids": set()})
        if group.get("category_reconciliation"):
            entry["category_reconciliation"] = group["category_reconciliation"]
        source_list = entry["source_issues"] if kind == "issues" else entry["source_requests"]
        for sid in group["source_ids"]:
            if sid in source_to_canonical:
                raise ValueError(f"source ID assigned to multiple canonical groups: {sid}")
            source_to_canonical[sid] = cid
            src = sources[sid]
            source_list.append({"source_id": sid, "aspect": src["aspect"], "description": src["description"],
                                "descriptions": src["descriptions"], "evidence": src["evidence"],
                                "review_ids": src["review_ids"]})
            entry["review_ids"].update(src["review_ids"])
    if set(source_to_canonical) != set(sources):
        raise ValueError("canonical mapping did not cover all sources")
    field = _signal_field(kind)
    for rid, result in updated.items():
        if not isinstance(result, Mapping) or str(result.get("status", "")).lower() != "success":
            continue
        for ai, aspect in enumerate(result.get("aspects", []) or []):
            if not isinstance(aspect, dict):
                continue
            for si, signal in enumerate(aspect.get(field, []) or []):
                if not isinstance(signal, dict):
                    continue
                sid = location_ids.get((str(rid), ai, si))
                signal["canonical_id"] = source_to_canonical.get(sid) if sid else None
        # One canonical scenario appears once per review. Retain each original
        # formulation and quote on the surviving signal for later audit.
        seen_review_ids: dict[str, dict] = {}
        for aspect in result.get("aspects", []) or []:
            if not isinstance(aspect, dict):
                continue
            retained = []
            for signal in aspect.get(field, []) or []:
                cid = signal.get("canonical_id") if isinstance(signal, dict) else None
                if cid and cid in seen_review_ids:
                    first = seen_review_ids[cid]
                    first.setdefault("alternate_formulations", []).append({
                        "description": signal.get("description"), "evidence": signal.get("evidence"),
                        "category": aspect.get("category"), "aspect": aspect.get("aspect")})
                else:
                    retained.append(signal)
                    if cid:
                        seen_review_ids[cid] = signal
            aspect[field] = retained
    catalog = []
    for entry in catalog_by_id.values():
        entry["review_ids"] = sorted(entry["review_ids"])
        entry["source_issues"].sort(key=lambda x: x["source_id"])
        entry["source_requests"].sort(key=lambda x: x["source_id"])
        catalog.append(entry)
    catalog.sort(key=lambda x: x["canonical_id"])
    return updated, catalog
