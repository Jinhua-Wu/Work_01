"""Reproducible experiments, blind rating, and interactive evidence extraction."""
import argparse
import csv
import json
import platform
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from statistics import mean

from engine import (ROOT, DESIGN_TASK, MANUAL, MANUAL_SOURCE, Client, canonical,
                    compile_prompt, digest, finalize, fingerprint, jsonl, make_input,
                    run_case, save_json, validate)
from energy import candidates

VARIANTS = ("manual", "direct", "auto", "no_knowledge", "no_loop")
MAIN_GROUPS = ("manual", "direct", "auto")
GENERATE = "You design reusable prompts for scientific extraction. Respond in one concise English paragraph."
CODE_NAMES = {"manual": "published question", "direct": "one-shot LLM",
              "auto": "retrieval and feedback", "no_knowledge": "feedback only", "no_loop": "retrieval only"}


def aggregate(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    denom = 2 * tp + fp + fn
    pred_n, gold_n = tp + fp, tp + fn
    return {"n": len(rows), "correct_n": sum(r["correct"] for r in rows),
            "format_n": sum(r["format"] for r in rows),
            "raw_format_n": sum(r["raw_format"] for r in rows),
            "valid_n": sum(r["valid"] for r in rows), "accuracy": mean(r["correct"] for r in rows),
            "format": mean(r["format"] for r in rows), "raw_format": mean(r["raw_format"] for r in rows),
            "valid": mean(r["valid"] for r in rows), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / denom if denom else 0,
            "value_f1": 2 * sum(r["value_tp"] for r in rows) / denom if denom else 0,
            "anchor_f1": 2 * sum(r["anchor_tp"] for r in rows) / denom if denom else 0,
            "method_on_anchored": tp / sum(r["anchor_tp"] for r in rows) if any(r["anchor_tp"] for r in rows) else None,
            "precision": tp / pred_n if pred_n else 0, "recall": tp / gold_n if gold_n else 0,
            "prompt_tokens": mean(r["prompt_tokens"] for r in rows),
            "output_tokens": mean(r["output_tokens"] for r in rows),
            "tokens": mean(r["prompt_tokens"] + r["output_tokens"] for r in rows),
            "latency": mean(r["seconds"] for r in rows)}


def assess(client, pool, cases, variant, policy, post_verify=True):
    return list(pool.map(partial(run_case, client, variant=variant, policy=policy,
                                 post_verify=post_verify), cases))


def failures_for_reflection(dev, predictions):
    failed = []
    for case, r in zip(dev, predictions):
        if r["correct"]:
            continue
        expected_tags = []
        for value, unit, method, sid in case["gold"]:
            tag = next((m["id"] for m in r["mentions"] if m["sentence"] == sid
                        and m["value"] == value and m["unit"] == unit), None)
            expected_tags.append([tag, method, sid])
        selected = r["selection"] or {}
        by_id = {m["id"]: m for m in r["mentions"]}
        selected_sids = [by_id[k]["sentence"] for k in selected if k in by_id]
        sids = sorted(set(selected_sids + [g[-1] for g in expected_tags]))
        # Keep reflection context compact while retaining source sentences for the failing choices.
        ctx = [{"sentence": sid, "text": case["sentences"][sid-1][:240]} for sid in sids[:3]]
        category = ("schema" if not r["format"] else "origin" if r["anchor_tp"] > r["tp"]
                    else "missed" if r["fn"] else "distractor")
        failed.append({"category": category, "target": case["material"], "evidence": ctx,
                       "expected": expected_tags, "model": r["raw"][:170],
                       "errors": r["errors"]})
    return failed


def policy_text(reply):
    words = " ".join(reply["raw"].strip().split())
    if not words or reply["finish"] == "length" or len(words.split()) > 120:
        raise ValueError("Unusable generation response. Inspect the local model and retry.")
    return words


def optimize(client, pool, dev, variant, initial, rounds):
    trace = []
    best = None
    candidate = initial
    for step in range(rounds + 1):
        predictions = assess(client, pool, dev, variant, candidate, post_verify=False)
        metrics = aggregate(predictions)
        objective = (0.55 * metrics["accuracy"] + 0.30 * metrics["f1"]
                     + 0.10 * metrics["format"] + 0.05 * metrics["valid"])
        key = (round(objective, 10), -metrics["tokens"], -step)
        current = {"round": step, "policy": candidate, "sha256": digest(candidate),
                   "objective": objective, "metrics": metrics, "predictions": predictions}
        trace.append(current)
        if best is None or key > best[0]:
            best = (key, current)
        print(f"{variant} round {step}: J={objective:.4f} strict={metrics['correct_n']}/{len(dev)}", flush=True)
        if step == rounds:
            break
        champion = best[1]
        failures = failures_for_reflection(dev, champion["predictions"])
        picked = []
        by_category = {}
        for f in failures:
            by_category[f["category"]] = by_category.get(f["category"], 0) + 1
        remaining = sorted(by_category, key=lambda k: (-by_category[k], k))
        # Rotate categories across rounds so the feedback is not the same prefix each time.
        if remaining:
            offset = step % len(remaining)
            remaining = remaining[offset:] + remaining[:offset]
            picked = [next(f for f in failures if f["category"] == c) for c in remaining[:2]]
        reflection = client.call(
            "You diagnose mistakes in selecting evidence tags and provenance for band gaps.",
            "Compare expected tags and model JSON. Name two transferable errors and concise fixes. "
            "Focus on composition and provenance, not values to memorize. "
            f"Error counts {canonical(by_category)}. Examples {canonical(picked)}",
            128, seed=50+step, temperature=0)
        rewrite = client.call(
            GENERATE,
            "Rewrite the general extraction procedure using the critique. "
            "Use 35–75 words, no candidate IDs, names of specific materials, or worked answers. "
            "Explain target matching, source types, and conservative evidence checks. "
            "The JSON protocol is supplied separately. Return just the instruction.\n"
            f"Current: {champion['policy']}\nCritique: {reflection['raw']}",
            150, seed=70+step, temperature=0.40)
        candidate = policy_text(rewrite)
        current.update({"feedback_categories": by_category, "reflection": reflection, "rewrite": rewrite})
    winner = best[1]
    return winner["policy"], {"winner_round": winner["round"], "trace": trace}


def design_calls(trace):
    calls = []
    for row in trace["trace"]:
        calls.extend(row["predictions"])
        calls.extend(row[k] for k in ("reflection", "rewrite") if k in row)
    return calls


def tokens(calls):
    return sum(r["prompt_tokens"] + r["output_tokens"] for r in calls)


def paired_ci(full, other, draws=3000):
    a = {r["id"]: r["correct"] for r in full}
    b = {r["id"]: r["correct"] for r in other}
    diffs = [int(a[k]) - int(b[k]) for k in a]
    rng = random.Random(63271)
    dist = sorted(mean(rng.choices(diffs, k=len(diffs))) * 100 for _ in range(draws))
    return {"accuracy_delta_pp": mean(diffs)*100,
            "paired_bootstrap_95_pp": [dist[int(draws*0.025)], dist[int(draws*0.975)]],
            "only_auto": diffs.count(1), "only_other": diffs.count(-1)}


def expert_case_ids(test, n=10):
    rng = random.Random(20261112)
    pool = list(test)
    rng.shuffle(pool)
    selected = []
    for positive in (True, False):
        group = [c for c in pool if bool(c["gold"]) == positive]
        materials = dict.fromkeys(c["material"] for c in group)
        portion = n//2 if positive else n-n//2
        chosen = [next(c for c in group if c["material"] == material)
                  for material in materials][:portion]
        seen = {c["id"] for c in chosen}
        chosen += [c for c in group if c["id"] not in seen][:portion-len(chosen)]
        selected.extend(c["id"] for c in chosen)
    return selected


def export_review(out, test, rows):
    focused = set(expert_case_ids(test))
    rng = random.Random(20261113)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    cases = {case["id"]: case for case in test}
    mapping, review, minimum = {}, [], []
    header = ["review_id", "material", "source", "numbered_passage", "observation_rows", "validation",
              "expert_score_1_5", "reviewer_code", "reviewer_role", "notes"]
    for i, r in enumerate(shuffled, 1):
        case = cases[r["id"]]
        rid = f"{'C' if case['split'] == 'confirm' else 'T'}-{i:03}"
        mapping[rid] = [r["id"], r["variant"]]
        passage = "\n".join(f"{j}. {s}" for j, s in enumerate(case["sentences"], 1))
        status = "valid" if r["valid"] else "rejected"
        artifact = canonical(r["rows"]) if r["valid"] else r["raw"]
        entry = [rid, case["material"], case["source"], passage, artifact, status, "", "", "", ""]
        review.append(entry)
        if r["id"] in focused and r["variant"] in MAIN_GROUPS:
            minimum.append(entry)
    for name, entries in (("expert_blind.csv", review), ("expert_priority.csv", minimum)):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(entries)
    save_json(out / "expert_key.json", {"items": mapping, "priority_case_ids": sorted(focused),
                                         "priority_review_ids": [r[0] for r in minimum]})


def report_experts(paths, key_path):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    items, priority = key["items"], set(key["priority_review_ids"])
    per_variant = {v: {} for v in VARIANTS}
    roles = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row["expert_score_1_5"].strip()
                if not value:
                    continue
                reviewer, role = row["reviewer_code"].strip(), row["reviewer_role"].strip()
                if value not in {"1", "2", "3", "4", "5"} or not reviewer or not role:
                    raise ValueError("Each score needs an integer 1–5, a reviewer code and a declared role.")
                if reviewer in roles and roles[reviewer] != role:
                    raise ValueError(f"Reviewer role differs for {reviewer}")
                roles[reviewer] = role
                rid = row["review_id"]
                _, variant = items[rid]
                reviews = per_variant[variant].setdefault(rid, {})
                if reviewer in reviews:
                    raise ValueError(f"Duplicate review: {rid}, {reviewer}")
                reviews[reviewer] = int(value)
    report = {}
    for v, entries in per_variant.items():
        preferred = [mean(scores.values()) for rid, scores in entries.items() if rid in priority]
        report[v] = {"n_rated_outputs": len(entries), "n_ratings": sum(map(len, entries.values())),
                     "mean_1_5": mean(mean(x.values()) for x in entries.values()) if entries else None,
                     "priority_n": len(preferred), "priority_mean_1_5": mean(preferred) if preferred else None}
    return {"groups": report, "declared_reviewers": roles,
            "priority_complete": all(report[v]["priority_n"] == len(key["priority_case_ids"])
                                     for v in MAIN_GROUPS)}


def compare_replay(client, pool, cases, archived, variant, policy):
    original = {r["id"]: r for r in archived if r["variant"] == variant}
    repeated = assess(client, pool, cases, variant, policy)
    audit = []
    for row in repeated:
        prev = original[row["id"]]
        audit.append({"id": row["id"], "same_input": row["input_sha256"] == prev["input_sha256"],
                      "same_prompt": row["prompt_sha256"] == prev["prompt_sha256"],
                      "same_raw": row["raw"] == prev["raw"],
                      "same_records": (row["records"], row["valid"], row["errors"]) ==
                                      (prev["records"], prev["valid"], prev["errors"]),
                      "same_accuracy": row["correct"] == prev["correct"],
                      "run": row})
    return {"variant": variant, "n": len(audit),
            "identical_raw": sum(a["same_raw"] for a in audit),
            "identical_records": sum(a["same_records"] for a in audit), "audit": audit}


def resume_design(out, dev, rounds):
    policies = json.loads((out / "policies.json").read_text(encoding="utf-8"))
    details = json.loads((out / "optimization.json").read_text(encoding="utf-8"))
    for variant in ("auto", "no_knowledge"):
        opt = details[variant]
        if len(opt["trace"]) != rounds + 1 or opt["trace"][opt["winner_round"]]["policy"] != policies[variant]:
            raise ValueError("Frozen design does not match requested rounds or winning policies")
        for entry in opt["trace"]:
            if len(entry["predictions"]) != len(dev):
                raise ValueError("Frozen design used a different development split")
            for case, row in zip(dev, entry["predictions"]):
                prompt, _ = compile_prompt(case, variant, entry["policy"])
                user = make_input(case, candidates(case["sentences"]))
                if (row["id"], row["input_sha256"], row["prompt_sha256"]) != (
                        case["id"], fingerprint(case), digest(canonical([prompt, user]))):
                    raise ValueError("Frozen design used different data or compiler")
    return policies, details["direct_design"], details["initial_generation"], {
        v: details[v] for v in ("auto", "no_knowledge")}


def bench(args):
    data = jsonl(args.data)
    dev = [x for x in data if x["split"] == "dev"]
    test = [x for x in data if x["split"] == "test"]
    probes = jsonl(args.probes)
    if not dev or len(test) < 30 or len({x["id"] for x in data + probes}) != len(data + probes):
        raise ValueError("Require dev examples, ≥30 test examples, and unique IDs.")
    if {x["source_key"] for x in dev} & {x["source_key"] for x in test}:
        raise ValueError("Source leakage between dev and test")
    client = Client(args.endpoint)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows, probe_rows, summaries, probe_summaries = [], [], {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.resume_design:
            policies, direct, start, optimizations = resume_design(out, dev, args.rounds)
        else:
            direct = client.call(GENERATE, DESIGN_TASK, 160, seed=11, temperature=0.35)
            start = client.call(GENERATE, DESIGN_TASK + "\nFor the decision procedure, emphasize a precise "
                                "material-and-sample relation and a local evidence trail.", 160,
                                seed=19, temperature=0.35)
            policies = {"manual": MANUAL, "direct": policy_text(direct), "no_loop": policy_text(start)}
            optimizations = {}
            for variant in ("auto", "no_knowledge"):
                policies[variant], optimizations[variant] = optimize(
                    client, pool, dev, variant, policies["no_loop"], args.rounds)
            save_json(out / "policies.json", policies)
            save_json(out / "optimization.json", {"manual_source": MANUAL_SOURCE, "direct_design": direct,
                                                   "initial_generation": start, **optimizations})
        if args.design_only:
            return
        for variant in VARIANTS:
            got = assess(client, pool, test, variant, policies[variant])
            rows.extend(got)
            summaries[variant] = aggregate(got)
            print(f"{variant} test: {summaries[variant]['correct_n']}/{len(test)}", flush=True)
            got = assess(client, pool, probes, variant, policies[variant])
            probe_rows.extend(got)
            probe_summaries[variant] = aggregate(got)
            save_json(out / "predictions.json", rows)
            save_json(out / "probe_predictions.json", probe_rows)
        repeat = [c for c in test if c["id"] in set(expert_case_ids(test, n=8))]
        replay = compare_replay(client, pool, repeat, rows, "auto", policies["auto"])
    write_report(out, args, test, dev, probes, rows, probe_rows, replay,
                 summaries, probe_summaries, direct, start, optimizations, policies)


def write_report(out, args, test, dev, probes, rows, probe_rows, replay,
                 summaries, probe_summaries, direct, start, optimizations, policies):
    cost = {"manual": [], "direct": [direct], "no_loop": [start],
            **{v: [start] + design_calls(o) for v, o in optimizations.items()}}
    for variant, calls in cost.items():
        s = summaries[variant]
        s["design_tokens"] = tokens(calls)
        s["amortized_tokens"] = s["tokens"] + s["design_tokens"]/s["n"]
    deltas, comparisons, strata = {}, {}, {}
    for v in VARIANTS:
        group = [r for r in rows if r["variant"] == v]
        strata[v] = {
            key: {"n": len(filtered), "correct": sum(r["correct"] for r in filtered)}
            for key, relevant in (("positive", {x["id"] for x in test if x["gold"]}),
                                  ("negative", {x["id"] for x in test if not x["gold"]}))
            if (filtered := [r for r in group if r["id"] in relevant])
        }
        if v != "auto":
            comparisons[v] = paired_ci([r for r in rows if r["variant"] == "auto"], group)
        if v in ("no_knowledge", "no_loop"):
            deltas[v] = {k: (summaries["auto"][k]-summaries[v][k])*100
                         for k in ("accuracy", "format", "valid", "f1")}
    save_json(out / "replay.json", replay)
    save_json(out / "summary.json", {
        "pv600_test": summaries, "supplementary_probes": probe_summaries, "strata": strata,
        "pre_verifier": {v: aggregate([{**r, **r["before_verifier"]} for r in rows if r["variant"] == v])
                         for v in VARIANTS},
        "auto_minus_ablation_pp": deltas, "paired_comparisons": comparisons,
        "configuration": {"rounds": args.rounds, "workers": args.workers,
                          "python": platform.python_version(), "platform": platform.platform(),
                          "server_model": rows[0]["model"], "dev": len(dev), "test": len(test),
                          "probes": len(probes), "dataset_sha256": digest(args.data.read_text(encoding="utf-8")),
                          "policy_sha256": digest(canonical(policies))}})
    export_review(out, test, rows)
    save_json(out / "expert_summary.json", report_experts([out / "expert_priority.csv"], out / "expert_key.json"))


def reprocess(args):
    data = jsonl(args.data)
    dev = [r for r in data if r["split"] == "dev"]
    test = [r for r in data if r["split"] == "test"]
    probes = jsonl(args.probes)
    cases = {r["id"]: r for r in test + probes}
    src, out = args.from_results, args.out
    out.mkdir(parents=True, exist_ok=True)
    policies, direct, start, opt = resume_design(src, dev, args.rounds)
    for name in ("policies.json", "optimization.json"):
        save_json(out / name, json.loads((src / name).read_text(encoding="utf-8")))

    def apply(run):
        case = cases[run["id"]]
        if fingerprint(case) != run["input_sha256"]:
            raise ValueError(f"Frozen prediction has the wrong input {run['id']}")
        checked = validate(run["raw"], case, candidates(case["sentences"]), run["finish"])
        return finalize(case, {**run, **checked})

    rows = [apply(r) for r in json.loads((src/"predictions.json").read_text(encoding="utf-8"))]
    probe_rows = [apply(r) for r in json.loads((src/"probe_predictions.json").read_text(encoding="utf-8"))]
    previous = json.loads((src/"replay.json").read_text(encoding="utf-8"))
    base = {r["id"]: r for r in rows if r["variant"] == previous["variant"]}
    audit = []
    for a in previous["audit"]:
        r = apply(a["run"])
        original = base[r["id"]]
        audit.append({**a, "same_records": (r["records"], r["valid"], r["errors"]) ==
                      (original["records"], original["valid"], original["errors"]),
                      "same_accuracy": r["correct"] == original["correct"], "run": r})
    replay = {**previous, "audit": audit, "identical_records": sum(x["same_records"] for x in audit)}
    summaries = {v: aggregate([r for r in rows if r["variant"] == v]) for v in VARIANTS}
    probe_summaries = {v: aggregate([r for r in probe_rows if r["variant"] == v]) for v in VARIANTS}
    save_json(out/"predictions.json", rows)
    save_json(out/"probe_predictions.json", probe_rows)
    write_report(out, args, test, dev, probes, rows, probe_rows, replay,
                 summaries, probe_summaries, direct, start, opt, policies)


def confirm(args):
    cases = jsonl(args.data)
    prior = jsonl(args.existing)
    blocked = {r["source_key"] for r in prior}
    blocked.update(json.loads(args.prior.read_text(encoding="utf-8"))["excluded_source_keys"])
    if len(cases) < 30 or len({r["source_key"] for r in cases}) != len(cases):
        raise ValueError("Need at least 30 fresh cases with one case per source.")
    if blocked & {r["source_key"] for r in cases}:
        raise ValueError("Confirmation overlaps development or diagnostic sources.")
    optdir = args.design
    resume_design(optdir, [r for r in prior if r["split"] == "dev"], args.rounds)
    policies = json.loads((optdir/"policies.json").read_text(encoding="utf-8"))
    client = Client(args.endpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    measures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v in VARIANTS:
            r = assess(client, pool, cases, v, policies[v])
            rows.extend(r)
            measures[v] = aggregate(r)
            print(f"{v} confirmation: {measures[v]['correct_n']}/{len(cases)}", flush=True)
            save_json(args.out/"confirm_predictions.json", rows)
    full = [r for r in rows if r["variant"] == "auto"]
    save_json(args.out/"confirm_summary.json", {
        "groups": measures,
        "pre_verifier": {v: aggregate([{**r, **r["before_verifier"]} for r in rows if r["variant"] == v])
                         for v in VARIANTS},
        "paired_comparisons": {v: paired_ci(full, [r for r in rows if r["variant"] == v])
                               for v in VARIANTS if v != "auto"},
        "positive": sum(bool(r["gold"]) for r in cases),
        "negative": sum(not r["gold"] for r in cases),
        "dataset_sha256": digest(args.data.read_text(encoding="utf-8")),
        "policy_sha256": digest(canonical(policies)),
    })
    export_review(args.out, cases, rows)
    save_json(args.out/"expert_summary.json",
              report_experts([args.out/"expert_priority.csv"], args.out/"expert_key.json"))


def valid_case(raw):
    if not isinstance(raw, dict):
        raise ValueError("Input must be a JSON object")
    case = {k: raw[k] for k in ("id", "material", "sentences", "pv_alias", "source") if k in raw}
    material, sentences = case.get("material"), case.get("sentences")
    if not isinstance(material, str) or not material.strip() or not isinstance(sentences, list) \
       or not sentences or not all(isinstance(s, str) and s.strip() for s in sentences):
        raise ValueError("Input needs a target material and nonempty ordered sentences")
    if any(k in case and not isinstance(case[k], str) for k in ("id", "pv_alias", "source")):
        raise ValueError("Optional id, pv_alias and source must be strings")
    return case


def interactive_input():
    print("输入现场给出的新材料和段落，每行一句，以空行结束。", flush=True)
    material = input("目标材料：").strip()
    source = input("文献标识或来源（可留空）：").strip()
    sentences = []
    while line := input().strip():
        sentences.append(line)
    return valid_case({"id": "LIVE", "material": material, "source": source, "sentences": sentences})


def extract(args):
    case = (interactive_input() if args.command == "live" else
            valid_case(json.loads(args.input.read_text(encoding="utf-8-sig"))))
    policies = json.loads(args.policies.read_text(encoding="utf-8"))
    result = run_case(Client(args.endpoint), case, args.variant, policies[args.variant])
    result["reference_status"] = "reference_not_requested"
    if args.reference:
        ref = next((r for r in jsonl(args.reference) if fingerprint(r) == fingerprint(case)), None)
        if ref is None:
            result["reference_status"] = "same_source_passage_not_indexed"
        else:
            from engine import evaluate
            comparison = evaluate(result, ref["gold"])
            result.update({"reference_status": "matches_reference" if comparison["correct"]
                           else "mismatch_with_reference", "reference_origin": ref.get("origin"),
                           "reference_metrics": comparison})
    if args.audit:
        from datetime import datetime, timezone
        save_json(args.audit, {"case": case, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                               "command": args.command, **result})
    fields = ("id", "variant", "cards", "format", "valid", "errors", "rows", "records",
              "reference_status", "raw", "seconds")
    print(json.dumps({k: result[k] for k in fields}, ensure_ascii=False, indent=2))


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="Generate, optimize and evaluate all groups")
    b.add_argument("--data", type=Path, default=ROOT / "data/pv600_subset.jsonl")
    b.add_argument("--probes", type=Path, default=ROOT / "data/probes.jsonl")
    b.add_argument("--out", type=Path, default=ROOT / "reproduced")
    b.add_argument("--rounds", type=int, default=2)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--endpoint", default="http://127.0.0.1:8279")
    modes = b.add_mutually_exclusive_group()
    modes.add_argument("--design-only", action="store_true", help="Freeze policies without test inference")
    modes.add_argument("--resume-design", action="store_true", help="Verify and reuse a frozen design")
    replay = sub.add_parser("reprocess", help="Revalidate existing model calls without extra inference")
    replay.add_argument("--data", type=Path, default=ROOT / "data/pv600_subset.jsonl")
    replay.add_argument("--probes", type=Path, default=ROOT / "data/probes.jsonl")
    replay.add_argument("--from-results", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--rounds", type=int, default=2)
    replay.add_argument("--workers", type=int, default=2)
    conf = sub.add_parser("confirm", help="Evaluate the frozen verifier and prompts on new sources")
    conf.add_argument("--data", type=Path, default=ROOT/"data/pv600_confirm.jsonl")
    conf.add_argument("--existing", type=Path, default=ROOT/"data/pv600_subset.jsonl")
    conf.add_argument("--prior", type=Path, default=ROOT/"data/prior_sources.json")
    conf.add_argument("--design", type=Path, default=ROOT/"results")
    conf.add_argument("--out", type=Path, default=ROOT/"confirmation")
    conf.add_argument("--workers", type=int, default=2)
    conf.add_argument("--rounds", type=int, default=2)
    conf.add_argument("--endpoint", default="http://127.0.0.1:8279")
    for name in ("extract", "live"):
        parser = sub.add_parser(name, help="Extract and verify a fresh passage")
        if name == "extract":
            parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--policies", type=Path, default=ROOT / "results/policies.json")
        parser.add_argument("--variant", choices=VARIANTS, default="auto")
        parser.add_argument("--endpoint", default="http://127.0.0.1:8279")
        parser.add_argument("--audit", type=Path)
        parser.add_argument("--reference", type=Path)
    r = sub.add_parser("expert-report", help="Aggregate blinded human scores from UTF-8 CSV files")
    r.add_argument("--ratings", type=Path, nargs="+", required=True)
    r.add_argument("--key", type=Path, default=ROOT / "confirmation/expert_key.json")
    r.add_argument("--out", type=Path)
    export = sub.add_parser("expert-export", help="Regenerate blinded sheets from archived predictions")
    export.add_argument("--data", type=Path, default=ROOT / "data/pv600_subset.jsonl")
    export.add_argument("--results", type=Path, default=ROOT / "results")
    export.add_argument("--confirm", action="store_true")
    a = p.parse_args()
    if a.command == "bench":
        if a.rounds < 1 or a.workers < 1:
            p.error("rounds and workers must be positive")
        bench(a)
    elif a.command == "reprocess":
        reprocess(a)
    elif a.command == "confirm":
        confirm(a)
    elif a.command in ("extract", "live"):
        extract(a)
    elif a.command == "expert-export":
        test = [c for c in jsonl(a.data) if c["split"] == ("confirm" if a.confirm else "test")]
        prediction_name = "confirm_predictions.json" if a.confirm else "predictions.json"
        predictions = json.loads((a.results / prediction_name).read_text(encoding="utf-8"))
        export_review(a.results, test, predictions)
        save_json(a.results / "expert_summary.json",
                  report_experts([a.results / "expert_priority.csv"], a.results / "expert_key.json"))
    else:
        metrics = report_experts(a.ratings, a.key)
        if a.out:
            save_json(a.out, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
