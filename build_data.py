"""Rebuild a source-isolated benchmark from the versioned PV600 CSV."""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from energy import MENTION, decimal

ROOT = Path(__file__).resolve().parent
MATERIALS = {"MAPI": "MAPbI3", "FAPI": "FAPbI3", "MAPB": "MAPbBr3",
             "CsPbI3": "CsPbI3", "CsPbBr3": "CsPbBr3"}
LABELS = {"Experimental_bandgap": "exp", "Computational_bandgap": "calc",
          "Bandgap_from_literature": "ref", "Bandgap_from_unknown_source": "unk"}
BOUNDARY = re.compile(r"[.!?]\s+(?=[A-Z0-9(])|\n+")
ABBREVIATION = re.compile(r"\b(?:Fig|Figs|Ref|Refs|Eq|Eqs|e\.g|i\.e|et al|vs)\.$", re.I)
SEED = 20261111
CONFIRM_SEED = 20261116


def source_key(source):
    s = source.strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    m = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)([^?#]+)", s)
    if m:
        s = "arxiv:" + re.sub(r"v\d+$", "", m[1].removesuffix(".pdf"))
    return s


def spans(text):
    starts = [0]
    for match in BOUNDARY.finditer(text):
        if not ABBREVIATION.search(text[max(0, match.start()-10):match.start()+1]):
            starts.append(match.end())
    cuts = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        chunk = text[a:b]
        if chunk.strip():
            cuts.append((a + len(chunk) - len(chunk.lstrip()), a + len(chunk.rstrip())))
    return cuts


def transform(row, max_chars=2550):
    text = row["Text"]
    if len(text) > max_chars or not text.strip():
        return None
    cuts = spans(text)
    gold, origins = [], []
    matches = {m.start("value"): m for m in MENTION.finditer(text)}
    for i in range(1, 7):
        label = row[f"Annotation_{i}"]
        if not label:
            continue
        start, end = (int(row[f"{field}_index_{i}"]) for field in ("Start", "End"))
        if text[start:end] != label:
            return None
        m = matches.get(start)
        # PV600 has table, interval and shared-unit annotations. The task takes explicit prose scalars only.
        if m is None or end not in (m.end("value"), m.end("uncertainty")):
            return None
        sid = next((j for j, (a, b) in enumerate(cuts, 1) if a <= start < m.end() <= b), None)
        if not sid:
            return None
        v = float(decimal(m["value"]))
        method = LABELS[row[f"Bandgap_type_{i}"]]
        gold.append([v, m["unit"], method, sid])
        origins.append({"pv_label": label, "raw_offset": [start, end],
                        "span": [start - cuts[sid-1][0], m.end() - cuts[sid-1][0]]})
    if len({tuple(g) for g in gold}) != len(gold):
        return None
    return {"pv_id": row["Snippet_name"], "source": row["Article_identifier"],
            "source_key": source_key(row["Article_identifier"]),
            "material": MATERIALS[row["Material"]], "pv_alias": row["Material"],
            "sentences": [text[a:b] for a, b in cuts], "gold": gold,
            "gold_spans": origins, "origin": "PV600"}


def stratum(item):
    gold = item["gold"]
    if len(gold) > 1:
        return "multi"
    if gold:
        return gold[0][2]
    return "hard_negative" if any(MENTION.search(s) for s in item["sentences"]) else "negative"


def choose(pool, prior):
    by_id = {x["pv_id"]: x for x in pool}
    dev = [by_id[x] for x in prior["dev_snippets"] if x in by_id]
    if len(dev) != len(prior["dev_snippets"]):
        raise ValueError("Some fixed development rows could not be mapped to the new schema")
    unexposed = [x for x in pool if x["source_key"] not in set(prior["excluded_source_keys"])]
    shuffled = sorted(unexposed, key=lambda x: x["pv_id"])
    rng = random.Random(SEED)
    rng.shuffle(shuffled)
    chosen = []
    seen = set()

    def take(predict):
        row = next((x for x in shuffled if x["source_key"] not in seen and predict(x)), None)
        if row is not None:
            seen.add(row["source_key"])
            chosen.append(row)
        return row

    # Aim for a diverse balanced sample, with shortfall filled from unused sources.
    for material in MATERIALS.values():
        for group in ("calc", "exp", "ref", "multi", "unk"):
            if sum(x["material"] == material for x in chosen) >= 4:
                break
            take(lambda x: x["material"] == material and stratum(x) == group)
        while sum(x["material"] == material for x in chosen) < 4:
            if not take(lambda x: x["material"] == material and x["gold"]):
                break
    while len(chosen) < 20:
        if not take(lambda x: bool(x["gold"])):
            raise ValueError("Insufficient independent positive sources")
    for material in MATERIALS.values():
        for group in ("hard_negative", "hard_negative", "hard_negative", "negative"):
            if not take(lambda x: x["material"] == material and stratum(x) == group):
                if not take(lambda x: x["material"] == material and not x["gold"]):
                    raise ValueError(f"Insufficient independent negative sources for {material}")
    rng.shuffle(chosen)
    return [
        {**r, "id": f"{split[0].upper()}{i:02d}", "split": split, "stratum": stratum(r)}
        for split, rows in (("dev", dev), ("test", chosen))
        for i, r in enumerate(rows, 1)
    ]


def choose_confirmation(pool, prior, used_cases):
    blocked = set(prior["excluded_source_keys"]) | {x["source_key"] for x in used_cases}
    left = sorted((r for r in pool if r["source_key"] not in blocked), key=lambda r: r["pv_id"])
    rng = random.Random(CONFIRM_SEED)
    rng.shuffle(left)
    picked = []

    def add(r):
        picked.append(r)
        blocked.add(r["source_key"])

    for r in left:
        if r["gold"] and r["source_key"] not in blocked and len(picked) < 12:
            add(r)
    if len(picked) != 12:
        raise ValueError("Confirmation requires twelve independently sourced positives")
    for material in MATERIALS.values():
        for want in ("hard_negative", "hard_negative", "negative"):
            options = [r for r in left if not r["gold"] and r["material"] == material
                       and r["source_key"] not in blocked]
            add(next((r for r in options if stratum(r) == want), options[0]))
    for r in left:
        if len(picked) == 30:
            break
        if not r["gold"] and r["source_key"] not in blocked:
            add(r)
    if len(picked) != 30:
        raise ValueError("Confirmation requires thirty independently sourced rows")
    rng.shuffle(picked)
    return [{**r, "id": f"C{i:02d}", "split": "confirm", "stratum": stratum(r)}
            for i, r in enumerate(picked, 1)]


def write_dataset(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--prior", type=Path, default=ROOT / "data" / "prior_sources.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pv600_subset.jsonl")
    p.add_argument("--confirm-out", type=Path)
    args = p.parse_args()
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        pool = [y for x in csv.DictReader(f) if (y := transform(x))]
    rows = choose(pool, prior)
    write_dataset(args.out, rows)
    if args.confirm_out:
        write_dataset(args.confirm_out, choose_confirmation(pool, prior, rows))
    print(f"{len(rows)} rows; source SHA256 {hashlib.sha256(args.csv.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
