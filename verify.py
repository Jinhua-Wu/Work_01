"""Conservative post-inference checks for explicit evidence and provenance."""
import re

from energy import GAP_CUE, scientific_hints

PV_FORMULAE = {
    "mapbi3": "MAPbI3", "mapi": "MAPbI3", "ch3nh3pbi3": "MAPbI3",
    "fapbi3": "FAPbI3", "fapi": "FAPbI3", "ch(nh2)2pbi3": "FAPbI3",
    "mapbbr3": "MAPbBr3", "mapb": "MAPbBr3", "ch3nh3pbbr3": "MAPbBr3",
    "cspbi3": "CsPbI3", "cspbbr3": "CsPbBr3", "cs4pbbr6": "Cs4PbBr6",
}
CITE_TRAIL = re.compile(r"^\s*[\])]*(?:\s*[.,)]?\s*\[\s*\d{1,3}(?:\s*[-,]\s*\d+)*\s*\]"
                        r"|\.\d{1,3}(?![.\d])(?=\s|[),;\]]|$))")
CITE_LEAD = re.compile(r"\b[A-Za-z]+,\d{1,3}\s+$")
CITE_AUTHOR = re.compile(r"^\s*by\s+[A-Z][a-zA-Z'’-]+\s+et al\.", re.I)
CITE_LOCAL = re.compile(r"\b(?:previously reported|previous studies?|previous work|"
                        r"the literature|in the literature|literature value|reported by|"
                        r"earlier work|from ref(?:erence)?|authors? reported)\b", re.I)
OTHER_TAIL = re.compile(r"^\s*(?:away from|above|below|relative to)\s+"
                        r"(?:the\s+)?(?:fermi level|valence band|conduction band|vbm|cbm)\b", re.I)
LASER = re.compile(r"\b(?:pump(?:[- ]diode)? laser|laser|cw pump|excitation photon(?: energy)?)\b", re.I)
DESIRED = re.compile(r"\b(?:ideal|optimal|optimum|desirable|targeted)\s+"
                     r"(?:optical\s+|energy\s+|fundamental\s+)?band[\s-]*gap\b", re.I)
NEAR_EXP = re.compile(r"\b(?:measured|measurement(?:s)?|experimentally (?:determined|obtained|measured)|"
                      r"experimental band[\s-]*gap|tauc|optical spectroscopy|tunneling spectroscopy|"
                      r"absorption (?:spectroscopy|spectrum|spectra|curve|data|edge)|"
                      r"(?:pl|sspl|trpl|photoluminescence) spectra?)\b", re.I)
NEAR_CALC = re.compile(r"\b(?:dft|pbesol|pbe|hse(?:06)?|g[0o]w[0o]|gw|"
                       r"computed|computational|cal[- ]?culated|calculation(?:s)?|"
                       r"density functional theory|ab initio|ab-initio|first principles)\b", re.I)
METHOD_TAIL = re.compile(r"^\s*(?:[\])])?[\s,]*(?:from|using|by|via|as (?:obtained|measured|computed) by)\b", re.I)
CONTRAST = re.compile(r"\b(?:whereas|while|however|rather than|in contrast|compared with|in agreement with)\b|[;]", re.I)


def material_patterns(case):
    aliases = PV_FORMULAE.copy()
    target = aliases.get(case["material"].casefold(), case["material"])
    aliases[case["material"].casefold()] = target
    if case.get("pv_alias"):
        aliases[case["pv_alias"].casefold()] = target
    terms = "|".join(re.escape(k) for k in sorted(aliases, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{terms})(?![A-Za-z0-9])", re.I)
    return target, aliases, pattern


def local_piece(sentence, anchor):
    before, after = sentence[:anchor["start"]], sentence[anchor["end"]:]
    local = CONTRAST.split(before)[-1][-150:]
    return local, after[:105]


def foreign_material(case, anchor, mentions):
    sentence = case["sentences"][anchor["sentence"] - 1]
    target, aliases, pattern = material_patterns(case)
    hits = list(pattern.finditer(sentence))
    local, after = local_piece(sentence, anchor)
    left = list(pattern.finditer(local))
    right = list(pattern.finditer(after[:50]))
    left_ids = [aliases[m[0].casefold()] for m in left]
    right_ids = [aliases[m[0].casefold()] for m in right]
    eligible = [m for m in mentions if m["sentence"] == anchor["sentence"] and not m["blocked"]]
    if "respectively" in sentence.casefold() and len(hits) == len(eligible) and len(hits) > 1:
        ids = [aliases[m[0].casefold()] for m in hits]
        if len(set(ids)) == len(ids):
            return ids[eligible.index(anchor)] != target
    # A named foreign subject plus a gap cue in the same clause is strong negative evidence.
    if left_ids and target not in left_ids and GAP_CUE.search(local) and target not in right_ids:
        return True
    if right and right[0].start() < 28 and re.match(r"^\s*(?:\)?\s*)?(?:for|of|in)\s", after, re.I):
        return right_ids[0] != target
    return False


def assess_anchor(case, m, mentions, model_origin):
    s = case["sentences"][m["sentence"]-1]
    local, after = local_piece(s, m)
    before = s[:m["start"]]
    if OTHER_TAIL.match(after):
        return None, "relative_band_edge_or_fermi_level"
    if LASER.search(local[-50:]) and not GAP_CUE.search(local[-28:]):
        return None, "excitation_energy"
    if LASER.match(after.lstrip()[:30]):
        return None, "excitation_energy"
    if DESIRED.search(local[-100:]) or re.search(r"\b(?:desirable|ideally)\b.{0,60}\bband[\s-]*gap\b", local, re.I):
        return None, "target_design_value"
    if re.search(r"\bconduction bandgaps?\b", local, re.I) and re.search(r"\bvalence band\b", before, re.I):
        return None, "relative_band_edge_or_fermi_level"
    if foreign_material(case, m, mentions):
        return None, "different_material"
    # The segment after the preceding tagged value can identify a cited comparator.
    previous = [a["end"] for a in mentions if a["sentence"] == m["sentence"] and a["end"] <= m["start"]]
    narrow = before[max(previous, default=0):][-115:]
    if (CITE_LOCAL.search(narrow) or CITE_AUTHOR.match(after)
            or ((CITE_TRAIL.match(after) or CITE_LEAD.search(narrow))
                and not re.search(r"\b(?:our|we)\b", narrow, re.I))):
        return "ref", "local_attribution_or_citation"
    nearby = local + (" " + after[:65] if METHOD_TAIL.match(after) else "")
    measured = list(NEAR_EXP.finditer(nearby))
    calculated = list(NEAR_CALC.finditer(nearby))
    if measured and (not calculated or "tauc" in nearby.casefold()
                     or measured[-1].start() > calculated[-1].start()):
        return "exp", "local_experimental_cue"
    if calculated:
        return "calc", "local_computational_cue"
    # Cross-sentence provenance is weaker. Keep it only if the preceding sentence corroborates
    # the model's experimental or computational label and this sentence refers back to that evidence.
    if m["sentence"] > 1 and re.search(r"\b(?:these|those|thus|from the|accordingly|therefore)\b", local, re.I):
        prev = case["sentences"][m["sentence"]-2]
        check = NEAR_EXP if model_origin == "exp" else NEAR_CALC if model_origin == "calc" else None
        if check and check.search(prev):
            return model_origin, "adjacent_method_cue"
    return "unk", "origin_unresolved"


def verify_selection(case, initial):
    mentions = initial["mentions"]
    eligible = {m["id"]: m for m in mentions if not m["blocked"]}
    if not eligible:
        # No explicitly admissible scalar is enough for a reproducible abstention.
        return {"records": [], "rows": [], "valid": True, "errors": [],
                "verification": [{"action": "abstain", "reason": "no_eligible_energy_mention"}]}
    if not initial["format"] or "generation_truncated" in initial["errors"]:
        return {"records": [], "rows": [], "valid": False, "errors": initial["errors"],
                "verification": [{"action": "reject_response", "reason": "schema_or_truncation"}]}
    records, audit = [], []
    supported_tag_seen = False
    for tag, label in sorted(initial["selection"].items(), key=lambda p: (len(p[0]), p[0])):
        anchor = eligible.get(tag)
        if anchor is None:
            audit.append({"tag": tag, "action": "reject_tag", "reason": "ineligible_or_unknown_tag"})
            continue
        supported_tag_seen = True
        method, reason = assess_anchor(case, anchor, mentions, label)
        if method is None:
            audit.append({"tag": tag, "action": "reject_tag", "reason": reason})
            continue
        sentence = case["sentences"][anchor["sentence"]-1]
        records.append({"material": case["material"], "band_gap": anchor["value"],
                        "unit": anchor["unit"], "method": method, "band_gap_eV": anchor["band_gap_eV"],
                        "sentence": anchor["sentence"], "evidence": sentence, "source": case.get("source"),
                        "span": [anchor["start"], anchor["end"]], "mention": anchor["mention"], "tag": tag,
                        "model_origin": label, "origin_check": reason, **scientific_hints(sentence)})
        audit.append({"tag": tag, "action": "retain" if method == label else "revise_origin",
                      "model_origin": label, "verified_origin": method, "reason": reason})
    rows = [[r["band_gap"], r["unit"], r["method"], r["sentence"]] for r in records]
    if not supported_tag_seen and initial["selection"]:
        return {"records": [], "rows": [], "valid": False, "errors": ["no_output_tag_is_eligible"],
                "verification": audit}
    return {"records": records, "rows": rows, "valid": True, "errors": [], "verification": audit}
