"""Find auditable energy mentions and mark unambiguous non-observations."""
import re
from decimal import Decimal, InvalidOperation

J_PER_EV = Decimal("1.602176634e-19")
UNIT = r"(?P<unit>meV|eV|J)\b"
PLAIN = r"[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)"
EXP = r"(?:[eE][+\-−]?\d+|\s*[×x]\s*10(?:\s*\^?\s*[+\-−]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+))"
NUM = rf"{PLAIN}{EXP}?"
MENTION = re.compile(rf"(?<![\w.])(?P<value>{NUM})"
                     rf"(?:\s*(?:±|\+/-)\s*(?P<uncertainty>{NUM})\s*\)?)?"
                     rf"\s*(?:-\s*)?{UNIT}")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻−", "0123456789+--")
RANGE_LEFT = re.compile(rf"(?<!\w)(?:{NUM})(?:\s*(?:meV|eV|J)\b)?\s*[-–—−]\s*$"
                        rf"|\bbetween\s+(?:{NUM})(?:\s*(?:meV|eV|J)\b)?\s+and\s*$"
                        rf"|\b(?:{NUM})\s+to\s*$", re.I)
RANGE_RIGHT = re.compile(rf"\s*[-–—−]\s*(?:{NUM})\s*(?:meV|eV|J)\b"
                         rf"|\s+and\s+(?:{NUM})\s*(?:meV|eV|J)\b", re.I)
BOUND = re.compile(r"(?:[<>≤≥]|(?:at least|at most|more than|less than|upper limit of|lower limit of|above|below|exceed(?:s|ing)?))\s*~?\s*$", re.I)
NON_GAP = re.compile(r"\b(?:exciton(?:ic)? binding energ(?:y|ies)|activation energ(?:y|ies)|"
                     r"fermi level|fermi energy|stokes shift|blue[- ]?shift|red[- ]?shift|"
                     r"gap (?:shift|correction)|gap (?:increased|decreased|changes?) by|"
                     r"offset|broadening|linewidth|barrier height)\b", re.I)
GAP_CUE = re.compile(r"\b(?:band[\s-]*gap|energy gap|optical gap|electronic gap|"
                    r"quasi[- ]?particle gap|e\s*g)\b", re.I)
METHOD_HINTS = [
    ("GW", re.compile(r"(?<![A-Za-z])(?:G[0O]W[0O]|GW)(?![A-Za-z])")),
    ("DFT", re.compile(r"\b(?:DFT|density functional theory|PBEsol|PBE|HSE06|HSE)\b", re.I)),
    ("Tauc", re.compile(r"\bTauc\b", re.I)),
    ("absorption", re.compile(r"\babsorption(?: spectroscopy| spectra?)?\b", re.I)),
    ("PL", re.compile(r"\b(?:photoluminescence|PL)\b", re.I)),
    ("STS", re.compile(r"\b(?:scanning tunneling spectroscopy|STS)\b", re.I)),
]
OPTICAL = re.compile(r"\boptical\s+(?:band[\s-]*gap|gap)\b", re.I)
ELECTRONIC = re.compile(r"\b(?:electronic|quasi[- ]?particle)\s+(?:band[\s-]*gap|gap)\b", re.I)


def decimal(raw):
    raw = raw.translate(SUP)
    value = re.sub(r"\s*[×x]\s*10\s*\^?\s*", "e", raw)
    return Decimal(value)


def ev_number(raw_value, unit):
    n = decimal(raw_value)
    if unit == "meV":
        return n / 1000
    if unit == "J":
        return n / J_PER_EV
    return n


def exclusion(sentence, match, normalized):
    start, end = match.span()
    before, after = sentence[max(0, start - 110):start], sentence[end:end + 90]
    if not 0 <= normalized <= 15:
        return "outside_0_15_eV"
    if RANGE_LEFT.search(before) or (RANGE_RIGHT.match(after)
                                       and ("between" in before.lower()[-35:] or after.lstrip().startswith(("-", "–", "—", "−")))):
        return "range_endpoint"
    if BOUND.search(before):
        return "bound"
    # The closest explicit quantity in the local clause controls the exclusion.
    local = re.split(r"[;:(),]|\b(?:while|whereas|but)\b", before, flags=re.I)[-1]
    labels = list(NON_GAP.finditer(local))
    gaps = list(GAP_CUE.finditer(local))
    if labels and len(local) - labels[-1].end() <= 30 and (not gaps or gaps[-1].start() < labels[-1].start()):
        return "related_energy"
    return None


def candidates(sentences):
    output = []
    for sid, sentence in enumerate(sentences, 1):
        for match in MENTION.finditer(sentence):
            start, end = match.span()
            # A shared unit after an uncertainty belongs to its central value.
            raw, unit = match["value"], match["unit"]
            try:
                normalized = ev_number(raw, unit)
                blocked = exclusion(sentence, match, normalized)
                if blocked == "outside_0_15_eV":
                    value, ev = None, None
                else:
                    value, ev = float(decimal(raw)), float(normalized)
            except (InvalidOperation, OverflowError, ValueError):
                value, ev, blocked = None, None, "nonfinite_numeric"
            output.append({"id": f"g{len(output)+1}", "sentence": sid, "start": start,
                           "end": end, "mention": match[0], "value_text": raw, "unit": unit,
                           "value": value, "band_gap_eV": ev, "blocked": blocked})
    return output


def annotate(sentences, mentions):
    labeled = []
    for i, s in enumerate(sentences, 1):
        for m in reversed([m for m in mentions if m["sentence"] == i and not m["blocked"]]):
            cut = m["end"]
            s = s[:cut] + " [" + m["id"] + "]" + s[cut:]
        labeled.append(f"{i}. {s}")
    return "\n".join(labeled)


def scientific_hints(sentence):
    optical = bool(OPTICAL.search(sentence))
    electronic = bool(ELECTRONIC.search(sentence))
    kind = "optical" if optical and not electronic else "electronic" if electronic and not optical else None
    return {"gap_kind_cue": kind, "method_cues": [name for name, pattern in METHOD_HINTS
                                                 if pattern.search(sentence)]}
