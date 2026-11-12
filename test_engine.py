"""Unit tests for the public extraction protocol and the experiment utilities."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_data import source_key
from energy import annotate, candidates
from engine import ROOT, KB, MANUAL_SOURCE, compile_prompt, evaluate, finalize, fingerprint, jsonl, validate
from run import expert_case_ids, report_experts, valid_case


class ScientificExtraction(unittest.TestCase):
    def check(self, raw, *sentences):
        return validate(raw, {"material": "GaN", "sentences": sentences}, candidates(sentences))

    def test_mev_and_source_anchor(self):
        s = "Our measured GaN optical gap is 3420 ± 20 meV."
        result = self.check('{"g1":"exp"}', s)
        self.assertTrue(result["valid"])
        r = result["records"][0]
        self.assertEqual(r["band_gap"], 3420)
        self.assertAlmostEqual(r["band_gap_eV"], 3.42)
        self.assertEqual(s[slice(*r["span"])], "3420 ± 20 meV")
        self.assertEqual(r["gap_kind_cue"], "optical")

    def test_joule_unicode_exponent(self):
        result = self.check('{"g1":"exp"}', "Measured GaN gap 2.403264951 × 10⁻¹⁹ J.")
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["records"][0]["band_gap_eV"], 1.5)

    def test_missing_units_and_wavelength_do_not_get_tags(self):
        ss = ["A PBE gap of 1.5.", "Our measurement shows an onset at 365 nm."]
        self.assertEqual(candidates(ss), [])
        self.assertTrue(self.check("{}", *ss)["valid"])

    def test_range_and_related_energy_are_blocked(self):
        ss = ["The gap lies between 1.2 and 1.4 eV.",
              "The exciton binding energy was 30 meV.",
              "The measured GaN optical gap was 3.4 eV."]
        ms = candidates(ss)
        self.assertEqual([x["blocked"] for x in ms], ["range_endpoint", "related_energy", None])
        self.assertNotIn("[g1]", annotate(ss, ms))
        self.assertTrue(self.check('{"g3":"exp"}', *ss)["valid"])
        self.assertIn("blocked_tag:range_endpoint", self.check('{"g1":"unk"}', *ss)["errors"])

    def test_bound_out_of_domain_and_unknown_id(self):
        ss = ["The gap is below 3.5 eV.", "A second gap is 17 eV."]
        self.assertEqual([m["blocked"] for m in candidates(ss)], ["bound", "outside_0_15_eV"])
        self.assertIn("unknown_tag", self.check('{"g9":"exp"}', *ss)["errors"])

    def test_wrapper_and_duplicate_keys(self):
        self.assertFalse(self.check('{"g1":"exp","g1":"calc"}', "The gap is 3.4 eV.")["format"])
        self.assertFalse(self.check('{"g1":"DFT"}', "The gap is 3.4 eV.")["format"])
        f = chr(96)*3 + 'json\n{"g1":"exp"}\n' + chr(96)*3
        result = self.check(f, "The gap is 3.4 eV.")
        self.assertTrue(result["format"])
        self.assertFalse(result["raw_format"])

    def test_criterion_checks_method_after_value_and_anchor(self):
        r = self.check('{"g1":"calc"}', "Our measurement yields 3.4 eV.")
        g = [[3.4, "eV", "exp", 1]]
        self.assertEqual(evaluate(r, g)["anchor_tp"], 1)
        self.assertFalse(evaluate(r, g)["correct"])

    def test_provenance_verification_preserves_raw_and_corrects_citation(self):
        s = "Our calculation gives 2.23 eV, in agreement with the experimental value of 2.25 eV.37"
        c = {"material":"CsPbBr3", "sentences":[s]}
        checked = validate('{"g1":"calc","g2":"exp"}', c, candidates(c["sentences"]))
        verified = finalize(c, checked)
        self.assertEqual(verified["before_verifier"]["rows"][1][2], "exp")
        self.assertEqual(verified["rows"][1][2], "ref")
        self.assertEqual(verified["records"][1]["origin_check"], "local_attribution_or_citation")

    def test_decimal_after_comma_is_not_a_citation(self):
        s = "The calculated band gaps are 1.5 eV, 1.67 eV, and 2.14 eV."
        c = {"material":"FAPbI3", "sentences":[s]}
        checked = validate('{"g1":"calc"}', c, candidates(c["sentences"]))
        self.assertEqual(finalize(c, checked)["rows"][0][2], "calc")

    def test_named_foreign_target_is_removed_with_reason(self):
        c = {"material":"CsPbI3", "sentences":["FAPI has a bandgap of 1.48 eV."]}
        r = finalize(c, validate('{"g1":"exp"}', c, candidates(c["sentences"])))
        self.assertEqual(r["rows"], [])
        self.assertEqual(r["verification"][0]["reason"], "different_material")

    def test_term_card_and_no_knowledge(self):
        c = valid_case({"material": "GaN", "sentences": ["Our HSE calculation gives a gap of 3.1 eV."]})
        _, cards = compile_prompt(c, "auto", "Use explicit evidence.")
        _, missing = compile_prompt(c, "no_knowledge", "Use explicit evidence.")
        self.assertIn("K03", cards)
        self.assertEqual(missing, [])
        for card in KB["cards"]:
            self.assertTrue(set(card["terms"]) <= KB["glossary"].keys())
            self.assertTrue(set(card["example"]["answer"].values()) <= {"exp","calc","ref","unk"})
        self.assertIn("s43246", MANUAL_SOURCE)
        self.assertEqual(fingerprint(c), fingerprint(dict(reversed(list(c.items())))))

    def test_source_isolation_and_priority_review(self):
        data = jsonl(ROOT / "data/pv600_subset.jsonl")
        dev = [c for c in data if c["split"] == "dev"]
        test = [c for c in data if c["split"] == "test"]
        old = json.loads((ROOT/"data/prior_sources.json").read_text(encoding="utf-8"))
        keys = {source_key(c["source"]) for c in test}
        self.assertEqual(len(keys), 40)
        self.assertFalse(keys & {source_key(c["source"]) for c in dev})
        self.assertFalse(keys & set(old["excluded_source_keys"]))
        selected = set(expert_case_ids(test))
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(bool(c["gold"]) for c in test if c["id"] in selected), 5)
        fresh = jsonl(ROOT / "data/pv600_confirm.jsonl")
        fresh_sources = {source_key(c["source"]) for c in fresh}
        self.assertEqual(len(fresh_sources), 30)
        self.assertEqual(sum(bool(c["gold"]) for c in fresh), 12)
        self.assertFalse(fresh_sources & {source_key(c["source"]) for c in data})
        self.assertFalse(fresh_sources & set(old["excluded_source_keys"]))

    def test_expert_import_requires_declared_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            key = {"items":{"R1":["T01","manual"]}, "priority_review_ids":["R1"],
                   "priority_case_ids":["T01"]}
            (p/"key.json").write_text(json.dumps(key), encoding="utf-8")
            with (p/"review.csv").open("w", encoding="utf-8", newline="") as stream:
                w=csv.writer(stream)
                w.writerow(["review_id","expert_score_1_5","reviewer_code","reviewer_role"])
                w.writerow(["R1","4","test_id","test_role"])
            result=report_experts([p/"review.csv"], p/"key.json")
            self.assertEqual(result["groups"]["manual"]["mean_1_5"], 4)
            self.assertIsNone(result["groups"]["auto"]["mean_1_5"])
            self.assertFalse(result["priority_complete"])


if __name__ == "__main__":
    unittest.main()
