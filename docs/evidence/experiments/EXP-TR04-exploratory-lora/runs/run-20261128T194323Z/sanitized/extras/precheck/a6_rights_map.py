"""Read-only rights-mapping check for the EXP-TR04 kept rows (writes nothing; prints JSON)."""
import json, re, sys
from collections import Counter
from pathlib import Path
from axbench_tr import tr04, tr04_rights, tr04_heldout

export = Path(sys.argv[1])
train, valid = tr04.load()
by_id = {r["id"]: r for r in train + valid}
ids = tr04.kept_ids()
doc = tr04_rights.load_izmir()
status = tr04_rights.izmir_status(doc)
allowed = {"Izmir-Metropolitan-Municipality-License", "cc-by"}
out = {"kept": {k: len(v) for k, v in ids.items()}, "problems": []}
fam_pkg = {}
for split, lst in ids.items():
    for i in lst:
        r = by_id.get(i)
        if r is None:
            out["problems"].append(f"missing_record:{i}"); continue
        prov = r["provenance"]
        if prov["origin"] != "synthetic_template": out["problems"].append(f"origin:{i}:{prov['origin']}")
        if prov.get("schema_source_id") != "SRC-O06": out["problems"].append(f"source:{i}:{prov.get('schema_source_id')}")
        st = status.get(r["family"])
        if st is None or not st["usable"]:
            out["problems"].append(f"family_not_usable:{i}:{r['family']}:{st}")
        else:
            pk = st["package"]; fam_pkg[r["family"]] = pk
            rec = doc["packages"][pk]
            if rec["license_id"] not in allowed or rec["flags"]:
                out["problems"].append(f"package_bad:{i}:{pk}")
        d = tr04_rights.__dict__  # noqa
        why = tr04.rights_decision(r, status)
        if why: out["problems"].append(f"rights_decision:{i}:{why}")
        if "translation_unverified" in r.get("tags", []): out["problems"].append(f"tag:{i}")
out["families"] = len(fam_pkg)
out["packages_used"] = len(set(fam_pkg.values()))
out["package_licences"] = dict(Counter(doc["packages"][p]["license_id"] for p in set(fam_pkg.values())))
out["packages_used_with_flags"] = sum(bool(doc["packages"][p]["flags"]) for p in set(fam_pkg.values()))
# held-out ids / export contents
held = {r["id"] for r in tr04_heldout.load_rows()}
ex_ids = json.loads((export / "ROW_IDS.json").read_text())
exported = {i for v in ex_ids.values() for i in v}
out["export_rows"] = {k: len(v) for k, v in ex_ids.items()}
out["heldout_ids"] = len(held)
out["heldout_ids_in_export"] = sorted(held & exported)
if held & exported: out["problems"].append("heldout_in_export")
out["export_ids_equal_keep_lists"] = {
    "train": ex_ids["tr04/train.jsonl"] == ids["train"], "validation": ex_ids["tr04/valid.jsonl"] == ids["validation"]}
if not all(out["export_ids_equal_keep_lists"].values()): out["problems"].append("export_ids_differ_from_keep")
# heldout text overlap (exact user request strings) with exported rows
def rows(p): return [json.loads(l) for l in Path(p).read_text("utf-8").splitlines()]
email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
abspath = re.compile(r"(?<![\w.])/(?:Users|home|private|tmp|var|opt|etc|Volumes|root)/|[A-Za-z]:\\\\|file://")
hits = {"email": 0, "abs_path": 0}
for f in sorted(export.rglob("*")):
    if f.is_file():
        t = f.read_text("utf-8")
        hits["email"] += len(email.findall(t)); hits["abs_path"] += len(abspath.findall(t))
out["scan_hits"] = hits
if any(hits.values()): out["problems"].append("scan_hits")
out["problem_count"] = len(out["problems"]); out["problems"] = out["problems"][:20]
print(json.dumps(out, indent=1, sort_keys=True))
