"""EXP-TR03 gate checker: strict, fail-closed validation of human ledgers (no model, no training).

Status: **prepared, not exercised on any human ledger; nothing here approves anything.** This
module reads files that people fill in under ``data/axbench-tr/tr03-prep/ledgers/`` (that directory
does not exist in the repository, on purpose) and reports, for each gate G1 to G7, one of:

* ``absent``: the file is not there;
* ``invalid``: the file contradicts the frozen worklist, the roster or a rule (the reasons are
  listed); one invalid cell fails the whole ledger, nothing is repaired or inferred;
* ``incomplete``: nothing is invalid but some item is not yet decided, awaits its second review,
  or has an unadjudicated disagreement;
* ``criteria_not_met``: complete and valid, but a numerical criterion fails;
* ``consistent_unverified``: complete, valid, criteria met. This is the best a program can say.

**It is never "passed".** The tool cannot authenticate a person, verify a rights opinion, judge
an evaluation set or confirm a clearance: a name in a file is a string. ``research_training_open``
is therefore always ``false`` here, whatever the files contain; opening execution is a separate,
immutable clearance by the technical lead and the owner on GitHub, which this code neither reads
nor replaces. File presence never changes any state by itself: every file's content is parsed and
bound by SHA-256 to the exact corpus, protocol and tooling.

Rules (all proposed, to be approved by the owner and the technical lead; see PROTOCOL.md):

* The roster is a record of the owner's designation with an attestation text; roles are
  ``reviewer``, ``adjudicator``, ``author``, ``checker``, ``owner``.
* A review ledger must have exactly the frozen header and exactly the frozen items (no missing,
  duplicate or foreign key); the immutable identity cells must equal the frozen worklist.
* Each item has a first review. A second review by a *different* person with the reviewer role
  is required for every item that is not a clean ``keep`` and for the pre-planned random tenth.
  If first and second judgements differ in any field the item is disputed and needs an
  adjudication by a third person with the adjudicator role. The effective judgement is the first
  review if undisputed, otherwise the adjudication. A dispute never resolves itself.
* Every judge must be human, on the roster with the role, competent in the item's language
  (native Turkish for Turkish, fluent English for English), and dated within the designation.
* ``keep`` is only valid for a clean item. Only effective ``keep`` items may train; ``fix`` is
  excluded because no verified correction mechanism exists.
* Denominators are the frozen counts; a blank is a failure, never a smaller n.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from axbench.stats import wilson

from . import tr03

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "docs/evidence/experiments/EXP-TR03-research-protocol/PROTOCOL.md"
TRAINING_CONFIG_PATH = REPO / "training/tr03/config.json"
TOOLING_PATHS = (
    "src/axbench_tr/tr03.py",
    "src/axbench_tr/tr03_gates.py",
    "training/tr03/tr03.py",
)
LEDGER_DIR_NAME = "ledgers"

YES_NO = ("yes", "no")
NATURAL = ("yes", "fixable", "no")
DECISIONS = ("keep", "fix", "drop")
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,31}$")
RESERVED_ID_TOKENS = frozenset(
    {"ai", "claude", "gpt", "llm", "bot", "model", "assistant", "tool", "script", "auto"}
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
GITHUB_COMMENT_RE = re.compile(
    r"^https://github\.com/sergthedev/axiom-local/(pull|issues)/[0-9]+#issuecomment-[0-9]+$"
)
ROSTER_SCHEMA = "axbench-tr.tr03-roster/2"
MIN_KEPT_TRAIN_TEMPLATES_PER_CLASS = 2
MIN_KEPT_VALID_TEMPLATES_PER_CLASS = 1
SOURCE_MIN_ROWS = 10

# Held-out evaluation structure (proposed; a different value is a prospective amendment).
HELDOUT_MIN_TR = 300
HELDOUT_MIN_EN = 100
HELDOUT_MIN_AUTHORS = 4
HELDOUT_MAX_AUTHOR_SHARE = 0.35
HELDOUT_MIN_TABLE_FAMILIES = 20
HELDOUT_MAX_FAMILY_SHARE = 0.10
HELDOUT_ACTIONS = ("PLAN", "CLARIFY", "REJECT")
HELDOUT_TABLE_SOURCES = ("independent_fictional", "licensed_unused_layout", "licensed_new")
HELDOUT_ATTESTATIONS = (
    "authors_blind_to_training_templates",
    "no_ai_authored",
    "no_copy_from_dataset_or_model_output",
    "gold_checked_by_second_person",
    "sealed_test_not_read",
)


class LedgerError(ValueError):
    """Raised only by helpers whose callers want an exception; reports use problem lists."""


# ----------------------------------------------------------------------------------- helpers


def _normalized_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tooling_sha256(repo: Path = REPO) -> str:
    """One hash over the three tooling files (line endings normalized to LF)."""
    h = hashlib.sha256()
    for rel in TOOLING_PATHS:
        data = (repo / rel).read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
    return h.hexdigest()


def current_bindings(prep_dir: Path = tr03.PREP_DIR) -> dict[str, str]:
    """The hashes a preregistration, held-out freeze or clearance must quote to be current."""
    out = {
        "protocol_sha256": _normalized_sha(PROTOCOL_PATH),
        "config_sha256": _raw_sha(TRAINING_CONFIG_PATH),
        "tooling_sha256": tooling_sha256(),
    }
    for key, name in (
        ("plan_sha256", "TRAINING_PLAN.json"),
        ("prep_manifest_sha256", "MANIFEST.json"),
    ):
        path = prep_dir / name
        out[key] = _raw_sha(path) if path.exists() else ""
    return out


def parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not DATE_RE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _is_id(value: Any) -> bool:
    if not isinstance(value, str) or not ID_RE.match(value):
        return False
    tokens = re.split(r"[-_]", value.lower())
    return value.lower() not in RESERVED_ID_TOKENS and not set(tokens) & RESERVED_ID_TOKENS


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise LedgerError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _no_constants(name: str) -> Any:
    raise LedgerError(f"non-finite JSON constant {name}")


def load_json_strict(data: bytes) -> Any:
    """JSON with duplicate keys and NaN or Infinity refused; BOM tolerated."""
    return json.loads(
        data.decode("utf-8-sig"),
        object_pairs_hook=_no_duplicates,
        parse_constant=_no_constants,
    )


def read_csv_strict(data: bytes) -> tuple[list[str], list[list[str]], list[str]]:
    """Header, rows and structural problems. A row with a wrong cell count is a problem."""
    problems: list[str] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], [], ["not_utf8"]
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        return [], [], ["empty_file"]
    header, body = rows[0], rows[1:]
    for i, r in enumerate(body, 2):
        if len(r) != len(header):
            problems.append(f"line_{i}_has_{len(r)}_cells_expected_{len(header)}")
    return header, [r for r in body if len(r) == len(header)], problems


def _wilson_lower(good: int, n: int) -> float | None:
    if n <= 0:
        return None
    lo, _ = wilson(good, n) or (0.0, 0.0)
    return round(lo, 4)


# ------------------------------------------------------------------------------------ roster


def roster_problems(roster: Any, purpose: str = "training") -> list[str]:
    """Why a roster cannot support ``purpose`` ('training' or 'heldout'). Empty means consistent
    (still unauthenticated: the tool cannot tell a real person from a string)."""
    if roster is None:
        return ["roster_missing"]
    if not isinstance(roster, dict):
        return ["roster_not_an_object"]
    problems: list[str] = []
    if roster.get("schema") != ROSTER_SCHEMA:
        problems.append("roster_schema_wrong")
    by = roster.get("designated_by")
    if not isinstance(by, str) or not by.strip():
        problems.append("designated_by_missing")
    if parse_date(roster.get("designated_on")) is None:
        problems.append("designated_on_invalid")
    att = roster.get("attestation")
    if not isinstance(att, dict):
        problems.append("attestation_missing")
    else:
        if att.get("statement") != tr03.ROSTER_STATEMENT:
            problems.append("attestation_statement_altered_or_missing")
        if att.get("statement_accepted") is not True:
            problems.append("attestation_not_accepted")
        if att.get("attested_by") != by or not by:
            problems.append("attestation_not_by_designator")
        if parse_date(att.get("attested_on")) is None:
            problems.append("attestation_date_invalid")
    people = roster.get("people")
    if not isinstance(people, list) or not people:
        return [*problems, "no_people"]
    seen: set[str] = set()
    valid_people: list[dict[str, Any]] = []
    for p in people:
        if not isinstance(p, dict):
            problems.append("person_not_an_object")
            continue
        pid = p.get("id")
        tag = pid if isinstance(pid, str) else "?"
        if set(p) != set(tr03.ROSTER_PERSON_FIELDS):
            problems.append(f"person_fields_wrong:{tag}")
            continue
        ok = True
        if not _is_id(pid):
            problems.append(f"person_id_invalid_or_reserved:{tag}")
            ok = False
        elif pid.lower() in seen:
            problems.append(f"duplicate_person_id:{pid}")
            ok = False
        else:
            seen.add(pid.lower())
        if p["type"] != "human":
            problems.append(f"person_not_human:{tag}")
            ok = False
        roles = p["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or len(set(roles)) != len(roles)
            or any(r not in tr03.ROSTER_ROLES for r in roles)
        ):
            problems.append(f"roles_invalid:{tag}")
            ok = False
        for flag in ("native_turkish", "fluent_english", "independent_of_corpus_authoring"):
            if not isinstance(p[flag], bool):
                problems.append(f"{flag}_not_boolean:{tag}")
                ok = False
        if ok:
            if set(p["roles"]) - {"owner"} and p["independent_of_corpus_authoring"] is not True:
                problems.append(f"not_independent_of_corpus_authoring:{tag}")
            if "author" in p["roles"] and set(p["roles"]) & {"reviewer", "adjudicator"}:
                problems.append(f"author_also_reviews_training_data:{tag}")
            valid_people.append(p)

    def with_role(*roles: str, **flags: bool) -> list[dict[str, Any]]:
        return [
            p
            for p in valid_people
            if set(p["roles"]) & set(roles) and all(p[k] is v for k, v in flags.items())
        ]

    if purpose == "training":
        if not with_role("owner"):
            problems.append("no_owner")
        for lang, flag in (("tr", "native_turkish"), ("en", "fluent_english")):
            reviewers = with_role("reviewer", **{flag: True})
            adjudicators = with_role("adjudicator", **{flag: True})
            judges = {p["id"] for p in (*reviewers, *adjudicators)}
            if len(reviewers) < 2:
                problems.append(f"fewer_than_two_{lang}_competent_reviewers")
            if not adjudicators:
                problems.append(f"no_{lang}_competent_adjudicator")
            if len(judges) < 3:
                problems.append(f"fewer_than_three_{lang}_competent_judges")
    else:
        authors = with_role("author", native_turkish=True)
        if len(authors) < HELDOUT_MIN_AUTHORS:
            problems.append("too_few_native_turkish_authors")
        if len(with_role("checker")) < 2:
            problems.append("fewer_than_two_checkers")
    return problems


@dataclass
class RosterIndex:
    people: dict[str, dict[str, Any]]
    designated_on: date

    @classmethod
    def build(cls, roster: Any) -> RosterIndex | None:
        if roster_problems(roster, "training"):
            return None
        return cls(
            {p["id"]: p for p in roster["people"]},
            parse_date(roster["designated_on"]),  # type: ignore[arg-type]
        )

    def judge_problems(self, pid: str, role: str, language: str | None) -> list[str]:
        p = self.people.get(pid)
        if p is None:
            return [f"not_on_roster:{pid}"]
        out = []
        if role not in p["roles"]:
            out.append(f"lacks_role_{role}:{pid}")
        if language in ("tr", "both") and p["native_turkish"] is not True:
            out.append(f"not_native_turkish:{pid}")
        if language in ("en", "both") and p["fluent_english"] is not True:
            out.append(f"not_fluent_english:{pid}")
        return out


# ------------------------------------------------------------------------- review ledgers


@dataclass(frozen=True)
class LedgerSpec:
    kind: str
    key: str
    identity: tuple[str, ...]
    judgments: tuple[str, ...]
    columns: tuple[str, ...]
    enums: Mapping[str, tuple[str, ...]]

    def stages(self) -> list[tuple[str, tuple[str, ...]]]:
        first, second, adjudication = tr03.review_blocks(self.judgments)
        return [("first", first), ("second", second), ("adjudication", adjudication)]


TEMPLATE_SPEC = LedgerSpec(
    "template",
    "lineage_id",
    tr03.TEMPLATE_IDENTITY,
    tr03.TEMPLATE_JUDGMENTS,
    tr03.TEMPLATE_COLUMNS,
    {
        "tr_natural": NATURAL,
        "en_natural": NATURAL,
        "gold_rule_correct": YES_NO,
        "serious_error": YES_NO,
        "decision": DECISIONS,
    },
)
ROW_SPEC = LedgerSpec(
    "row",
    "sample_id",
    tr03.ROW_IDENTITY,
    tr03.ROW_JUDGMENTS,
    tr03.ROW_COLUMNS,
    {
        "request_natural": NATURAL,
        "gold_correct": YES_NO,
        "serious_error": YES_NO,
        "decision": DECISIONS,
    },
)


def clean_keep_problems(spec: LedgerSpec, j: Mapping[str, str]) -> list[str]:
    """A judgement must be internally consistent: keep needs a clean item, a serious error is
    dropped, an irreparable request is dropped."""
    out = []
    if spec.kind == "row":
        natural = [j["request_natural"]]
        correct = j["gold_correct"]
    else:
        natural = [j["tr_natural"], j["en_natural"]]
        correct = j["gold_rule_correct"]
    if j["decision"] == "keep" and not (
        correct == "yes" and j["serious_error"] == "no" and all(n == "yes" for n in natural)
    ):
        out.append("keep_on_an_item_that_is_not_clean")
    if j["serious_error"] == "yes" and j["decision"] != "drop":
        out.append("serious_error_not_dropped")
    if "no" in natural and j["decision"] != "drop":
        out.append("irreparable_request_not_dropped")
    return out


def is_clean_keep(spec: LedgerSpec, j: Mapping[str, str]) -> bool:
    return j["decision"] == "keep" and not clean_keep_problems(spec, j)


@dataclass
class Item:
    key: str
    state: str = "not_reviewed"  # not_reviewed, awaiting_second, disputed, resolved, invalid
    effective: dict[str, str] | None = None
    first: dict[str, str] | None = None
    reasons: list[str] = field(default_factory=list)
    second_required: bool = False
    disputed: bool = False


@dataclass
class LedgerResult:
    spec: LedgerSpec
    sha256: str
    expected: int
    problems: list[str]
    items: dict[str, Item]

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def complete(self) -> bool:
        return self.valid and all(i.state == "resolved" for i in self.items.values())

    def counts(self) -> dict[str, int]:
        c = Counter(i.state for i in self.items.values())
        return {
            k: c.get(k, 0)
            for k in ("not_reviewed", "awaiting_second", "disputed", "resolved", "invalid")
        }


def _stage_values(
    row: Mapping[str, str], cols: tuple[str, ...]
) -> tuple[str, dict[str, str] | None]:
    """('blank'|'complete'|'partial', values). Notes may stay empty in a complete stage."""
    required = cols[:-1]
    cells = {c: row[c] for c in cols}
    filled = [c for c in required if cells[c] != ""]
    if not filled and cells[cols[-1]] == "":
        return "blank", None
    if len(filled) == len(required):
        return "complete", cells
    return "partial", None


def evaluate_ledger(
    spec: LedgerSpec,
    data: bytes,
    frozen: Sequence[Mapping[str, str]],
    roster: RosterIndex | None,
    today: date | None = None,
) -> LedgerResult:
    """Validate a review ledger against the frozen worklist. Never repairs or infers anything."""
    today = today or date.today()
    sha = hashlib.sha256(data).hexdigest()
    header, raw_rows, problems = read_csv_strict(data)
    result = LedgerResult(spec, sha, len(frozen), problems, {})
    if roster is None:
        result.problems.append("roster_missing_or_invalid")
        return result
    if tuple(header) != spec.columns:
        result.problems.append("header_differs_from_the_frozen_worklist")
        return result
    frozen_by_key = {r[spec.key]: r for r in frozen}
    seen: set[str] = set()
    for cells in raw_rows:
        row = dict(zip(header, cells, strict=True))
        key = row[spec.key]
        if key in seen:
            result.problems.append(f"duplicate_key:{key}")
            continue
        seen.add(key)
        ref = frozen_by_key.get(key)
        if ref is None:
            result.problems.append(f"foreign_key:{key}")
            continue
        item = Item(key)
        result.items[key] = item
        diff = [c for c in spec.identity if row[c] != ref[c]]
        if diff:
            item.state = "invalid"
            item.reasons.append("identity_differs_from_frozen_worklist:" + ",".join(diff))
            result.problems.append(f"{key}:identity_differs:{','.join(diff)}")
            continue
        _evaluate_item(spec, row, ref, roster, today, item)
        for reason in item.reasons:
            if item.state == "invalid":
                result.problems.append(f"{key}:{reason}")
    for key in sorted(set(frozen_by_key) - seen):
        result.problems.append(f"missing_key:{key}")
    return result


def _judge_ok(
    spec: LedgerSpec,
    cells: Mapping[str, str],
    cols: tuple[str, ...],
    role: str,
    language: str | None,
    roster: RosterIndex,
    today: date,
    item_reasons: list[str],
    label: str,
) -> dict[str, str] | None:
    """Validate one complete stage; return its judgements or None (reasons appended)."""
    start = len(item_reasons)
    pid, ptype, on, *rest = cols[:-1]
    if cells[ptype] != "human":
        item_reasons.append(f"{label}_type_not_human")
    item_reasons += [f"{label}:{p}" for p in roster.judge_problems(cells[pid], role, language)]
    d = parse_date(cells[on])
    if d is None:
        item_reasons.append(f"{label}_date_invalid")
    elif d < roster.designated_on or d > today:
        item_reasons.append(f"{label}_date_outside_designation_to_today")
    judgments: dict[str, str] = {}
    for base, col in zip(spec.judgments, rest, strict=True):
        value = cells[col]
        if value not in spec.enums[base]:
            item_reasons.append(f"{label}_{base}_not_in_enum")
        judgments[base] = value
    if len(item_reasons) == start:
        item_reasons += [f"{label}:{p}" for p in clean_keep_problems(spec, judgments)]
    return judgments if len(item_reasons) == start else None


def _evaluate_item(
    spec: LedgerSpec,
    row: Mapping[str, str],
    ref: Mapping[str, str],
    roster: RosterIndex,
    today: date,
    item: Item,
) -> None:
    language = ref["language"] if spec.kind == "row" else "both"
    (_, first_cols), (_, second_cols), (_, adj_cols) = spec.stages()
    s1, c1 = _stage_values(row, first_cols)
    s2, c2 = _stage_values(row, second_cols)
    s3, c3 = _stage_values(row, adj_cols)
    for name, status in (("first", s1), ("second", s2), ("adjudication", s3)):
        if status == "partial":
            item.reasons.append(f"{name}_stage_partially_filled")
    if "partial" in (s1, s2, s3):
        item.state = "invalid"
        return
    if s1 == "blank":
        if s2 != "blank" or s3 != "blank":
            item.state = "invalid"
            item.reasons.append("later_review_without_first")
        return  # not_reviewed
    assert c1 is not None
    first = _judge_ok(
        spec, c1, first_cols, "reviewer", language, roster, today, item.reasons, "first"
    )
    if first is None:
        item.state = "invalid"
        return
    item.first = first
    planned = ref["second_review_planned"] == "yes"
    item.second_required = planned or not is_clean_keep(spec, first)
    if s2 == "blank":
        if s3 != "blank":
            item.state = "invalid"
            item.reasons.append("adjudication_without_second_review")
        elif item.second_required:
            item.state = "awaiting_second"
        else:
            item.state = "resolved"
            item.effective = first
        return
    assert c2 is not None
    second = _judge_ok(
        spec, c2, second_cols, "reviewer", language, roster, today, item.reasons, "second"
    )
    if second is None:
        item.state = "invalid"
        return
    if c2[second_cols[0]] == c1[first_cols[0]]:
        item.state = "invalid"
        item.reasons.append("second_reviewer_is_the_first_reviewer")
        return
    item.disputed = first != second
    if not item.disputed:
        if s3 != "blank":
            item.state = "invalid"
            item.reasons.append("adjudication_without_dispute")
            return
        item.state = "resolved"
        item.effective = first
        return
    if s3 == "blank":
        item.state = "disputed"
        return
    assert c3 is not None
    adjudicated = _judge_ok(
        spec, c3, adj_cols, "adjudicator", language, roster, today, item.reasons, "adjudication"
    )
    if adjudicated is None:
        item.state = "invalid"
        return
    if c3[adj_cols[0]] in (c1[first_cols[0]], c2[second_cols[0]]):
        item.state = "invalid"
        item.reasons.append("adjudicator_is_a_reviewer_of_the_item")
        return
    latest = max(parse_date(c1[first_cols[2]]), parse_date(c2[second_cols[2]]))  # type: ignore[type-var]
    if parse_date(c3[adj_cols[2]]) < latest:  # type: ignore[operator]
        item.state = "invalid"
        item.reasons.append("adjudicated_before_the_reviews")
        return
    item.state = "resolved"
    item.effective = adjudicated


# ------------------------------------------------------------------------------ gate reports


def _rate_block(good: int, n: int) -> dict[str, Any]:
    return {
        "ok": good,
        "of_frozen_n": n,
        "rate": round(good / n, 4) if n else None,
        "wilson_lower_95_iid_not_cluster_adjusted": _wilson_lower(good, n),
    }


def _problems_view(problems: Sequence[str], limit: int = 40) -> dict[str, Any]:
    return {"count": len(problems), "first": list(problems[:limit])}


def row_gate_report(result: LedgerResult, frozen: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Row gates on the frozen sample. Denominators are the frozen counts, never the answered."""
    by_key = {r["sample_id"]: r for r in frozen}
    strata: dict[str, list[str]] = defaultdict(list)
    by_lang: dict[str, list[str]] = defaultdict(list)
    by_source: dict[str, list[str]] = defaultdict(list)
    for k, r in by_key.items():
        strata[f"{r['language']}:{tr03.frame_class(r['frame'])}"].append(k)
        by_lang[r["language"]].append(k)
        by_source[r["schema_source_id"]].append(k)

    def eff(k: str) -> dict[str, str] | None:
        it = result.items.get(k)
        return it.effective if it is not None and it.state == "resolved" else None

    def gold(keys: Iterable[str]) -> int:
        return sum((eff(k) or {}).get("gold_correct") == "yes" for k in keys)

    def natural(keys: Iterable[str]) -> int:
        return sum((eff(k) or {}).get("request_natural") in ("yes", "fixable") for k in keys)

    def block(keys: list[str]) -> dict[str, Any]:
        return {
            "frozen_n": len(keys),
            "gold_correct": _rate_block(gold(keys), len(keys)),
            "natural_or_fixable": _rate_block(natural(keys), len(keys)),
        }

    all_keys = list(by_key)
    tr_strata = {k: v for k, v in strata.items() if k.startswith("tr:")}
    serious = [k for k in all_keys if (eff(k) or {}).get("serious_error") == "yes"]
    serious_kept = [k for k in serious if (eff(k) or {}).get("decision") != "drop"]
    sources_ok = all(len(v) >= SOURCE_MIN_ROWS for v in by_source.values())
    ready = result.complete
    gates = {
        "G3_ledger_valid_and_complete": ready,
        "G3o_origin_gold_ge_95pct_all_languages": ready
        and gold(all_keys) / len(all_keys) >= tr03.GOLD_CORRECT_MIN,
        "G3o_origin_natural_or_fixable_ge_90pct_all_languages": ready
        and natural(all_keys) / len(all_keys) >= tr03.NATURAL_OR_FIXABLE_MIN,
        "G3l_english_gold_ge_95pct": ready
        and gold(by_lang["en"]) / len(by_lang["en"]) >= tr03.GOLD_CORRECT_MIN,
        "G3l_english_natural_or_fixable_ge_90pct": ready
        and natural(by_lang["en"]) / len(by_lang["en"]) >= tr03.NATURAL_OR_FIXABLE_MIN,
        "G3t_turkish_gold_ge_95pct": ready
        and gold(by_lang["tr"]) / len(by_lang["tr"]) >= tr03.GOLD_CORRECT_MIN,
        "G3t_turkish_natural_or_fixable_ge_90pct": ready
        and natural(by_lang["tr"]) / len(by_lang["tr"]) >= tr03.NATURAL_OR_FIXABLE_MIN,
        "G3f_every_turkish_frame_stratum_has_59_frozen_rows_and_gold_ge_95pct": ready
        and len(tr_strata) == len(tr03.FRAME_CLASSES)
        and all(
            len(v) >= tr03.GATE_MIN_N and gold(v) / len(v) >= tr03.GOLD_CORRECT_MIN
            for v in tr_strata.values()
        ),
        "G3s_every_schema_source_has_at_least_10_reviewed_rows": ready and sources_ok,
        "G3d_zero_serious_errors_kept": ready and not serious_kept,
    }
    return {
        "ledger_sha256": result.sha256,
        "frozen_items": len(frozen),
        "item_states": result.counts(),
        "problems": _problems_view(result.problems),
        "second_review_required": sum(i.second_required for i in result.items.values()),
        "disputed": sum(i.disputed for i in result.items.values()),
        "serious_errors_found": len(serious),
        "serious_errors_kept": len(serious_kept),
        "by_language": {k: block(v) for k, v in sorted(by_lang.items())},
        "by_stratum_reported": {k: block(v) for k, v in sorted(strata.items())},
        "by_schema_source_reported": {k: block(v) for k, v in sorted(by_source.items())},
        "gates": gates,
        "all_row_gates_met": all(gates.values()),
        "limits": "Wilson bounds treat rows as independent; rows of one template are not. A met "
        "gate supports using these templates for training only; it is not a quality claim about "
        "any model and not a population guarantee",
    }


def template_gate_report(
    result: LedgerResult, frozen: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    decided = {
        k: i.effective["decision"]
        for k, i in result.items.items()
        if i.state == "resolved" and i.effective is not None
    }
    by_key = {r["lineage_id"]: r for r in frozen}
    coverage: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    for split in coverage:
        for cls in tr03.FRAME_CLASSES:
            keys = [
                k
                for k, r in by_key.items()
                if r["split"] == split and tr03.frame_class(r["frame"]) == cls
            ]
            if keys:
                coverage[split][cls] = sum(decided.get(k) == "keep" for k in keys)
    coverage_ok = (
        result.complete
        and all(n >= MIN_KEPT_TRAIN_TEMPLATES_PER_CLASS for n in coverage["train"].values())
        and all(n >= MIN_KEPT_VALID_TEMPLATES_PER_CLASS for n in coverage["validation"].values())
    )
    kept = sorted(k for k, d in decided.items() if d == "keep")
    return {
        "ledger_sha256": result.sha256,
        "templates_frozen": len(frozen),
        "item_states": result.counts(),
        "problems": _problems_view(result.problems),
        "decisions": dict(sorted(Counter(decided.values()).items())),
        "kept_lineage_ids": kept if result.complete else [],
        "kept_templates_by_frame_class": coverage,
        "gates": {
            "G2_every_frozen_template_has_a_resolved_human_decision": result.complete,
            "G2c_every_frame_class_keeps_enough_templates": coverage_ok,
        },
        "rule": f"a template trains only if its effective decision is keep; fix and drop remove "
        f"all its rows; each frame class must keep at least {MIN_KEPT_TRAIN_TEMPLATES_PER_CLASS} "
        f"train and {MIN_KEPT_VALID_TEMPLATES_PER_CLASS} validation templates (frozen before "
        "any decision); dropped templates are never replaced or regenerated in this experiment",
    }


# ------------------------------------------------------------------------------------ rights

RIGHTS_DECISIONS = ("include", "exclude")


def rights_gate_report(
    data: bytes,
    frozen: Sequence[Mapping[str, str]],
    roster: RosterIndex | None,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    problems: list[str] = []
    header, raw_rows, struct = read_csv_strict(data)
    problems += struct
    included: list[str] = []
    decided = 0
    if roster is None:
        problems.append("roster_missing_or_invalid")
    elif tuple(header) != tr03.RIGHTS_COLUMNS:
        problems.append("header_differs_from_the_frozen_worklist")
    else:
        frozen_by = {r["source_id"]: r for r in frozen}
        seen: set[str] = set()
        identity = tr03.RIGHTS_COLUMNS[: tr03.RIGHTS_COLUMNS.index("owner_decision")]
        for cells in raw_rows:
            row = dict(zip(header, cells, strict=True))
            sid = row["source_id"]
            if sid in seen:
                problems.append(f"duplicate_key:{sid}")
                continue
            seen.add(sid)
            ref = frozen_by.get(sid)
            if ref is None:
                problems.append(f"foreign_key:{sid}")
                continue
            if any(row[c] != ref[c] for c in identity):
                problems.append(f"{sid}:identity_differs_from_frozen_worklist")
                continue
            decision = row["owner_decision"]
            if decision == "" and not row["decided_by"] and not row["decided_on"]:
                continue
            if decision not in RIGHTS_DECISIONS:
                problems.append(f"{sid}:decision_not_in_enum")
                continue
            p = roster.people.get(row["decided_by"])
            if p is None or "owner" not in p["roles"]:
                problems.append(f"{sid}:decided_by_is_not_an_owner_on_the_roster")
                continue
            d = parse_date(row["decided_on"])
            if d is None or d < roster.designated_on or d > today:
                problems.append(f"{sid}:decided_on_invalid")
                continue
            if decision == "include":
                if sid not in tr03.ELIGIBLE_SOURCES:
                    problems.append(f"{sid}:include_is_not_allowed_for_a_source_excluded_by_rule")
                    continue
                if not row["notes"].strip():
                    problems.append(f"{sid}:include_needs_notes_on_how_the_caveat_is_addressed")
                    continue
                included.append(sid)
            decided += 1
        problems += [f"missing_key:{k}" for k in sorted(set(frozen_by) - seen)]
    complete = not problems and decided == len(frozen)
    return {
        "ledger_sha256": hashlib.sha256(data).hexdigest(),
        "sources_frozen": len(frozen),
        "sources_decided": decided,
        "problems": _problems_view(problems),
        "included_sources": sorted(included) if complete else [],
        "complete": complete,
        "note": "an owner decision is recorded as typed; the tool neither judges the licences nor "
        "verifies the person, and gives no legal opinion",
    }


# -------------------------------------------------------------------- derived training sets


def research_manifest(
    templates: LedgerResult,
    rows: LedgerResult,
    rights: Mapping[str, Any],
    eligible_train: Sequence[Mapping[str, Any]],
    eligible_valid: Sequence[Mapping[str, Any]],
    row_frozen: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """The filtered train and validation id lists a research run could use, derived only from
    resolved human decisions. Refuses (``ok`` false) unless every ledger is complete and valid."""
    reasons = []
    if not templates.complete:
        reasons.append("template_ledger_not_complete_and_valid")
    if not rows.complete:
        reasons.append("row_ledger_not_complete_and_valid")
    if not rights.get("complete"):
        reasons.append("rights_ledger_not_complete_and_valid")
    if reasons:
        return {"ok": False, "reasons": reasons}
    kept = {k for k, i in templates.items.items() if (i.effective or {}).get("decision") == "keep"}
    included = set(rights["included_sources"])
    sample = {r["sample_id"]: r for r in row_frozen}
    row_dropped: set[str] = set()
    template_removed_by_rows: set[str] = set()
    for sid, item in rows.items.items():
        eff = item.effective or {}
        rid, lineage = sample[sid]["record_id"], sample[sid]["lineage_id"]
        if eff.get("decision") != "keep":
            row_dropped.add(rid)
        if eff.get("gold_correct") == "no" or eff.get("serious_error") == "yes":
            template_removed_by_rows.add(lineage)

    def pick(rows_: Sequence[Mapping[str, Any]], check_rows: bool) -> tuple[list[str], Counter]:
        out: list[str] = []
        why: Counter = Counter()
        for r in rows_:
            lin = r["lineage_id"]
            if lin not in kept:
                why["template_not_kept"] += 1
            elif r["provenance"]["schema_source_id"] not in included:
                why["rights_source_not_included"] += 1
            elif check_rows and lin in template_removed_by_rows:
                why["template_removed_by_a_reviewed_row_finding"] += 1
            elif check_rows and r["id"] in row_dropped:
                why["row_not_kept_by_review"] += 1
            else:
                out.append(r["id"])
        return sorted(out), why

    train_ids, train_why = pick(eligible_train, True)
    valid_ids, valid_why = pick(eligible_valid, False)
    by_class = Counter(
        tr03.frame_class(r["frame"]) for r in eligible_train if r["id"] in set(train_ids)
    )
    ok = bool(train_ids) and bool(valid_ids) and set(by_class) == set(tr03.FRAME_CLASSES)
    return {
        "ok": ok,
        "reasons": [] if ok else ["a_frame_class_or_split_is_empty_after_filtering"],
        "derived_from_unverified_human_ledgers": True,
        "train_ids": train_ids,
        "validation_ids": valid_ids,
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(valid_ids).encode()).hexdigest(),
        "counts": {
            "train": len(train_ids),
            "validation": len(valid_ids),
            "train_excluded_by_reason": dict(sorted(train_why.items())),
            "validation_excluded_by_reason": dict(sorted(valid_why.items())),
            "train_by_frame_class": dict(sorted(by_class.items())),
        },
        "rule": "keep only rows of effective-keep templates, from rights-included sources, "
        "whose sampled reviews (if sampled) are effective keep, and drop whole templates that "
        "have a reviewed row with wrong gold or a serious error; nothing is inferred or corrected",
    }


# -------------------------------------------------------- G5 to G7: structural file checks


def _binding_problems(obj: Mapping[str, Any], expected: Mapping[str, str]) -> list[str]:
    bound = obj.get("bindings")
    if not isinstance(bound, dict):
        return ["bindings_missing"]
    out = []
    for key, value in expected.items():
        got = bound.get(key)
        if not isinstance(got, str) or not SHA_RE.match(got):
            out.append(f"binding_missing_or_not_sha256:{key}")
        elif got != value:
            out.append(f"stale_binding:{key}")
    return out


def _file_state(path: Path) -> tuple[str, Any, list[str]]:
    if not path.exists():
        return "absent", None, []
    try:
        obj = load_json_strict(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        return "invalid", None, [f"not_strict_json:{exc}"]
    if not isinstance(obj, dict) or not obj:
        return "invalid", None, ["empty_or_not_an_object"]
    return "present", obj, []


def _verdict(problems: Sequence[str]) -> str:
    return "invalid" if problems else "consistent_unverified"


def heldout_problems(
    obj: Mapping[str, Any], roster: Any, bindings: Mapping[str, str], today: date
) -> list[str]:
    problems = []
    if obj.get("schema") != "axbench-tr.tr03-heldout-frozen/1":
        problems.append("schema_wrong")
    problems += _binding_problems(
        obj, {k: bindings[k] for k in ("protocol_sha256", "prep_manifest_sha256")}
    )
    if not isinstance(obj.get("tasks_file_sha256"), str) or not SHA_RE.match(
        obj["tasks_file_sha256"]
    ):
        problems.append("tasks_file_sha256_missing")
    rp = roster_problems(roster, "heldout")
    problems += [f"roster:{p}" for p in rp]
    if rp or not isinstance(roster, dict):
        return problems
    people = {p["id"]: p for p in roster["people"]}
    att = obj.get("attestation")
    if not isinstance(att, dict):
        problems.append("attestation_missing")
    else:
        for k in HELDOUT_ATTESTATIONS:
            if att.get(k) is not True:
                problems.append(f"attestation_not_true:{k}")
        owner = people.get(att.get("attested_by", ""))
        if owner is None or "owner" not in owner["roles"]:
            problems.append("attested_by_is_not_the_owner")
    d = parse_date(obj.get("frozen_on"))
    if d is None or d > today:
        problems.append("frozen_on_invalid")
    tasks = obj.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return [*problems, "tasks_missing"]
    ids: set[str] = set()
    langs: Counter = Counter()
    authors: Counter = Counter()
    families: Counter = Counter()
    for t in tasks:
        if not isinstance(t, dict):
            problems.append("task_not_an_object")
            continue
        tid = t.get("task_id")
        if not isinstance(tid, str) or not tid or tid in ids:
            problems.append(f"task_id_missing_or_duplicate:{tid}")
            continue
        ids.add(tid)
        lang = t.get("language")
        if lang not in ("tr", "en") or t.get("action") not in HELDOUT_ACTIONS:
            problems.append(f"{tid}:language_or_action_invalid")
            continue
        if t.get("table_source") not in HELDOUT_TABLE_SOURCES:
            problems.append(f"{tid}:table_source_invalid")
        fam = t.get("table_family_id")
        if not isinstance(fam, str) or not fam:
            problems.append(f"{tid}:table_family_missing")
        author, checker = people.get(t.get("author_id", "")), people.get(t.get("checker_id", ""))
        if author is None or "author" not in author["roles"]:
            problems.append(f"{tid}:author_not_an_author_on_the_roster")
            continue
        if author["native_turkish"] is not True or (
            lang == "en" and author["fluent_english"] is not True
        ):
            problems.append(f"{tid}:author_not_competent")
        if checker is None or "checker" not in checker["roles"] or checker["id"] == author["id"]:
            problems.append(f"{tid}:checker_missing_wrong_role_or_the_author")
        for flag in ("layout_held_out", "intent_held_out", "near_miss"):
            if not isinstance(t.get(flag), bool):
                problems.append(f"{tid}:{flag}_not_boolean")
        langs[lang] += 1
        authors[author["id"]] += 1
        families[fam] += 1
    if langs["tr"] < HELDOUT_MIN_TR or langs["en"] < HELDOUT_MIN_EN:
        problems.append(f"too_few_tasks_tr_{langs['tr']}_en_{langs['en']}")
    total = sum(authors.values())
    if len(authors) < HELDOUT_MIN_AUTHORS or (
        total and max(authors.values()) / total > HELDOUT_MAX_AUTHOR_SHARE
    ):
        problems.append("author_clusters_too_few_or_too_concentrated")
    if len(families) < HELDOUT_MIN_TABLE_FAMILIES or (
        total and max(families.values()) / total > HELDOUT_MAX_FAMILY_SHARE
    ):
        problems.append("table_family_clusters_too_few_or_too_concentrated")
    return problems


def preregistration_problems(
    obj: Mapping[str, Any], bindings: Mapping[str, str], today: date
) -> list[str]:
    problems = []
    if obj.get("schema") != "axbench-tr.tr03-preregistration/1":
        problems.append("schema_wrong")
    problems += _binding_problems(obj, bindings)
    cfg = json.loads(TRAINING_CONFIG_PATH.read_text("utf-8"))
    comp = obj.get("comparators")
    if not isinstance(comp, dict):
        problems.append("comparators_missing")
    else:
        c0 = comp.get("C0")
        if not isinstance(c0, dict) or c0.get("revision") != cfg["model"]["revision"]:
            problems.append("C0_revision_not_the_pinned_base")
        c1 = comp.get("C1")
        if not isinstance(c1, dict) or c1.get("rules") != "axiom.rules/0.1":
            problems.append("C1_not_axiom_rules_0.1")
        c2 = comp.get("C2")
        if c2 is not None and not (
            isinstance(c2, dict)
            and SHA_RE.match(str(c2.get("adapter_sha256", "")))
            and COMMIT_RE.match(str(c2.get("inference_commit", "")))
            and c2.get("retrained") is False
            and c2.get("used_for_selection") is False
        ):
            problems.append("C2_present_but_not_a_verified_frozen_secondary_comparator")
    ana = obj.get("analysis")
    if not isinstance(ana, dict) or (
        ana.get("bootstrap_replicates") != 2000
        or ana.get("seed") != 2026
        or ana.get("family_alpha") != 0.05
        or ana.get("coprimary_contrasts") != 2
        or ana.get("per_contrast_one_sided_alpha") != 0.025
    ):
        problems.append("analysis_not_the_protocol_values")
    for key in ("approved_by",):
        if not isinstance(obj.get(key), str) or not obj[key]:
            problems.append(f"{key}_missing")
    d = parse_date(obj.get("approved_on"))
    if d is None or d > today:
        problems.append("approved_on_invalid")
    return problems


def clearance_problems(
    obj: Mapping[str, Any],
    bindings: Mapping[str, str],
    ledger_hashes: Mapping[str, str],
    today: date,
) -> list[str]:
    problems = []
    if obj.get("schema") != "axbench-tr.tr03-clearance/1":
        problems.append("schema_wrong")
    if not isinstance(obj.get("commit"), str) or not COMMIT_RE.match(obj["commit"]):
        problems.append("commit_not_a_40_hex_sha")
    problems += _binding_problems(obj, bindings)
    lh = obj.get("ledger_sha256")
    if not isinstance(lh, dict):
        problems.append("ledger_sha256_missing")
    else:
        for name, value in ledger_hashes.items():
            if lh.get(name) != value:
                problems.append(f"stale_or_missing_ledger_hash:{name}")
    for key in ("cleared_by", "owner_authorization"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            problems.append(f"{key}_missing")
    for key in ("clearance_comment_url", "owner_authorization_url"):
        if not isinstance(obj.get(key), str) or not GITHUB_COMMENT_RE.match(obj[key]):
            problems.append(f"{key}_not_a_github_comment_url")
    for key in ("macos_ci_run_url", "windows_ci_run_url", "m5_no_weight_preflight_sha256"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            problems.append(f"{key}_missing")
    d = parse_date(obj.get("cleared_on"))
    if d is None or d > today:
        problems.append("cleared_on_invalid")
    return problems


# ----------------------------------------------------------------------------- the report


def frozen_worklists(version: str = tr03.CURRENT_VERSION) -> dict[str, list[dict[str, str]]]:
    train, valid, _, _ = tr03.load(version)
    elig_t, _ = tr03.split_eligible(train)
    elig_v, _ = tr03.split_eligible(valid)
    return {
        "template": tr03.template_worklist(elig_t + elig_v),
        "row": tr03.row_worklist(elig_t),
        "rights": tr03.rights_worklist(),
    }


def _frozen_drift(prep_dir: Path, frozen: Mapping[str, list[dict[str, str]]]) -> list[str]:
    out = []
    for name, key, cols in (
        ("TEMPLATE_REVIEW_WORKLIST.csv", "template", tr03.TEMPLATE_COLUMNS),
        ("ROW_REVIEW_WORKLIST.csv", "row", tr03.ROW_COLUMNS),
        ("RIGHTS_WORKLIST.csv", "rights", tr03.RIGHTS_COLUMNS),
    ):
        path = prep_dir / name
        if not path.exists() or path.read_bytes() != tr03._csv_bytes(cols, frozen[key]):
            out.append(f"committed_worklist_differs_from_a_fresh_build:{name}")
    return out


def gates_report(
    prep_dir: Path = tr03.PREP_DIR,
    today: date | None = None,
    ledgers: Path | None = None,
    include_ids: bool = False,
) -> dict[str, Any]:
    """State of every gate from the files in ``ledgers`` (default ``prep_dir/ledgers``). Never
    returns an open execution gate."""
    today = today or date.today()
    ledgers = ledgers if ledgers is not None else prep_dir / LEDGER_DIR_NAME
    frozen = frozen_worklists()
    drift = _frozen_drift(prep_dir, frozen)
    bindings = current_bindings(prep_dir)
    gates: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}

    def file_hash(name: str) -> str | None:
        p = ledgers / name
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    # G1 roster
    roster_state, roster, rp = _file_state(ledgers / "HUMAN_ROSTER.json")
    index: RosterIndex | None = None
    if roster_state == "present":
        rp = roster_problems(roster, "training")
        index = RosterIndex.build(roster)
        gates["G1_roster"] = {"status": _verdict(rp), "problems": rp}
        hashes["roster"] = file_hash("HUMAN_ROSTER.json") or ""
    else:
        gates["G1_roster"] = {"status": roster_state, "problems": rp}

    def review_gate(
        name: str, filename: str, spec: LedgerSpec, fro: list[dict[str, str]]
    ) -> LedgerResult | None:
        path = ledgers / filename
        if not path.exists():
            gates[name] = {"status": "absent"}
            return None
        if drift:
            gates[name] = {"status": "invalid", "problems": drift}
            return None
        res = evaluate_ledger(spec, path.read_bytes(), fro, index, today)
        hashes[spec.kind] = res.sha256
        report = (
            template_gate_report(res, fro) if spec.kind == "template" else row_gate_report(res, fro)
        )
        if not res.valid:
            status = "invalid"
        elif not res.complete:
            status = "incomplete"
        elif not all(report["gates"].values()):
            status = "criteria_not_met"
        else:
            status = "consistent_unverified"
        gates[name] = {"status": status, "report": report}
        return res

    t_res = review_gate(
        "G2_templates", "TEMPLATE_REVIEW_RESULTS.csv", TEMPLATE_SPEC, frozen["template"]
    )
    r_res = review_gate("G3_rows", "ROW_REVIEW_RESULTS.csv", ROW_SPEC, frozen["row"])

    # G4 rights
    rpath = ledgers / "RIGHTS_DECISIONS.csv"
    rights_report: dict[str, Any] = {"complete": False}
    if not rpath.exists():
        gates["G4_rights"] = {"status": "absent"}
    elif drift:
        gates["G4_rights"] = {"status": "invalid", "problems": drift}
    else:
        rights_report = rights_gate_report(rpath.read_bytes(), frozen["rights"], index, today)
        hashes["rights"] = rights_report["ledger_sha256"]
        gates["G4_rights"] = {
            "status": "consistent_unverified"
            if rights_report["complete"]
            else ("invalid" if rights_report["problems"]["count"] else "incomplete"),
            "report": rights_report,
        }

    # G5 held-out, G6 preregistration
    for gate, filename, hash_key, checker in (
        ("G5_heldout", "HELDOUT_FROZEN.json", "heldout", heldout_problems),
        (
            "G6_preregistration",
            "PREREGISTRATION.json",
            "preregistration",
            lambda o, r, b, t: preregistration_problems(o, b, t),
        ),
    ):
        state, obj, problems = _file_state(ledgers / filename)
        if state == "present":
            problems = checker(obj, roster, bindings, today)
            hashes[hash_key] = file_hash(filename) or ""
            gates[gate] = {"status": _verdict(problems), "problems": problems}
        else:
            gates[gate] = {"status": state, "problems": problems}

    # G7 clearance binds the exact hash of every other file
    state, obj, problems = _file_state(ledgers / "CLEARANCE.json")
    if state == "present":
        needed = {"roster", "template", "row", "rights", "heldout", "preregistration"}
        problems = clearance_problems(obj, bindings, {k: hashes.get(k, "") for k in needed}, today)
        problems += [f"no_ledger_to_bind:{k}" for k in sorted(needed - set(hashes))]
        gates["G7_clearance"] = {"status": _verdict(problems), "problems": problems}
    else:
        gates["G7_clearance"] = {"status": state, "problems": problems}

    derived: dict[str, Any] = {"ok": False, "reasons": ["ledgers_missing"]}
    if t_res is not None and r_res is not None and rights_report.get("complete"):
        train, valid, _, _ = tr03.load()
        elig_t, _ = tr03.split_eligible(train)
        elig_v, _ = tr03.split_eligible(valid)
        derived = research_manifest(t_res, r_res, rights_report, elig_t, elig_v, frozen["row"])
        if not include_ids:
            derived = {k: v for k, v in derived.items() if k not in ("train_ids", "validation_ids")}
    consistent = all(g["status"] == "consistent_unverified" for g in gates.values())
    return {
        "tool": "axbench_tr.tr03_gates",
        "identity_authenticated": False,
        "bindings_current": bindings,
        "frozen_worklist_problems": drift,
        "gates": gates,
        "derived_training_set": derived,
        "all_gates_consistent_unverified": consistent,
        "research_training_open": False,
        "execution_gate": {
            "open": False,
            "reason": "this tool never opens execution: a program cannot authenticate people, "
            "verify rights, judge an evaluation set or confirm a clearance; execution needs the "
            "technical lead's immutable clearance and the owner's authorization on GitHub, "
            "checked by people",
        },
    }


def exit_code(report: Mapping[str, Any]) -> int:
    """1 while any gate is not consistent; 2 when all are (still not execution); never 0."""
    return 2 if report["all_gates_consistent_unverified"] else 1
