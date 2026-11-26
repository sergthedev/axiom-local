"""EXP-010 training plan, completion boundary and stop rules (pure logic, no MLX, no OS calls).

The training harness (``envs/model/exp010_train.py``) tokenizes, trains and samples the operating
system, and passes plain values in, so every decision here is unit-tested in Core CI on every
platform. Operational definitions are in the EXP-010 pre-run record
(``docs/evidence/experiments/EXP-010-lora-pilot/PRERUN.md``); they are fixed before any training.

Accounting (verified against the installed MLX-LM 0.32.2 trainer, PRERUN E1-E3): one *micro-step*
is one forward/backward pass on one example (micro-batch 1); one *optimizer update* applies the
mean of the gradients of ``GRAD_ACCUM`` consecutive micro-steps. MLX-LM's ``iters`` counts
micro-steps, so ``updates`` optimizer updates need ``updates * GRAD_ACCUM`` micro-steps.

Completion boundary (PRERUN D2): the trained sequence is the inference prompt (chat template,
``enable_thinking=False``, generation prompt) followed by the compact gold JSON and the end-of-turn
token ``<|im_end|>``. The template's newline after ``<|im_end|>`` is dropped because generation
stops at ``<|im_end|>``. Loss is taken on every token after the prompt and on nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

TRAIN_PLAN_VERSION = "exp010.train-plan/1"

SEED = 2026
MICRO_BATCH = 1
GRAD_ACCUM = 4
LEARNING_RATE = 1e-5
NUM_LAYERS = 8  # the last 8 of Qwen3-1.7B's 28 transformer blocks
LORA_PARAMETERS = {
    "rank": 8,
    "scale": 20.0,  # MLX-LM 0.32.2 default; multiplies the low-rank product directly (not alpha/r)
    "dropout": 0.0,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
}
PHASE_UPDATES = {"smoke": 20, "pilot": 200}
EVAL_EVERY_UPDATES = 25  # pilot: validation loss at updates 0, 25, ..., 200
SAVE_EVERY_UPDATES = 50  # pilot: adapter saved at updates 50, 100, 150, 200
SEQ_LENGTH_CAP = 1024  # above this the protocol stops and reports
IM_END = 151645  # <|im_end|>
NEWLINE = 198  # "\n" after <|im_end|> in the Qwen3 chat template

THROUGHPUT_WINDOW = 10  # optimizer updates per window
SUSTAINED_WINDOWS = 3
SLOWDOWN_FRACTION = 0.5


class PlanError(Exception):
    """A pre-training condition the protocol treats as a stop (never silently repaired)."""


# ------------------------------------------------------------------ schedule


@dataclass(frozen=True)
class Schedule:
    phase: str
    updates: int
    grad_accum: int
    micro_steps: int
    eval_updates: tuple[int, ...]
    save_updates: tuple[int, ...]
    epochs: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "optimizer_updates": self.updates,
            "grad_accumulation_steps": self.grad_accum,
            "micro_batch": MICRO_BATCH,
            "micro_steps": self.micro_steps,
            "mlx_lm_iters": self.micro_steps,
            "eval_after_updates": list(self.eval_updates),
            "save_after_updates": list(self.save_updates),
            "epochs_over_train": self.epochs,
        }


def schedule(phase: str, n_train: int = 400) -> Schedule:
    """Micro-steps, evaluation and save points for a phase, in optimizer updates.

    Evaluation and saving happen only at update boundaries (after the optimizer step), never between
    micro-steps of one accumulation group. Update 0 is the untrained adapter (LoRA B is zero).
    The smoke evaluates and saves only at its start and end.
    """
    if phase not in PHASE_UPDATES:
        raise ValueError(f"unknown phase {phase!r}")
    updates = PHASE_UPDATES[phase]
    micro = updates * GRAD_ACCUM
    if phase == "pilot":
        evals = tuple(range(0, updates + 1, EVAL_EVERY_UPDATES))
        saves = tuple(range(SAVE_EVERY_UPDATES, updates + 1, SAVE_EVERY_UPDATES))
    else:
        evals = (0, updates)
        saves = (updates,)
    return Schedule(phase, updates, GRAD_ACCUM, micro, evals, saves, micro * MICRO_BATCH / n_train)


class UpdateLedger:
    """Maps MLX-LM's per-report iteration numbers to optimizer updates and checks alignment.

    The harness reports every ``GRAD_ACCUM`` micro-steps, so report ``u`` must carry iteration
    ``u * GRAD_ACCUM``; anything else means updates and reports are out of step (a stop).
    """

    def __init__(self, sched: Schedule) -> None:
        self.sched = sched
        self.updates = 0

    def advance(self, iteration: int) -> int:
        expected = (self.updates + 1) * self.sched.grad_accum
        if iteration != expected:
            raise PlanError(f"report at micro-step {iteration}, expected {expected}")
        self.updates += 1
        if self.updates > self.sched.updates:
            raise PlanError(f"update {self.updates} exceeds the planned {self.sched.updates}")
        return self.updates

    def complete(self) -> bool:
        return self.updates == self.sched.updates


# ------------------------------------------------------------------ completion boundary


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    tokens: tuple[int, ...]
    offset: int  # first trained position = number of prompt tokens

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def completion_tokens(self) -> int:
        return self.length - self.offset


def completion_example(
    example_id: str, prompt_ids: Sequence[int], full_ids: Sequence[int]
) -> TrainingExample:
    """Build one training sequence from the inference prompt and the full chat rendering.

    Requires, token by token: the full rendering starts with the inference prompt; it ends with
    ``<|im_end|>`` and the template newline; ``<|im_end|>`` occurs exactly once after the prompt.
    The trailing newline is dropped. Any violation raises :class:`PlanError`.
    """
    p, f = list(prompt_ids), list(full_ids)
    if f[: len(p)] != p:
        raise PlanError(
            f"{example_id}: training rendering does not start with the inference prompt"
        )
    if len(f) < len(p) + 3 or f[-2:] != [IM_END, NEWLINE]:
        raise PlanError(f"{example_id}: rendering does not end with <|im_end|> and a newline")
    tokens = f[:-1]
    if tokens[len(p) :].count(IM_END) != 1:
        raise PlanError(f"{example_id}: <|im_end|> must occur exactly once in the completion")
    return TrainingExample(example_id, tuple(tokens), len(p))


def target_positions(offset: int, length: int) -> range:
    """Sequence positions whose token is a training target (PRERUN D2): the completion only."""
    return range(offset, length)


def mlx_default_mask_positions(offset: int, length: int, padded_length: int) -> list[int]:
    """Positions selected by MLX-LM 0.32.2 ``default_loss`` (for the record, PRERUN E4).

    ``steps = arange(1, padded_length)``; ``mask = steps >= offset and steps <= length``. Position
    ``length`` is the first padding token, so the installed loss trains one padding target per
    example. The harness uses :func:`target_positions` instead.
    """
    return [s for s in range(1, padded_length) if offset <= s <= length]


def mlx_padded_length(length: int, max_seq_length: int, pad_to: int = 32) -> int:
    """Batch width chosen by MLX-LM 0.32.2 ``iterate_batches`` for a batch whose longest is
    ``length``: one plus the next multiple of 32, capped at ``max_seq_length``."""
    return min(1 + pad_to * ((length + pad_to - 1) // pad_to), max_seq_length)


def max_seq_length(lengths: Sequence[int]) -> int:
    """The protocol rule: smallest power of two at least the longest full sequence (train and
    validation); above ``SEQ_LENGTH_CAP`` the experiment stops and reports."""
    longest = max(lengths)
    value = 1 << max(0, math.ceil(math.log2(longest)))
    if value < longest:  # guard against floating-point rounding in log2
        value <<= 1
    if value > SEQ_LENGTH_CAP:
        raise PlanError(f"longest sequence {longest} needs {value} > {SEQ_LENGTH_CAP} tokens")
    return value


def check_lengths(examples: Sequence[TrainingExample], limit: int) -> None:
    """No example may be truncated: any sequence longer than ``limit`` is a stop."""
    over = [e.example_id for e in examples if e.length > limit]
    if over:
        raise PlanError(f"{len(over)} sequences exceed {limit} tokens: {over[:5]}")


def length_summary(values: Sequence[int]) -> dict[str, Any]:
    v = sorted(values)
    n = len(v)
    mid = n // 2
    median = v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2
    return {"n": n, "min": v[0], "median": median, "max": v[-1], "sum": sum(v)}


# ------------------------------------------------------------------ stop rules


def non_finite(name: str, value: float | None) -> str | None:
    """Stop reason when a loss is missing, NaN or infinite."""
    if value is None or not math.isfinite(value):
        return f"non-finite {name}: {value!r}"
    return None


@dataclass
class UpdateThroughputWatch:
    """Protocol slowdown rule in optimizer updates per second of training compute.

    Windows of ``THROUGHPUT_WINDOW`` consecutive updates; the baseline is the first window (its
    throughput = updates / summed compute seconds, evaluation and saving excluded). The run stops
    when ``SUSTAINED_WINDOWS`` consecutive later windows fall below ``SLOWDOWN_FRACTION`` of it.
    With 20 updates (the smoke) there is only one later window, so the rule cannot fire there.
    """

    seconds: list[float] = field(default_factory=list)
    windows: list[float] = field(default_factory=list)
    slow_windows: int = 0
    stopped: str | None = None

    def add(self, update_seconds: float) -> str | None:
        if self.stopped:
            return self.stopped
        if not update_seconds > 0:
            raise ValueError("update compute time must be positive")
        self.seconds.append(update_seconds)
        if len(self.seconds) % THROUGHPUT_WINDOW:
            return None
        window = THROUGHPUT_WINDOW / sum(self.seconds[-THROUGHPUT_WINDOW:])
        self.windows.append(window)
        if len(self.windows) == 1:
            return None
        base = self.windows[0]
        self.slow_windows = self.slow_windows + 1 if window < SLOWDOWN_FRACTION * base else 0
        if self.slow_windows >= SUSTAINED_WINDOWS:
            self.stopped = (
                f"sustained slowdown: {SUSTAINED_WINDOWS} consecutive windows of "
                f"{THROUGHPUT_WINDOW} updates below {SLOWDOWN_FRACTION:.0%} of the first window "
                f"({base:.4f} updates/s)"
            )
        return self.stopped

    def summary(self) -> dict[str, Any]:
        return {
            "window_updates": THROUGHPUT_WINDOW,
            "baseline_updates_per_second": self.windows[0] if self.windows else None,
            "windows_updates_per_second": self.windows,
            "updates_timed": len(self.seconds),
            "stopped": self.stopped,
        }


# ------------------------------------------------------------------ completion checks (rev. 1)
# Lead pre-run review, corrections B and C: a run is "completed" only when its persisted records
# and adapter files pass these checks; a self-reported status flag is never sufficient.

RECORD_KINDS = ("eval", "update", "save")


def expected_record_sequence(sched: Schedule) -> list[tuple[str, int]]:
    """The exact ``(kind, update)`` order the harness writes: the loss at update 0, then for every
    update its record, its save (if planned) and its evaluation (if planned)."""
    seq: list[tuple[str, int]] = [("eval", 0)] if 0 in sched.eval_updates else []
    for u in range(1, sched.updates + 1):
        seq.append(("update", u))
        if u in sched.save_updates:
            seq.append(("save", u))
        if u in sched.eval_updates:
            seq.append(("eval", u))
    return seq


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def check_training_records(records: Sequence[dict[str, Any]], sched: Schedule) -> list[str]:
    """Problems with a training log (empty list = complete and consistent).

    Every planned update, evaluation and save record must be present exactly once and in the
    planned order; micro-steps must equal ``update * grad_accum``; losses must be finite; every
    save must name ``update_NNNN.safetensors`` with a SHA-256.
    """
    problems: list[str] = []
    got: list[tuple[str, int]] = []
    for i, r in enumerate(records):
        kind, update = r.get("kind"), r.get("update")
        if kind not in RECORD_KINDS or not isinstance(update, int) or isinstance(update, bool):
            problems.append(f"record {i}: unexpected kind/update {kind!r}/{update!r}")
            continue
        got.append((kind, update))
        if kind == "update":
            if r.get("micro_steps") != update * sched.grad_accum:
                problems.append(f"update {update}: micro_steps {r.get('micro_steps')!r}")
            if not _finite(r.get("train_loss")):
                problems.append(f"update {update}: train loss {r.get('train_loss')!r}")
            if not (_finite(r.get("update_seconds")) and r["update_seconds"] > 0):
                problems.append(f"update {update}: update_seconds {r.get('update_seconds')!r}")
        elif kind == "eval" and not _finite(r.get("loss")):
            problems.append(f"eval {update}: loss {r.get('loss')!r}")
        elif kind == "save":
            sha = r.get("sha256")
            if r.get("file") != f"update_{update:04d}.safetensors":
                problems.append(f"save {update}: file {r.get('file')!r}")
            if not (isinstance(sha, str) and len(sha) == 64):
                problems.append(f"save {update}: sha256 {sha!r}")
    expected = expected_record_sequence(sched)
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        dup = sorted({x for x in got if got.count(x) > 1})
        problems.append(
            f"record sequence differs from the plan: missing {missing[:5]}, unexpected "
            f"{extra[:5]}, duplicated {dup[:5]}, {len(got)} records, {len(expected)} planned"
        )
    return problems


def check_adapter_tensors(
    tensors: dict[str, dict[str, Any]], expected_shapes: dict[str, list[int]]
) -> list[str]:
    """Problems with saved LoRA tensors: ``{name: {"shape", "finite", "max_abs"}}``.

    Names and shapes must equal the planned trainable tensors, every value must be finite and
    every ``lora_b`` tensor must be non-zero (``lora_b`` starts at zero, so an all-zero ``lora_b``
    means the adapter was not trained).
    """
    problems: list[str] = []
    if set(tensors) != set(expected_shapes):
        diff = sorted(set(tensors) ^ set(expected_shapes))
        problems.append(f"tensor names differ from the plan: {diff[:6]}")
    for name in sorted(set(tensors) & set(expected_shapes)):
        t = tensors[name]
        if list(t.get("shape", [])) != list(expected_shapes[name]):
            problems.append(f"{name}: shape {t.get('shape')} != {expected_shapes[name]}")
        if t.get("finite") is not True:
            problems.append(f"{name}: non-finite values")
        if name.endswith("lora_b") and not (_finite(t.get("max_abs")) and t["max_abs"] > 0):
            problems.append(f"{name}: lora_b is zero (adapter not trained)")
    return problems


def check_request_records(
    records: Sequence[dict[str, Any]], expected_ids: Sequence[str]
) -> list[str]:
    """Problems with inference records (the smoke reload check, correction B).

    Exactly the expected example IDs, once each and in order; every record ``status == "ok"`` with
    a string ``raw_output``. A missing, duplicated, failed or empty record is a problem.
    """
    problems: list[str] = []
    ids = [r.get("example_id") for r in records]
    if len(set(expected_ids)) != len(expected_ids):
        problems.append("expected IDs are not unique")
    if ids != list(expected_ids):
        problems.append(f"record IDs {ids} differ from the expected {list(expected_ids)}")
    for r in records:
        if r.get("status") != "ok":
            problems.append(f"{r.get('example_id')}: status {r.get('status')!r} "
                            f"({r.get('error')!r})")  # fmt: skip
        elif not isinstance(r.get("raw_output"), str):
            problems.append(f"{r.get('example_id')}: no raw output")
    return problems
