# ADR 0003: Input guard, dataset snapshot and provisional resource caps

- Status: Proposed (awaiting technical review and owner approval)
- Date: 2026-11-25
- Issue / PR: Issue #2 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230), item 1)

## Context

Everything downstream depends on exactly what the parser accepted. Silent coercion (locale guessing, NA tokens,
precision loss in large identifiers) would make results irreproducible or wrong without any error.

## Decision

Implemented in `src/axiom/ingest.py` and `src/axiom/limits.py`.

| Topic | Policy |
|-------|--------|
| Parsing | Python 3.12 `csv.reader` with `strict=True`, `"` quote, doubled-quote escaping, no whitespace skipping (REF-008). Records must have exactly the header's width. |
| Explicit profile | `encoding` (`utf-8` or `utf-8-sig`), `delimiter` (`,` or `;`), `decimal` (`.` or `,`; comma only with `;`), and a schema listing **every** header column in header order as `numeric` or `text`. Nothing is inferred. A BOM under plain `utf-8` is rejected; NUL characters are rejected. |
| Missing | An empty field is the only missing value. `NA`, `null`, whitespace and similar tokens in numeric columns are errors, never coerced. |
| Numeric tokens | ASCII digits only, optional sign, optional decimal mark and exponent; no thousands separators; NaN/Infinity tokens, overflow to infinity and nonzero values that underflow to zero are rejected. `-0` is stored as `0.0`. Subnormal values are accepted. Conversion is to the nearest binary64 (decimal inputs with more than ~17 significant digits are rounded; this is inherent and documented). |
| Identifiers | Integer tokens in numeric columns beyond +/-2^53 are rejected (use a `text` column). |
| Header | NFC-normalized; empty names and post-normalization collisions are rejected; case and non-ASCII letters are preserved. Schema names are NFC-normalized for comparison. |
| Text values | Preserved exactly (no NFC, no trimming, no case folding). |
| Rows | Physically blank lines are skipped and counted (`n_blank_lines`); all-empty records such as `,` are kept as rows of missing values and counted. Header-only input is zero rows; empty input or an empty/invalid header is an error. `row_ordinal` is the 1-based index of a data record (not the physical line number) and is used in every exclusion report. |
| Snapshot | `Dataset` is an immutable frozen dataclass of tuples behind a read-only mapping. The raw SHA-256 covers exactly the bytes that were parsed (the file is read once, bounded to the byte cap). The *logical* SHA-256 covers a versioned, length-prefixed binary serialization (`axiom.dataset/1`) of schema, row count and typed values in source order. |
| Order | Source order is preserved. No permutation-invariant hash and no permutation-invariant K-means partition is promised. |
| Caps | See table below. |

Provisional caps (engineering guards chosen before any profiling; **not** scientific constants; revisit after profiling):

| Cap | Value | Rationale |
|-----|-------|-----------|
| `max_bytes` | 32 MiB | Bounds whole-input allocation; the file is read once with `read(max_bytes + 1)`. |
| `max_rows` | 1,000,000 | Keeps typed tuples within a few hundred MB. |
| `max_columns` / `max_header_chars` | 256 / 256 | Bounded schemas and plans. |
| `max_field_chars` | 65,536 | Bounds a single field. |
| `max_groups` | 10,000 | Bounds Compare output size. |
| `max_k` / `max_cluster_features` | 64 / 64 | Bounds K-means memory and batch sizes. |
| `max_reported_ordinals` | 100 | Bounds excluded-row lists; the true count is always reported. |

## Alternatives considered

- Locale or delimiter sniffing (rejected: guessing changes results silently).
- `csv.QUOTE_NOTNULL` to distinguish quoted empty strings (rejected: the lead decision keeps empty-field-is-missing simple).
- Streaming row-by-row typing without a raw-bytes hash (rejected: the lead decision requires hashing exactly the bytes parsed).

## Tradeoffs and consequences

- The strictness rejects some files that other tools would accept; each rejection is a structured error with a record ordinal.
- A quote character inside an *unquoted* field is accepted literally by Python's `csv` module even with `strict=True`;
  this is a known limitation of the parser, not detected by AXIOM.
- Holding typed tuples in memory costs roughly 32 bytes per numeric cell; the caps above are not yet profiled.

## Unvalidated assumptions

- The caps are adequate for the intended desktop use (no profiling yet; only the timings in EXP-001 exist).
- Behaviour on CSV files exported by specific spreadsheet programs has not been surveyed.

## References

REF-008
