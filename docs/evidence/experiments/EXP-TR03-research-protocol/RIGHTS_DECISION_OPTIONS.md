# EXP-TR03 rights: evidence and decision options for the owner (not a decision, not legal advice)

Status: **prepared for the owner; no decision is recorded or implied.** The owner records decisions in
`ledgers/RIGHTS_DECISIONS.csv` (header and rows exactly as in `RIGHTS_WORKLIST.csv`: `owner_decision` is `include` or
`exclude`, with `decided_by` the owner's roster code, `decided_on` an ISO date and, for `include`, a `notes` text saying how the
caveat is addressed). The checker refuses `include` for SRC-O03, SRC-O04, SRC-O05 and external-adapted records, which stay
excluded by rule unless a prospective amendment says otherwise. Only table and column **names** of these sources reach the
training text; no observations (rows of data) do. Evidence below is quoted from
`data/axbench-tr/sources/LICENCE_EVIDENCE_ADDENDUM_1.json` (retrieved 2026-11-27); pages can change.

| Source | Layouts in eligible rows | What the repository holds | Open caveat | Options for the owner |
|--------|--------------------------|---------------------------|-------------|-----------------------|
| SRC-O01 World Bank WDI (Türkiye and 14 economies) | 1,773 train rows (34.8%) and 865 validation rows (67.9%); 176 of the 518 sampled rows; the top table family `worldbank/train` is all of it | AD-01: dataset page labels CC BY 4.0; AD-02: CC BY 4.0 is the catalogue default; AD-03: site terms default to non-commercial "unless otherwise stated". Attribution required. | Several indicators are compiled from IMF, ILO, UN, OECD or Eurostat; their upstream terms were **not** read, so use rests on the WDI dataset label | (a) include, citing AD-01 and recording the upstream caveat as accepted; (b) include only schemas whose indicators are World Bank's own (needs a per-indicator list; not prepared); (c) exclude, which removes 1,773 train and 865 validation rows (two thirds of validation) and the WDI economic layouts |
| SRC-O02 Eurostat (Türkiye: unemployment, HICP, GDP) | 105 train rows (2.1%) and 27 validation rows (2.1%); 15 of the 518 sampled rows; 3 datasets | AD-05: reuse for commercial or non-commercial purposes authorised with acknowledgement; AD-04: the commercial-use exception list names countries other than EU, EFTA and official acceding and candidate countries; Türkiye is a candidate country, so not in that list | no dataset-specific copyright note was found for the three datasets; the exception list could change; only names reach examples | (a) include with the Eurostat acknowledgement; (b) exclude, which loses the smallest source (2.1% of rows), little volume but a few Türkiye economic series |
| SRC-O06 İzmir Metropolitan Municipality open data (CKAN) | 3,213 train rows (63.1%) and 382 validation rows (30.0%); 327 of the 518 sampled rows | AD-06: licence page says all materials CC BY 4.0 unless stated otherwise (commercial use included, attribution required); 93 packages carry the municipality licence id (92) or `cc-by` (1) | per-package free-text notes were **not** read for exceptions | (a) include; (b) include after someone reads the free-text notes of the packages used (about 93; not done); (c) exclude, which removes 3,213 train rows (63.1%) and leaves the corpus with SRC-O01 and SRC-O02 layouts only |
| SRC-O03 ECB reference rates, SRC-O04 OECD SDMX, SRC-O05 TCMB daily rates | 309 train and 26 validation rows | nothing (terms not verified in this session) | terms unverified | keep excluded (default). Including any would need terms read and recorded as new evidence first, a protocol amendment, and a new frozen worklist |
| external adapted records (Cognipeer and other upstream sources) | 473 train and 96 validation rows | see `DATA_SOURCES.md`: upstream generation terms unresolved, SRC-N03 conflict, unverified translations | unresolved | keep excluded (default) |

Attribution to preserve if anything is used (source, publisher, licence id, evidence ids) is in `INVENTORY.json` >
`source_attribution`. Excluding a source after the review is allowed: its rows are removed from the training and validation
lists by the derivation (`research_manifest`) and the frozen review sample and its denominators do not change. The review
sample was not stratified by source; if the owner excludes SRC-O02, its 15 sampled rows were still reviewed.

What I did not do: read any provider's terms beyond the addendum, check whether CC BY 4.0 on a dataset page covers every
indicator, assess whether training a model is a permitted reuse of any of these licences, or consider any non-English
jurisdiction. Counsel can be asked those; the checker only records the owner's typed decision.
