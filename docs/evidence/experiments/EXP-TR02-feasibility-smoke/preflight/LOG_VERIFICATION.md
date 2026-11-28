# Verification of the M5 no-weight preflight logs (two runs)

Status: **evidence, no training, no weights.** The owner-authorized review of PR #13 asked that the transcribed preflight record
be checked against the original raw logs and that sanitized copies with hashes be committed. This file records what was
checked and how to verify it.

## How the evidence was obtained
- Each preflight ran once on the owner's MacBook Air M5 from a Remote Control session started from this thread. Each command
  wrote its own log on the device (a `CMD:` line, a `START:` UTC line, the output, the full `/usr/bin/time -l` block and
  `EXIT_STATUS`). The **original logs stay on the device**, unedited, under a work directory outside every repository; they are
  not committed because they contain the home directory path and the Homebrew Python path.
- The SHA-256 of every original was measured on the device twice (identical both times); the inventories are below.
- Sanitized copies were made on the device with a fixed rule set: home directory prefix to `$HOME`, any other absolute local
  path to `<local-path>`, device names to `<device>`, a standalone account name to `<user>`, credential-like tokens to
  `<redacted-credential>`. No device name, account name or credential was present in any log (those counts are 0); only home
  directory prefixes and one Homebrew Python path were replaced. The device wrote a `REDACTIONS.json` with the original and
  sanitized hashes, byte counts and redaction counts.
- The sanitized bytes reached this repository as base64 text; every file was decoded here and its SHA-256 and byte count
  compared with `REDACTIONS.json`. All files of both runs matched. A transcription typo in the base64 transport of one file
  (run 2, `03_verify_model.log`) was caught by this check and the file was re-sent (the committed copy matches the device hash). The committed files are
  therefore byte-identical to the sanitized files on the device. `.gitattributes` marks this directory `-text` so Windows
  checkouts keep the same bytes.
- What is **observed raw evidence**: every value in the sanitized logs. What is **not** raw evidence: anything not in these
  files (listed per run below); it was reported by the device session from live terminal output and is a transcription.

## Run 1: head 96cbdf7 (`at-96cbdf7-sanitized-logs/`)
`REDACTIONS.json` SHA-256 `69dcbfabaf83b52f25d321742ff7257cb82e3ed4841c1e48a290fb930613045d`.

| File | Original SHA-256 | Original bytes | Sanitized SHA-256 | Sanitized bytes | Redactions (home prefix / other path) |
| --- | --- | ---: | --- | ---: | --- |
| `01_export.log` | `ecf54d21333a29e43785cf46e8366bc9cc3800f72f53010adbdd54141bfed58b` | 1252 | `3536e412bd41750666a6c5510e1bba16db6f5ac23643c72297d05c6b54a43d40` | 1212 | 1 / 1 |
| `02_verify_inputs.log` | `25ebccb950c101b24d25315f6707fd3101939f68dd5658c1d8502a087fcdd465` | 983 | `2406174586290d6664dec073259ec8fad158d7b6a39c5e52267976f2f97ba057` | 975 | 1 / 0 |
| `03_verify_model.log` | `cd9edef73780ffdc19327454b1e652001928125be498fb1bf463080a7ce281eb` | 982 | `23892f080bfd3519d36eff9519fc854e51888e24a2daca0cb54485e6fdf77eae` | 974 | 1 / 0 |
| `04_check_data.log` | `349a17b1068af4bdf29aeae4426a03ed6015d45d9c89574c71cb177ba833a1a7` | 1988 | `e92b0a6a1b963bce2940b2ff084172aa1a52e93b7b4f61ad0b88aa98ffe07d5e` | 1972 | 2 / 0 |
| `05_check_iterator_probe.log` | `127a433e976a73b7cf4a4a3027f1f514d3ba495e9c78ae5a3f86d4ebfd088f74` | 1967 | `aab09daee53c6977d4efe73cea2eebd339857ef2ad76369246b19dc8bcb12364` | 1951 | 2 / 0 |
| `05_pressure.txt` | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | 0 / 0 |
| `06_check_iterator_smoke.log` | `373a0355ee2f2157785cd430239f1476dd237f697263878490be2e49f90c1357` | 1968 | `4cd61ddc22185dae897df4baf7777cc240262603aaef7a0199be342fa70184f9` | 1952 | 2 / 0 |
| `06_pressure.txt` | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | 0 / 0 |
| `07_preflight_probe.log` | `1f8c4cfb185d0588b3b3098ebc36b0410e1898866b42a1d8677918a5bedf2cfd` | 1624 | `dc91f921215fd5cebe97e811fe036ad6e1178fd820f8edc732f64eba26dbadb7` | 1600 | 3 / 0 |

Verification result: the raw logs match the earlier transcribed record (`PREFLIGHT_NOWEIGHT.md`) in every step: exit status 0,
wall time, maximum resident set size, the full JSON of steps 2 to 7, the per-file supervised token statistics, all eight mask
samples for probe and smoke, `problem_count` 0, and the step 7 record (commit, mlx 0.32.4, mlx-lm 0.32.2, python 3.12, offline,
manifest and per-example hashes, model revision, `after_complete_sha256` empty). No discrepancy in a number was found. Things
the first record left out and the logs show: step 1's venv-creation lines from uv (and that its 16.14 s includes creating the
virtual environment), the peak memory footprint (17.6 MB for step 1, about 10 MB for steps 2 to 7, much smaller than the maximum
resident set size) and 0 swaps in every step; step 6's terminal output was discarded and exists only in its log; the tool wrapper
showed "Exit code 1" for step 7 only because of a trailing `ls` of the not-created output directory, while the step itself
logged `EXIT_STATUS: 0`.

Not present in any raw log of run 1 (reported from live terminal output, not independently preserved): the AC power reading
and initial memory pressure; the empty `ps` check; macOS, hardware model, uv version and date; the MLX device line; the
worktree HEAD and clean-tree confirmation; the checkpoint provenance revision; the check that the output directory did not
exist; the `uv sync` output; and the SHA-256 of `ROW_IDS.json` (re-measured on the device afterwards and equal to the reported
value). No device-information file was saved in run 1.

## Run 2: head d5f6cd5 (`at-d5f6cd5-sanitized-logs/`)
Run once after the licensing revision. `REDACTIONS.json` SHA-256 `42fe43aa9155d02fcc50472395f8e5a5f57950549587b9bdf6e23b62469a008e`. Run 2 also saved the safety checks and the export
file hashes in `00_environment.txt`, and logged `uv sync` in `00b_uv_sync.log`, closing most gaps listed for run 1.

| File | Original SHA-256 | Original bytes | Sanitized SHA-256 | Sanitized bytes | Redactions (home prefix / other path) |
| --- | --- | ---: | --- | ---: | --- |
| `00_environment.txt` | `d322335af87c516a71149a51ca7080b0d29e13443cba5db2b0c200a1b5b73910` | 1711 | `d322335af87c516a71149a51ca7080b0d29e13443cba5db2b0c200a1b5b73910` | 1711 | 0 / 0 |
| `00b_uv_sync.log` | `ca5491cf77b162a58fb75c0141f79ed2ef9f54658502aaf9f98e5e86580975e7` | 2205 | `c27ced7efe80f63ee64fdc6024c8525e7e61d4e347710a67e25d303939a406bd` | 2149 | 3 / 1 |
| `01_export.log` | `b15a798358288f7ea77db5904ba1f0e7c5aa038c938b80be00e3286e1795fee4` | 1254 | `cae92986b4cafc7aa4970efc58af7a7119bfe700601aa37d683720a0b7fa531e` | 1214 | 1 / 1 |
| `02_verify_inputs.log` | `ffb0fb329836f5134a8707d04998d712a2e78fd8b29e5336887ae08912b74c1e` | 985 | `cd757d51f30f37e516bd8891ca7bb4a685d95fc3efe7d009a06111177ae7be0c` | 977 | 1 / 0 |
| `03_verify_model.log` | `ee3ce563bc4df5a0398b17d80b7be0681c21ac11b7d4929de9d7671ca086d7ef` | 982 | `7827fad69ca95ec1a54ce1723629ae7724feb158c568bfa155d3419f7b76018a` | 974 | 1 / 0 |
| `04_check_data.log` | `c6064c8bc4d4c1d75216613ac03ece4af034b5f0684dcd4b85ce712d6394b6dd` | 1990 | `ff5f0260630893fc1c5ec5e1c91cf227b04f2c163f9e98ffeca68613a78b9a36` | 1974 | 2 / 0 |
| `05_check_iterator_probe.log` | `35456dd7307729deabdb7c11f6bbf11599006487fd64e9cab602a5dbfe42e765` | 1969 | `e4b142fc9aa30f93bed35dce4581187daaa6fb1b6d84723131115df97f82b8cd` | 1953 | 2 / 0 |
| `05_pressure.txt` | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | 0 / 0 |
| `06_check_iterator_smoke.log` | `7f0afc251d86d86fd4e5d3fdaa50ef8986f9331e4092ac9d1798820e862c9e9f` | 1970 | `2add95a0db158c405ab908c7e7d2e972590433ce41146b2876e4cac80891dcd0` | 1954 | 2 / 0 |
| `06_pressure.txt` | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | `37bc57022bf6242b990b4a539316849e4d34a944b21156b7e91ba6b4c571f8ee` | 37 | 0 / 0 |
| `07_preflight_probe.log` | `1b919fbc597597d38653d9a8dd92515b8fa4b6f208be76e197b07d6fea138fd9` | 1628 | `acf3c473835eae3cf80a31b1f2a8ff1757272e29413f0bda47766cda2fcdd941` | 1604 | 3 / 0 |

Not in any raw log of run 2: the end-of-run worktree status and HEAD check (empty, HEAD unchanged), the final process count,
the check that the probe output directory was not created, and the programmatic comparison of the export hashes with the
expected values (done correctly on the second attempt after a scripting slip in the first comparison; the expected values
also match the hash list in `00_environment.txt`).

## Limits
Hashes prove that these files are the bytes that were reviewed, not that the device did what the logs say; the logs are
written by the commands themselves on the owner's machine and the original files remain there for any further check. The
logs do not contain GPU memory or training measurements: no model was loaded.
