> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Complete Markdown tables: original + aug1 corpus

This version replaces the internal JSON table references with real Markdown pipe
tables at their page positions. The raw TARs, JSON, images, historical labels and
checkpoints remain unchanged. The earlier `fix/table-targets-v2` branch retains
the inline-HTML approach; this branch is `fix/markdown-tables-orig-aug1`.

## Exact source selection and layout

Pinned dataset revision: `4cf05556e21bc54aca54a8b1f81ce172f3ef0477`.
Select all 103 original TARs and all 103 `aug1` TARs, excluding `aug2` and `aug3`.
The historical integrity checks accepted 1,019,145 originals and 1,019,145 aug1
pages. Their unchanged splits contain **1,841,540 training**, **99,318 validation**
and **97,432 test** records. The 660 historical rejected views remain accounted
for; this is not an arbitrary exactly-one-million sample cap.

New MinIO bucket/prefix:

```
s3://qwen35-9b-ocr-finetuning/dataset/processed/ocr-markdown-v3-orig-aug1/
  CONFIG.json
  originals/shard_001/sample_0009924.md
  aug1/shard_001/sample_0009924_01.md
  manifests/data__shard_001.jsonl
  manifests/data_aug__shard_001_aug1.jsonl
  audits/data__shard_001.jsonl
  reports/data__shard_001.json
  pilot/result.json
  failures/
  COMPLETE.json
```

The `.md` objects are real complete page targets. Training manifests embed those
same SHA-256-verified texts and point to the original image bytes by TAR range.
Audits retain the raw annotation address, hash, table IDs, cell text, span geometry
and conversion policy. No full image dataset is duplicated on the PVC or moved
back through the laptop VPN. Each output `.md`, manifest and audit is read back
from MinIO before its shard is marked complete.

## Conversion and validation

- Resolve each reference against that annotation's `tables` by ID. Reject missing,
  duplicate, malformed or unplaced table entries; never silently omit a cell.
- Preserve source DOM column order. RTL preview uses `direction: rtl`.
- Markdown cannot represent merged-cell geometry. Write each source cell text once
  at its anchor, leave covered grid slots empty, and retain spans in the audit.
- Use an actual header only when the first row contains header cells. Otherwise
  add an empty Markdown header and keep every source data row.
- Escape literal pipes, Markdown punctuation and HTML characters in cell text.
  Normalize only cell whitespace; preserve Arabic letters, digits and diacritics.
- Render every generated table with independent Markdown-it and compare every
  cell against the audited rectangular grid. Check all original/aug1 target and
  split pairs. Preserve v1 splits rather than hashing the new targets.
- Run the actual model processor on every selected page. Retain full targets;
  never truncate to the unchanged combined 8,192-token context. Any oversize page
  blocks training readiness even though its complete `.md` has been saved.
- Four indexed CPU workers resume only at completed shard boundaries with a pinned
  code/configuration hash. Incomplete or failed preparation cannot launch training.

## Corrected training phase

Initialize adapter weights from the verified completed checkpoint **14355**,
SHA `e32116d77a324a8b970503e23440dbc83434a21b171a2ec3bcb87d14e6c503e6`.
The previous job's dataset cursor is at the end of its old pass: start a **new full
deterministic pass** of the corrected training split with a fresh AdamW optimizer.
The original checkpoint including its optimizer/cursor is preserved.

| Argument | Corrected phase |
|---|---|
| GPUs / strategy | Four unused H200 by default; explicit `--world-size 1` supports one unused H200, PyTorch DDP |
| Microbatch / accumulation / global batch | 1 / 96 / 384 on four GPUs; 1 / 384 / 384 on one GPU |
| Context / image thumbnail | 8,192 combined tokens / 1,024 pixels |
| LoRA rank / alpha / trainable parameters | 16 / 32 / 29,097,984 |
| Base / adapter computation | BF16 SDPA / original FP32 custom LoRA |
| AdamW / weight decay / clip | Fresh state / 0.01 / 1.0 |
| LR / warmup | 100 updates to 2e-5, cosine to 1e-5 |
| Coverage | One pass of all 1,841,540 selected training views |
| Monitoring loss | 128 held-out original table pages + 128 non-table pages |
| Checkpoints | First update, then 250 updates or one hour, final/stop |
| Safety | Four-GiB disk reserve, STOP, coordinated nonfinite checks, validation degradation |

The microbatch starts at one because supervision is now longer; the previous
global batch is preserved. No unmeasured BF16 LoRA or larger microbatch change is
mixed into the data correction. Intermediate DDP microsteps use `no_sync()`;
data readiness and finite gradients are checked collectively per update.

Before real training, every requested rank must pass a real update, finite gradients,
identical adapters, actual FLA/conv execution, SHA-verified PVC/S3 checkpoint and
exact checkpoint replay (parameter and loss errors below 1e-5, same cursor). The
gate is rolled back before the new pass. Kernels remain isolated: Triton 3.7.1,
FLA core 0.5.2, causal-conv1d 1.7.0. No base image modification.

The same eight held-out table pages and four controls are generated before and
after training with image-only inputs. Raw predictions, EOS status, normalized
CER/WER and strict canonical Markdown cell errors are saved separately. Predicted
file links are never filled with ground-truth cells. Strict slot metrics are not
TEDS; Markdown span loss and synthetic headers affect structure interpretation.
Lower loss alone does not establish that all table problems are fixed.

Outputs use the separate PVC directory and S3 prefix
`training-markdown-v3-orig-aug1/`. All checkpoints are retained and S3-verified.
GPU telemetry is recorded every five seconds. A failed training job is not
automatically restarted. The VM controller only runs the conversion dependencies
and submits training once; it is not a recurring Codex heartbeat.
