> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Complete table targets and honest OCR evaluation

The raw OCR annotations contain `markdown`, `tables` and `blocks`. A local link such
as `[tbl-0.html](tbl-0.html)` names a table in the same annotation. Training only on
the Markdown link omits the cells from the supervised answer.

## What this branch changes

- `annotation_targets.py` replaces each internal table reference at its position
  with that table's complete, unchanged HTML. It preserves headers, cells,
  `rowspan`/`colspan`, surrounding text and Markdown reading order.
- `blocks` checks content identity and table reading indices; blocks are not
  appended a second time. Distinct tables with identical contents are supported.
- Missing IDs, duplicate IDs/references, unplaced tables, conflicting block order
  and malformed HTML are quarantined. They are never silently skipped.
- The original JSON/images are unchanged. The v2 preparer uses a new index,
  evidence directory and MinIO prefixes. The v1 files and checkpoint are retained.
- Split membership is hashed from the **original** Markdown, exactly as in v1.
  Hashing the expanded answer would move some previously trained pages into val.
- The training encoder rejects unresolved table references. V2 rows use an
  explicit inline-table OCR prompt; the 8192 combined training-token limit, LoRA,
  optimizer and batch settings have not been increased.

## Create the isolated manifests (not yet executed on the cluster)

Use the existing project runtime/credentials on the VM or in an owned CPU job:

```sh
python scripts/prepare_dataset_tables.py --limit 1
# Inspect table audit/rejections and token lengths before the complete preparation.
python scripts/prepare_dataset_tables.py
```

Expected new locations, created only when that preparation is actually run:

```text
s3://qwen35-9b-ocr-finetuning/dataset/processed/ocr-manifest-v2-tables/
s3://qwen35-9b-ocr-finetuning/dataset/manifests/ocr-manifest-v2-tables/COMPLETE.json
<project>/preparation/ocr-manifest-v2-tables/index.sqlite
<project>/evidence/ocr-manifest-v2-tables/
```

The COMPLETE marker is written only after every shard has been processed and each
derived upload verified. Rejected records/counts are reported; do not reuse the
old hard-coded train count if v2 quarantines additional records.

For staging metadata and refreshing the same held-out page IDs:

```sh
python scripts/stage_training.py --manifest-version ocr-manifest-v2-tables
export Q35OCR_DATA_DIR=/work/q35ocr-20260921-p560/data-tables-v2
export Q35OCR_VALIDATION_FILE=/work/q35ocr-20260921-p560/data-tables-v2/validation.jsonl
```

`refresh_table_selection.py` joins the old selection by exact ID, verifies the
unchanged image hash/group/split and writes a new selection, preserving its order.
It refuses a missing/changed page. The old 256-page file is not overwritten.

## The existing checkpoint has NOT been corrected by editing labels

The completed checkpoint 14355 still has its existing weights. This change is not
proof that it now reads complete tables. The proposed first job is a bounded
diagnostic of that checkpoint: one **unused** GPU, at most four supplied pages,
300 seconds/page, greedy generation, stopping on `im_end` or `endoftext`.
The inference script refuses a GPU with a compute process and never trains.

After inspecting this baseline, a separately named, bounded correction phase can
start from the final adapter with complete training/validation targets. Validate
all combined sequence lengths before taking GPUs; reject/review oversize pages,
never silently truncate cells or raise the training context limit. This is a new
objective/phase, not an exact continuation of the old full-stream cursor.
Do not run the historical full-data or resume launchers blindly on the v2 data.
No full-scale correction training or deployment is launched by this branch.

## Diagnostic inputs and scorer

```sh
python scripts/prepare_table_examples.py --annotations /path/sample_0009976.json \
  /path/sample_0009947.json /path/sample_0009924.json --output-dir /path/new-examples
```

`inference-inputs.json` contains only IDs, images, hashes, dimensions and the common
prompt. `references.json` is separate and is not read by the GPU inference script.
Images without JSON may be transcribed but must not receive reference-based scores.
These source samples are **not certified independent of previous training**.

```sh
python scripts/test_table_checkpoint.py \
  --model-dir /work/q35ocr-20260921-p560/model \
  --checkpoint /work/q35ocr-20260921-p560/training-fast-gpu4-m3-from12995-v1/checkpoint-14355.pt \
  --checkpoint-sha256 e32116d77a324a8b970503e23440dbc83434a21b171a2ec3bcb87d14e6c503e6 \
  --inputs /path/new-examples/inference-inputs.json --output-dir /path/new-diagnostic
```

The script preserves original PNG resolution (the saved processor may resize),
uses the same custom FP32 LoRA forward path, verifies every adapter tensor and
records limits/EOS. The checkpoint/PVC is read-only. A GPU job should have a
30-minute deadline, backoffLimit 0, a single scheduler-chosen GPU, priority 0 and
separate output storage. Never free someone else's GPU or resume an old trainer.

```sh
python -m pip install -r requirements-evaluation.txt
python scripts/score_table_ocr.py --annotation /path/sample_0009976.json \
  --prediction /path/new-diagnostic/predictions/sample_0009976.md \
  --output /path/new-diagnostic/score-sample_0009976.json
```

The scorer resolves tables only from the **reference's own annotation**. It never
fills a model's predicted placeholders with ground-truth HTML. A structured model
prediction can be scored with `--prediction-json`; only its own `tables` is used.

Reported values: whole-page normalized CER/WER; reference/predicted table and cell
counts; exact structural matches; cell CER/WER in strict table/row/cell order;
unresolved links and parse errors. Missing cells are deletions, extra cells are
insertions; repetitions/rates above 100% are not clipped. Invalid predicted HTML
has an explicit error and no table rate, rather than a made-up score. These are
**not TEDS** and do not perform optimal table matching across reordered tables.

## Inspect exact historical training examples without GPUs

`inspect_training_tables.py` scans a bounded number of original manifests from
the existing PVC `data/COMPLETE.json`, checks their SHA-256 against MinIO, and
can verify that the final checkpoint records this exact dataset hash. It looks
for internal table links, inline HTML and a multi-line pipe heuristic. Pipes
alone are not evidence that an annotation contains a table.

Selected original annotations/images are retrieved from the raw TAR with exact
S3 ranges, requiring valid TAR checksums and exact member IDs. Images must match
the training manifest's hash, and raw Markdown must match its supervised target.
The script emits IDs, groups, splits, source keys, table/cell counts and separate
old/corrected targets. It never modifies the source bucket, manifests or weights.

```sh
python scripts/inspect_training_tables.py \
  --complete-file /work/q35ocr-20260921-p560/data/COMPLETE.json \
  --output-dir /inspection-out/new-audit --max-shards 1 --max-examples 10
```

The scan is a diagnostic, not a dataset-wide census or a correction training set.
Keep validation/test pages out of any training subset. Preserve original and
augmented group membership when making the separately versioned v2 labels.

## Local verification

```sh
PYTHONPATH=scripts:tests python -m unittest test_annotation_targets test_table_scoring \
  test_prepare_table_targets_integration test_refresh_table_selection
```

Synthetic identity/missing-table cases test the code, not model accuracy. The
provided real JSONs are processed in a private output directory and not committed.
