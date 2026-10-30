> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Dataset

## Pinned source

- Hugging Face dataset: `OCR-Data/ocr_data`
- Revision: `4cf05556e21bc54aca54a8b1f81ce172f3ef0477`
- Inventory: 427,825,934,227 bytes across 414 files, including 103 original and 309 augmented TAR shards
- The training split used in the saved October 2026 run: 3,683,080 examples

The original JSON `markdown` field supplies the OCR target, including Arabic text, diacritics and mathematical notation. The source also carries layout and language metadata. Raw dataset archives and prepared images are stored outside this repository.

## Preparation and integrity

`scripts/prepare_dataset.py` reads the pinned source manifest, inspects TAR members, validates image/annotation pairs and uses image and annotation hashes to preserve integrity. Project storage contains the raw and prepared data; neither the files nor the object credentials are committed here.

The trainer rejects sequences longer than 8,192 combined tokens instead of silently truncating targets. It preserves aspect ratio while resizing images to fit within 1,024 × 1,024. These are the recorded run settings, not evidence that all small text remains legible. Measure CER/WER and page completeness on held-out documents before making OCR-quality claims.

Keep documents and augmented variants grouped in the same split. Do not place raw source annotations, customer documents or OCR images in GitHub. The numerical metric tables in `../metrics/` contain no source text or image content.
