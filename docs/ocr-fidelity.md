> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# OCR fidelity correction — investigation, October 2026

The historical checkpoint 4796 shows hallucinated continuations, unwanted Arabic
on French pages, Arabic-shaped digit conversion, and special-character damage.
The original base model without the learned LoRA updates performs better on
several identical diagnostic inputs. A stricter prompt alone does not fix this.

The new `scripts/infer_ocr_fidelity.py` preserves raw text, uses literal-copy
instructions, checks adapter/base provenance, and records explicit review flags
for loops, unexpected Arabic on monolingual pages, and interrupted generations.
It returns exit code 2 for flagged output. An EOS or an unflagged response does
not prove image grounding. Uniform near-white blank pages are handled before
resizing; faint marks are retained. Digits and symbols are never post-converted.

Run `python scripts/infer_ocr_fidelity.py --check` to inspect installed packages
without installation, network access, or model loading. Actual inference requires
a local verified original base, FP32 LoRA adapter, its snapshot metadata, and the
verified `base-files.json`. Use `--language mixed` for genuinely mixed pages.
All raw predictions are saved separately from the JSON audit.

The corrective corpus contains 300 training and 75 validation examples, exactly
100/25 per language (Arabic, French, English). It combines 299 public real pages,
61 deterministic synthetic literal controls, and 15 synthetic blank pages.
The old 31-image comparison documents are excluded. Real pages are split by
document/article. Publisher annotations with corrupted text, repeated long
paragraphs, or a manually observed missing stamp are quarantined. Publisher
transcriptions have not all been independently verified line by line.

Sources are NOD/Yarmouk (CC BY 4.0) and NayanaOCRBench_Natural (CC BY-NC 4.0).
The Nayana noncommercial restriction is retained in the corpus manifest. Source
rights and SHA-256 hashes accompany the images and exact Unicode references.

The bounded training script prepares a **new** LoRA from the verified original
base, rather than inheriting the regressed adapter. It retains context 8192,
rank 16, alpha 32, FP32 adapter computation, and a BF16 base. It checks zero-update
identity, real gradients, finite parameters, validation loss by language, and
image-only generations. No historical checkpoint is overwritten and no candidate
is automatically deployed.

As of 2026-10-28 the filtered corpus and training code are prepared locally.
The first HPC connection times out; corrective training has **not** started.
The filtered corpus still needs its remote CPU encoder/EOS check and the new
candidate needs generation validation. No corrected weights or universal
hallucination-free claim are supplied at this stage.
