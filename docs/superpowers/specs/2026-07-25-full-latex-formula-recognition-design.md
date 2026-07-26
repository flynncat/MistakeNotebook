# Full LaTeX Formula Recognition Design

Date: 2026-07-25
Status: Approved in conversation; pending implementation

## 1. Goal

Add local recognition for common printed mathematical notation, from elementary fractions through calculus and higher-mathematics expressions. The system must preserve formula position, export editable LaTeX/Word equations, and never replace an uncertain formula with an unverified guess.

The selected recognition tier is UniMERNet Tiny (about 441 MB) to prioritize interactive performance on the user's M4 Pro. A manual formula editor with live preview compensates for the accuracy difference from the Base checkpoint.

## 2. Supported Scope

The recognition path targets printed:

- fractions and nested fractions;
- powers, subscripts, roots, and grouped expressions;
- sums, products, limits, derivatives, and integrals;
- relations, arrows, Greek letters, and common operators;
- matrices, determinants, aligned expressions, and piecewise functions.

"Full LaTeX" means the model may emit the broad mathematical syntax present in its tokenizer and training data. It does not mean arbitrary executable TeX, custom packages, document commands, or guaranteed perfect recognition. Unsupported or unsafe commands are rejected for preview/export and retain the source formula crop.

Handwritten formulas are outside this iteration.

## 3. Root Cause

The current macOS Vision OCR is a text OCR engine. On `_DSC6580.JPG` it reports the stacked fractions as `3`, `?`, `8`, and `?o1`; denominators and two-dimensional structure are lost.

The current `content_blocks.py` is not formula OCR. It only converts already-linearized strings such as `2ù10` into a small LaTeX subset. Therefore a dedicated mathematical formula detection and recognition stage is required before text reconstruction.

## 4. Architecture

Introduce an isolated formula pipeline with four units:

1. `FormulaDetector`
   - Locates inline and display formula boxes on the normalized high-resolution question image.
   - Uses Pix2Text MFD 1.5 ONNX from `breezedeus/pix2text-mfd-1.5` at Hugging Face revision `f470a885e0fca1d3d2bfa2a54991db7ae01f1861` (MIT, 80,311,115 repository bytes).
   - Uses score threshold `0.55`, selected by the `_DSC6580.JPG` benchmark to remove two logo/footer false positives while retaining all four fractions; NMS IoU is `0.45`, and boxes are clipped to image bounds.
   - Runs once on the oriented full page before page segmentation. Split boundaries that intersect a detected formula are moved to the nearest whitespace outside the box. It runs again on each normalized child crop to obtain final child-relative boxes.
   - Returns pixel boxes and detector confidence.

2. `FormulaRecognizer`
   - Uses UniMERNet source revision `5a2c80d96b1d2dba447ff18d873e5fb73ba03c35` and `wanderkid/unimernet_tiny` revision `3f09ac4b1cd583be47ea20a7d7daef839473028a` (Apache-2.0, 430,075,701 repository bytes).
   - Runs two deterministic image variants: normalized grayscale and high-contrast grayscale.
   - Returns LaTeX, token count, per-token selected probability, EOS/UNK status, elapsed time, and agreement status.
   - Loads lazily and remains warm for the process lifetime.

3. `FormulaMerger`
   - Removes generic OCR fragments overlapping formula boxes.
   - Merges text and formulas by baseline and horizontal position.
   - Produces ordered version-2 content blocks without embedding sentinel strings into the canonical OCR text.
   - Every formula receives a deterministic `formula_id` from the normalized crop SHA-256 plus its top-left child-relative pixel coordinate.
   - Every block records top-left-origin child-image pixel coordinates. Formula blocks additionally record `display`, `row_index`, and `baseline_y`.
   - Components belong to one row when their vertical overlap is at least 50% of the smaller height or their centers differ by at most 0.6 times the median OCR text height. Rows sort by `(top, left)` and components within a row sort by `(left, top, formula_id)`. A formula is display math only when it forms its own row or is taller than 2.5 times the row's median text height.

4. `FormulaValidator`
   - Normalizes harmless LaTeX variations.
   - Rejects document-level commands, file/network access, macro definitions, and unknown unsafe commands.
   - Requires syntactic conversion to MathML.
   - Accepts automatic output only when both image variants normalize to the same expression, EOS is reached, no UNK token appears, mean selected-token probability is at least `0.80`, fifth-percentile token probability is at least `0.35`, and syntax validation succeeds.
   - Any failed condition marks the formula `needs_review` and uses the clean formula crop as the faithful display fallback.

Formula recognition is independent of problem classification and existing text dual-OCR diagnostics.

## 5. Model Management

- Pin the exact source/model revisions above in `models/v2_manifest.json`. The setup script calculates artifact SHA-256 and refuses startup use when the installed hash differs from the manifest generated during the first verified download.
- Record licenses, byte sizes, and SHA-256 checksums.
- Extend the existing model setup script rather than downloading weights during a web request.
- The service must start without the model. When missing, formula regions remain image blocks and the UI reports an actionable installation message.
- First attempt the pinned adapters in the existing Python 3.11 environment. If dependency resolution conflicts with the existing app, create `<root>/.formula-venv` and a supervised JSON-lines worker that loads both models once, enforces a 120-second recognition timeout, restarts once after a crash, and never mutates the main environment. Do not downgrade existing application dependencies.
- Benchmark warm latency, cold load time, peak memory, and MPS/CPU correctness on `_DSC6580.JPG`.

## 6. Content Blocks Version 2

Each formula block stores:

```json
{
  "formula_id": "formula-a1b2c3d4-412-96",
  "type": "latex",
  "latex": "\\frac{3}{5}",
  "model_latex": "\\frac{3}{5}",
  "source_text": "3/5",
  "source_box": [x1, y1, x2, y2],
  "original_crop_asset": "formula-01-original.png",
  "clean_crop_asset": "formula-01-clean.png",
  "detector_confidence": 0.98,
  "mean_token_probability": 0.94,
  "p05_token_probability": 0.81,
  "eos_reached": true,
  "display": false,
  "row_index": 2,
  "baseline_y": 318,
  "recognition_state": "auto_verified",
  "recognizer": "unimernet-tiny",
  "edited_at": null
}
```

Allowed states:

- `auto_verified`: both deterministic passes agree and validation succeeds.
- `needs_review`: disagreement, invalid syntax, truncation, or missing end token.
- `human_verified`: user saved a valid edit.
- `image_fallback`: no safe LaTeX is available.

Version-1 blocks remain readable. Reprocessing upgrades a problem to version 2. No destructive bulk migration is required.

All existing review/accept/reconstruction write paths obey these preservation rules:

- If OCR text is unchanged, version-2 formula blocks and `human_verified` edits are retained byte-for-byte.
- If text or source imagery changes, the operation must explicitly invalidate affected formulas and rerun formula recognition; it may not silently call the version-1 block builder.
- Explicit full reprocessing archives the previous version-2 JSON in problem metrics before replacement.

## 7. Reconstruction and Ordering

- Formula boxes are cropped from the normalized source before handwriting removal can damage thin bars or symbols. Both the original normalized crop and the white-background clean crop are retained.
- Crops are background-normalized and binarized onto white, preserving printed pixels without paper color.
- Text OCR observations that substantially overlap a formula box are excluded from canonical text.
- Ordered blocks use geometric reading order, including formulas embedded between two text fragments on one line.
- The white-background `question.png` uses the cleaned crop for formulas, so visual fidelity does not depend on LaTeX rendering.
- Markdown and DOCX consume the LaTeX value when verified; otherwise they embed the clean formula crop.

## 8. Manual Formula Review

Each problem card gains a formula review section:

- clean source crop;
- original normalized formula crop or an explicit original-image link;
- editable LaTeX field;
- debounced live preview;
- validation/error message;
- reset-to-model action;
- per-formula state badge.

Preview uses a local backend conversion to sanitized MathML and native browser MathML rendering. It does not use a CDN or execute TeX.

Saving formula edits:

- validates all submitted expressions;
- updates version-2 content blocks atomically;
- sets edited formulas to `human_verified`;
- refreshes Markdown/DOCX output immediately;
- retains `model_latex` so reset-to-model remains available;
- executes a single SQLite transaction with `UPDATE problems SET ... WHERE id=? AND updated_at=?`;
- returns HTTP 409 when the compare-and-swap affects zero rows;
- serializes against review acceptance and reprocessing with the existing storage lock. Review/reprocessing also changes `updated_at`, so a stale formula save cannot overwrite newer content.

A problem with any `needs_review` formula cannot be automatically accepted.

## 9. Exports

Markdown:

- verified formulas render as inline `$...$` or display `$$...$$`;
- unverified formulas render as image attachments.

DOCX:

- use `latex2mathml==3.81.0` (MIT) followed by `mathml-to-omml==1.0.3` (MIT) to produce native OMML;
- support fractions, scripts, radicals, n-ary operators, matrices, and piecewise structures;
- never invoke a TeX shell;
- test the converter against fractions, scripts, radicals, n-ary limits, matrices, determinants, `cases`, and `aligned`;
- if conversion fails, insert `clean_crop_asset` and add the formula ID and reason to export metadata and the API response's `formula_fallbacks` list.

The existing deterministic simple-formula converter remains only as a compatibility fallback.

## 10. API and Storage

Use existing `content_blocks_json`, `content_blocks_version`, and `content_source_sha256`.

Add:

- `POST /api/formulas/preview` for validated, sanitized MathML preview;
- `PUT /api/problems/{problem_id}/formulas` for atomic formula corrections.

Requests have strict size/count limits. Formula assets must remain basename-only files under the problem artifact directory and follow existing traversal/symlink protections.

Concrete limits:

- at most 64 formula blocks per problem;
- at most 8,192 UTF-8 bytes per LaTeX expression;
- at most 2,048 lexer tokens and nesting depth 32;
- preview request body at most 16 KiB and conversion timeout 2 seconds;
- formula-edit request body at most 512 KiB;
- recognition output at most 1,536 tokens.

LaTeX safety uses a positive `SAFE_MATH_COMMANDS` and `SAFE_MATH_ENVIRONMENTS` allowlist stored in source control. It includes mathematical symbols/operators, font/style commands, and `matrix`, `pmatrix`, `bmatrix`, `vmatrix`, `Vmatrix`, `cases`, `aligned`, `gathered`, and `array`. Any command/environment outside the list is rejected. In particular, macro definitions, document commands, includes, file/network access, links, HTML, and shell escapes are never allowed.

MathML is parsed with `defusedxml`; DTDs, entities, processing instructions, XInclude, and external resources are forbidden before browser preview or OMML conversion.

## 11. Acceptance Tests

The local, gitignored fixture `Sample/Latex/_DSC6580.JPG` exists and is the manual end-to-end acceptance page. During implementation, copy only its four tightly bounded printed formula crops into `tests/fixtures/formulas/` so automated tests do not depend on ignored sample directories. The full page must:

- split into the expected problems without losing formula boxes;
- detect all four fractions;
- produce exactly `\frac{3}{5}`, `\frac{2}{3}`, `\frac{5}{8}`, and `\frac{7}{9}`;
- preserve adjacent Chinese text;
- show white-background faithful formula crops;
- export native LaTeX to Markdown and editable fractions to DOCX.

Additional fixtures cover:

- nested fractions and radicals;
- limits and multi-integrals;
- sums/products with upper and lower limits;
- matrices and determinants;
- piecewise functions;
- long expressions and malformed/truncated crops;
- disagreement between preprocessing passes;
- manual edit, stale edit rejection, and image fallback.

Run the full existing regression suite. Formula processing must not alter ordinary text-only questions, figure preservation, page segmentation, classification, PDF compatibility, or existing Markdown/DOCX downloads.

## 12. Performance

- Lazy model initialization.
- Batch all formula crops from one problem/page.
- Cache recognition by SHA-256 of the normalized crop plus model revision.
- Record detector, recognizer, and total timings in problem metrics.
- No fixed latency claim is accepted until measured on this M4 Pro.
- The benchmark reports simple fraction, medium calculus, and long matrix latency separately.
