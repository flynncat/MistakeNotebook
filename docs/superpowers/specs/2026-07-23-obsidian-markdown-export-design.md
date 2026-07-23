# Obsidian Markdown Export Design

## Goal

Add an independent Obsidian Markdown export for the current batch without
changing the existing A4 PDF workflow. The result must remain editable in
Obsidian and include each accepted problem's clean reconstructed image.

## Confirmed Scope

- Keep the existing PDF controls.
- Add a separate "Export Obsidian Markdown" control.
- Export the current batch only.
- Include only eligible, confirmed problems.
- Download a ZIP containing one combined Markdown note and an attachment
  directory.
- Do not create one Markdown note per problem.
- Include reconstructed images only; never include original photographs.

## Archive Layout

```text
mistake-book-<short-batch-id>.zip
|-- mistake-book-<short-batch-id>.md
`-- attachments/
    |-- <short-problem-fingerprint>.png
    `-- ...
```

Attachment names use
`problem-<first-16-fingerprint-hex>-<full-problem-id>.png`. When the stored
fingerprint is absent, the exporter computes SHA-256 from normalized question
text. The complete database problem ID guarantees uniqueness even when two
records share the same content. Every archive path must be a generated relative
path. Absolute paths, parent traversal components, and unsafe separators are
forbidden.

## Markdown Contents

The combined note starts with YAML frontmatter:

```yaml
---
type: olympiad-mistake-book
batch_id: <full batch id>
created_at: <ISO 8601 timestamp>
problem_count: <number of exported problems>
tags:
  - mistake-book
  - olympiad
---
```

The body contains:

1. A title and total problem count.
2. A clickable table of contents grouped by primary and secondary category.
3. Level-two headings for primary categories.
4. Level-three headings for secondary categories.
5. A level-four heading for each problem.
6. For every problem:
   - Obsidian image embed: `![[attachments/<filename>.png]]`
   - Editable question text
   - Category tags, source name, and acceptance time
   - Empty sections for "My Answer", "Error Analysis", "Correct Solution",
     and "Review Log"

Every category and problem heading is preceded by an explicit ASCII HTML
anchor. Its ID is `mb-` plus the first 12 hexadecimal characters of SHA-256
over the heading type, category path, and problem ID. The table of contents
uses standard Markdown links such as `[Label](#mb-0123abcdef45)`. This avoids
Obsidian's locale-sensitive automatic heading-slug rules and duplicate-title
ambiguity.

Question text and source names must be escaped where Markdown control
characters could alter document structure. Inline category tags use the form
`#mistake-book/<primary>/<secondary>`. A tag segment retains only Unicode
letters and numbers plus ASCII `_` and `-`; every other code point becomes
`-`. Repeated dashes are collapsed, leading and trailing dashes are removed,
and an empty result becomes `uncategorized`. Chinese letters are preserved by
the Unicode letter rule.

## Eligibility

The Markdown exporter deliberately uses stricter eligibility than PDF because
the user requested manually confirmed, reconstructed content:

- `review_status == accepted`;
- `selected_kind == reconstructed`;
- `selected_artifact` exists, is readable, and its basename is `question.png`;
- category and OCR text are present;
- `accepted_at` is present. For migrated accepted records only, `updated_at`
  is the documented fallback value.

If any problem is ineligible and partial export is not explicitly allowed, the
export fails with HTTP 409 and reports the number of unresolved problems.
An empty eligible set always fails.

Records accepted through the normalized-image action are ineligible because
they are not reconstructed images. The error detail must identify this reason
instead of silently exporting a non-reconstructed image.

The first UI version does not add a second "partial Markdown export" button.
The API retains an `allow_partial` field for parity and future use, but the
independent Markdown button sends `false`.

## Backend Components

Add a focused `markdown_export.py` module that:

- filters eligible problems;
- groups them by primary and secondary category;
- generates deterministic safe attachment names;
- copies reconstructed PNG files into a temporary export tree;
- renders a UTF-8 Markdown note;
- creates the ZIP using generated relative paths only;
- atomically replaces the final archive;
- removes temporary files after success or failure.

Add these routes:

- `POST /api/batches/{batch_id}/export-markdown`
  - accepts `{ "allow_partial": false }`;
  - returns a Markdown archive download URL;
  - returns 409 for unresolved or empty exports.
- `GET /api/batches/{batch_id}/markdown`
  - returns `application/zip`;
  - sets a user-friendly Obsidian ZIP download name.

The archive can use a deterministic path under `data/exports`; no database
schema change is required. The PDF path and PDF download route remain
unchanged.

`public_problem` adds a boolean `markdown_exportable`, computed on the server
from the strict eligibility fields without exposing `selected_artifact`.

## Frontend

- Add a separate "Export Obsidian Markdown" button beside the PDF controls.
- Do not change the existing PDF buttons or their behavior.
- Disable the Markdown button when there is no current batch, the batch is
  processing, or no problem has `markdown_exportable == true`.
- The button is available only in the current-batch view; switching to the
  processed-assets view does not change its batch-scoped meaning.
- Disable the button while an export request is in flight and re-enable it
  after success or failure.
- On click, call the Markdown export endpoint and navigate to its returned
  download URL.
- Display backend 409 details directly so the user sees the exact unresolved
  count.

## Error Handling and Security

- Reject an archive when unresolved problems remain in strict mode.
- Reject an archive with no eligible problems.
- Treat a missing or unreadable reconstructed image as ineligible.
- Never use source-provided strings as archive paths.
- Reject or sanitize absolute paths, separators, and `..`.
- Write to a temporary archive and use atomic replacement.
- Remove temporary output after any failure.
- Do not emit local absolute paths or session-token URLs into Markdown.

## Tests

- A one-problem and a multi-problem batch each produce a valid ZIP.
- The ZIP contains exactly one Markdown note and reconstructed attachments.
- YAML, Unicode categories, table of contents, templates, and Obsidian embeds
  are correct.
- Duplicate problem titles cannot overwrite attachments.
- Markdown metacharacters and hostile source paths are escaped.
- Pending problems, empty batches, and missing artifacts return clear errors.
- API authentication and ZIP download headers are correct.
- Existing PDF, single-problem, multi-problem, and asset publication tests
  continue to pass.

## Acceptance Criteria

After extracting the ZIP into any Obsidian vault:

- the combined note opens as UTF-8;
- category navigation works;
- every reconstructed image renders;
- question text is editable;
- every problem has the correction and review template;
- no running mistake-book service or machine-specific absolute path is needed.
