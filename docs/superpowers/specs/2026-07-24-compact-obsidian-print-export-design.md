# Compact Obsidian and Print Export Design

## Goals

1. Keep searchable question content as Markdown text.
2. Store only irreducible visual resources as images.
3. Preserve the original reading order between text, formulas, and figures.
4. Make both PDF and Obsidian output suitable for printing and solving again.
5. Fix Obsidian table-of-contents navigation.
6. Give exported archives descriptive, time-based names.

## Confirmed Product Decisions

- Ordinary question text is emitted as Markdown text.
- Reliably recognized formulas are emitted as inline or display LaTeX.
- Formulas that cannot be represented reliably, figures, diagrams, and tables
  are emitted as cropped PNG resources.
- Resource order must follow the source layout. A resource may not be silently
  appended after the question.
- Semantic or AI-authored redrawing is forbidden. Every image block requires
  source-pixel support.
- PDF may keep its proven clean question-image rendering to avoid regression.
- PDF and Markdown remove answer-analysis templates and correction labels.
- Each problem occupies approximately half an A4 page and receives plain white
  answer space without labels, rules, grids, or borders.
- At most two problems appear on one printed page.
- Archive naming order is local time, primary-category statistics, random ID.
- At most three primary categories are listed in the filename.

## Ordered Content Model

Each reconstructed problem gains dedicated, versioned storage columns:

- `content_blocks_version INTEGER`;
- `content_blocks_json TEXT`;
- `content_source_sha256 TEXT`.

Version 1 is a JSON object, not a free-form metrics entry:

```json
{
  "version": 1,
  "source": {
    "artifact": "cleaned.png",
    "sha256": "<hex>",
    "width": 1600,
    "height": 2200,
    "coordinate_system": "top-left-normalized",
    "ocr_lines": [
      {
        "id": "l1",
        "stable_index": 0,
        "text": "...",
        "box": [0.1, 0.1, 0.9, 0.15]
      }
    ]
  },
  "blocks": [
    {
      "id": "b1",
      "type": "text",
      "text": "...",
      "source_spans": [{"line_id": "l1", "start": 0, "end": 4}]
    },
    {
      "id": "b2",
      "type": "latex",
      "latex": "...",
      "display": false,
      "source_text": "...",
      "source_spans": [
        {"line_id": "l2", "start": 3, "end": 8, "box": [0.2, 0.2, 0.4, 0.25]}
      ]
    },
    {
      "id": "b3",
      "type": "image",
      "asset": "resources/b3.png",
      "alt": "question figure",
      "source_box": [0.1, 0.2, 0.8, 0.5],
      "source_spans": []
    }
  ]
}
```

Supported block types:

- `text`: Unicode Markdown text.
- `latex`: a deterministic, syntax-validated LaTeX fragment.
- `image`: a clean crop backed by a source bounding box.

The versioned OCR line list is authoritative for revalidating order and
content conservation. `stable_index` preserves detector order for final ties.
Text and formula blocks reference half-open Unicode code-point ranges. One OCR
line may therefore become leading text, a formula replacement, and trailing
text without losing surrounding prose.

Image asset paths are artifact-directory-relative, use generated ASCII names,
and may not contain separators other than their fixed `resources/` prefix.
Coordinates are normalized `[x0, y0, x1, y1]` values relative to the declared
clean source, with origin at the top left. `content_source_sha256` must equal
the source object's SHA-256. The exporter rejects unknown schema versions.

Block order is derived from OCR line boxes and visual-resource boxes in reading
order. The exporter never guesses an image position from question semantics.

The deterministic ordering algorithm is:

1. Detect full-height whitespace gaps wider than 8 percent of page width and
   split true columns with XY-cut.
2. Visit columns left to right.
3. Within one column, cluster items into rows when vertical overlap is at least
   50 percent of the smaller item height.
4. Visit rows top to bottom and items in one row left to right.
5. Resolve ties by normalized `y0`, then `x0`, then stable source index.
6. OCR character spans covered by a formula replacement are consumed by that
   replacement. Figure boxes suppress only OCR spans proven to be labels inside
   the figure; surrounding prose remains text.

Validation rejects a sequence when non-replacement blocks overlap by more than
20 percent of the smaller area, when one OCR code-point range is consumed more
than once, when a resource is emitted more than once, or when any accepted OCR
code point or detected visual resource is left uncovered. Validation runs
against the persisted, source-hashed OCR line list.

## Formula Policy

Version 1 supports numbers, Latin variables, Chinese units wrapped in
`\text{}`, `+ - = < > <= >=`, multiplication, division, ratios, balanced round
parentheses, simple `a/b` fractions, integer superscripts, and integer
subscripts. Geometry diagrams, matrices, cases, roots with ambiguous spans, and
handwritten notation are not converted in version 1.

Common, deterministic expressions may become LaTeX after normalization of
operators, fractions, superscripts, and delimiters. Conversion is accepted only
when:

- every source token is consumed;
- delimiter balance and the supported grammar validate;
- numbers and variables round-trip to the source token sequence.

The round trip canonicalizes whitespace and equivalent operator glyphs, then
requires exact token equality. Formula detection records character-level spans
and boxes, including when a formula occurs inside prose such as
`if x=2, then ...`. If validation fails, only the formula span box plus fixed
padding is cropped from the declared `cleaned.png`, using the same dimensions
and SHA-256 stored in the schema. The image block records the original box,
source span, and original formula text. Only that source span is consumed;
leading and trailing prose become separate text blocks. No language model
correction is allowed in this fallback.

## Figure and Table Policy

Existing `figure-selected.png` resources remain the preferred figure assets.
Their stored source boxes determine insertion order. Additional printed tables
or non-formula graphics require a clean crop and source box.

Old records are not rewritten during database startup. Migration is lazy and
uses these rules:

- An accepted record may be backfilled as one text block only when it carries a
  compatible versioned detector result proving no visual or formula candidate,
  or a human `compact_text_only_verified` marker.
- Every other old record requires explicit reprocessing from its source image;
  absence of an old visual keyword is not proof of a text-only question.
- Reprocessing writes a candidate block sequence without changing the accepted
  OCR text or published asset. Normalize both texts with Unicode NFKC and
  whitespace collapsing, then require complete character-for-character
  equality. Any non-whitespace difference returns the record to human review.
  Matching numbers alone is insufficient. All block validation must also pass.
- A failed reprocessing attempt records `compact_export_status` and a concrete
  reason. It requires renewed human review and remains ineligible.

A maintenance operation may batch-reprocess ineligible accepted records, but
must never auto-accept changed text or visuals. New records write version-1
blocks during normal V2 processing. Compact export accepts only schema version
1 with a matching source SHA-256. It never falls back to a full-question
screenshot.

## Markdown Rendering

- A pure-text problem creates no attachment.
- Image blocks create individual PNG attachments.
- Text, LaTeX, and image embeds are emitted in `content_blocks` order.
- Inline LaTeX uses `$...$`; display LaTeX uses `$$...$$`.
- Obsidian image embeds use `![[attachments/<name>.png]]`.

The ZIP includes `.obsidian/snippets/mistake-book-print.css`, and the note
frontmatter contains `cssclasses: [mistake-book-print]`. The note explains that
the user must enable this CSS snippet once in Obsidian before printing.

The exporter does not wrap Markdown in raw HTML. After each problem's final
content block it emits one standalone marker:

```html
<div class="mb-answer-space mb-normal"></div>
```

Long problems use `mb-long`; a standalone
`<div class="mb-page-break"></div>` is inserted before or after a problem when
the measured layout requires it. These empty markers do not contain Markdown,
LaTeX, images, headings, or block IDs, so Obsidian parses all actual content
normally. The CSS styles only marker height and page breaks.

A shared `layout_measure.py` computes physical content height before either
exporter runs. It uses the verified print font, fixed 11 pt size, 1.5 line
height, 180 mm content width, actual font glyph advances for wrapped text,
14 mm for each display-math row, the line height for inline math, and the
rendered aspect ratio of image blocks at their CSS/PDF width. A 10 percent
safety margin is added. The packaged CSS fixes the same font size, line height,
width, and image constraints. The exporter assigns `normal`, `long`, or
`continuation` classes from this measured value before writing Markdown; CSS
does not attempt to measure content.

The print font is open-source Noto Sans SC under the SIL Open Font License. The
repository stores the verified source font and license. Each export subsets it
to glyphs used by that note, packages a WOFF2 file plus the license, and uses
that exact subset through CSS `@font-face`. `layout_measure.py` uses the same
source font file. Export fails rather than silently changing metrics when the
verified font is unavailable. Proprietary Microsoft YaHei is not redistributed.

For A4 with 15 mm top and bottom margins, printable height is 267 mm. Two normal
problem slots are 128 mm each with an 11 mm shared allowance for page headers
and separation. A normal slot reserves at least 60 mm of plain answer space, so
question content may use at most 68 mm.

If question content exceeds 68 mm, the problem receives a full 267 mm page and
at least 100 mm of plain answer space. If content itself exceeds 167 mm, content
continues on as many single-problem pages as required and is followed by a
separate 128 mm plain answer-space block. No content or answer space may
overlap. CSS uses `break-inside: avoid` where a block fits and explicit page
break classes after two normal slots or around long problems. There is no
answer-space heading or decoration.

## PDF Rendering

The PDF keeps using the accepted clean question image. Remove the `Correction`
label and ruled correction area. `layout_measure.py` assigns the same slot
class. A normal image is scaled proportionally into at most 68 mm content
height, followed by at least 60 mm blank space. A long image is scaled
proportionally into at most 167 mm, followed by at least 100 mm blank space.

The PDF never crops a question image. If fitting into 167 mm would reduce the
measured printed glyph height below 8 pt, the whole image is fitted
proportionally on a content-only page and a separate 128 mm blank answer block
follows on the next page. This may use more pages but cannot cut text, formulas,
or figures. If fitting the whole image into that full content page would still
reduce measured glyph height below 8 pt, PDF export is rejected with a concrete
request to reprocess or split the question; it never crops or emits unreadable
text. Continue enforcing no more than two normal problems per page.

## Obsidian Navigation

The existing HTML-ID targets are not reliable Obsidian link destinations.
Replace them with Obsidian block IDs placed at the end of the actual heading
line:

```markdown
## Counting ^mb-0123abcdef45
```

Table-of-contents entries use:

```markdown
[[#^mb-0123abcdef45|Visible title]]
```

IDs remain deterministic 16-hex SHA-256 prefixes derived from block type,
category path, and problem ID. The generator maintains a set and rejects any
collision. Link aliases replace `|`, `]`, `#`, and control characters with
visually equivalent full-width or safe characters before insertion into a
wikilink. Tests must verify that every target has one matching link, every link
has one target, and a real Obsidian-compatible parser resolves representative
links.

## Naming

Example:

```text
20260724-1123_Counting3-Application2-plus2categories_K7M2.zip
```

The actual category names remain Unicode Chinese in user-facing filenames.
Rules:

- local timestamp uses `datetime.now().astimezone()` and format
  `YYYYMMDD-HHmm`;
- the primary category field is `category_group`;
- categories are sorted by descending problem count, then by their order in
  the versioned `classification.TAXONOMY`; unknown groups follow in Unicode
  code-point order;
- include at most three categories as `<name><count>`;
- append `plus-N-categories` in the localized form when categories remain;
- append a four-character random code;
- random alphabet is `23456789ABCDEFGHJKMNPQRSTUVWXYZ`;
- random generation uses Python `secrets.choice`;
- ZIP and combined Markdown use the same basename;
- names are normalized with Unicode NFKC;
- separators, control characters, trailing spaces/dots, and Windows-reserved
  device names are replaced safely;
- the basename is limited to 180 UTF-8 bytes; category portions are truncated
  on code-point boundaries while timestamp and random ID are always preserved.

Each export creates a new random code. The final generated archive path is
returned by the API instead of being recomputed from only the batch ID. This
new rule applies to the Markdown ZIP and its internal Markdown file. Existing
PDF naming remains unchanged unless a separate PDF naming change is approved.

## API and State

Add an `exports` table with `id` (UUID), `batch_id`, `kind`, `minute_key`,
`display_code`, canonical `file_path`, user-facing `filename`, `created_at`,
and `expires_at`. A unique database index on
`(kind, minute_key, display_code)` reserves display codes atomically. The
Markdown export endpoint creates the archive under a newly generated UUID
storage name, inserts the row transactionally, and returns
`/api/exports/{export_id}` plus the user-facing filename.

The download route looks up the row, requires `kind == markdown`, checks expiry,
resolves the canonical path, requires its parent to be the canonical exports
directory, and rejects symbolic links. It never accepts a filename from the
request. Concurrent exports use distinct UUIDs and files.

Four-character display-code insertion conflicts trigger a transaction rollback
and atomic retry up to 16 times; no query-then-insert uniqueness assumption is
used. Exhaustion returns a clear 503.
Expired exports and database rows are deleted on startup and before each new
export. The retention period is 30 days.

No absolute local path or session token is written into Markdown.

## Error Handling

- Reject compact export when content block order cannot be proved.
- Reject an image block without an existing clean asset and source box.
- Reject an invalid or partially consumed LaTeX conversion.
- Reject duplicate attachment paths.
- Reject ZIP traversal and unsafe generated names.
- Return concrete problem IDs and reasons for records requiring reprocessing.

Markdown text, wikilink aliases, HTML attributes, YAML values, and LaTeX are
escaped by context-specific functions; one generic Markdown escape function is
not reused across contexts. Image assets must resolve inside that problem's
canonical artifact directory, must be regular files, and must not be symbolic
links. ZIP writers receive generated relative paths only. Download responses
expose the user-facing filename but never the storage path or session token.

## Testing

- pure-text questions create no attachments;
- valid formulas become LaTeX;
- unsupported formulas become positioned image blocks;
- figures remain between the correct surrounding text blocks;
- no full-question screenshot is included;
- PDF and Markdown contain plain half-page answer space;
- no answer-analysis or correction headings remain;
- every Obsidian block link resolves to exactly one target;
- generated names follow time-category-random ordering and length limits;
- category ties are deterministic;
- old pure-text records backfill safely and old visual records remain
  ineligible until successful reprocessing;
- multi-column, same-row, overlap, tie, omission, duplication, and text/resource
  conservation cases validate deterministically;
- formula fallback consumes exactly the replaced source lines and uses the
  recorded clean-source hash and box;
- actual PDF page geometry is inspected for normal and long questions;
- the packaged Obsidian print CSS and representative rendered print layout are
  tested;
- hostile Markdown, HTML, LaTeX, Unicode filenames, traversal, and symlink
  inputs are rejected or escaped;
- random-code collision retries and concurrent exports remain isolated;
- existing PDF, single-question, multi-question, review, and asset tests pass.

## Acceptance Criteria

After extracting an archive into an Obsidian vault:

- ordinary question text is searchable and selectable;
- supported mathematics renders as LaTeX;
- only necessary visual resources are attachments;
- visual resources appear in their original reading positions;
- directory links jump correctly in Obsidian;
- print output contains at most two problems per page with plain answer space;
- the archive name identifies creation time and major content categories.
