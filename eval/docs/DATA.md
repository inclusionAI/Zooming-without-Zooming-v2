# Data preparation and layout

No dataset, checkpoint or credential is bundled. Official source references live in
`downloads.py`. Download without `--execute` prints the plan only; `--revision` is
required for execution and is resolved to an immutable Hugging Face commit. The manifest
contains commit, file sizes and SHA-256. Interrupted downloads can resume only into the
same recorded source/revision directory. No benchmark files are downloaded during tests.

For the twelve platform tasks and MIMIC, use the strict local raw converters in
[`convert`/`prepare-suite`](../README.md), or import the authorized
project-normalized release. The raw converters must reconstruct the recorded
historical annotation fingerprints; arbitrary upstream conversions are not accepted.
Exact source filenames, counts, preparation commands and image relocation are in
[PLATFORM_BENCHMARKS.md](DATA.md). Their download plans explain this
requirement instead of inventing public URLs for private normalized annotations.
All 25 entries have an explicit recipe in `examples/datasets.json`; native-layout
entries are audited/registered in place. MindCube's raw converter generates the
preserved official Tiny raw_qa prompt from the extracted original JSONL and
verifies the historical 1,050-input projection; the old prepare command still
expects already-generated prompts.

```bash
zwz-eval download --bench memlens --target data/memlens/raw --revision COMMIT
zwz-eval download --bench memlens --target data/memlens/raw --revision COMMIT --execute
zwz-eval extract --archive data/example.tar.gz --target data/example-unpacked
```

`extract` requires a new target and rejects traversal, symlinks, hard links and device
files. It retains the archive. Inspect the actual release archive layout before placing
the extracted root: downloads do not assume every archive has an identical top directory.
The [download catalogue](DOWNLOADS.md) covers all 25 HF entries and clearly separates
official releases, converted/community mirrors and Chartography's inaccessible
anonymous URL. Downloaded data must still pass the project profile audit.

Expected roots after extraction/preparation:

```text
data/
  mmlongbench/
    mmlb_data/vrag/{infoseek,viquae}_K*_dep*.jsonl
    mmlb_data/NIAH/{retrieval-image,reasoning-image,vh_multi*}*.jsonl
    images/mmlb_image/...
  milebench/
    raw/<Task>/<Task>.json
    raw/<Task>/images/...
    normalized/milebench_<Task>_test.json
  memlens/
    raw/dataset_{32k,64k,128k,256k}.json
    release_images/...
    normalized/memlens_{32k,64k,128k,256k}.json
  agentvista/val.json
  agentvista/images/...
  screenspot_pro/annotations/*.json
  screenspot_pro/images/...
  ui_vision/annotations/element_grounding/element_grounding_<split>.json
  ui_vision/images/...
```

MMLongBench upstream archive names include `0_mmlb_data.tar.gz`, `1_vrag_image.tar.gz`,
`2_vh_image.tar.gz`, `2_mm-niah_image.tar.gz`. These cover the selected visual families;
the downloader currently snapshots the requested dataset revision, potentially including
other large files. Budget storage before execution. This is not a claim that archive
names will never change; pin and audit the actual revision.

Conversion commands:

```bash
zwz-eval prepare --bench milebench --source data/milebench/raw --target data/milebench/normalized
zwz-eval prepare --bench memlens --source data/memlens/raw --image-root data/memlens/release_images --target data/memlens/normalized
zwz-eval prepare --bench agentvista --source data/agentvista-download --target data/agentvista
zwz-eval audit --config examples/all.json --bench memlens
```

AgentVista preparation reads the pinned Parquet shards, requires 209 questions and
exports their encoded image bytes unchanged (no recompression). The generated
`val.json` maps `problem`, `answer` to the retained agent's `problem`, `solution`
schema. Its image names are relative to the generated `images/` directory. It writes
a conversion manifest and requires a new target instead of overwriting an old dataset.

Conversions are not a license to select new rows or change the historical subset. Keep
the normalized annotation manifests with reported results. Wrong known denominators,
malformed annotations and duplicate IDs fail before inference. New platform preparation
defaults to annotation-only `image_mode: on-demand`: original images are read and
preprocessed per inference request with the unchanged Qwen pixel policy and lossless
PNG transport, without an all-images PNG staging barrier. Missing/corrupt images fail
when read; preparation alone does not certify full image integrity. Explicit `audit`
reads all referenced image bytes. `--materialize-images` optionally requests offline
platform PNG preparation in a new directory; old materialized bundles retain their
existing assets and hashes without redecoding or replacement on reuse. Other raw/native
converters retain their own storage rules. MMLongBench
audits the deterministically selected depth/material rows and their context/choice images.
The UI-Vision projection is generated inside its run directory and does not modify raw data.

Paths in config are resolved relative to the config file, not the terminal's current
directory. `image_root` is the prefix of the paths actually stored in annotations; do
not double-prefix `images/`. If importing authorized normalized annotations from another
machine, rebase only the image paths, preserve question/reference fields, and record a
new manifest. The CLI never falls back to somebody else's cluster directory.

### Independent image storage

An explicit `datasets.<bench>.image_root` is honored by audit, inference and native
command plans, including MMLongBench, ScreenSpot-Pro and UI-Vision. Annotation and
image storage can therefore live on different disks without copying the dataset.
Image names/order/pixels are not changed by this relocation. If omitted, the shared
defaults are:

| Profile | Default image root |
| --- | --- |
| MMLongBench | `<root>/images/mmlb_image` |
| MileBench, MemLens, AgentVista, ScreenSpot-Pro, UI-Vision | `<root>/images` |
| Platform 12, MIMIC, MMSI, ViewSpatial, MindCube, Chartography, SPAR, ERQA | `<root>` |

These are prefixes for annotation-relative image names, not filename search paths.
Explicitly set the root when a release uses a different relative layout. No basename
search, decoy substitution or missing-image fallback is introduced. MMLongBench's
`serve-plan` grants local-media access to the same selected image root; the inference
server must see that exact path too.

---

## Benchmark annotation format and protocols

The adapters preserve the selected project evaluate_v3 message construction,
Qwen image processing, answer parsing, rule accuracy and local A3B review chain.
They do not combine unrelated latest upstream protocols. Original questions,
options, system messages, gold answers and custom judge prompts remain intact.

Supported tasks are MMStar, BabyVision, VStar, HR-4K, HR-8K, MME-RealWorld-en/cn,
ZoomBench, Perception-Bench, TreeBench, O3-Bench and CountQA. Exact denominators and
source identities are in the protocol/source registries. MME-RealWorld is not MME.

### Required annotation format

The benchmarks listed above require a JSON array or JSONL with one record per
line in the following format. All other benchmarks use their original downloaded
format and the documented preparation entrypoints. Each record contains:

| Field | Processed representation |
| --- | --- |
| `images` | Ordered list of image-path strings, using the user's own paths. |
| `query` | Original question and choices, or the original ordered list of user turns. |
| `response` | Original reference answer, including its original option-label form. |
| `system` | Optional original system message; retain it if present, including an empty string. |
| `LLM_JUDGE_PROMPT` | Optional per-record original scoring prompt; preserve it unchanged. |

```json
{
  "images": ["/path/to/images/image.png"],
  "query": "<image>Original question and answer choices",
  "response": "Reference answer"
}
```

This is a schema illustration, not a benchmark case. Keep `<image>` placeholders,
image and question order, option punctuation, sample IDs and all additional source
fields. A `query` list denotes separate original user turns; do not flatten it.
Image roots/maps change path prefixes only. Missing images, conflicting identities
or incorrect fixed denominators fail validation rather than silently changing data.

Set the dataset manifest's `format` to `normalized` for the benchmarks listed above.
All other benchmarks use their original downloaded formats. Both paths prepare
data separately from model evaluation and must preserve the selected project protocol.

### Separate data preparation

Provide the annotations in the required format. The importer
checks annotation fingerprints, question order, fields and image identities.
Provide explicit image roots/maps; never guess by basename. Install
`.[platform-prepare]` without replacing your existing accelerator runtime.

Default `image_mode: on-demand` records annotations and original-image references.
Each inference request reads its images in order, applies the retained Qwen
processing and sends lossless PNG. Preparation does not read all images first.
Missing/corrupt images fail on access; no replacement black/gray image is inserted.
Use explicit `audit` / `doctor --audit-data` for full asset checks. Optional
`--materialize-images` creates a separate offline prepared directory. Existing
materialized manifests retain their asset hashes and are not silently migrated.

### Inference and scoring

Main inference retains native templates, no-thinking, temperature 0, top_p 1,
16384 generated tokens, 32768 context and at most 10 images. Image processing uses
min_pixels 65536, max_pixels 16777216 and patch size 14. It preserves user turns
and placeholder interleaving; no new letter-only instruction is added.

The original answer-tag parser and fallback rules are retained. Rule scoring uses
trimmed case-insensitive exact comparison and the original short multiple-choice
branch. Only initial errors go to Qwen3.5-35B-A3B, using question/gold/extracted
answer text, not images. Preserve per-row custom judge prompts. The native
tokenizer renders raw `/v1/completions` input with the empty-think suffix;
temperature is 0 and the cap is 96 tokens. Final accuracy uses the retained
four-decimal rounding before percentage conversion.

Judge failures do not become negative completed verdicts. Cached judgments bind
to input hashes. Successful wrong, empty or malformed answers are not resampled.
`score.json` requires the complete expected questions and required verdicts.
On-demand/offline synthetic equivalence tests are not full real-data/GPU acceptance.

---

## Normalized input provenance

This source-only GitHub submission does not bundle benchmark annotations or images.
Obtain authorized data from the sources in [DOWNLOADS.md](DOWNLOADS.md).
Platform/MIMIC normalized inputs require the verified annotation identity expected
by their importer; an arbitrary upstream download is not automatically equivalent.

Preserve question order, query/system/gold, extra fields, image ordering and repeated
references. Resolve image paths through explicit roots or exact image maps; do not
match ambiguous basenames, silently drop rows, replace missing images or deduplicate
questions sharing an image. Source provenance and image-byte validation are separate.
Read [data formats](DATA.md), [platform benchmarks](DATA.md), and the
dataset source catalogue before importing. Dataset licenses are separate from the
source license. No private normalized-data export is included in this package.
