# Evaluation contract

This is the project-adapted reference protocol. It is not a claim that every
local parser, generation setting, or tool backend is the authors' default evaluation.
Any change to the following contract needs a new version and recomputation for all
models being compared. Do not selectively repair only a model's wrong predictions.

## Shared requirements

Use the checkpoint's native chat template unless the reference explicitly specifies
otherwise. Do not replace a trained template with a base template silently. Archive
the model revision, tokenizer/chat-template hash, judge revision, serving arguments,
runtime versions and normalized annotation manifest. The API model ID is checked
against `/v1/models`; this alone does not prove the tokenizer or template is correct.

Do not truncate histories, silently drop images, lower resolution to avoid an error,
or let service failures become incorrect answers. The caller preserves raw outputs,
finish reasons and extracted answers for the shared inference path. Prompt/metric
implementations for the remaining native adapters are retained as versioned sources.

The new runner namespaces MileBench sample IDs across tasks to isolate the legacy
extractor cache. This changes cache identity, not the question, answer or metric.
All score.json scores are percentages, and task macros are equal weighted.
Native MemLens/MileBench/AgentVista judge caches are additionally bound
to input file hashes and scorer command identity. Modified model responses cannot
silently inherit earlier verdicts. Unbound historical caches are rejected, not deleted
or automatically certified. Use a new run ID for a changed input/scorer configuration.

## MMLongBench (25 tasks)

Five families: **InfoSeek, ViQuAE, MM-NIAH Retrieval-image, MM-NIAH Reasoning-image,
and Visual Haystack Multi-image**, each at **K8, K16, K32, K64 and K128**.
Counting and text-only families are excluded. These K values are dataset context
settings, not the response length or the server context-window limit.

Seed 42; greedy generation; 2,048 new tokens; no-thinking. Source-ID caps: 100 for
InfoSeek/ViQuAE/VH-multi, 50 for the two MM-NIAH families. Retain every depth/material
row associated with each selected ID; these are **not** caps of 100/50 JSON rows.
Image names containing double quotes are excluded in V-RAG by the retained loader
before deterministic ID selection. The source selection and prompt templates are unchanged.

Metrics: substring EM for InfoSeek/ViQuAE; MC accuracy for image retrieval/reasoning;
binary accuracy for VH-multi. Preserve the upstream last-answer/parser behavior and
VH fallback sequence `[0,1,1,0]`; this fallback is a limitation, not an LLM judge.
The reported macro is the mean of 25 task scores. No Counting-specific prompt is used.

The fallback is now assigned by each row's zero-based position **after source-ID
selection**, repeating that same sequence. It no longer depends on concurrent API
completion order or post-processing calls made while restoring a checkpoint.
This is an explicit reproducibility fix: valid Yes/No parsing is unchanged, but
unparseable VH responses can differ from old stateful/concurrent scores. Use a new
run ID for the corrected protocol; never silently relabel or overwrite old scores.
Native `.json.cache` files are included in score hashes. `summarize` rejects changed,
missing or previously unbound native caches rather than certifying stale scores.

The preserved upstream writer excludes empty parsed outputs from averaged metrics.
**This runner refuses to label such a run a complete final result**, rather than
silently shrinking its denominator. Its upstream JSON/cache remain available for
auditing; an archived historical score that excluded empties must be labeled as such.
Do not compare it as if it were automatically equivalent to a complete new run.
In the inspected reference archive, Retrieval-image K128 has 300 cache rows but 299
final JSON rows: one nonempty 2,048-token response has an empty parsed answer.
The archived 68.341226% macro therefore does not pass this wrapper's completion
gate. Historical artifacts are preserved; no gold-assisted parser repair is applied.

The native API wrapper now raises on transport/malformed-response failure instead
of saving a fabricated empty answer with negative token counts. Valid explicit
empty content retains actual token accounting and remains a model output. The
launcher selects strict task-error propagation, so a swallowed native exception
cannot quietly advance to the next task and appear successful.

Default transport uses `file://` image URLs. The inference service must see the same
image filesystem and permit the configured image root. This is not an arbitrary
remote-host API adapter. Do not point it at a hosted endpoint lacking these files.

## MileBench (29 test tasks)

13 temporal + 13 semantic + 3 diagnostic tasks; `adv` is excluded from formal scoring.
The exact task list is in `protocols.py`. Native template, no-thinking, 8,192 new tokens,
262,144 serving context and up to 256 images. No image downsampling is introduced by
the common formatter. Task instruction and original image order are retained.

MC questions use the retained Chinese option-extraction prompt with the configured
judge, temperature 0, 10 output tokens and no-thinking, then exact option comparison.
This is **not** a semantic correctness fallback. The extractor sees the first 500
question characters and first 1,000 response characters, as in the reference code.
Open-ended responses use the retained ROUGE-L F1; Needle and MMCoQA use their specific
matching rules. Final score is mean of 29 task means, not mean of three category means.

The shared CLI passes the selected `model_id` to the scorer as `MILEBENCH_MODELS`.
If invoking the vendored scorer directly, set that environment variable explicitly
to the space-separated model IDs to score. There is no built-in historical model
roster and no inference from whatever model directories happen to exist.

`prepare` converts raw official annotations; matching historical subset identity still
requires the historical normalized manifest. Merely downloading the latest release
does not establish that identity. New output directories cannot reuse old extract caches.

## MemLens (four contexts)

32k/64k/128k/256k, exactly 789 questions each. The converter retains the full session
history, turn roles, dates and images. The system instructs the model to answer from
history and say **"Insufficient information"** if it cannot be answered. The final
instruction requests a direct answer. Flattening all history into one user message is
not equivalent. Native template, no-thinking; 8,192 new tokens; 262,144 context; 256 images.

The retained official-derived judge routes by question type/subtype (including refusal,
knowledge update and multi-session reasoning), extracts the prediction and applies the
binary rubric. Judge: no-thinking, temperature 0, 2,048 output tokens. Existing prediction
parsing and the >500-word auto-zero rule are retained. A judge parse failure is 0 in this
reference implementation and is not silently rescued by a new semantic fallback.
Four-context macro is mean of four LLM-judge accuracies; exact match is only diagnostic.

## AgentVista (adapted API/free-tools)

209 questions; free `web_search`, `image_search`, `visit`; total tool budget 12;
30 turns; up to 100 images; total-token budget 65,536; 1,024 completion tokens per call.
Thinking uses the native main model default during tool use, and is disabled on the
forced final turn when no tools remain. Plain final text is accepted. The reference
tool implementations, registration and agent prompts are included, not replaced by
a generic question-answer loop. Free tools differ from paid original search services;
live web content and search results mean exact rerun determinism is not promised.
The 12-tool setting is a next-turn threshold, not a certified strict per-call cap
for multi-tool responses. Effective CLI limits, visit text limits, no reverse-image
upload, HTTP fallbacks and the remaining reference-launcher verification are detailed
in [the free-tool guide](PROTOCOLS.md). Public tool availability has not
been certified by the offline tests.

Final-answer postjudge receives question, reference and final response. Temperature
0.2, 8,192 tokens, native/default judge thinking (no forced false). This is distinct
from later experimental dual-judge runs. The denominator remains 209, including
recorded inference failures; `accuracy_on_judged` is not the table metric. Missing or
unresolved judgment files block final aggregation. Do not reuse a different tool backend
and call it the same protocol. The runner does not execute the code-interpreter tool.

## ScreenSpot-Pro / UI-Vision

Preserved native Qwen-family grounding backend (`qwen3_5_vllm` only): tool-call prompt,
1,000×1,000 coordinate convention, retained tool-call parser, no-thinking, greedy,
8,192 tokens, processor max_pixels=8,294,400. It uses the checkpoint's native tokenizer;
no generic OpenAI grounding prompt is substituted. Other model families need a separately
validated adapter, not a misleading model-name change.

ScreenSpot-Pro: all 1,581 positive English instruction tasks. UI-Vision: element grounding
only, basic 1,772 / functional 1,772 / spatial 1,935; no action/task grounding. Normalize
coordinates against original dimensions and apply inclusive point-in-box matching.
Invalid format is incorrect and stays in the denominator. UI-Vision main score is the
equal-weight mean of three split accuracies, not the pooled-example micro score.

See [the GUI reproduction/audit guide](PROTOCOLS.md) for explicit coordinate
units, historical GLM incompatibilities, native trace fields and the separately
declared portable API adapter. Native primary reference parsing is not silently
replaced by the optional tolerant diagnostic parser.

---

## MMSI, ViewSpatial and MindCube portable adapters

These three adapters use the same configuration, run contract, dataset audit,
API identity check, locking, inference journal and score output layout as the
existing framework. They are source-audited ports; real-data/GPU numerical parity
is still an acceptance gate. No private NAS paths or private platform dependency
is needed. Do not call them unmodified upstream implementations.

### Import original data

Obtain the authorized original datasets first. Their HF sources and observed
revisions are now recorded in [DOWNLOADS.md](DOWNLOADS.md); `download` can fetch
an explicitly selected revision. It does not automatically extract every archive
or generate prompts during download. The new `convert --bench mindcube` / one-click
preparation entry generates the preserved official Tiny raw_qa prefix directly from
the extracted `data/raw/MindCube_tinybench.jsonl`; it does not need cognitive-map
scaffolding. All 1,050 IDs, questions, golds, ordered image identities and generated
prompts must match the historical keyed fingerprint. Row order is preserved from
the supplied raw file; historical row-order/image-byte equality is not asserted.
See [QUICKSTART.md](../README.md). The legacy `prepare` example below expects
already-generated prompts.

Run from the repository root, with the `data` and `api` extras installed:

```bash
zwz-eval prepare --bench mmsi \
  --source data/mmsi/original/test.parquet --target data/mmsi/prepared
zwz-eval prepare --bench viewspatial \
  --source data/viewspatial/original/test.json --target data/viewspatial/prepared
zwz-eval prepare --bench mindcube \
  --source data/mindcube/original/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --target data/mindcube/prepared
```

These are example **local import paths**, not claims about downloaded archive
filenames. Pass your actual original annotation file to `--source`. Preparation
refuses existing targets. Expected sample counts are fixed: MMSI 1,000,
ViewSpatial 5,712, MindCube-Tiny 1,050. Partial imports and duplicate identities
are rejected. Original question/gold fields are preserved. MMSI embedded image
bytes are exported unchanged; the other two retain original relative image paths.
Set `image_root` to the root those paths are relative to.

Prepared layout:

```text
data/<benchmark>/prepared/
  records.json          # deterministic schema projection
  preparation.json     # original annotation + projected annotation hashes
  images/              # MMSI embedded images only
```

### Run through the common CLI

Edit [examples/spatial.json](../examples/spatial.json), recording the actual model
revision and native chat-template hash. The judge for the established reviewed
ViewSpatial profile is Qwen3.6-35B-A3B, no-thinking. Do not substitute another judge
while claiming same-protocol comparability.

```bash
export MAIN_API_KEY=EMPTY JUDGE_API_KEY=EMPTY
zwz-eval audit --config examples/spatial.json --bench viewspatial
zwz-eval serve-plan --config examples/spatial.json --bench viewspatial --role main
## Start the correctly configured main service separately, then:
zwz-eval infer --config examples/spatial.json --bench viewspatial
## If GPUs are shared, finish the main service and start the judge service separately:
zwz-eval serve-plan --config examples/spatial.json --bench viewspatial --role judge
zwz-eval score --config examples/spatial.json --bench viewspatial
```

For MMSI/MindCube, replace `--bench`; their primary scores require no LLM judge.
Service plans never start/kill processes. They request the reference context and
image limits and fail if the chosen model/build cannot support them; no automatic
truncation or context reduction is added. Verify actual no-thinking template
behavior on your server, not just the boolean in the request.

### Exact generation and parsing differences

| Profile | Generation / preprocessing | Primary score |
| --- | --- | --- |
| MMSI | Original image bytes first, original question plus literal backtick-option suffix; 4,096 tokens; no-thinking | 2026-10-19 final-answer parser, then exact gold match. This is explicitly versioned separately from the older first-regex parser. |
| ViewSpatial | Images first, resized RGB 420×280 LANCZOS/JPEG95; original question + choices + exact source suffix; 1,024 tokens; no-thinking | Last complete answer tag or source final-line option extractor, then wrong-only text review. Judge receives question, choices, gold and raw prediction, **no images**; 16 output tokens, no-thinking. |
| MindCube | Original image order then unmodified `input_prompt`; 1,536 tokens; top_p=1, reasoning_effort=low; no-thinking service | Source extractor's pattern priority; exact `gt_answer` comparison. Translation settings excluded from primary Overall, but included in separately reported unfiltered diagnostic. |

The original MindCube client uses `message.reasoning` if formal `content` is empty.
This port preserves that behavior, saves both channels and reports the fallback
count. It does **not** silently switch to a newly invented answer extraction rule
or a different `reasoning_content` field. A model-only empty/EOS answer is retained
and scored, not repeatedly sampled. Transport failures remain resumable.

The audited reference Tiny inputs contain exactly 250 around, 200 rotation and 600
among rows (1,050 total; zero translation rows). Data audit now enforces that
reference distribution. A different Tiny selection must be a separately versioned
profile, not an unchanged reference run. Frozen historical-output replay reproduced
557/1,050 and every setting count/correct total; this does not accept new GPU generation.

### Outputs and failure accounting

```text
outputs/<model>/<run>/<benchmark>/
  contract.json
  dataset_manifest.json
  predictions.jsonl
  inference_summary.json
  inference_complete.json
  scores/
    reviews.jsonl       # ViewSpatial wrong-only judgments, resumable
    scored_records.json # original/parser/review decisions per sample
    metrics.json        # initial + reviewed counts, filtered denominator, diagnostics
  score.json            # final primary score with source-output hashes
```

Unparseable or failed judge requests do not become incorrect answers: final
scoring is withheld until all required reviews succeed. Correctly completed
reviews are reused only when the request hash still matches. Initial predictions
are never overwritten by review. Invalid or missing gold letters are rejected,
so two missing parses cannot incorrectly count as correct. Dataset/source/config
changes require a new run ID.

---

## Chartography, SPAR and ERQA project profiles

These portable adapters use the existing CLI and immutable run/output contract.
They are tested with synthetic inputs and preserved reference functions. Actual
GPU, full-dataset import and numerical parity acceptance remain pending.

### Data preparation and configuration

Acquire authorized original data. Example local import commands (replace source
filenames with your actual files; these are not download-layout assertions):

```bash
zwz-eval prepare --bench chartography \
  --source data/chartography/original/data.parquet --target data/chartography/prepared
zwz-eval prepare --bench spar \
  --source data/spar/original --target data/spar/prepared
zwz-eval prepare --bench erqa \
  --source data/erqa/original/erqa.tfrecord --target data/erqa/prepared
```

Chartography retains original chart paths. Set its `image_root` to the original
directory those paths are relative to. SPAR requires four `data/*.parquet` shards
and materializes original embedded image bytes. ERQA materializes the reference
TensorFlow `decode_image` → PIL PNG representation; no resizing or JPEG conversion
is added. Prepared roots contain `records.json` and `preparation.json`. Targets
must be new; prior data is not overwritten.

**ERQA preparation environment:** TensorFlow is needed only for TFRecord import.
Use a separate CPU preparation environment with `.[erqa-prepare]` if its protobuf
constraints conflict with your GPU/PPU runtime. Copy prepared outputs to the run
environment, which only needs API/Pillow dependencies. Never replace the PPU
torch/vLLM/protobuf installation just to read TFRecords.

Configuration additions to `datasets`:

```json
{
  "chartography": {"root": "../data/chartography/prepared", "image_root": "../data/chartography/original"},
  "spar": {"root": "../data/spar/prepared"},
  "erqa": {"root": "../data/erqa/prepared"}
}
```

Use the real Qwen3.6-35B-A3B judge, native model templates and the recorded model /
judge revisions for same-protocol comparisons. `serve-plan` prints commands only;
it does not start services, submit tasks or kill other processes. Separate main
and judge stages if they share GPUs.

```bash
zwz-eval audit --config examples/all.json --bench chartography
zwz-eval infer --config examples/all.json --bench chartography
## After making the correct judge service available:
zwz-eval score --config examples/all.json --bench chartography
```

Replace `--bench` with `spar` or `erqa` as appropriate. SPAR's **primary score**
does not need a judge. Its optional existing LLM diagnostic is separate:

```bash
zwz-eval review --config examples/all.json --bench spar
```

This command adds audit outputs and recomputes the public score report with the
audit attached under diagnostics; it does **not** replace the primary macro.

### Chartography: exact profile

- 100 original questions, text first then original chart image; no answer-format
  suffix. Native no-thinking request, 16,384 output tokens, 32,768 service context.
- Preserve raw content; use the source client's stripped content for judging.
  Retain any returned `reasoning_content` and report its presence.
- Judge sees question, stripped gold and response, **not the image**. The preserved
  rubric handles final revisions, intervals, units, category sets and multipart
  all-or-nothing correctness. Judge budget 8,192, no-thinking, original JSON schema.
- Preserve source verdict parsing, all-criteria pass test and contradiction/range
  guard. A contradictory or unparseable verdict gets up to five parse attempts.
- Empty model output is a scored zero without a judge request. Judge transport /
  parse failures remain errors, not scored zeros. Save raw malformed verdicts.
- The source's rule-matching result is a diagnostic only, never the primary score.

### SPAR: actual main-table profile, not the obsolete 512 run

The read-only source/result check on 2026-10-19 matched the main table to
`answer2048_llmjudge_v2`, not the earlier 512-token launcher. Base Overall
41.465008069491795 and reference Overall 45.69163206517503 match the arithmetic mean
of each result's 20 task scores. Source hashes are recorded in
[audit_evidence.json](audit_evidence.json). Directory naming alone was not used as
evidence of an LLM-judged primary score.

- 7,211 samples, 20 task types; 2,048 output tokens; no-thinking service; top_p=1.
- Preserve the task-specific source prompts and the shared answer-tag system
  prompt verbatim, including its visible reasoning wording. Do not “simplify” it
  because the service is no-thinking.
- Text before original image bytes. Image MIME declaration follows the source;
  no new image resize is applied. Reference context 131,072, maximum 3 images.
- Preserve first complete answer-tag extraction and task-specific final-answer
  recovery. MC exact match, numeric MRA and signed movement-delta MRA retain
  fractional values. Primary Overall = **equal-weight macro over all 20 tasks**.
- `official_micro` is diagnostic. Optional A3B audit reviews only scores below 1,
  upgrades to 1 only for an explicit `correct` verdict, and retains other fractional
  values. Its historical Overall is sample-micro; the port labels both reviewed
  micro and reviewed task-macro separately. Neither silently becomes primary.
- Audit uses task/question/gold/raw response, no images; 160 output tokens,
  no-thinking. Failed audits remain resumable errors.

### ERQA: source inference plus conservative bidirectional review

- 400 original TFRecord examples. Preserve decoded PNGs and question character
  `visual_indices` interleaving exactly. No extra question suffix/system message;
  4,096 output tokens; native no-thinking; reference context 131,072 / 16 images.
- Reject mismatched nonempty index/image lists instead of silently truncating a
  `zip`. Preserve source sorting, zero-index images and out-of-range-index behavior.
- Initial answer extraction is the **actual project's heuristic extractor**, not
  a newly substituted strict first-letter rule. Preserve its answer-tag priority.
- Audit every initially wrong and initially correct answer using their respective
  original text prompts. No images are sent to the judge. Initial-wrong budget is
  **1,024**; initial-correct budget is **160**. The source's literal `\\n` delimiters
  in the wrong-answer prompt are preserved, not silently rewritten.
- Revised count = initial correct + initially-wrong `correct` − initially-correct
  `incorrect`. Explicit semantic `ambiguous` retains the initial decision.
- Transport/JSON-parse failures are **not** semantic ambiguity. After three parse
  attempts they withhold final scoring and remain resumable. This deliberately
  does not certify old runs that retained failed verdicts as ambiguous.

### Outputs and acceptance

All three retain `predictions.jsonl`, per-request source hashes and finish reasons.
`scores/` holds per-sample decisions, raw judge journals and metrics. `score.json`
requires complete fixed denominators and records input/output hashes. Changed
data, source, template, judge or model configuration requires a new run ID.
Client/preprocessing library versions are included in the run contract, so a
Pillow change cannot silently reuse outputs rendered by another version. This
does not discover a remote server's runtime: record its model/template/runtime
identity separately in configuration and validate the service itself.
Text-only judge service plans use zero image slots and a separate 32,768-token
context; they do not inherit the main VLM's large multimodal warm-up allocation.

Automatic download URLs/revisions remain disabled for these additions until
verified. A working local import path is not a claim that a public download or
third-party redistribution license has been cleared. See the release gate.

---

## ScreenSpot-Pro and UI-Vision: reproduction and audit

These are **single-screenshot, single-click, offline grounding** evaluations.
They are not browser roll-outs, action-sequence success, answer similarity, or
LLM-judged results. Ground-truth rectangles are used only after generation.

### Native reference protocol

Use the normal `zwz-eval run --config model.json --bench screenspot_pro`
or `--bench ui_vision` entry point. The fixed reference is:

| Item | ScreenSpot-Pro | UI-Vision |
|---|---|---|
| Tasks | All 1,581 English positive instruction tasks | Element grounding only: basic 1,772; functional 1,772; spatial 1,935 |
| Input | Original instruction and one screenshot; image before text | Original `prompt_to_evaluate` and one screenshot; image before text |
| Generation | Native checkpoint template, `enable_thinking=false`, temperature 0, 8,192 output tokens | Same |
| Image handling | RGB, factor-32 smart resize, minimum 1,024 pixels, maximum 8,294,400; native PIL resize (BICUBIC); no crop/zoom/dual view | Same |
| Coordinate contract | Tool-call coordinates on a 1,000×1,000 reference screen, divide x,y by 1,000 | Same |
| Primary parser | Historical reference `<tool_call>\n{JSON}\n</tool_call>` parser, two-number point or four-number rectangle centre | Same |
| Hit | Inclusive point-in-ground-truth-box in normalized original-image coordinates | Same |
| Aggregate | Correct / 1,581 | Equal mean of the three split accuracies; **not** correct / 5,479 |
| Invalid generation | Incorrect; kept in denominator | Same |
| API/worker failure | Incomplete; no publishable final score until resolved | Same |
| Judge | Deterministic geometry, no A3B/LLM fallback | Same |

For an original screenshot `(W,H)`, prediction `(x,y)` on the 1,000 grid means
`(x/1000, y/1000)`. Original pixel box `[x1,y1,x2,y2]` becomes
`[x1/W,y1/H,x2/W,y2/H]`. Both inequalities include their endpoints.
Do **not** divide by screenshot dimensions a second time. If an API adapter
declares input-image pixels instead, divide by the **sent image's dimensions**,
not 1,000 and not the pre-resize dimensions.

UI-Vision upstream sorts reversed bbox endpoints. The local native reference
converter retains original bbox order; the 5,479 released annotation rows audited
on 2026-10-22 had **zero** reversed boxes, so this did not change those results.
The separate API loader explicitly implements upstream endpoint sorting.

The upstream UI-Vision example defaults to 256 generated tokens; 8,192 is the
project's reference adaptation, not an upstream default. A larger allowance does
not mean a model has to consume it. Report actual finish reasons/tokens.

### Updated audit traces, without silently changing historical scores

Native results now retain sample index/ID, original dimensions, resized dimensions,
actual rendered prompt, generation limits, finish reason/token count, and a
**separate** tolerant tool-call diagnostic point. The primary parser/score remains
the historical reference parser. Compact XML/JSON may therefore be diagnostic-parseable
but historically `wrong_format`; do not call the diagnostic a primary score.
This audit does not retroactively certify old runs that lack request traces.

Native execution manages its own local model backend, not an external `serve` pool.
Keep the selected GPUs free before launching it. Resumption is **split-level only**:
a complete split with a matching contract can be reused, but an interrupted split
is rerun as a whole. There is no native per-question continuation guarantee; an
explicit retry of an incomplete split may therefore regenerate already answered
items in that split. Do not describe this as the per-request recovery implemented
by the separate API adapter below. Full end-to-end GPU acceptance remains required.

`tools/audit_gui_predictions.py` reads old GLM journals and compares run-wide
coordinate interpretations **diagnostically**. It does not change source outputs
or pick whichever coordinate convention happens to hit each target.

### Portable API adapter (including GLM)

#### Recommended for a new external-model run

For endpoints supporting native function calls, use
`--prompt-contract explicit --coordinate-space normalized_1000 --parser tool_call --native-tools`
in a **new output directory**. This adds the actual sent image dimensions and the
conversion formula to the system prompt. The task instruction, screenshot order,
pixel budget, target boxes and inclusive hit rule stay unchanged. The parser accepts
the same unambiguous tool call with or without XML newlines; it does not search
reasoning or guess units. Out-of-range coordinates remain incorrect, with a separate
diagnostic count. No invalid answer is regenerated based on scoring feedback.

`--native-tools` supplies `computer_use` through the actual API `tools` and forced
`tool_choice` fields, with one `left_click` and two coordinates in [0,1000]. It
removes manually embedded XML tool signatures from the system prompt. This avoids
mixing Qwen-style XML with GLM/DeepSeek's native tool syntax. The server must support
these fields; an unsupported request fails rather than silently switching protocols.
Model coordinates still pass the same range and point-in-box checks. The streamed
response is decoded as UTF-8 bytes, including when no charset header is supplied.
Without `--native-tools`, manual XML tool prompting remains available explicitly.

This is a declared API adaptation, **not identical to the historical reference prompt
and strict parser**. To reproduce that historical protocol, explicitly use
`--prompt-contract historical --parser native_xml`. Defaults retain that behavior.
Do not overwrite historical scores or label newly adapted scores exact reproductions.
For a fair adapter comparison, fix these settings before the run and use them for
all models in that comparison. An API may still internally resize images.

`metrics.json` reports `unparseable_tool_call`, `coordinate_out_of_range`, and
`format_only_legacy_failure` alongside the score. These are diagnostics, never
automatic score corrections. Start with `--limit 20`, inspect the saved requests and
responses without choosing units by gold hits, then resume the same directory without
`--limit`. A smoke test reports incomplete and exits 2 by design.

Install `pip install -e '.[api]'`. Example with a supported OpenAI-compatible API:

```bash
set -eu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
## Set MAIN_API_KEY in your environment if required. Never put a key in a report.
python -m zwz_eval.gui_api \
  --bench screenspot_pro --data-root /data/screenspot-pro \
  --api https://your-endpoint.example/v1 --model GLM-5.3-Flash \
  --reasoning-effort low --coordinate-space normalized_1000 --parser tool_call --prompt-contract explicit --native-tools \
  --max-tokens 8192 --max-pixels 8294400 --workers 8 \
  --out-root /results/glm/screenspot_pro/fixed_contract

python -m zwz_eval.gui_api \
  --bench ui_vision --data-root /data/ui-vision \
  --api https://your-endpoint.example/v1 --model GLM-5.3-Flash \
  --reasoning-effort low --coordinate-space normalized_1000 --parser tool_call --prompt-contract explicit --native-tools \
  --max-tokens 8192 --max-pixels 8294400 --workers 8 \
  --out-root /results/glm/ui_vision/fixed_contract
```

Historical mode preserves the native **task/system text, grid and parser**;
the recommended explicit mode clarifies the system text and relaxes formatting only.
An API's
hidden server template, visual encoder, reasoning implementation and tokenizer
cannot be made identical to a Qwen checkpoint. GLM `reasoning_effort=low` is
**not** evidence of no-thinking. Report this model-specific distinction.

An endpoint genuinely using pixel-native computer-use coordinates may instead use
`--coordinate-space image_pixels --parser tool_call` with a **new output directory**.
The system then declares the actual sent resolution. This is an explicitly
different prompt/parser adapter, not an exact reference-text reproduction. Fix the
choice for the whole run before evaluating; never infer units by magnitude or gold.

The API adapter uses lossless PNG at the same factor-32 pixel budget, streams SSE,
captures actual native tool calls as well as text, saves content and reasoning
separately, and **never mines reasoning as an answer fallback**. It saves the
request JSON with image SHA-256 instead of bulky base64/authentication, and hashes
the original/sent images. Partial runs produce `status=incomplete, accuracy=null`
and exit 2. Failed API requests are resumable; malformed model answers stay scored
incorrect. A `--limit` smoke test does not reduce full-data denominators.

Resuming requires the same contract (annotations, source hashes, prompt, model,
units, parser, limits and Pillow). Historical GLM directories without a contract
are rejected. Do not reuse the old GLM queue's predictions under the new settings.

### Why old external-model scores may look strange

The historical GLM runner audited on 2026-10-22 had a 1,000-grid system prompt,
but a later parser treated most coordinates as original pixels, automatically
treated small `[0,1]` values as fractions, accepted loose coordinate prose, and
could use the reasoning channel when content was empty. Its summaries divided
only by successful API returns and could average only available UI splits.
One old run used approximately 1MP JPEG images and 8,192 tokens; a later run used
8.2944MP JPEG images, 2,048 tokens and `reasoning_effort=low`. These are **not one
stable reference configuration**. A stored `official_metrics.json` name does not
prove completeness or upstream equivalence.

The audit tool can inspect a caller-supplied prediction journal under declared
coordinate interpretations. Private case-study rows are not bundled in this source
export. Such diagnostic interpretations are not replacement leaderboard scores;
the audit does not rerun inference.

### Official sources and scope

- [ScreenSpot-Pro official evaluator](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/eval_screenspot_pro.py): original-pixel xyxy bbox normalization and inclusive hit rule.
- [UI-Vision official element evaluator](https://github.com/uivision/UI-Vision/blob/main/eval/grounding/eval_uivision.py): element task, pixel xyxy, endpoint sorting and geometric hits.
- [UI-Vision project](https://github.com/uivision/UI-Vision): Basic/Functional/Spatial and Final Avg are distinct from layout/action tasks.
- Dataset downloads: [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro), [UI-Vision](https://huggingface.co/datasets/ServiceNow/ui-vision).

Official source checked on 2026-10-22. Project reference settings are specified above;
do not describe all of them as upstream defaults or compare unqualified numbers
from different resolutions, prompts, parser revisions or task sets.

---

## MIMIC, MMStar and BabyVision: adapted reference paths

These adapters reproduce selected project implementations, not a claim of unmodified
upstream evaluation. Import authorized project-normalized annotations; do not substitute
an arbitrary upstream JSON with different prompts. Automatic normalized-release download
is not certified. No private credentials, model weights or dataset images are distributed.

### MIMIC

**Distributed protocol: complete 5,000-record Counting, not the obsolete
2,500-record raw-ID-deduplicated variant.** Historical data audits are
provenance notes only; they do not select or reduce the evaluation subset.
The package includes executable preparation, inference/resume, judge and scoring
code. Regression tests cover 5,000 distinct full identities sharing the same raw
ID, all 8,000 records across four tasks, and rejection of a 2,500-record legacy
dataset or incomplete legacy prediction journal.

Prepare `mimic_common.json`, `mimic_counting.json`, `mimic_listing.json` and
`mimic_oddone.json` through `zwz-eval prepare --bench mimic` (see CLI help).
Required counts are 1,000 / 5,000 / 1,000 / 1,000. Preserve question, response and
ordered images. Optional exact image maps relocate paths without fuzzy basename matching.
Counting identity includes question and images: repeated raw IDs must not collapse rows.

Generation uses the adapted shared multimodal formatter, no-thinking, 8,192 output
tokens and up to 256 images. The dedicated profile has a 32,768 context limit;
historical shared fleets could use a larger context. Do not relabel unlike runs as identical.
Scoring retains answer extraction, string matching, optional mathruler, and the source
text-only LLM rubric. The judge output cap is 8,192. Listing is micro-pooled object F1;
the final score is the equal-weight macro of the four task scores.

`datasets.mimic.mathruler_mode` must explicitly be `disabled` or `required`.
The latter also requires an exact `mathruler_version`. The example uses `disabled`
as an explicit example, **not certification of historical policy**. The selected
source permits environment-dependent mathruler imports; a historical log records
mathruler unavailable and a 2,500-row Counting subset. This package deliberately
requires all 5,000 Counting identities, so it must not be described as numerically
identical to that incomplete historical result.

Historical audits found that similarly named runs can use different weights or
chat templates. A task status, model roster or table value alone does not prove
completion or model identity. Verify the requested checkpoint and tokenizer,
all prediction IDs, the judge records and the final denominator before reporting
a result. Do not borrow missing verdicts from another model or template run.
Private deployment paths, model aliases, run identifiers and operational records
are not included in this public documentation. Non-identifying source hashes and
protocol evidence remain in `audit_evidence.json`.

### MMStar and BabyVision

Required counts: MMStar 1,500 and BabyVision 388. Preparation consumes the project's
normalized JSON/JSONL, preserving `query`, `system`, gold and custom `LLM_JUDGE_PROMPT`.
Images are preprocessed with the source qwen-vl-utils settings: min pixels 65,536,
max pixels 16,777,216, patch size 14. By default preparation records the annotations
and original image references (`image_mode: on-demand`); each inference request
reads and preprocesses its own ordered images, encodes lossless PNG in memory and
can start generation without an all-dataset PNG export barrier. This is not a
claim that all source images were already read or verified. Explicit `audit` reads
all referenced images; `--materialize-images` requests the optional offline PNG
export into a new preparation directory. Existing materialized bundles are never
decoded again or replaced by ordinary reuse. Use `.[platform-prepare]` for image
processing; do not replace a vendor PPU torch installation. Unlike the old
black-placeholder fallback, preprocessing errors fail closed. Native processor
parity still needs GPU acceptance. MIMIC uses its own preparation path unchanged.

Main inference preserves the empty system role and separate query turns. Generation:
16,384 output tokens, temperature 0, top-p 1, no-thinking, context 32,768, max 10 images.
The parser uses the **first** complete answer tag, then `Answer:` or the full output;
case-insensitive exact matching precedes the source short-MCQ first-letter fallback.
Do not replace this with the shared last-answer extractor.

Only initial wrong answers receive LLM review. The audited reference platform configuration
uses **Qwen3.5-35B-A3B**, not an interchangeable Qwen3.6 judge. It renders the native
tokenizer chat template without an `enable_thinking` argument, then appends exactly
`<think>\n\n</think>\n\n`. The rendered prompt is sent to `/v1/completions` with
temperature 0 and **96 output tokens**. This is not a 96-token chat-completions call.
Questions, gold, parsed predictions and per-item custom judge prompts retain source routing.

Select `judge_profile: platform_local96` as shown in `examples/all.json`. Supply the
actual served model ID, endpoint, revision and local complete tokenizer assets; the
tokenizer hash is part of the immutable run contract. Install `.[native-judge]` for
rendering. A generic global judge must not silently override this profile.
Reported accuracy is rounded to four decimals before conversion to percent, matching
the platform report; exact counts and unrounded accuracy are also retained.

### Completion and acceptance

Transport failures remain resumable and withhold a final score. Valid empty model
answers are retained, not repeatedly resampled. Data, runtime, tokenizer and protocol
changes require a new run ID. Local tests use synthetic data and mocked GPU boundaries;
they do not establish full real-dataset/model acceptance or redistribution clearance.

---

## AgentVista: adapted free-tool profile

### Standard preparation and evaluation workflow

AgentVista uses the same `zwz_eval` CLI and shared shell entrypoints as
other benchmarks. The complete retained implementation, including the agent loop,
prompts, tools and final judge, is packaged under
`zwz_eval/resources/vendor/agentvista/`. There is no separate installation
tree or independent protocol copy.

The executing computer or container must be able to reach its configured model
APIs and public search/visit sites. No SSH tunnel, container management or local
GPU model startup is performed by this adapter. Install from `eval/`:

```bash
python -m pip install -e '.[agentvista,data]'
```

First download the authorized release listed in [DOWNLOADS.md](DOWNLOADS.md),
then prepare its Parquet directory separately from inference:

```bash
python -m zwz_eval prepare --bench agentvista \
  --source /path/to/downloaded/agentvista --target /path/to/prepared/agentvista
```

Preparation produces the 209-question `val.json` and original image assets; it
does not call a model or implicitly download data. For this converter's output,
the data root is the prepared directory and the image root is its `images/`
subdirectory. See the existing-image-layout exception below.

Use [examples/agentvista.json](../examples/agentvista.json) as a configuration
example. All data/output paths, model identities and endpoints are user supplied:

```bash
export AGENTVISTA_DATA_ROOT=/path/to/prepared/agentvista
export AGENTVISTA_IMAGE_ROOT=/path/to/prepared/agentvista/images
export EVAL_OUTPUT_ROOT=/path/to/results
export MAIN_MODEL_ID=my-model
export MAIN_BASE_URL=https://model-api.example/v1
export MAIN_MODEL_REVISION=REPLACE_WITH_ACTUAL_REVISION
export MAIN_CHAT_TEMPLATE_SHA256=REPLACE_WITH_ACTUAL_SHA256
export JUDGE_MODEL_ID=Qwen3.6-35B-A3B
export JUDGE_BASE_URL=https://judge-api.example/v1
export JUDGE_MODEL_REVISION=REPLACE_WITH_ACTUAL_REVISION
```

These are placeholders, not a verified model release. Base URLs end in `/v1`;
do not append `/chat/completions`. Record the actual model/judge revisions and
native-template hash. Supply `MAIN_API_KEY` and `JUDGE_API_KEY` securely through
the environment, never in committed JSON. `EMPTY` applies only to intentionally
unauthenticated endpoints. Distinct model/protocol runs need distinct identities.

```bash
python -m zwz_eval plan --config examples/agentvista.json --bench agentvista
python -m zwz_eval doctor --config examples/agentvista.json --bench agentvista \
  --phase all --audit-data --check-tools --check-endpoints
bash scripts/evaluate.sh examples/agentvista.json agentvista infer
bash scripts/evaluate.sh examples/agentvista.json agentvista score
python -m zwz_eval summarize --root /path/to/results
```

Alternatively include `agentvista` in a selected list passed to
`scripts/run_benchmarks.sh`, provided the required endpoints are already available.
No separate launcher or GPU scheduler is added. The generic `workers` field does
not promise that many simultaneous agents; the retained native loop and its
eight-worker final-judge executor determine effective concurrency.

Inference and final-answer judging remain separate stages. Reinvoke compatible
commands for the retained resume behavior; do not resample completed wrong or
malformed answers. Missing results/deferred judgments remain visible, and a partial
native `judge_summary.json` is not a completed contract-bound `score.json`. Use the
shared `score` entrypoint, not a bare vendor invocation that bypasses validation.
The table score is correct final judgments divided by **209**, as a percentage,
not `accuracy_on_judged`.

### Retained protocol and tool profile

This is the project's adapted backend, **not the official paid search backend**.
The identified reference direct8 launcher and its inner runner have now been checked:
they select `multi_tool_agent_search_only`, 30 turns, 100 images, 65,536 total tokens,
1,024 completion tokens, search limits 100/100, threshold 12 and five API retries.
The wrapper now supplies the prompt key and search limits explicitly, instead of
inheriting the native CLI's different prompt default. The rendered prompt matches
all 209 saved reference injected-system-prompt files byte for byte. Evidence hashes
and the 36/209 archived result are recorded in `audit_evidence.json`.
No claim of live service availability is made by offline tests.

The wrapper sets `AGENTVISTA_FREE_TOOLS=1`. Importing the native tools package then
replaces the registry entries for the three permitted tool names. Merely finding
Serper/Jina classes or empty keys in the YAML does not mean they are used.

| Tool | Effective free implementation | Important limitations |
| --- | --- | --- |
| `web_search` | DuckDuckGo public HTML, titles/URLs/snippets, at most 10 results | No Serper key. HTTP failure returns an error; unrecognized/empty HTML can appear as no results. |
| `image_search` | DuckDuckGo image token + image JSON; Bing HTML fallback for request/JSON exceptions | Text query only in practice. Reverse-only requests do not upload images; the model is asked to describe the image and retry text search. A missing query token returns an error without the Bing fallback. |
| `visit` | Direct HTTP fetch and text stripping; public Jina Reader fallback on request failure | No authenticated Jina key or LLM summary. Not a browser/JavaScript renderer. Empty extracted text returns an error. |

The free registry does not invoke the paid classes' Serper, image-upload or LLM
summary routines. YAML `use_llm_summary: true` applies to the other Visit class,
not FreeVisit. In the currently packaged YAML the effective free visit character
limit is **150,000**, from `visit.max_content_length`; the 8,000-character class
fallback applies only when configuration supplies no limit. An inherited
`AGENTVISTA_FREE_VISIT_CHARS` could override it in the bare native script. The
retained reference trajectories contain 43 feedback messages longer than 8,000 and
two visit outputs at exactly 150,000 characters plus the 23-character feedback
prefix. Together with the YAML, this supports 150,000 rather than 8,000 for the
identified run. The portable wrapper now sets 150,000 explicitly to prevent an
ambient override. The complete shell environment was not snapshotted historically.

The wrapper supplies a total-tool threshold of 12. The packaged CLI defaults each
search tool's limit to **100**; the handler's web-5/image-3 defaults apply only when
those CLI attributes are absent. The wrapper explicitly pins both to 100, so 5/3 must
not be reported as its effective limits. The total budget is checked before the
next model turn; it is not a proven strict per-call cap for a multi-tool response.
These settings match the inspected reference launch record.
Other settings are 30 turns, up to 100 images, and a 65,536 total-token budget. Main calls request 1,024
completion tokens. Main thinking follows the adapted native/tool-use behavior,
with no-thinking on forced final calls without tools. No code-interpreter tool is
enabled. Final-answer judging is separate; all 209 questions remain in the main
denominator. Do not report accuracy-on-judged as the table score.

### Dependencies and failures

Before starting a full run, check the actual free-tool backends from the machine
that will execute the agent:

```bash
python -m zwz_eval doctor --config my-config.json --bench agentvista \
  --phase all --audit-data --check-tools
```

`--check-tools` makes three public example search/visit requests using the packaged
backends. It does not call a model or download benchmark data. A failed probe makes
`doctor` exit nonzero. The default doctor remains offline unless live checks are
explicitly requested. A passing probe establishes current connectivity only; it
does not guarantee every later web request will succeed.

For an existing historical dataset whose `val.json` image names start with
`images/`, set **both** `datasets.agentvista.root` and `image_root` to the directory
containing `val.json`. For the package's Parquet converter, names omit `images/`,
so `image_root` is the converter's `images` subdirectory (also the default).
The audit checks all 209 records and their image references. Saved trajectory
files named `original_image.jpg` may be resized/re-encoded diagnostics; use the
original dataset assets for reproduction.

- Search does not require paid search credentials; **model inference and judging
  still require a reachable model endpoint**, and may have monetary costs.
- Outbound access is needed to DuckDuckGo, Bing (fallback), Jina Reader (fallback),
  visited sites and retrieved image hosts. A deployment may require allowlisting.
- Public HTML endpoints can change, rate-limit or challenge clients. Free is not
  guaranteed availability and does not imply deterministic results across dates.
- No fallback to paid search is selected by this wrapper. Tool error strings remain
  agent observations, not automatically successful search results.
- Only public/non-sensitive queries and URLs should be sent to external services.
  Preserve trajectories, effective tool configuration and backend source hashes
  for audit; do not distribute private questions, keys or browsing content by default.

`tests/test_agent_free_tools.py` verifies the actual registry, expected HTTP routes,
failure observations, no reverse-image upload and no free-Visit LLM-summary call,
using mocked HTTP with no paid keys, plus the saved reference prompt hash. It does
**not** verify current live search availability or full reference trajectory parity.
Do not replace backends to improve a score
without separately versioning and rerunning all compared models.

---

## Final project-adapted script selection

The reference is **official benchmark logic plus the selected project
adaptations**, tied to recorded reference invocations. It is not the upstream's
current defaults, a generic newest launcher, or a scorer selected to improve a
score. A new GPU run is not needed to select and inspect this source chain.
Runtime acceptance and third-party redistribution review are separate matters.

### Selected source chains for all supported benchmarks

The paths below identify the retained project files, not commands that depend on
a private server. Complete native snapshots are under
`zwz_eval/resources/vendor/`; preserved definitions are under
`zwz_eval/resources/audited/`. Their manifests record source and packaged
SHA-256 values. Portable execution uses the common CLI, not the old deployment
shell scripts. Read the linked benchmark guides for the exact settings.

| Benchmark | Retained official/project source chain | Selected adaptation, not an upstream default |
| --- | --- | --- |
| Chartography | `run_chartography_all.sh` → `run_chartography_multi.py` → `judge_chartography.py` + `judge_guard.py`; audited `chartography*` | Bare chart/question prompt; 16,384 tokens; text-only A3B rubric plus contradiction guard; 100 questions. See CHART_SPAR_ERQA.md. |
| MIMIC | `run_mimic_eval.sh` / full-UID Counting repair → shared `infer.py` → `judge_qwenlm.py`, `cal_mimic_listing_f1.py`, macro scorer; audited `mimic_*` | **5,000 Counting**, full question/image identity; 1,000 each other task; Listing micro-F1 and four-task macro. Explicit mathruler branch, never auto-select from installed packages. See MIMIC_MMSTAR_BABYVISION.md. |
| MMSI-Bench | `infer_mmsi.py` → project rerun `score_mmsi.py`; audited `mmsi_final` and `mmsi_original` | 1,000 questions; 4,096 tokens; retained final-answer recovery, not a universal parser. Both extractor versions remain distinguishable. See SPATIAL_ADAPTERS.md. |
| SPAR-Bench | `run_sparbench_selected7_grouped3_llm.sh` → `run_sparbench_answer512.sh` with the actual 2,048-token override → `run_sparbench_vllm_answer.py`; audited `spar` | `answer2048_llmjudge_v2` evidence selects 2,048, despite the wrapper's old filename; 20-task rule macro is primary. LLM audit is separate. See CHART_SPAR_ERQA.md. |
| MMLongBench | `run_mmlongbench_visual_core_16gpu.sh` with recorded reference overrides → `eval_api_batch.py`, `data.py`, `utils.py`, `utils_prompt.py` | Non-Counting visual25; official prompts, 2,048 output tokens, five K lengths, family-specific 100/50 unique-ID caps; no extra robust-output prompt. See PROTOCOLS.md. |
| MileBench | `run_milebench_eval.sh` → shared adapted `infer.py`; `run_milebench_llm.sh` → `cal_milebench_with_llm.py` | 29 test tasks; 8,192 output tokens; original task metrics and MC letter extraction, not generic semantic replacement. See PROTOCOLS.md. |
| MemLens | `run_memlens_eval.sh` → shared adapted `infer.py`; `run_memlens_judge_official.sh` → `judge_memlens_official.py` | Four contexts × 789; 8,192 output tokens; official task rubric, normalization and auto-zero branches; not exact-match as the reported score. See PROTOCOLS.md. |
| ERQA | `run_erqa_16gpu.sh` → `run_erqa.py` → `judge_erqa_wrong_answers_a3b.py` and `judge_erqa_initial_correct_answers_a3b.py`; audited `erqa*` | TFRecord visual-index interleaving; 4,096 tokens; original heuristic parse then conservative bidirectional review. See CHART_SPAR_ERQA.md. |
| ViewSpatial-Bench | Historical inference → initial scoring → wrong-only model review; packaged adapter: audited `viewspatial` | 420×280 LANCZOS/JPEG95; 1,024 tokens; original option parser, wrong-only review. See SPATIAL_ADAPTERS.md. |
| MindCube-Tiny | `run_mindcube_qwen35_16gpu.sh` → `infer_openai_pool.py` → `scripts/run_evaluation.py`, `core/extractors.py`, `core/base_metrics.py` | raw_qa 1,050 with 250/200/600 settings; 1,536 tokens; source pattern priority; semantic review is not primary. See SPATIAL_ADAPTERS.md. |
| AgentVista | the selected direct-API launcher → adapted agent/free tools; `agentvista_postjudge.py` → `general_qa_tool.py`, packaged as final-answer scorer | Search-only system prompt, 30 turns, 1,024 tokens/call, threshold 12, free search/visit, final-answer score over 209; native-default judge thinking, not a later dual-judge experiment. See AGENTVISTA_FREE_TOOLS.md. |
| ScreenSpot-Pro | `run_screenspot_pro_one_model.sh` → `eval_grounding_topology.py` → `models/qwen3_5.py` | en/positive/instruction, native no-thinking, 8,192 tokens, 8,294,400 pixels, retained tool-call coordinates, inclusive point-in-box; 1,581 examples. See PROTOCOLS.md. |
| UI-Vision | `run_ui_vision_one_model.sh` → retained element conversion → same GUI evaluator/model adapter | basic/functional/spatial, original coordinate and pixel policy; three-split macro, not sample-micro. See PROTOCOLS.md. |
| MMStar | Saved project `evaluate_v3` task config → `core/inference.py`, `plugins/parsers.py`, metrics → `core/llm_client_local.py` | Normalized 1,500 records; 16,384 tokens; first-answer-tag parser, short-MCQ branch, wrong-only native-completion 96-token local judge. See MIMIC_MMSTAR_BABYVISION.md. |
| BabyVision | Saved project `evaluate_v3` task config → same platform source chain, preserving per-item judge prompts | Normalized 388 records; same platform ACC path, not the separate upstream boxed-answer evaluator. See MIMIC_MMSTAR_BABYVISION.md. |
| VStar | the selected project platform launcher → open/technical `run_single_eval.sh` → evaluate_v3 inference, parser, ACC, local judge | 191 normalized rows, 16,384 tokens, wrong-only local-96 A3B. See PLATFORM_BENCHMARKS.md. |
| HR-4K | Same platform chain; `hrbench-4k-shuf-imgplaceholder.jsonl` | 800 normalized rows, original prompts and platform scoring. |
| HR-8K | Same platform chain; `hrbench-8k-shuf-imgplaceholder.jsonl` | 800 normalized rows, original prompts and platform scoring. |
| MME-RealWorld-en | Same platform chain; `mme-realworld-shuf-NasPic-imgplaceholder.jsonl` | 23,609 rows; not classic MME or a substituted upstream metric. |
| MME-RealWorld-cn | Same platform chain; `mme-realworld-cn-shuf-NasPic-imgplaceholder.jsonl` | 5,917 rows; original Chinese prompts and platform scoring. |
| ZoomBench | Same platform chain; `pbench-shuf-NasPic-imgplaceholder.jsonl` | 845 rows; not an added crop/search inference method. |
| Perception-Bench | Same platform chain; `perceptionbench.jsonl` | 3,000 rows, original normalized release. |
| TreeBench | Same platform chain; `treebench-shuf-imgplaceholder.jsonl` | 405 rows, original normalized release. |
| O3-Bench | Same platform chain; `insight-o3.jsonl` | 345 rows; preserve original extra layout fields. |
| CountQA | Same platform chain; `countqa-shuf-imgplaceholder.jsonl` | 1,528 rows; platform ACC with wrong-only local judge, not a new numeric scorer. |

### Current shell defaults are not the reference invocation

Read-only inspection on 2026-10-20 found that the current canonical
`run_mmlongbench25.sh` defaults `MMLONGBENCH_ROBUST_OUTPUT` to **1**, while the
older `run_mmlongbench_visual_core_16gpu.sh` defaults output length to **512**
and robust output to **1**. Neither may be run bare and called the selected
reference setup. Recorded reference args select **2,048**, non-Counting visual25 and
**robust-output disabled**. The portable wrapper explicitly selects those
settings and clears inherited extra-prompt switches. The old scripts are not
copied into the package as competing public entrypoints.

The portable VH fallback retains `[0,1,1,0]` but now binds it to selected-row
position, fixing the retained source's concurrency/resume-dependent state. This
changes only the fallback for unparseable VH outputs, not the prompt, subset or
valid Yes/No parser. It is recorded in vendor provenance and the run contract;
old results are preserved and are not silently converted into corrected scores.

Likewise, `answer512` in a SPAR filename does not override its recorded
2,048-token invocation. A later no-thinking AgentVista judge does not override
the retained native-default final-answer judge. A Qwen3-VL template workaround
for another checkpoint does not become the reference template. These are precisely
why the invocation, model identity and downstream scorer must be read together.

### Source verification versus unresolved historical scores

On 2026-10-20 all **44** previously recorded private source/config/summary
artifacts were read again from their recorded locations and their SHA-256 values
were unchanged. This confirms source-snapshot stability, not that every artifact
is a script or that a current file existed unchanged at an earlier execution.
Native snapshots and extracted scoring definitions remain separately hashed.
The source checks did not launch evaluation or modify historical outputs.

The owner's final instruction selects complete 5,000-row MIMIC Counting even
while its later reference result location is being set aside. The MileBench
table/cache discrepancy and MMLongBench filtered row are historical result
questions, not permission to replace official/project scoring logic. Some
historical environments were not fully snapshotted: this document does not
invent missing template hashes or claim complete reconstruction of every old
execution. No numerical rerun acceptance is claimed by this script selection.
