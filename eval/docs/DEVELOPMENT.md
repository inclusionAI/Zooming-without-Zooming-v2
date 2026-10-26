# Implementation status

Development version 0.1.0.dev0 — not yet approved for public release or a claim of end-to-end parity.

| Component | Implemented | Remaining acceptance gate |
| --- | --- | --- |
| Portable user setup | Installed-wheel `init` configuration scaffolding and read-only `doctor` path/dependency/data/endpoint checks; no per-model scripts | Model IDs/local metadata do not attest the actual remote serving template or working GPU runtime |
| Protocol registry (25 benchmarks) | Explicit task, generation and aggregation identities | Real-data/GPU acceptance is separate from source selection |
| Downloaded-data / selected-bench workflow | `convert`, `prepare-suite`, `run-suite`; two public Bash entrypoints; strict raw MIMIC/platform annotation recipes; MindCube Tiny raw_qa generation; native datasets registered without copying | Synthetic conversion/dispatch and historical metadata reconstruction are not full upstream-payload, historical image-byte or GPU acceptance |
| Local single-machine service lifecycle | Explicit `serve` uses per-replica ports, bounded startup-only collision retries, owned cleanup and caller GPU allocation; examples specify 16 main / 8 TP2 judge endpoints | `run-suite` does not yet automate per-benchmark main/infer/release/judge/score/release; callers must manage role changes. CPU fake-service tests are not GPU acceptance |
| Native GUI resume | Complete split output is validated and reused | Interrupted splits do not yet persist per-question resume journals; no claim of exact per-question recovery |
| Owner's complete table | All 25 execution adapters wired; explicit `coverage` command | Real-data/GPU acceptance is separate from implementation coverage |
| Ten additional platform benchmarks | Fixed normalized counts, Qwen preprocessing, journal inference/resume, wrong-only native-96 A3B judge and rounded ACC | Requires authorized normalized annotations/images and matching runtime; synthetic dispatch checks are not a real-data run |
| MIMIC / MMStar / BabyVision | Preparation, hash audit, resumable inference and distinct source-derived judges | MIMIC historical mathruler policy; normalized release provenance; native image/tokenizer and real GPU parity |
| Source audit | Source-hashed parser/metric/rubric excerpts and regression tests | Real runtime and redistribution review remain separate |
| MMSI / ViewSpatial / MindCube | CLI preparation, strict data audit, inference journal, per-sample parsing, ViewSpatial review and aggregate | Real data import and numerical/GPU parity acceptance; automatic download revision verification |
| Chartography / SPAR / ERQA | Preparation, immutable inference journal, distinct native metrics/rubrics, resumable judgments; SPAR optional review stays diagnostic | Real data import and numerical/GPU parity; original TFRecord decode acceptance; source revision/redistribution checks |
| Common configuration | Portable JSON, environment credential references, endpoint validation | Additional nested-schema validation |
| Data audit | Required annotations, counts, IDs, image existence | Complete MMLongBench family-specific reference checks |
| MileBench / MemLens inference | Journal, resume, immutable contract, raw output retention | Real endpoint and GPU integration tests |
| Scoring | Unified invocations, denominator checks, preserved parser fixtures; deterministic selected-row VH fallback; native-cache integrity checks and concurrent interruption/resume regression | Real-data and judge acceptance; old stateful VH scores are not automatically certified |
| Other five inference adapters | Portable wrappers, explicit task routing, completion checks | Full GPU/native runtime integration tests |
| Downloads | HF entries for all 25, source provenance/access observations, pinned revisions, file manifests, resumable downloads, safe extraction | Chartography anonymous access returns 401; community/converted mirrors are explicit; not every raw HF schema has a project-normalization converter |

Never substitute an unfinished adapter with a generic prompt. Never turn API failures
into model errors, silently exclude empty answers, or promote partial scores to final scores.

---

## Contributions

1. Keep benchmark adapters separate from scheduling, storage mounts and model serving.
2. A prompt, parser, judge, sample selection or aggregation change is a **protocol
   change**, not an invisible bug fix. Version it and rerun/recompute every comparison
   model consistently. Never tune parsers using the gold answer to rescue selected cases.
3. Keep raw outputs, failures, denominators and source/data hashes auditable. Do not hide
   API failures as wrong answers or report a partial macro as a completed benchmark.
4. Add synthetic regression fixtures. Do not commit benchmark images, large datasets,
   checkpoints, credentials or private evaluation responses.
5. Third-party code needs upstream attribution, original license text and permission to
   redistribute; Apache-2.0 for new code does not override third-party terms.
6. Run `python -m unittest discover -s tests -v`, `zwz-eval check-source`, and
   `zwz-eval release-check`. The last command intentionally blocks public release
   until the outstanding rights and real-run acceptance evidence are reviewed.

The source export tools are maintainer utilities, not part of the evaluation pipeline.
Do not regenerate snapshots from a different source pack while retaining the same
protocol ID. Preserve original and exported hashes when recording reviewed changes.

---

## Security and private data

Do not include API keys, private endpoints, checkpoints, dataset images, raw model
outputs or private run logs in an issue or public release. Environment variable names
can be shared; their values cannot. Source-archive builds exclude .env, data, models,
outputs, build artifacts and logs. The automated scan is a heuristic, not proof that
no secret exists; manually review the final archive before publication. Source exports
use an explicit directory/file-type allowlist, reject unexpected binary/large files
and scan docs, examples and tools as well as runtime code. Exporting an unpacked
source archive regenerates its manifests instead of recursively including old ones.

Keep personal evaluation projects under the repository's ignored `local/` directory
or outside the checkout. Both Git defaults and the source builder exclude root
`local/`; the original files remain available for local use. Unknown directories
are not automatically trusted exports. Do not commit personal configs under
`examples/` or `zwz_eval/`, and never store real API keys in config files.

Evaluate trusted checkpoints/datasets only. Model loaders can execute installed model
code, and grounding requires a GPU runtime. AgentVista makes model-directed network
requests to live websites. Run it in an isolated environment with restricted network
egress; do not expose host services, metadata endpoints, private mounts or unrelated
credentials to the agent. Its free-tool implementation is an evaluation compatibility
snapshot, not a hardened general-purpose browser sandbox.

Archive extraction rejects links, devices and path traversal. Large valid archives can
still exhaust disk capacity; check storage/quota and inspect uncompressed sizes first.
The project never automatically deletes existing outputs or kills GPU services.

Report suspected vulnerabilities privately to the repository maintainer; no maintainer
contact address has been assigned in this development snapshot.

---

## What is and is not validated

Run `python tools/validate.py` to regenerate the machine-readable validation report.
Tests use temporary synthetic annotations and a local HTTP stub, not benchmark scores.

- Protocol task identity (25 / 29 / 14 / four-context configuration).
- Immutable contracts, run locks, safe path components and recoverable torn journals.
- OpenAI-compatible served-model validation, thinking fields and non-retryable 404s.
- Source parser fixtures: answer boxes, final-prefix extraction, refusal rubric routing,
  strict paper conclusion, inclusive point-in-box and invalid-format denominators.
- Safe archive rejection of traversal and symbolic links.
- Command plans for all 25 adapters, source-script existence and credential-free argv.
- All ten added platform routes have small synthetic preparation/inference/judge/
  scoring/resume tests, checking 16,384-token no-thinking main calls, first-answer
  parsing, wrong-only 96-token raw-completion judging and no duplicate resume calls.
  Their complete normalized denominators are checked separately. These are not
  full real-data or GPU tests.
- Synthetic MileBench 29-task end-to-end inference/scoring/resume using the real
  preserved formatter and scorer subprocess. Two MC examples reuse the same original
  ID across tasks with different correct letters, testing cache isolation.
- MemLens converter/materializer: session roles, dates and interleaved images survive.
- Four synthetic MemLens splits × 789 completed inference records through the actual
  preserved judge subprocess and local HTTP stub: 3,156 judge calls, ten task-key
  routes, 2,048-token/no-thinking parameters, full macro, no additional judge calls
  on resume, and rejection of a corrected answer under an old native cache. This
  does not generate 3,156 new model answers or test real judge quality.
- Native judge cache bindings reject changed input/scorer settings, unbound legacy
  caches and source files modified during a scorer subprocess.
- Missing API response fields are service errors; explicit empty/EOS model outputs
  remain completed outputs and are not resampled.
- Native GUI lifecycle regression tests cover stable GPU assignment, exact sample
  coverage, startup failure/hang, worker crash, inference timeout, shutdown errors,
  model close and cleanup of owned engine children on normal exit or SIGTERM.
  They use CPU subprocess fixtures, including a separate unrelated process that
  must survive cleanup. Run them through `python -m unittest discover -s tests -p 'test_gui*.py'`
  so spawned workers can import the test module; these checks
  need permission to inspect their child processes with `ps`.

Install the `api` extra before running the full suite (Pillow is used by synthetic
fixtures). Individual optional-boundary skips are listed in the validation report;
a skipped test is not acceptance evidence. The CI workflow installs that extra.
API constraints in constraints-api.txt record the
tested library versions, not an entire GPU/runtime lockfile.

CI also builds a wheel, replaces the editable install with that wheel, and checks
its resource manifest and command entrypoints from outside the checkout with
Python isolated mode. This is deliberately separate from source-tree tests:
an editable installation can mask a missing native adapter file. These installed
package checks do not import torch/vLLM or establish native GPU compatibility.

Still required before publishing reproducibility claims:

1. Full normalized dataset manifests and permission to distribute any imported data.
2. Real native GPU runs for MMLongBench and the Qwen GUI backend on the supported runtime.
3. Live AgentVista tools and the exact intended judge checkpoint/revision.
4. Comparison against archived reference per-example decisions and all task denominators.
5. Third-party and adaptation licensing clearance.

### Separate frozen historical-output evidence

On 2026-10-19, read-only reference output replay verified SPAR's 7,211 rows/20 tasks
(0 per-example differences against current reference code, macro 45.691632%),
MMSI's 1,000 saved per-example decisions (323 correct, both retained parser
versions matched this file), and MindCube's 1,050 inputs (557 correct, all setting
counts/correct totals matched). Archived MemLens verdicts also reaggregate to
42.585551% over four 789-row splits. Hashes and exact scope are in
`audit_evidence.json`; private questions, images and model outputs are not bundled.
These checks are distinct from synthetic unit tests and do not establish generation
parity, new LLM judge quality, or all-16 acceptance.

Two historical warnings are deliberately not labeled as acceptance: one reference
MMLongBench Retrieval-image K128 response was filtered after parsing (300 cache /
299 final rows), and the inspected 29-task MileBench summary averages 70.420005%
while the existing table says 70.08%. Neither historical table nor outputs were
changed; the latter discrepancy still needs authoritative-source reconciliation.
See the audit evidence for artifact hashes and scope.

The subsequent MileBench replay used all 6,440 saved answers and historical cached
letter extractions, without API calls. All 29 task metrics reaggregate to
70.420005%; no multiple-choice gold was outside its choices and no extraction was
missing. The reference loader skips malformed cache lines (300 in the inspected
cross-model files); replay followed that behavior. This confirms arithmetic, not
that every legacy cached extraction was bound to the same answer revision. The
70.08% table discrepancy therefore remains unresolved rather than silently fixed.

A source-package build or `--help` check cannot establish any of these. Keep CUDA/PPU
runtime acceptance separate; torch/vLLM installation is platform-specific. This turn
does not start evaluation jobs, modify private scores or claim a 16-suite real-data run.

---

## Public-release checklist

### Confirmed

- New framework code: Apache-2.0, with the root LICENSE and NOTICE retained.
- On 2026-10-21 the human owner confirmed public-distribution permission for
  owner-authored evaluation adaptations and authorized contributor changes.
  The machine-readable scope is `resources/release_permissions.json`.
- Portable `init` templates are packaged in the wheel. Local paths, model/service
  identities and output locations are configured, not copied into model-specific scripts.
- The public source builder excludes datasets, weights, private outputs, `.env`
  secrets, local launchers, build leftovers and the private `bundled_data/` tree.
- Synthetic contract/pipeline tests and installed-wheel setup checks are recorded
  separately from real GPU/full-dataset acceptance.

### Still required before calling this a cleared public release

The local service startup update (2026-10-25) adds per-replica communication port
isolation, startup-only bounded collision retries and owned process cleanup.
See [service startup](SETUP.md). CPU/fake-service tests do not establish
native GPU acceptance; existing jobs must not be interrupted or called patched
merely because a new source archive was built.

Platform image preparation now defaults to on-demand requests instead of a
full-dataset preprocessed-PNG barrier. Annotation manifests, per-question image
order, Qwen preprocessing settings, prompts and score rules are retained. Explicit
offline materialization and full-image auditing remain available; old prepared
assets are not redecoded or replaced on reuse. CPU/synthetic equivalence tests
do not establish full real-data/runtime acceptance for this execution change.

Commands:

```bash
python tools/validate.py
python -m zwz_eval release-check
## Review-only export while the release check remains nonzero:
python tools/build_source_archive.py --draft --output /tmp/ZwZ-v2-Evaluation-Source.tar.gz
```

`--draft` does not bypass source integrity or secret scanning. It acknowledges
unresolved release gates, not legal clearance. Do not remove the gates to imply
that authorization or real-model acceptance has occurred.

Before exporting, the builder may set `ZWZ_PRIVATE_IDENTIFIERS` in its
private environment to a comma-separated list of personal names, private model/run
aliases and internal service addresses to forbid. The scanner checks both member
names and text content, case-insensitively, without printing the matched value.
Keep this list outside the repository, package and logs; the source does not
contain anyone's personal denylist. Generic path, run-ID and credential checks
remain enabled without it. Static scanning is a release aid, not a guarantee
that every possible private string or credential format has been recognized.

---

## English source presentation

Public documentation is English. Non-English benchmark prompts, expected answers,
regex patterns and parsing constants retain their original runtime values and are
represented by Unicode escapes where needed. Translating benchmark content would
change evaluation semantics and is intentionally not done.

Explanatory non-English comments are replaced with an English protocol reference;
original docstring values use Unicode escapes. Full executable AST comparison
includes docstrings and preserves every literal value. Original vendor/source hashes remain in
provenance metadata; packaged hashes reflect this presentation-only transformation.
This does not certify numerical reproduction or clear redistribution release gates.
