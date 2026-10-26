# Run the same protocol on another machine

The public framework is path-configurable, not tied to the owner's cluster.
Use the same benchmark releases, task selection, prompts, image ordering,
preprocessing and judges. Changing a filesystem path is supported; changing
questions, gold answers or the scoring profile does not establish comparability.

## 1. Install in a clean environment

Python 3.10+ is required. From the source root:

```bash
python -m pip install -e '.[api,data]'
python -m zwz_eval --help
```

Install the platform-matched torch/vLLM build separately. In particular, do not
replace a PPU build with an ordinary CUDA wheel. Data preparation can use a
separate CPU environment. The API client does not need to load the main model.

| Selected work | Additional requirements |
| --- | --- |
| Local raw platform image conversion | `.[platform-prepare]`; torch must match the platform |
| ERQA TFRecord conversion | `.[erqa-prepare]` in a preparation environment |
| MMLongBench native inference | `.[mmlongbench]`, plus platform torch and compatible transformers |
| Platform wrong-answer judge | `.[native-judge]` and the exact local Qwen3.5-35B-A3B tokenizer |
| AgentVista inference | `.[agentvista]`; outbound access for the free search/visit tools |
| ScreenSpot-Pro / UI-Vision inference | Platform torch/vLLM, compatible transformers and Pillow; retained Qwen backend only |
| Other API inference / native score clients | `.[api]`; see `doctor` for selected runtime metadata checks |

These are component requirements, not a GPU lockfile. Inspect the recorded source
and runtime compatibility before installing/upgrading packages in a working model
environment. `constraints-api.txt` records tested API-layer versions only.

## 2. Initialize one local project

The command works with both an installed wheel and an editable checkout; it does
not require this repository's `examples/` or `scripts/` directory at runtime.

```bash
python -m zwz_eval init --directory local/my-eval \
  --bench mmlongbench,milebench,memlens \
  --model-id my-model --model-path /srv/models/my-checkpoint \
  --data-root /srv/benchmarks --output-root /srv/eval-results
```

It creates only three files, never a model-specific launcher:

```text
local/my-eval/
  model.json       # model/judge identity, service endpoints, hardware, output root
  datasets.json    # downloaded source files, image roots and conversion options
  README.md        # the commands to prepare, check, infer, score and summarize
```

Omit `--data-root`/`--output-root` to use `data/` and `outputs/` relative to the
generated project. Relative paths are always resolved against their JSON file,
not the shell working directory. Paths containing spaces are supported. Existing
projects are never overwritten. Omit `--bench` to generate all 25 entries; select
only the benchmarks whose data you have actually prepared.

Edit `model.json`: model and judge names must match their served IDs; record actual
checkpoint/revision identities and the main model's native chat-template hash.
Do not leave `RECORD_ACTUAL_*` in published experiment metadata. Keep credentials
in the named environment variables, not JSON. The judge is not a freely
interchangeable model if the reported results must match this protocol.

## 3. Point the manifest at your downloaded data

The generated manifest uses the reviewed release layout. If your copy has another
top-level directory, edit its `source`/`image_root`, not the conversion/scoring code:

```json
{
  "output_root": "/srv/benchmarks/prepared",
  "datasets": {
    "memlens": {
      "source": "/data/my-memlens-release/annotations",
      "image_root": "/data/my-memlens-release/images",
      "format": "raw"
    }
  }
}
```

Keep all selected entries in your real manifest. The snippet illustrates path
relocation only. `normalized` imports support an exact image prefix or explicit
image map where documented; basename-only fuzzy matching is prohibited. Native
MMLongBench/GUI data is audited and registered without copying the tree.
See [data formats](DATA.md), [raw conversion](../README.md) and the complete
[download catalogue](DOWNLOADS.md). Use pinned revisions. Some releases require
manual archive extraction or access authorization; no dataset is downloaded by
`init`, `doctor`, or an evaluation run.

The MIMIC/platform raw converters retain historical annotation fingerprints. An
arbitrary custom dataset is not accepted under an official benchmark's name just
because it has `images/query/response`. A genuinely new dataset needs an explicit
adapter, expected denominator, prompt and scorer with tests; see CONTRIBUTING.md.
Do not remove fingerprint checks to make a different set appear comparable.

```bash
python -m zwz_eval prepare-suite --manifest local/my-eval/datasets.json \
  --config local/my-eval/model.json --output-config local/my-eval/ready.json
python -m zwz_eval doctor --config local/my-eval/ready.json \
  --bench mmlongbench,milebench,memlens --phase infer --audit-data
```

Preparation preserves original data and publishes a result only after the checks
for its input mode. New platform inputs use annotation-only on-demand preparation:
they do not pre-read or export every image. Original images are preprocessed per
inference request using the retained settings. `--materialize-images` explicitly
selects offline platform PNG preparation; existing prepared bundles are not
converted on reuse. `audit` or `doctor --audit-data` remains the separate full-image
check, not a required preprocessing barrier before generation.
`ready.json` is the runnable path projection; do not edit the generated dataset
receipt to bypass its hashes. Reuse the same full config across inference and
scoring. A changed selection/config/code requires a new run ID, not mixed caches.

## 4. Start compatible services, then infer and score separately

```bash
python -m zwz_eval serve-plan --config local/my-eval/ready.json \
  --bench memlens --role main
```

This prints the exact argv/environment; it does not execute them. Start those
commands in your own service terminals. Provision the selected benchmark's full
context/image limits and native model template. Do not silently truncate inputs
or change image resolution to fit. Multiple endpoints mean multiple independent
servers, not automatic replicas. A server suitable for a short-context platform
task is not automatically suitable for MemLens 256k or MMLongBench 800 images.
The packaged template lists 16 TP=1 main endpoints and 8 TP=2 judge endpoints
for a 16-GPU machine used by one role at a time; adjust all three together for
your hardware. The optional foreground `serve` command starts one declared
benchmark/role pool. `run-suite` only calls available services in sequence; it
does not automatically switch main/judge roles or manage an entire GPU suite.

```bash
export MAIN_API_KEY=EMPTY JUDGE_API_KEY=EMPTY  # only for unauthenticated local services
python -m zwz_eval doctor --config local/my-eval/ready.json \
  --phase infer --check-endpoints
python -m zwz_eval run-suite --config local/my-eval/ready.json --phase infer
# Release your main service resources yourself, then prepare the required judges.
python -m zwz_eval serve-plan --config local/my-eval/ready.json \
  --bench memlens --role judge
python -m zwz_eval doctor --config local/my-eval/ready.json \
  --phase score --check-endpoints
python -m zwz_eval run-suite --config local/my-eval/ready.json --phase score
python -m zwz_eval summarize --root /srv/eval-results
```

Run only a compatible service group at once using `--bench`. The platform's native
96-token judge uses its declared model and raw completions; other tasks retain
their own policies. See [judge setup](SETUP.md).
GUI inference starts its own local GPU replicas instead of using your main API.
The default MMLongBench API uses `file://`: service and client must see the same
image root at the same path, with the service's local-media permission enabled.
Hosted APIs without that filesystem are not equivalent backends for that adapter.

`doctor` is read-only. Without the optional flags it checks local paths, selected
dependency metadata, judge routing and local template metadata. `--audit-data`
hashes the complete selected data; `--check-endpoints` queries `/v1/models` only,
without generating answers. For AgentVista, add `--check-tools` to exercise the
packaged free search/image-search/visit backends with public example queries.
This does not download benchmark data or call a model. Exit 2 means a requested check failed. Neither a
model ID nor a local hash proves what a remote server actually loaded. Runtime
imports/GPU operation, exact processor behavior and real-model accuracy still
require platform-specific integration acceptance.

Results use `output_root/model_id/run_id/benchmark/`. Resume the same command with
the same configuration; failures do not silently become zero scores. Old outputs
are not deleted. Native GUI resumption reuses only complete matching splits;
an interrupted split is rerun in full, not continued per question. The separate
GUI API adapter has a per-request journal instead. Template or data changes need
new run IDs. See [protocol details](PROTOCOLS.md).

## 5. Public source versus private distribution

Publish source, not local datasets, model weights, secrets or evaluation outputs.
The source builder uses an allowlist and omits `bundled_data`, local launchers and
machine-specific leftovers. The owner's private archive may contain authorized
historical annotations/Chartography data; it is **not** the public source archive.

New framework code uses Apache-2.0. Vendored code and scoring excerpts retain their
original rights. `release-check` still blocks a claim of cleared public release
while contributor/third-party authorization and integration evidence are unresolved.
A draft source archive is for technical/legal review, not proof of legal clearance:

```bash
python -m zwz_eval release-check
python tools/build_source_archive.py --draft --output /tmp/ZwZ-v2-Evaluation-Source.tar.gz
```

Do not disable the checks or relicense third-party content to make the gate pass.
The current owner authorization and remaining actions are listed in
[release readiness](DEVELOPMENT.md).

---

## Local services: isolated ports and bounded startup retries

For normal evaluation use one terminal and one command:

```bash
bash scripts/evaluate.sh local/my-eval/ready.json hr8k
```

This manages main startup/readiness, inference, main cleanup, judge startup when
required, scoring and judge cleanup. Append `--services external` to use existing
APIs without lifecycle management. The manual commands below are advanced tools,
not required evaluation steps.

### Request concurrency versus GPU count

`workers` is the total number of client request threads, not the number per GPU.
For 16 single-GPU replicas and capacity 16 per replica, set `workers` to 256
and each service's `--max-num-seqs` to 16. Setting both GPU count and total
workers to 16 feeds roughly one request per replica. Image decoding, CPU memory,
KV-cache capacity and token lengths can reduce effective concurrency. API
continuous batching is not the same call boundary as native `generate(batch)`;
do not claim a fixed 256-item batch or a guaranteed 16-fold speedup.

Worker count belongs to the recorded run configuration. Changing it mid-run
requires an explicit, hash-bound execution receipt; never edit the original
contract or loosen prompt, data, model or scoring checks to resume.

These commands serve **one local host**. They do not submit cluster jobs, install
runtimes, start containers, or change benchmark prompts/scoring. Native GUI
benchmarks start their own backends and intentionally reject these API commands.

`serve-plan` still prints commands only. To execute a reviewed plan with startup
checks and cleanup, keep the explicit `serve` command in its own terminal:

```bash
python -m zwz_eval serve-plan --config configs/my-model.json --bench hr8k --role main
python -m zwz_eval serve --config configs/my-model.json --bench hr8k \
  --role main --log-dir local/services/hr8k-main
```

After the service reports `ready`, run the existing benchmark inference command
in another terminal. `ready` means the expected model is available at
`/v1/models`; it does **not** mean any benchmark answer has been generated.
After inference finishes, Ctrl-C the service terminal to clean up its owned
process groups. If main and judge share GPUs, start judge only after main exits:

```bash
python -m zwz_eval serve --config configs/my-model.json --bench hr8k \
  --role judge --log-dir local/services/hr8k-judge
```

### Initialization and local compilation caches

There is no separate benchmark warmup command. vLLM performs its own memory
profiling, KV-cache allocation and kernel initialization before the service is
ready. These dummy inputs are not evaluation questions and generate no benchmark
score. Further kernels can compile on the first real input with a new shape.

Keep compilation caches reusable and preferably on a local filesystem. Concurrent
replicas create, rename and remove many small cache files; a high-latency shared
filesystem can turn startup into a long filesystem wait. A changing PID or cache
file alone is not proof that evaluation is progressing. Check endpoint readiness,
successful completion requests and predictions. Do not hide a startup problem by
changing prompts, image limits, model context or generation settings.

Set your own cache directory before starting `serve`; child services inherit the
environment. For example, replace the placeholder with a short local path:

```bash
export EVAL_CACHE_ROOT=/path/to/local-cache
mkdir -p "$EVAL_CACHE_ROOT"/tmp "$EVAL_CACHE_ROOT"/triton \
  "$EVAL_CACHE_ROOT"/deep_gemm "$EVAL_CACHE_ROOT"/flashinfer "$EVAL_CACHE_ROOT"/vllm
export TMPDIR="$EVAL_CACHE_ROOT/tmp"
export TRITON_CACHE_DIR="$EVAL_CACHE_ROOT/triton"
export DG_CACHE_DIR="$EVAL_CACHE_ROOT/deep_gemm"
export FLASHINFER_WORKSPACE_BASE="$EVAL_CACHE_ROOT/flashinfer"
export VLLM_CACHE_ROOT="$EVAL_CACHE_ROOT/vllm"
```

The DeepGEMM/FlashInfer variables apply when those installed backends support them;
they are not additional package dependencies. Keep the temporary path short for
Linux Unix-domain sockets. Separate independent jobs' temporary directories and
GPU/port allocations. Persistent predictions and scores stay in the configured
output directory. A host-local cache may disappear when the machine is replaced;
this does not remove the persistent evaluation results.

### Why HTTP ports alone were insufficient

vLLM also opens internal distributed communication ports. In the inspected 0.23
runtime, both single-process TP1 and multiprocess TP2 choose an available port,
close the probe socket, and later bind TCPStore. Another process can take the
port in between. Giving every HTTP API a different port does not cover this race;
`MASTER_PORT` alone does not control these explicit TCP initialization paths.

Plans now set per-replica `VLLM_PORT` starts and loopback addresses, with separate
blocks for main/judge and each startup attempt. Defaults are main base **25000**,
judge base **29000**, **32 ports per block**, and at most **3 attempts** (initial
start plus two retries). For 16 main replicas, the reserved planning range is
25000–26535; eight judge replicas use 29000–29767. These are planning ranges, not
OS reservations. GPU allocation, tensor parallelism, context, images, thinking,
generation parameters and scoring are unchanged.

The CLI accepts `--main-internal-port-base`, `--judge-internal-port-base`,
`--internal-port-block-size`, `--service-start-attempts`, and `--startup-timeout`.
The communication options are launch-time settings, not evaluation configuration
changes. Use disjoint port ranges for separate jobs on the same host, and do not
assign the same GPUs to concurrent main/judge pools.

### Failure handling and limits

- Validate ranges, HTTP/internal overlap, existing listeners and the host's
  ephemeral port range before starting. Never kill a process because it happens
  to own a port.
- Start replicas in parallel. Retry only a failed **startup** with an explicit
  `EADDRINUSE` in that attempt's log, using its next isolated block. Old appended
  errors do not trigger retries. Other failures and startup timeouts fail clearly.
- Clean only process groups created by this launcher, including children left by
  an exited API parent. Healthy replicas are not restarted to retry another one.
- Once a service has become ready, a later failure is reported, not silently
  restarted. The launcher never resubmits an evaluation request or resamples an
  answer, including wrong or malformed successful answers.
- Preserve attempt records, current command metadata, PID, hostname, selected
  block and log offsets for diagnosis. Check real inference/judge outputs too.

`VLLM_PORT` is a scan start, not a hard upper bound: vLLM may increment it if a
port becomes occupied. Preflight checks cannot atomically reserve a future
TCPStore socket. Isolation and bounded retries reduce and recover the known
startup race; they are not a guarantee against every concurrent host process.
This launcher supports local independent replicas, not cross-node/data-parallel
or elastic deployments. Do not copy its loopback settings into a distributed
cluster launcher.

### Updating an existing evaluation

Do not replace code under a live inference/scoring process. Stage reviewed code,
wait for the controller and run locks to be released, then activate it. Retain the
original source manifest and run contract. Exact before/after execution receipts
may cover the service launcher, serving plan, CLI and runner; they cannot authorize
changes to data, prompt, decoding, model identity, metrics or the rest of the
framework. Completed predictions are retained and not sampled again.
On shared storage, also verify the actual task processes and resource ownership
on every participating host immediately before activation. Do not assume a NAS
`flock` observed on one host proves another host is idle.

### Testing the launcher without GPUs

The default test suite checks plans, error classification, retry bounds and
immutable execution receipts. On a Linux host that permits local loopback
listeners and owned process-group signals, opt into the real CPU process tests:

```bash
RUN_SERVICE_PROCESS_TESTS=1 python -m unittest discover -s tests -p 'test_serving_processes.py' -v
```

These use tiny fake model servers, not vLLM or a GPU. They exercise concurrent
startup, a real address-in-use failure, cleanup of an orphan child before retry,
and refusal to terminate an existing port owner. Passing them is not a claim of
full GPU or benchmark acceptance.

---

## Judge configuration and input consistency

Configure judge paths, revisions, native templates, served model IDs and endpoints
in your own model JSON. Example service aliases are not weight download locations.
The main judge example is Qwen3.6-35B-A3B; the platform-local96 profile uses
Qwen3.5-35B-A3B. They are not interchangeable. Model IDs must match `/v1/models`.

```bash
python -m zwz_eval serve-plan --config local/my-eval/ready.json \
  --bench memlens --role judge
```

`serve-plan` is read-only. To launch, use `serve` in a separate terminal; wait for
readiness and inspect successful score requests. Eight TP=2 judge replicas require
eight endpoints and occupy 16 GPUs. Stop main services before reusing those GPUs.

Preserve question text, image order and task-specific image transforms. See
[PROTOCOLS.md](PROTOCOLS.md) and benchmark guides for exact limits and aggregation.
