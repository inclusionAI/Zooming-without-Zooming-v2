# ZwZ-v2 Evaluation

Evaluation code for 25 benchmarks. Data preparation and model evaluation are
separate steps. All paths and model settings are supplied by the user.

## Installation

From the repository root:

```bash
cd eval
python -m pip install -e '.[api,data]'
```

Use Python 3.10+ and a working accelerator runtime for local models.
See [setup](docs/SETUP.md) for benchmark-specific dependencies.
AgentVista additionally needs `pip install -e '.[agentvista,data]'` and Internet
access for search/visit tools.

## 1. Configure

Create a working directory for the benchmarks you want to run:

```bash
python -m zwz_eval init --directory local/my-eval --bench hr4k,hr8k
```

Edit these two files before continuing:

| File | Settings |
| --- | --- |
| `local/my-eval/model.json` | Model/judge paths and identities, GPU settings and result directory. |
| `local/my-eval/datasets.json` | Downloaded annotation/image paths, input format and preparation directory. |

Replace the example paths and `RECORD_ACTUAL_*` identity placeholders with your
own values. Relative paths resolve against the JSON file, not the terminal.
Keep credentials in environment variables, not in committed files.
The default local setup uses 16 GPUs: 16 TP=1 main replicas and 8 TP=2 judge replicas.

## 2. Prepare data

Download and extract the selected benchmarks following the
[download catalogue](docs/DOWNLOADS.md), then run:

```bash
bash scripts/prepare_benchmarks.sh local/my-eval
```

This reads `datasets.json` and `model.json` and writes `ready.json` in the same
working directory. It does not start models or modify the downloaded originals.
Matching prepared data can be reused. Input requirements are listed
[below](#benchmark-data-format).

## 3. Evaluate

```bash
bash scripts/evaluate.sh local/my-eval
```

The command evaluates the configured benchmarks serially and saves predictions
and scores. Model startup, required judging and service cleanup are automatic.

To select only part of the prepared benchmarks, append their IDs:

```bash
bash scripts/evaluate.sh local/my-eval hr4k,hr8k
```

Results are written to `<output_root>/<model_id>/<run_id>/<benchmark>/`.
Rerun a compatible configuration to resume; successful wrong or malformed model
answers are not resampled. Native GUI recovery reuses complete splits only.

## Supported benchmarks

Use the exact benchmark IDs below in commands and configuration files.

| Benchmark ID | Benchmark |
| --- | --- |
| `vstar` | VStar |
| `hr4k` | HR-Bench 4K (HR-4K) |
| `hr8k` | HR-Bench 8K (HR-8K) |
| `mme_en` | MME-RealWorld English (MME-en) |
| `mme_cn` | MME-RealWorld Chinese (MME-cn) |
| `zoombench` | ZoomBench |
| `chartography` | Chartography |
| `perception` | Perception-Bench |
| `treebench` | TreeBench |
| `o3` | O3-Bench |
| `countqa` | CountQA |
| `mimic` | MIMIC |
| `mmsi` | MMSI-Bench |
| `spar` | SPAR-Bench |
| `mmlongbench` | MMLongBench visual-core (25 non-Counting tasks) |
| `milebench` | MileBench |
| `memlens` | MemLens |
| `erqa` | ERQA |
| `viewspatial` | ViewSpatial-Bench |
| `mindcube` | MindCube-Tiny |
| `agentvista` | AgentVista |
| `screenspot_pro` | ScreenSpot-Pro |
| `ui_vision` | UI-Vision |
| `mmstar` | MMStar |
| `babyvision` | BabyVision |

Select one ID or a comma-separated list, for example `hr4k,hr8k,memlens`.
Use `all` to run every benchmark configured in your file, not to add unconfigured
datasets. Multiple benchmarks run serially in the listed order. MME-en/cn refer to
MME-RealWorld, not the separate MME benchmark.

## Benchmark data format

MMStar, BabyVision, VStar, HR-4K, HR-8K, MME-RealWorld-en/cn, ZoomBench,
Perception-Bench, TreeBench, O3-Bench and CountQA require JSONL (one record per
line) or a JSON array in this format:

```json
{
  "images": ["/path/to/images/image.png"],
  "query": "<image>Original question and answer choices",
  "response": "Reference answer"
}
```

Set `format` to `normalized` in `datasets.json` for these benchmarks.
All other benchmarks use their original downloaded format and the preparation
entrypoint in the [download catalogue](docs/DOWNLOADS.md).

Preserve original questions, choices, image order, answers, IDs and optional
`system`, multi-turn `query` lists and `LLM_JUDGE_PROMPT` fields. Change image
path prefixes using your own image root/map; do not rewrite evaluation content.
Images are processed on demand during inference; whole-dataset image
materialization is not required. See [annotation details](docs/DATA.md).

## Evaluation subsets

The reported results use the following task selections. Preserve these selections
when reproducing the table.

| Benchmark | Evaluated selection |
| --- | --- |
| MMLongBench | Visual-core: InfoSeek, ViQuAE, MM-NIAH Retrieval-image, MM-NIAH Reasoning-image and Visual Haystack Multi-image, each at K8/16/32/64/128 (25 tasks). Counting and text-only tasks are excluded. Seed 42; 100 source IDs per InfoSeek/ViQuAE/VH-multi task and 50 per MM-NIAH task, retaining all associated depth/material rows. |
| MindCube | Tiny `raw_qa`: 1,050 examples (250 around, 200 rotation, 600 among). |

See [protocols](docs/PROTOCOLS.md) for selection and aggregation details.

## Options

Append `infer` or `score` after the benchmark list to run one phase.
For existing model APIs, configure the endpoints and append `--services external`.

## Reference

- [Downloads](docs/DOWNLOADS.md)
- [Data formats](docs/DATA.md)
- [Environment and services](docs/SETUP.md)
- [Benchmark protocols](docs/PROTOCOLS.md)
- [Development and validation](docs/DEVELOPMENT.md)

See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled component licenses.
