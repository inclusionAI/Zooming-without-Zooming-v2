# Multi-Region VQA Synthesis (`data_synthesis/`)

Pipeline for synthesizing multi-region visual grounding and VQA training data.
The only required input is a folder of images. The pipeline successively runs
image quality filtering, dense region extraction, question generation and MCQ
assembly, LLM self-checks, rejection sampling, and cross-model Qwen-VL
auditing, producing validated VQA records and exporting them to verl training
parquet.

## Entry points

```bash
# Stage 01 — scan an image folder into a manifest, filtered by content richness
python scripts/01_filter_manifest.py \
    --image_folder /path/to/images \
    --output_jsonl data/filtered_manifest.jsonl

# Continuous synthesis (MANIFEST defaults to the stage-01 output
# data/filtered_manifest.jsonl)
bash run_pipeline_continuous.sh \
  [TARGET_VALID] [BATCH_SIZE] [MAX_BATCHES] [MANIFEST]
```

Core library code lives in `mr_vqa/`; staged scripts live in `scripts/`.

## Pipeline

The pipeline runs, in order: image quality filtering
(`scripts/01_filter_manifest.py`, which screens by long-side resolution,
mean per-channel RGB Shannon entropy, Canny edge density, and luminance
standard deviation, discarding low-content images before they consume region
extraction budget), dense region extraction (`scripts/02_extract_regions.py`),
question generation (`scripts/03_generate_questions.py`), self-check and
validation (`scripts/04_validate_questions.py`), and finally
`scripts/05_export_dataset.py` exports the dataset to parquet. The run stops
automatically once `TARGET_VALID` valid records have been produced.

## Configuration

```bash
cp configs/local_env.sh.example configs/local_env.sh   # edit keys and model names
```

Set `MLLM_*`, `KIMI_*` and `MATRIX_API_KEY` in `configs/local_env.sh`.
Comma-separated API keys are scheduled concurrently by an in-process key pool.
