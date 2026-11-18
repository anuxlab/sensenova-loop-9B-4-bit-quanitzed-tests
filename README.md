# SenseNova Loop-9B: 4-bit Model Evaluation Suite

A reproducible, local-first harness for testing SenseNova Loop-9B from basic instruction following through reasoning, code generation, bounded tool use, critique/revision, long-context retrieval, and image understanding.

## Target checkpoint and runtime

The default target is [`nativ-community/SenseNova-Loop-9B-4bit`](https://huggingface.co/nativ-community/SenseNova-Loop-9B-4bit), an MLX-VLM 4-bit export for Apple Silicon. It is **not** the same artifact/runtime as the original `sensenova/SenseNova-Loop-9B` Transformers checkpoint. The community model card describes 4-bit affine quantization (group size 64), with the vision tower in BF16 and mHC routing tensors in FP32. Its reported weight files occupy about 5.96 GB (5.55 GiB).

The MLX backend is intended for Apple Silicon Macs. The optional Transformers backend is a reference path for the original Hugging Face checkpoint; it does not automatically load the MLX 4-bit quantization.

## Quick start: Apple Silicon

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -U mlx mlx-vlm

# Run the complete text benchmark
python -m sensenova_eval.runner --backend mlx --categories all

# Separate tracks
python -m sensenova_eval.runner --categories basic,reasoning
python -m sensenova_eval.runner --categories coding,agentic,self_improve
python -m sensenova_eval.runner --categories longcontext --thinking

# Image understanding (provide your own image)
python run_vision_test.py ./sample.jpg
```

First run downloads the model. Set `HF_TOKEN` in the environment only if the repository requires authenticated access. Do not commit tokens.

## Evaluation tracks

| Track | What it probes | Scoring |
|---|---|---|
| Basic | Arithmetic, instruction following, simple logic | Exact marker checks |
| Reasoning | Multi-step arithmetic and structured debugging plans | Expected markers; inspect full response |
| Coding | Python function generation | Executes unit tests with a timeout |
| Agentic | Planning plus a bounded calculator tool cycle | Valid tool action, tool result, final answer |
| Self-improvement | Draft → critique → revision | Revised code must pass tests |
| Long context | Needle-in-a-haystack retrieval | Exact key retrieval |
| Vision | Image description and visual grounding | Human review; no automatic vision score yet |

The bundled tasks are a **starter smoke-test suite**, not a standardized benchmark and not directly comparable to published benchmark scores. Add larger, licensed datasets and task-specific graders before drawing broad capability conclusions.

## Reports

Each run writes JSON to `reports/latest.json` by default. It contains model/backend metadata, per-task responses, elapsed time, optional token counts, pass/fail status, and a summary pass rate. Preserve reports alongside the exact model revision, runtime versions, hardware, and generation settings when comparing runs.

## GitHub Actions

- `Harness quality checks` runs lightweight unit tests on GitHub-hosted Ubuntu runners. It does not download model weights.
- `SenseNova 4-bit MLX benchmark` is manual-only and requires a trusted self-hosted Apple Silicon runner labelled `self-hosted`, `macOS`, `ARM64`, with `mlx-vlm` installed. It uploads the JSON report as an artifact.

Do not expose a self-hosted runner to untrusted pull requests: generated code tests execute model output locally. The harness uses a temporary directory and timeout, but subprocess execution is **not a security sandbox**. For untrusted prompts/models, run coding tests inside a disposable container or VM with no network and strict resource limits.

## Reproducibility and interpretation

- Use greedy decoding for deterministic baseline runs where supported.
- Keep thinking mode, model revision, backend, max tokens, and hardware fixed across comparisons.
- Run each task multiple times with sampling enabled only when measuring variance; this starter runner uses deterministic defaults.
- Review failed outputs, not only the aggregate pass rate. Marker-based grading can reward superficial matches.
- A critique/revision pass is an evaluation of one bounded self-correction cycle, not evidence that the model autonomously improves its weights.

## Extend the suite

Add task records to `sensenova_eval/tasks.py`, or implement dedicated graders for JSON schema validity, algorithmic correctness, tool-call traces, repository-level coding, data analysis, SQL, security reasoning, and multi-step workflows. Keep task prompts and graders versioned so regression results remain meaningful.

## License

This repository's original license is retained. Check the model checkpoint's model card for its license and attribution requirements.
