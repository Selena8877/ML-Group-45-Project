
# NOMADD-4 — Assignment 2: Model Integration and Checkpointing

**Scope:** Person 4 — Phase 2 Ablation Study  
**Main File:** `ablation/run_ablation_phase2.py`  

## 1. Overview

The Phase 2 ablation script was extended to support six model configurations and resumable checkpointing. The implementation focuses on model provider integration, error handling, checkpoint recovery, and validation without live model inference.

The shared script also includes data ingestion, prompt construction, and dry-run functionality developed as part of the overall assignment.

## 2. Model Provider Integration

Six model configurations are supported using the existing NOMADD-4 provider framework.

| Model | Provider |
|---|---|
| Gemini | GeminiProvider |
| OpenAI | OpenAIProvider |
| Claude | ClaudeProvider |
| Llama | LMStudioProvider |
| Gemma | LMStudioProvider |
| Qwen | LMStudioProvider |

### Implementation Details

- Dynamic provider initialization based on the selected model.
- Existing provider classes reused for consistency with Phase 1.
- Cloud models configured using environment variables.
- Local models integrated through LM Studio.
- Provider initialization and prediction failures logged.
- Unavailable local endpoints handled without crashing the entire script.
- Dry-run mode avoids initializing providers or making API calls.

## 3. Resumable Checkpointing

A fault-tolerant checkpoint system was implemented to preserve completed predictions and support interrupted evaluations.

**Checkpoint Directory:**

`ablation/checkpoints_phase2/`

**Checkpoint Structure:**

`<model>/<language>/<condition>.partial.json`

### Features

- Saves valid predictions every 50 attempted posts and at the end of each condition.
- Loads existing checkpoints when execution resumes.
- Validates stored predictions against post indices, text, ground-truth labels, and classification results.
- Skips previously completed valid predictions.
- Retries missing, failed, or incomplete predictions.
- Handles corrupted checkpoint files through warning messages.
- Uses temporary files when saving checkpoints to reduce the risk of incomplete writes.
- Keeps Phase 2 checkpoints separate from existing Phase 1 data.

## 4. Testing and Validation

The implementation was validated through Python syntax checks, the full Phase 2 dry-run, and simulated provider tests.

### Python Syntax Validation

Run from the `nomadd4_assignment1` directory:

```powershell
python -m compileall -q .\ablation\run_ablation_phase2.py .\src\models\
```

**Result:** Passed.

### Full Dry-Run Validation

```powershell
python .\ablation\run_ablation_phase2.py --dry-run
```

**Results:**

| Validation | Result |
|---|---|
| Model configurations | 6 |
| Languages | 4 |
| Ablation conditions | 7 |
| Prompt variants validated | 28 |
| Unique Phase 2 posts | 995 |
| Projected predictions | 41,790 |
| Dry-run status | Passed |

The dry-run confirmed successful dataset ingestion, prompt validation, checkpoint inspection, and projected model-call calculations.

No live API calls or model inference were performed.

### Simulated Checkpoint Tests

Simulated model-provider tests verified:

1. Successful checkpoint creation and saving.
2. Correct identification of completed predictions.
3. Skipping previously completed predictions during resumption.
4. Retrying failed predictions.
5. Graceful handling of corrupted checkpoint files.
6. Accurate checkpoint completion calculations.

These tests used mock providers and temporary checkpoint files rather than live models.

## 5. Dataset Validation Observations

The final dry-run loaded the following Phase 2 datasets:

| Language | Unique Posts | Phase 1 Matches |
|---|---:|---:|
| Arabic | 249 | 248/249 |
| Bengali | 250 | 250/250 |
| Chinese | 250 | 250/250 |
| Urdu | 246 | 246/246 |

The Arabic dataset contained one exact duplicate, while the Urdu dataset contained 13 exact duplicates, which were removed during loading.

All datasets passed the script's current validation tolerance. The reduced Arabic and Urdu counts and the unmatched Arabic record should be confirmed by the team before final submission.

## 6. Implementation Files

The following files were included in the implementation commit:

- `ablation/run_ablation_phase2.py`
- `scripts/run_phase1.py`
- `src/models/__init__.py`
- `src/models/base.py`
- `src/models/claude_provider.py`
- `src/models/deepseek_provider.py`
- `src/models/gemini_provider.py`
- `src/models/lm_studio_provider.py`
- `src/models/openai_provider.py`

The Phase 1 runner and model-provider modules were incorporated from the existing NOMADD-4 codebase to support the Phase 2 integration.

## 7. Completion Status and Limitations

Model integration, checkpointing, error handling, and validation were completed and committed to the shared GitHub repository.
