# digital_tissue

Digital Tissue is a local-first Python toolkit for building and running simplified tissue simulations and benchmarking LLM agents against them. Everything runs locally (no database required).

A tissue is a grid of cells where each cell contains different molecules (genes, RNA, proteins) governed by rules that determine how they change over time. The entire simulation state lives in a single JSON file (`gridstate.json`).

This repo includes:

- A **web-based editor** to create and modify tissues — make them healthy, give them diseases, evolve them toward desired states
- **LLM benchmark challenges** where an LLM agent interrogates the tissue through API endpoints (running experiments, requesting omics data, proposing interventions) to solve a problem
- **Evolutionary optimization** to search over initial conditions and find tissue states that maximize an objective

The main goal is to generate environments that are biology-like: many interacting variables, limited observability, noisy data, and open-ended problems that require reasoning to solve.

## Requirements

- Python `3.10+`
- macOS or Linux (Windows via WSL is likely fine)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quickstart (Editor UI, no LLM keys required)

Start the backend and UIs:

```bash
python3 -m backend.tools.run_ui
```

Open:

- `http://127.0.0.1:8000/` (portal)
- `http://127.0.0.1:8000/editor` (editor)

To change ports:

```bash
python3 -m backend.tools.run_ui --runtime-port 8000 --benchmarks-port 8001
```

Backend only (no Streamlit UI):

```bash
python3 -m backend.runtime_server 8000
```

## LLM benchmarks (optional)

To run LLM-driven benchmark episodes, you need API keys.

Create `keys.txt` at the repo root (it is gitignored):

```text
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
GEMINI_API_KEY=
```

Or point to a different file path with `DT_KEYS_FILE`.

Once the servers are running, open the Benchmarks UI:

- `http://127.0.0.1:8001/`

You can also run benchmark scripts directly:

```bash
python3 trials/run_llm_benchmark.py --help
python3 trials/run_llm_suite.py --help
```

## Outputs

- Benchmark run artifacts: `var/runs/llm_bench/` (e.g. `events.jsonl`, `report.json`, logs)
- Streamlit controller state: `var/runs/benchmarks/`
- Runtime logs: `var/log/`

## Repository layout

- `backend/`: runtime server + simulation engine
- `apps/editor/`: static editor UI served by the backend
- `apps/benchmarks/`: Streamlit benchmarks UI
- `benchmarks/challenges/`: fixture puzzles backing `/api/tests/*`
- `assets/`: models, examples, prompts
- `trials/`: CLI scripts for running/analyzing benchmarks
- `docs/`: deeper references (`LAYER_OPS.md`, `MEASUREMENTS.md`)


## Extended documentation

I highly recommend learning how to use this repo by talking to an AI assistant that has read it. However, below is an attempt to summarize the key concepts.

### What is a tissue simulation?

A tissue simulation is defined by a single JSON file (`gridstate.json`) that encodes:

- a **2D grid** of cells (dimensions `H × W`)
- a set of **layers** — named variables stored per grid cell (e.g. `gene_x`, `rna_y`, `protein_z`, `molecule_a`). Each layer is a flat `float32` array of length `H*W`, base64-encoded inside the JSON.
- **layer ops** — update rules that compute the next value of each layer from the current state every tick (time unit)
- **measurements** — expressions computed from layers to track aggregate metrics (e.g. total live cells, mean protein level)

The backend loads this JSON into memory, advances it step-by-step, and exposes the state through HTTP endpoints. The editor UI visualizes layers as heatmaps and plots measurements over time.

The point is not to perfectly model real biology — it is to have a **small, self-contained scenario** where you can test strategies, run optimizations, or benchmark LLM agents against a measurable goal (e.g. reduce cancer-like behavior, extend simulated lifespan).

---

### How the LLM benchmark works

In benchmark mode, an LLM acts as an automated agent that interacts with the simulator through the backend API:

1. The runner picks a challenge (e.g. "cancer").
2. The LLM receives instructions (scenario, scoring, allowed actions).
3. The LLM calls backend API endpoints to run experiments and propose interventions.
4. The run produces artifacts: `events.jsonl` (step-by-step transcript), `report.json` (score + summary), `story.md`, `issues.json`, `stdout.log` / `stderr.log`.

This lets you compare different LLM providers/models, prompts, or toolsets on identical puzzles.

---

### Running an LLM benchmark via the Streamlit UI

1. Create `keys.txt` (see [LLM benchmarks](#llm-benchmarks-optional)) with the relevant API key(s).
2. Start servers: `python3 -m backend.tools.run_ui`
3. Open `http://127.0.0.1:8001/` (or click **Benchmarks** from the portal).
4. **Settings** tab → choose a **Challenge** (cancer / aging / hereditary disease).
5. **Run** tab → choose **Provider** and **Model** → click **Start new**.
6. When finished, inspect the transcript, report, and score in the UI or on disk under `var/runs/llm_bench/`.

---

### Make your own simulation

1. Copy an example `gridstate.json` from `assets/examples/`.
2. Edit it (or look at built-in models in `assets/models/`).
3. Load it in the editor and run.

To make it available in the Benchmarks UI, place the file under `benchmarks/challenges/<challenge_name>/` and restart.

---

### Using the editor

1. Open `http://127.0.0.1:8000/editor`.
2. Click **Demo** to load a built-in example, or **Import** for your own file.
3. **Runtime** tab → **Reset** → **Step once** or **Start** to run.
4. Switch variables (layers) to view heatmaps; use **Inspect** for summary stats.
5. **Evolution** tab → choose algorithm (`cem_delta` or `affine`) → **Start** to search over initial conditions.

---

### Architecture overview

- **Backend** (`backend/runtime_server.py`): single-process `ThreadingHTTPServer`. Holds simulation state in memory as a JSON payload, updates it on `/api/runtime/step`. Serves the editor UI as static files (no frontend build step).
- **Editor UI** (`apps/editor/`): `index.html`, `app.js`, `style.css`, `portal.html`.
- **Benchmarks UI** (`apps/benchmarks/`): Streamlit app for running and inspecting benchmark episodes.
- **Simulation engine** (`backend/digital_tissue/apply_layer_ops.py`): core tick function `apply_layer_ops_inplace(payload, seed_offset=...)`.

The server sets `Cache-Control: no-store`, so edits to HTML/JS/CSS are picked up on refresh.

Logs go to `var/log/` (override with `DT_LOG_DIR`).

Supported LLM API key environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`. Optional base URL overrides: `XAI_BASE_URL`, `GEMINI_BASE_URL`.

API smoke test:

```bash
curl -s http://127.0.0.1:8000/api/health
```

---

### Key terms

- **step / tick**: one simulation update
- **grid**: 2D layout of the tissue
- **layer**: one variable per grid location (RNA / protein / molecule / state)
- **gridstate / payload**: the JSON model file containing the full simulation state
- **Evolution**: built-in optimization that searches over starting conditions

### The `gridstate.json` format

Simulation state is passed around as a single JSON object:

- `version`: must be `1`
- `H`, `W`: grid dimensions
- `layers`: list of `{ "name": "gene_x", "kind": "continuous" }` metadata objects
- `data`: dict mapping layer names to `{ "dtype": "float32", "b64": "..." }` buffers

Layer buffers are **base64-encoded float32 arrays** of length `H*W`. This keeps the JSON interface simple while staying compact and fast to decode with NumPy.

The optional `payload["event_counters"]["totals"]` dict tracks simulation events (divisions, deaths) used by Evolution fitness calculations. Evolution strips existing counters from the base payload before evaluations.

---

### Backend API

All endpoints are in `backend/runtime_server.py`.

#### Runtime

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/runtime/reset` | `{ payload }` | `{ ok, tick, H, W, layers }` |
| `POST /api/runtime/step` | `{ layers? }` | `{ tick, data, scalars, measurements, events }` |
| `POST /api/runtime/frame` | `{ layers? }` | same as `/step` without advancing |

`/step` calls `apply_layer_ops_inplace(payload, seed_offset=tick)` and increments tick.

#### Evolution

Evolution runs in a background thread so the UI stays responsive.

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/evolution/start` | `{ payload, config }` | `{ ok, job_id }` |
| `POST /api/evolution/stop` | — | `{ ok }` |
| `POST /api/evolution/status` | — | `{ running, progress, baseline, series, history, top, perf }` |
| `POST /api/evolution/candidate` | `{ id }` | `{ ok, id, fitness, metrics, genome, payload }` |

The `payload` in `/candidate` response is a fully reconstructed `gridstate.json` loadable into Runtime.

---

### Evolution algorithms

Evolution searches over the initial conditions of `gene_*`, `rna_*`, `protein_*` layers. The `cell`/`cell_type` layer is auto-detected for masking.

#### `affine` (per-layer scale/bias GA)

Each layer gets a `scale` and `bias`: `arr2 = arr * scale + bias`, clamped to `[0, ∞)`. Genomes mutate via log-normal noise on scale and additive noise on bias. Standard elitist GA structure.

Parameters: `mutation_rate`, `sigma_scale`, `sigma_bias`.

#### `cem_delta` (CEM per-cell delta-field)

Each layer gets a per-cell delta: `arr2 = arr + delta`. Deltas are sampled from a per-layer Gaussian (`mu`, `sigma`) and updated via Cross-Entropy Method toward the best candidates.

Parameters: `cem_sigma_init`, `cem_alpha`, `cem_sigma_floor`, `cem_mask` (`cell` or `all`).

This works better when the optimal solution requires heterogeneous per-cell specialization.

#### Fitness

Weighted sum of `alive` cells + event totals (`divisions`, `starvation_deaths`, `damage_deaths`). Weights set in the UI. A baseline from the unmodified payload is computed for comparison.

---

### Performance notes

Each evolution evaluation copies the payload, applies the genome, runs N ticks, and computes fitness. The `/api/evolution/status` response includes `perf` telemetry (`apply_s`, `ticks_s`, `decode_cell_s`, `total_s`) to diagnose bottlenecks.

The backend uses `ThreadPoolExecutor` for parallel evaluations. If `ticks_s` dominates and CPU doesn't scale with threads, consider process-based parallelism.

---

### Troubleshooting

**`python3 -m venv` fails on Ubuntu/Debian:**

```bash
sudo apt install -y python3-venv   # or python3.12-venv for 3.12
```

Then recreate `.venv` and reinstall deps.

**`externally-managed-environment` error:**
Use a virtual environment (`python3 -m venv .venv`). Avoid `--break-system-packages`.

**GitHub push auth:**
Use a Personal Access Token, SSH remote, or `gh auth login`.
