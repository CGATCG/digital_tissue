# digital_tissue

Digital Tissue is a toolkit for making and running **biological puzzles**.

The main goal of this repo is to create small, runnable tissue simulations (with “diseases” and measurable outcomes) and then **test how well LLMs can solve them**.

Everything runs locally by default. There is no database requirement.

This is a research and teaching tool. It is intentionally simplified and is not a medically validated model.

## What you can do (high level)

- Explore a tissue simulation in your browser.
- Create or edit a puzzle model (a JSON file) and rerun instantly.
- Run “disease challenge” scenarios (cancer / aging / hereditary disease).
- Run benchmark suites where an LLM acts like an agent that can:
  - read puzzle instructions
  - run experiments through the backend API
  - propose interventions
  - get scored on whether it improved the outcome

## How it works (short version)

- **Simulator backend (Python)**: runs the simulation and exposes an HTTP API.
- **Editor UI (browser)**: lets you load a model, run it step-by-step, and inspect what happened.
- **Benchmark runner + UI (optional)**: runs suites of LLM-driven puzzle attempts and saves reports.

You can use the editor without any LLM keys. LLM keys are only needed for LLM benchmarks.

---

## Table of contents

- [What you can do (high level)](#what-you-can-do-high-level)
- [How it works (short version)](#how-it-works-short-version)
- [Who this is for](#who-this-is-for)
- [What is a “biological puzzle”?](#what-is-a-biological-puzzle)
- [How the LLM benchmark works](#how-the-llm-benchmark-works)
- [Run an LLM benchmark (step-by-step)](#run-an-llm-benchmark-step-by-step)
- [What you get from a benchmark run](#what-you-get-from-a-benchmark-run)
- [Make your own puzzle](#make-your-own-puzzle)
- [Try it in 5 minutes (no LLM keys required)](#try-it-in-5-minutes-no-llm-keys-required)
- [Quick start (recommended)](#quick-start-recommended)
- [Using the editor (step-by-step)](#using-the-editor-step-by-step)
- [Where files and outputs go](#where-files-and-outputs-go)
- [What the system does](#what-the-system-does)
- [Repository layout](#repository-layout)
- [Technical reference](#technical-reference)
- [Key terms (optional)](#key-terms-optional)
- [The `gridstate.json` format](#the-gridstatejson-format)
- [Running the backend + UIs](#running-the-backend--uis)
- [Web UI overview](#web-ui-overview)
  - [Runtime screen](#runtime-screen)
  - [Evolution screen](#evolution-screen)
- [Benchmarks (optional)](#benchmarks-optional)
- [Backend API](#backend-api)
  - [Runtime API](#runtime-api)
  - [Evolution API](#evolution-api)
- [Evolution algorithms](#evolution-algorithms)
- [Performance and profiling](#performance-and-profiling)
- [Troubleshooting](#troubleshooting)

---

If you're new here, you can read the first sections and then jump straight to [Quick start](#quick-start-recommended). You do not need to understand the internal data format to use the editor.

## Who this is for

- If you are a student, you can use this repo to explore a simulation, change variables, and see how outcomes change.
- If you are a biology researcher, you can use this as a sandbox for toy models and controlled intervention experiments.
- If you are an AI/ML developer, you can use this as a benchmark harness: an LLM agent interacts with a simulator using tools and gets scored.

## What is a “biological puzzle”?

In this repo, a puzzle is a small simulation scenario with:

- a starting tissue state (cells + variables like molecules/RNA/proteins)
- a set of rules that update the tissue over time
- a goal you can measure (for example: improve “health” metrics, reduce cancer-like behavior, extend lifespan in the simulation)

The point is not to perfectly model real biology. The point is to have a **small, runnable scenario** where you can test strategies and compare outcomes.

---

## How the LLM benchmark works

In benchmark mode, the LLM acts as an automated agent and interacts with the simulator through the backend API.

A benchmark run looks like this:

1. The benchmark runner chooses a puzzle (for example: “cancer”).
2. The LLM gets a set of instructions:
   - what the disease scenario is
   - what success means (the score)
   - what actions are allowed
3. The LLM interacts with the simulation environment by calling the backend API.
4. The run produces artifacts you can inspect:
  - a step-by-step transcript of what the LLM did
  - measurements from the simulation
  - a final score and summary report

What “solving the puzzle” means depends on the challenge, but the common pattern is:

- start from a baseline tissue state
- apply changes (interventions)
- run the simulation
- compare outcomes to the baseline

This lets you compare:

- different LLM providers/models on the same puzzles
- different prompts or toolsets
- how robust an LLM is across repeated runs

You can run benchmarks through:

- the Streamlit UI (recommended)
- scripts under `trials/`

---

## Run an LLM benchmark (step-by-step)

This is the easiest way to run the “LLM tries to solve the puzzle” part.

1. Create `keys.txt` (see Quick start) and put in the API key(s) for the provider you want.
2. Start the servers:

```bash
python3 -m backend.tools.run_ui
```

3. Open the Benchmarks UI:

- from the portal: `http://127.0.0.1:8000/` then click **Benchmarks**, or
- directly: `http://127.0.0.1:8001/`

4. In the sidebar:

- open the **Settings** tab
- choose a **Challenge** (cancer / aging / hereditary disease)

5. In the sidebar:

- open the **Run** tab
- choose a **Provider** and **Model**
- (optional) choose an **Initial prompt**
- click **Start new**

6. When the run finishes, you can inspect:

- the step-by-step transcript
- the report and score
- any saved run artifacts on disk under `var/runs/llm_bench/`

---

## What you get from a benchmark run

Each run writes files under `var/runs/llm_bench/`.

You typically get:

- a run folder with a unique `run_id`
- `events.jsonl`: a step-by-step event stream of what the agent did
- `report.json`: a structured summary (including the final score)
- `stdout.log` / `stderr.log`: logs from the benchmark runner

This is designed so you can compare runs and build your own dashboards or analyses.

---

## Make your own puzzle

If you want to create your own scenario:

- Start from an existing example in `assets/examples/`.
- Look at built-in models in `assets/models/`.
- Look at the existing challenge fixtures in `benchmarks/challenges/`.

If you want your puzzle to show up in the **Benchmarks** UI, add your `gridstate.json` to the appropriate challenge folder under `benchmarks/challenges/` (for example: `benchmarks/challenges/cancer/`) and restart the servers.

The simplest workflow is:

1. Copy an example `gridstate.json`.
2. Edit it.
3. Load it in the editor and run.

---

## Try it in 5 minutes (no LLM keys required)

1. Install dependencies (see Quick start below).
2. Start the server + UIs:

```bash
python3 -m backend.tools.run_ui
```

3. Open:

- `http://127.0.0.1:8000/`

4. Click **Editor**.
5. Click **Demo**, then click **Reset**, then click **Start** (or **Step once**).

## Quick start (recommended)

These steps work on macOS and Linux.

### Requirements

- Python **3.10+**
- A browser (Chrome/Firefox/Safari)

### 1) Create a Python virtual environment and install dependencies

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 1.5) (Optional) Configure API keys for LLM benchmarks

If you plan to run LLM benchmarks, create a single keys file at the repo root named `keys.txt` and put your API keys there.

Example:

```text
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
GEMINI_API_KEY=
```

Quick way to create the file:

```bash
cat > keys.txt <<'EOF'
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
GEMINI_API_KEY=
EOF
```

Format is one entry per line:

- `KEY=VALUE`

It also accepts `export KEY=VALUE` (so you can copy/paste from shell-style files).

- `keys.txt` is **gitignored**.
- You can also point to a custom file path by setting `DT_KEYS_FILE`.

### 2) Start the servers

Recommended (starts the Python backend **and** the Streamlit benchmarks UI):

```bash
python3 -m backend.tools.run_ui
```

Defaults:

- Backend + editor portal: `http://127.0.0.1:8000/`
- Streamlit benchmarks UI: `http://127.0.0.1:8001/`

Override ports:

```bash
python3 -m backend.tools.run_ui --runtime-port 8000 --benchmarks-port 8001
```

Backend only (editor UI + APIs, without Streamlit):

```bash
python3 -m backend.runtime_server 8000
```

### 3) Open the UI

Open the printed URL in your browser.

Routes:

- `/` – portal (choose Editor vs Benchmarks)
- `/editor` – editor UI
- `/benchmarks` – redirect to the Streamlit benchmarks app (runs on a separate port)
- `/monitor` – alias for `/benchmarks`

If you use a non-default port, set:

```bash
export DT_BENCHMARKS_PORT=8001
```

Stopping:

- If you started with `python3 -m backend.tools.run_ui`, `Ctrl+C` stops the backend, and it will also stop Streamlit if it started it.
- If you started Streamlit separately, stop it with:

```bash
python3 -m backend.tools.benchmarks_ctl stop
```

---

## Using the editor (step-by-step)

1. Start the servers (see Quick start).
2. Open the portal:
   - `http://127.0.0.1:8000/`
3. Open the editor:
   - `http://127.0.0.1:8000/editor`
4. Load a model:
   - Click **Demo** to load a built-in example, or
   - Click **Import** to load your own `gridstate.json` file.
5. Run the simulation:
   - Open the **Runtime** tab.
   - Click **Reset** (syncs the current editor state to the backend runtime).
   - Click **Step once** or **Start**.
6. Inspect state:
   - Switch variables (layers) and view heatmaps.
   - Use **Inspect** to see per-layer summary stats and cursor values.
7. Optimize initial conditions:
   - Open the **Evolution** tab.
   - Choose an algorithm (e.g. `cem_delta` or `affine`).
   - Click **Start**.
   - Load or download any candidate from the Top Candidates table.

---

## Where files and outputs go

- `workspace/`
  - Local working data used by the backend (e.g. editor saves and runtime working state)
- `var/`
  - Generated outputs
  - `var/log/`: runtime server logs (`runtime_server.log`, `stderr.log`, `runtime_server_faulthandler.log`)
  - `var/runs/llm_bench/`: benchmark run artifacts (events, reports, logs)
  - `var/runs/benchmarks/`: Streamlit controller state (`streamlit.pid`, `streamlit.log`, `streamlit_meta.json`)

---

## What the system does

At a high level:

1. You load a model file (a `gridstate.json` JSON).
2. The backend updates the model step-by-step over time.
3. The editor UI visualizes the variables in the model (shown as “layers” in the UI) and computed measurements.
4. The Evolution system can automatically try many starting conditions and keep the ones that score best.

The backend exposes these capabilities as HTTP endpoints so you can drive it from:

- the built-in editor UI
- scripts
- benchmark harnesses

### For biologists

The core idea is that a model file is represented as:

- a 2D grid
- a set of variables over that grid (called “layers” in the UI)
- a set of update rules (Layer Ops) that compute the next state each step

Typical workflow:

1. Start the servers.
2. Open the editor at `/editor`.
3. Import a `gridstate.json` (or use a demo).
4. Use **Runtime** to step forward and inspect layer behavior.
5. Use **Evolution** to search for initial conditions that improve an objective.

### For AI / software developers

Architecture summary:

- `backend/runtime_server.py` is a single-process HTTP server.
- simulation state is a JSON payload in memory; the server updates it on `/api/runtime/step`.
- the browser editor is a static app served by the backend (no frontend build system).
- benchmarks are driven by scripts under `trials/` and a Streamlit UI under `apps/benchmarks/`.

Data model summary:

- layers are base64-encoded `float32` buffers (`H*W` elements)
- a payload is fully self-contained (it contains data, layer metadata, and optional measurement/op definitions)

API smoke test (no UI required):

```bash
curl -s http://127.0.0.1:8000/api/health
```

Reset runtime with an example payload and step once:

```bash
python3 - <<'PY'
import json
import urllib.request

payload = json.load(open('assets/examples/gridstate.json', 'r', encoding='utf-8'))

def post(path, obj):
    req = urllib.request.Request(
        'http://127.0.0.1:8000' + path,
        data=json.dumps(obj).encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode('utf-8', errors='replace'))

post('/api/runtime/reset', {'payload': payload})
post('/api/runtime/step', {})
PY
```

---

## Repository layout

High-level structure:

- `backend/runtime_server.py`
  - Single-process backend server
  - Serves the web UI
  - Implements `/api/runtime/*` and `/api/evolution/*`
  - Runs the simulation and evolution jobs

- `backend/digital_tissue/apply_layer_ops.py`
  - Core simulation update step: `apply_layer_ops_inplace(payload, seed_offset=...)`
  - Utilities for encoding/decoding layer arrays:
    - `_decode_float32_b64(b64, expected_len, layer_name)`
    - `_encode_float32_b64(arr)`

- `apps/editor/`
  - `index.html` – UI layout (Runtime + Evolution screens)
  - `style.css` – UI styling
  - `app.js` – UI logic, API calls, plotting, candidate table, etc.

- `apps/benchmarks/`
  - Streamlit app for running and inspecting benchmark suites

- `benchmarks/challenges/`
  - Benchmark fixture models used by the `/api/tests/*` endpoints (e.g. cancer / aging / hereditary disease)

- `backend/tools/`
  - Small command-line utilities for development
  - `backend.tools.run_ui`: starts Streamlit + backend together
  - `backend.tools.benchmarks_ctl`: start/stop/status for the Streamlit app

- `assets/examples/`
  - Example `gridstate.json` payloads and snapshots

- `assets/models/`
  - Built-in model JSON payloads (served by the models API)

- `assets/prompts/`
  - Prompt files used by benchmark runs

- `settings/`
  - Configuration used by the backend and benchmarks
  - Includes `settings/pricing.json` and omics panel definitions

- `var/`
  - Generated outputs (typically gitignored)
  - `var/runs/llm_bench/`: benchmark run artifacts
  - `var/runs/benchmarks/`: Streamlit state (pid/log/meta)

- `trials/`
  - Benchmark runner scripts (e.g. `trials/run_llm_benchmark.py`, `trials/run_llm_suite.py`) and other development artifacts

- `requirements.txt`
  - Python dependencies (primarily `numpy`; some scripts may use `streamlit` and `plotly`)

---

## Technical reference

### Key terms (optional)

- **step** (sometimes called a **tick** in code): one simulation update.
- **grid**: the 2D layout of the tissue.
- **variable / layer**: one value stored per grid location (RNA / protein / molecule / state, etc.).
- **gridstate / payload**: the JSON model file that contains the full simulation state.
- **Evolution**: built-in optimization that searches over starting conditions.

### The `gridstate.json` format

This project passes simulation state around as a single JSON object (the “payload”).

At minimum, a payload is expected to look like:

- `version`: must be `1`
- `H`, `W`: grid dimensions
- `layers`: list of layer metadata objects, typically:
  - `{ "name": "gene_x", "kind": "continuous" }`
- `data`: dictionary mapping layer names to buffers:
  - `{ "dtype": "float32", "b64": "..." }`

The layer buffers are stored as **base64-encoded float32 arrays**, length `H*W`.

#### Why base64 float32?

- Compact and portable representation inside JSON
- Fast enough to decode/encode with `numpy.frombuffer(...)` and `arr.tobytes()`
- Keeps the UI/backend interface simple (no binary protocol)

#### Event counters

The simulation tracks certain events in an optional structure:

- `payload["event_counters"]`
  - includes a `totals` dict used by Evolution fitness calculations (divisions, deaths, etc.)

Evolution always strips any existing `event_counters` from the base payload copy to avoid contaminating evaluations.

### Running the backend + UIs

#### Overview

The backend is a local development server (ThreadingHTTPServer). It serves the UI as static files and provides JSON APIs.

#### Starting the backend

Start it from the repo root:

```bash
python3 -m backend.runtime_server
```

Or start the backend and Streamlit together (recommended):

```bash
python3 -m backend.tools.run_ui
```

You should see output like:

- `Runtime server: http://127.0.0.1:8000/`

Open that URL.

#### Important behavior: no caching

The server sets `Cache-Control: no-store`, so editing `apps/editor/app.js` or `apps/editor/index.html` refreshes immediately.

#### Logs

By default, the runtime server writes logs under:

- `var/log/`

You can override the directory by setting `DT_LOG_DIR`.

### Benchmarks (optional)

#### Overview

This repo also includes an optional benchmark harness and a Streamlit UI (`apps/benchmarks/`) for running and inspecting benchmark runs.

#### What the benchmark system does

- Starts a backend server locally.
- Runs benchmark “episodes” that call the backend APIs.
- Stores run artifacts under `var/runs/llm_bench/` (runs, suites, logs, reports).

The benchmark-related simulation endpoints live under `/api/tests/*`. The fixture models backing these endpoints are stored in `benchmarks/challenges/`.

#### Running the Streamlit UI

If you are using the recommended `run_ui` command, Streamlit is started for you.

To run Streamlit by itself:

```bash
python3 -m backend.tools.benchmarks_ctl start --port 8001 --address 127.0.0.1
```

#### LLM API keys

LLM providers are only needed for LLM benchmark runs.

If you run an LLM benchmark, you must provide the corresponding API key(s). The recommended way is to put them in `keys.txt` (see Quick start), which is auto-loaded into environment variables when you start the backend/benchmarks.

The environment variables used are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

Some providers also support optional base URL overrides:

- `XAI_BASE_URL`
- `GEMINI_BASE_URL`

#### Running benchmark scripts (CLI)

The Streamlit app is the easiest entry point, but you can also run benchmark scripts directly:

```bash
python3 trials/run_llm_benchmark.py --help
python3 trials/run_llm_suite.py --help
```

### Web UI overview

#### Overview

The UI lives in `apps/editor/` and is served by the backend.

The Evolution UI was designed to remain compatible even as backend algorithms evolved:
- endpoints remain stable
- status payload stays consistent
- top candidates are always reconstructable into a full `gridstate.json`

#### Runtime screen

The Runtime screen is the main “run the simulation and visualize it” dashboard.

Core features:
- Load a gridstate JSON
- Step the simulation forward tick-by-tick or run continuously
- View selected layers as heatmaps
- View scalars and measurements derived from layers
- View event counters

Runtime uses:
- `/api/runtime/reset` to load a payload
- `/api/runtime/step` to advance the state
- `/api/runtime/frame` to fetch the current state without stepping

#### Evolution screen

The Evolution screen runs evolutionary search over the **initial gene/rna/protein layers**.

You can:
- choose the **base payload**:
  - `Runtime file (if loaded)`
  - `Current editor state`
- choose the **algorithm**:
  - `cem_delta` (CEM delta-field)
  - `affine` (GA scale/bias)
- tune evaluation parameters:
  - `Variants per generation`
  - `Ticks per evaluation`
  - `Generations`
  - `Elites`
  - `Replicates`
  - `Workers`
  - `Seed`
- tune algorithm-specific parameters (see [Evolution algorithms](#evolution-algorithms))
- tune fitness weights

During a run, the UI:
- polls `/api/evolution/status` periodically
- renders a live plot
- shows a Top Candidates table with per-candidate metrics

Candidates support:
- **Load**: resets the Runtime state with that candidate’s reconstructed payload
- **Download**: downloads the candidate `gridstate.json`

### Backend API

#### Overview

All backend logic is implemented in `backend/runtime_server.py`.

#### Runtime API

##### `POST /api/runtime/reset`

Body:

- `payload`: a `gridstate` object

Response:
- `{ ok, tick, H, W, layers }`

##### `POST /api/runtime/frame`

Body:
- `layers`: optional list of layer names to return in `data`

Response includes:
- `tick`
- `data` (selected layers)
- `scalars` (sum/mean/nonzero per decoded layer)
- `measurements` (expressions computed from the payload config)
- `events` (event counters)

##### `POST /api/runtime/step`

Body:
- `layers`: optional list of layer names to return

Behavior:
- calls `apply_layer_ops_inplace(payload, seed_offset=tick)`
- increments `tick`
- returns the same structure as `/frame`

#### Evolution API

Evolution is managed by a background thread (`_EvolutionJob`) so the UI stays responsive.

##### `POST /api/evolution/start`

Body:
- `payload`: base `gridstate` object
- `config`: evolution config object built by the UI

Response:
- `{ ok: true, job_id }`

##### `POST /api/evolution/stop`

Response:
- `{ ok: true }`

##### `POST /api/evolution/status`

Response:
- `running`, `error`, `job_id`
- `cfg`: the config used
- `progress`: generation/variant counters + total evaluations
- `baseline`: baseline evaluation (fitness + metrics)
- `series`: live per-evaluation series (fitness/best/mean) for plotting
- `history`: per-generation summaries (best/mean/p10/p90)
- `top`: top candidates table for UI
- `perf`: cumulative timing telemetry for diagnosing bottlenecks

##### `POST /api/evolution/candidate`

Body:
- `id`: candidate id

Response:
- `{ ok, id, fitness, metrics, genome, payload }`

`payload` is a fully reconstructed `gridstate.json` that can be loaded into Runtime.

### Evolution algorithms

#### Overview

Evolution searches over the **initial conditions** of layers matching:

- `gene_*`
- `rna_*`
- `protein_*`

The `cell`/`cell_type` layer is detected automatically and used for:
- determining which grid locations are initially “cells”
- (optionally) masking delta-field updates to only those locations

#### `affine`: per-layer scale/bias GA

This is the original evolutionary approach.

Genome representation:

- For each mutable layer name `nm`, store:
  - `scale` (float)
  - `bias` (float)

Apply step:

- For each mutated layer:
  - decode base layer `arr`
  - compute `arr2 = arr * scale + bias`
  - clamp to `[0, huge]`
  - if the layer kind is `counts`, round before clamping
  - encode and store back into payload

Search structure:

- For each generation:
  - sample `variants` genomes by mutating parents
  - evaluate each candidate for `ticks` steps
  - compute fitness
  - keep `elites` best genomes as the parent pool

Mutation parameters:

- `mutation_rate`: probability of mutating a layer
- `sigma_scale`: log-normal noise on scale (`scale *= exp(N(0, sigma_scale))`)
- `sigma_bias`: additive noise scaled by the layer’s observed std

Parallelism:

- Uses `ThreadPoolExecutor` to evaluate variants concurrently.

When to use:

- Good baseline
- Fast to implement
- But limited expressivity: relative spatial differences inside a layer don’t change much

#### `cem_delta`: CEM per-cell delta-field

This is the newer algorithm designed for rapid reward improvement when the optimal solution requires **heterogeneous per-cell specialization**.

High-level idea:

- Instead of global scale/bias per layer, learn a **delta per cell per layer**.
- Use Cross-Entropy Method (CEM): maintain a sampling distribution and update it towards the best samples.

Genome representation:

- For each layer `nm`, a candidate holds:
  - `delta_b64`: base64 float32 array length `H*W`

Apply step:

- For each mutated layer:
  - decode base layer `arr`
  - decode candidate delta `delta`
  - compute `arr2 = arr + delta`
  - clamp to `[0, huge]` (+ count rounding when needed)

Distribution state (per generation):

- For each layer `nm`:
  - `mu[nm]`: float32 array length `H*W`
  - `sig[nm]`: float32 array length `H*W`

Sampling:

- `delta = mu + sig * eps`, where `eps ~ N(0, 1)`

Update:

- Take the best `topK` candidates (derived from `Elites`)
- Compute mean/std of their deltas
- Exponentially smooth updates with:
  - `cem_alpha` in `[0, 1]`

Key parameters:

- `cem_sigma_init`: initial exploration magnitude (scaled by each layer’s std)
- `cem_alpha`: update aggressiveness (higher = faster updates, can be noisier)
- `cem_sigma_floor`: prevents sigma collapse / premature convergence
- `cem_mask`:
  - `cell` (default): apply deltas only where there are initially cells
  - `all`: allow deltas everywhere

Why this works better on complex tissues:

- Allows different locations to evolve independently.
- Can express spatial “programs” where neighboring cells need distinct values.

### Fitness function

Evaluation collects:
- `alive`: number of live cells after running `ticks`
- event totals (from `payload.event_counters.totals`):
  - `divisions`
  - `starvation_deaths`
  - `damage_deaths`

Fitness is a weighted sum:

```
fitness = w_alive * alive
        + w_divisions * divisions
        + w_starvation_deaths * starvation_deaths
        + w_damage_deaths * damage_deaths
```

Weights are provided by the UI in `fitness_weights`.

Baseline:
- Evolution computes and stores a baseline fitness from the unmodified base payload.
- The plot draws a dashed baseline line.

### Candidate storage and reconstruction

Candidates are stored server-side during a run (in memory):

- Each candidate has:
  - `id` (UUID)
  - `gen` (generation index)
  - `fitness`
  - `metrics`
  - `genome` (either affine or delta representation)

Reconstruction:
- `/api/evolution/candidate` takes an id and returns a complete `gridstate` payload.
- For `affine`, the genome contains per-layer `scale` and `bias`.
- For `cem_delta`, the genome contains per-layer `delta_b64` arrays.

The UI uses this for:
- loading a candidate into Runtime
- downloading candidate JSON

### Performance and profiling

Evolution can be computationally expensive.

#### The critical loop

Each evaluation does roughly:

1. Copy the base payload and apply the genome
2. Run `ticks` times:
   - `apply_layer_ops_inplace(payload, seed_offset=...)`
3. Decode the cell layer to count alive cells
4. Read event counters totals
5. Compute fitness

#### Built-in profiler telemetry

Evolution status includes:

- `perf.evals`: number of evals measured
- `perf.apply_s`: time spent in payload copy + genome application
- `perf.ticks_s`: time spent inside `apply_layer_ops_inplace` ticks
- `perf.decode_cell_s`: time spent decoding cell layer for fitness
- `perf.total_s`: total evaluation time

This is meant to answer:

- Are we bottlenecked by simulation ticks?
- Are we bottlenecked by JSON/base64 overhead?
- Does adding more workers actually improve throughput?

#### Notes on scaling to many cores

The backend currently uses **threads** (`ThreadPoolExecutor`). Depending on whether `apply_layer_ops_inplace` releases the GIL (or is dominated by NumPy operations), thread scaling may be limited.

If `perf.ticks_s` dominates and CPU utilization does not scale with threads, the next step is typically:
- process-based parallelism for evaluations, OR
- heavy optimization inside `apply_layer_ops_inplace`

### Development workflow

#### Typical git workflow

```bash
git status
git add -A
git commit -m "your message"
git pull --rebase
git push
```

#### Updating on another computer

If already cloned:

```bash
git pull --rebase
```

If you have local changes:

```bash
git stash
git pull --rebase
git stash pop
```

### Troubleshooting

#### Ubuntu/Debian: `python3 -m venv` fails / `ensurepip` missing

Install venv support:

```bash
sudo apt update
sudo apt install -y python3-venv
```

If you’re on Python 3.12 specifically:

```bash
sudo apt install -y python3.12-venv
```

Then recreate `.venv` and install deps.

#### Ubuntu/Debian: `externally-managed-environment`

This is PEP 668 protection. Fix is to use a virtual environment (`python3 -m venv .venv`) and install into it.

Avoid `--break-system-packages` unless you know what you’re doing.

#### GitHub push authentication

GitHub does not allow password auth for HTTPS pushes.

Use one of:
- a Personal Access Token (PAT) as your “password”
- SSH remotes
- GitHub CLI (`gh auth login`)

### Notes / open ends

- Multi-fidelity evaluation (short ticks for culling + long ticks for survivors) is a natural next improvement for speed.
- If you are running with 20–35 workers and it is still slow, use `perf` to identify whether the bottleneck is simulation ticks or overhead.
