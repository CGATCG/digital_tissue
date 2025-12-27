# digital_tissue

This repository contains a **grid-based “digital tissue” simulation** plus a lightweight **web editor + runtime dashboard** and a built-in **Evolution** tool for searching over *initial conditions* (gene/RNA/protein layers) to maximize a configurable reward.

The project is intentionally self-contained:
- The backend is a single Python HTTP server (`runtime_server.py`) that:
  - serves the web UI from `web_editor/`
  - exposes JSON APIs under `/api/runtime/*` and `/api/evolution/*`
  - runs simulations and evolutionary search in-process
- The UI is plain HTML/CSS/JS (no build step), served by the backend.

Storage is not treated as a constraint; the design favors clarity and iteration speed.

---

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [The `gridstate.json` format](#the-gridstatejson-format)
- [Running the backend + UI](#running-the-backend--ui)
- [Web UI overview](#web-ui-overview)
  - [Runtime screen](#runtime-screen)
  - [Evolution screen](#evolution-screen)
- [Backend API](#backend-api)
  - [Runtime API](#runtime-api)
  - [Evolution API](#evolution-api)
- [Evolution algorithms](#evolution-algorithms)
  - [`affine`: per-layer scale/bias GA](#affine-per-layer-scalebias-ga)
  - [`cem_delta`: CEM per-cell delta-field](#cem_delta-cem-per-cell-delta-field)
  - [Fitness function](#fitness-function)
  - [Candidate storage and reconstruction](#candidate-storage-and-reconstruction)
- [Performance and profiling](#performance-and-profiling)
- [Development workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)

---

## Quick start

### 1) Create a Python virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2) Run the server

```bash
python3 runtime_server.py
```

By default it serves on:

- `http://127.0.0.1:8000/`

You can pass a port:

```bash
python3 runtime_server.py 8001
```

### 3) Open the UI

Open the printed URL in your browser.

---

## Repository layout

High-level structure:

- `runtime_server.py`
  - Single-process backend server
  - Serves the web UI
  - Implements `/api/runtime/*` and `/api/evolution/*`
  - Runs the simulation and evolution jobs

- `apply_layer_ops.py`
  - Core simulation update step: `apply_layer_ops_inplace(payload, seed_offset=...)`
  - Utilities for encoding/decoding layer arrays:
    - `_decode_float32_b64(b64, expected_len, layer_name)`
    - `_encode_float32_b64(arr)`

- `web_editor/`
  - `index.html` – UI layout (Runtime + Evolution screens)
  - `style.css` – UI styling
  - `app.js` – UI logic, API calls, plotting, candidate table, etc.

- `array/`
  - Example `gridstate.json` payloads and snapshots

- `trials/`
  - Research/analysis scripts and stored experiment results used during development

- `requirements.txt`
  - Python dependencies (primarily `numpy`; some scripts may use `streamlit` and `plotly`)

---

## The `gridstate.json` format

This project passes simulation state around as a single JSON object (the “payload”).

At minimum, a payload is expected to look like:

- `version`: must be `1`
- `H`, `W`: grid dimensions
- `layers`: list of layer metadata objects, typically:
  - `{ "name": "gene_x", "kind": "continuous" }`
- `data`: dictionary mapping layer names to buffers:
  - `{ "dtype": "float32", "b64": "..." }`

The layer buffers are stored as **base64-encoded float32 arrays**, length `H*W`.

### Why base64 float32?

- Compact and portable representation inside JSON
- Fast enough to decode/encode with `numpy.frombuffer(...)` and `arr.tobytes()`
- Keeps the UI/backend interface simple (no binary protocol)

### Event counters

The simulation tracks certain events in an optional structure:

- `payload["event_counters"]`
  - includes a `totals` dict used by Evolution fitness calculations (divisions, deaths, etc.)

Evolution always strips any existing `event_counters` from the base payload copy to avoid contaminating evaluations.

---

## Running the backend + UI

The backend is a local development server (ThreadingHTTPServer). It serves the UI as static files and provides JSON APIs.

Start it from the repo root:

```bash
python3 runtime_server.py
```

You should see output like:

- `Runtime server: http://127.0.0.1:8000/`

Open that URL.

### Important behavior: no caching

The server sets `Cache-Control: no-store`, so editing `web_editor/app.js` or `web_editor/index.html` refreshes immediately.

---

## Web UI overview

The UI lives in `web_editor/` and is served by the backend.

The Evolution UI was designed to remain compatible even as backend algorithms evolved:
- endpoints remain stable
- status payload stays consistent
- top candidates are always reconstructable into a full `gridstate.json`

### Runtime screen

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

### Evolution screen

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

---

## Backend API

All backend logic is implemented in `runtime_server.py`.

### Runtime API

#### `POST /api/runtime/reset`

Body:

- `payload`: a `gridstate` object

Response:
- `{ ok, tick, H, W, layers }`

#### `POST /api/runtime/frame`

Body:
- `layers`: optional list of layer names to return in `data`

Response includes:
- `tick`
- `data` (selected layers)
- `scalars` (sum/mean/nonzero per decoded layer)
- `measurements` (expressions computed from the payload config)
- `events` (event counters)

#### `POST /api/runtime/step`

Body:
- `layers`: optional list of layer names to return

Behavior:
- calls `apply_layer_ops_inplace(payload, seed_offset=tick)`
- increments `tick`
- returns the same structure as `/frame`

### Evolution API

Evolution is managed by a background thread (`_EvolutionJob`) so the UI stays responsive.

#### `POST /api/evolution/start`

Body:
- `payload`: base `gridstate` object
- `config`: evolution config object built by the UI

Response:
- `{ ok: true, job_id }`

#### `POST /api/evolution/stop`

Response:
- `{ ok: true }`

#### `POST /api/evolution/status`

Response:
- `running`, `error`, `job_id`
- `cfg`: the config used
- `progress`: generation/variant counters + total evaluations
- `baseline`: baseline evaluation (fitness + metrics)
- `series`: live per-evaluation series (fitness/best/mean) for plotting
- `history`: per-generation summaries (best/mean/p10/p90)
- `top`: top candidates table for UI
- `perf`: cumulative timing telemetry for diagnosing bottlenecks

#### `POST /api/evolution/candidate`

Body:
- `id`: candidate id

Response:
- `{ ok, id, fitness, metrics, genome, payload }`

`payload` is a fully reconstructed `gridstate.json` that can be loaded into Runtime.

---

## Evolution algorithms

Evolution searches over the **initial conditions** of layers matching:

- `gene_*`
- `rna_*`
- `protein_*`

The `cell`/`cell_type` layer is detected automatically and used for:
- determining which grid locations are initially “cells”
- (optionally) masking delta-field updates to only those locations

### `affine`: per-layer scale/bias GA

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

### `cem_delta`: CEM per-cell delta-field

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

---

## Performance and profiling

Evolution can be computationally expensive.

### The critical loop

Each evaluation does roughly:

1. Copy the base payload and apply the genome
2. Run `ticks` times:
   - `apply_layer_ops_inplace(payload, seed_offset=...)`
3. Decode the cell layer to count alive cells
4. Read event counters totals
5. Compute fitness

### Built-in profiler telemetry

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

### Notes on scaling to many cores

The backend currently uses **threads** (`ThreadPoolExecutor`). Depending on whether `apply_layer_ops_inplace` releases the GIL (or is dominated by NumPy operations), thread scaling may be limited.

If `perf.ticks_s` dominates and CPU utilization does not scale with threads, the next step is typically:
- process-based parallelism for evaluations, OR
- heavy optimization inside `apply_layer_ops_inplace`

---

## Development workflow

### Typical git workflow

```bash
git status
git add -A
git commit -m "your message"
git pull --rebase
git push
```

### Updating on another computer

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

---

## Troubleshooting

### Ubuntu/Debian: `python3 -m venv` fails / `ensurepip` missing

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

### Ubuntu/Debian: `externally-managed-environment`

This is PEP 668 protection. Fix is to use a virtual environment (`python3 -m venv .venv`) and install into it.

Avoid `--break-system-packages` unless you know what you’re doing.

### GitHub push authentication

GitHub does not allow password auth for HTTPS pushes.

Use one of:
- a Personal Access Token (PAT) as your “password”
- SSH remotes
- GitHub CLI (`gh auth login`)

---

## Notes / open ends

- Multi-fidelity evaluation (short ticks for culling + long ticks for survivors) is a natural next improvement for speed.
- If you are running with 20–35 workers and it is still slow, use `perf` to identify whether the bottleneck is simulation ticks or overhead.
