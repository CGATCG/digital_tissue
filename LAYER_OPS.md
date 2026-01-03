# Layer Ops (digital_tissue) — LLM Context Document

This document describes how **Layer Ops** works in the `digital_tissue` project.

Layer Ops is the simulation “update step” system: each tick, a configured sequence of operations reads and writes named **layers** on a 2D grid. Layer Ops is defined by a JSON config embedded in the simulation payload (`gridstate.json` / “payload”), and executed by `apply_layer_ops.py`.

This doc is written to be used as **context for an LLM**.

---

## 1) Mental model

- A simulation state (“payload”) contains:
  - a grid size `H x W`
  - a set of named layers (`payload.layers` metadata + `payload.data` buffers)
  - optional counters (`payload.event_counters`)
  - optional `layer_ops_config` describing how to update layers each tick

- Each **layer** is stored as a flat `float32` buffer length `H*W`, but Layer Ops evaluates expressions on **2D views** of those layers (`H x W`) for convenience.

- A Layer Ops program is a list of **steps** executed in order. Steps may:
  - modify one or many layers (type `op`)
  - create temporary variables (type `let`)
  - expand loops over layers (type `foreach`)
  - run built-in “physics/biology-ish” operations (e.g. `transport`, `diffusion`, `divide_cells`, `pathway`)

- The Layer Ops engine is intentionally constrained:
  - expressions are parsed with Python `ast` and only a limited set of syntax is allowed
  - no attribute access (`foo.bar`) is allowed
  - only whitelisted functions are callable

---

## 2) Where the authoritative behavior lives

- **Executor**: `apply_layer_ops.py`
  - entry point: `apply_layer_ops_inplace(payload, export_let_layers=False, seed_offset=0) -> int`
  - loads layers from payload (`_extract_layers`)
  - expands `foreach` into concrete steps
  - evaluates steps in order
  - writes updated layers back into payload as base64 float32 buffers

- **UI** (editor/configuration): `web_editor/app.js`
  - user edits a structured Layer Ops config
  - the UI performs lightweight validation and provides conveniences like autocomplete

---

## 3) Payload format basics

A payload is a JSON object with (at minimum):

- `version`: must be `1`
- `H`, `W`: grid dimensions
- `layers`: list of metadata objects like:

```json
{ "name": "gene_x", "kind": "continuous" }
```

- `data`: object mapping layer names to encoded buffers:

```json
"gene_x": { "dtype": "float32", "b64": "..." }
```

Layer Ops reads/writes the `data` buffers.

### Layer kinds

The engine tracks a `kind` per layer from `payload.layers`:

- `continuous`: treated as float fields
- `counts`: treated as **non-negative integers** stored in float32 buffers
- `categorical`: treated as **integers from a discrete allowed set** stored in float32 buffers

Some built-in steps require specific kinds (notably molecule layers for diffusion/transport require `counts`).

---

## 4) Layer Ops configuration

Layer Ops configuration lives at:

- `payload.layer_ops_config`

There are currently two config versions:

### Version 1 (legacy)

```json
{
  "version": 1,
  "ops": [
    { "target": "gene_a", "expr": "where(cell==1, gene_a, 0)" }
  ]
}
```

Internally, v1 ops are converted into v2-like steps of type `op`.

### Version 2 (current)

```json
{
  "version": 2,
  "steps": [
    { "type": "op", "target": "gene_a", "expr": "where(cell==1, gene_a, 0)" }
  ]
}
```

Each `step` is a JSON object. Most steps also support:

- `enabled`: if `false`, step is skipped
- `name`: optional label (also used for some event counting)
- `group`: optional grouping label (primarily for UI organization)

---

## 5) Execution order and tick semantics

When the simulation advances by one tick, the runtime calls:

- `apply_layer_ops_inplace(payload, seed_offset=...)`

Within that call:

1. Layers are decoded into numpy arrays.
2. `foreach` steps are expanded into concrete `op`/`let` steps.
3. Steps are executed sequentially in the resulting order.
4. Modified layers are written back to payload.

Steps later in the list see the results of earlier steps in the same tick.

---

## 6) Expression engine (for `op` and `let`)

### Environment

During evaluation, the expression environment contains:

- Every layer name mapped to a **2D array** (`H x W`) view of the underlying buffer.
- Any `let` variables created earlier in the tick (also `H x W` arrays).
- Boolean constants: `True`, `False`
- A limited set of functions.

### Allowed functions

From `apply_layer_ops.py` the core functions are:

- `where(a, b, c)` → elementwise selection (numpy `where`)
- `clip(x, lo, hi)`
- `abs(x)`
- `sqrt(x)`
- `exp(x)`
- `log(x)`
- `minimum(a, b)`
- `maximum(a, b)`

There are also **layer-group helpers** that accept glob patterns:

- `sum_layers(pat)`
- `mean_layers(pat)`
- `min_layers(pat)`
- `max_layers(pat)`

These pattern helpers operate over all layer names matching a glob pattern.

There are also random-field helpers:

- `rand_beta(alpha=1.0, beta=1.0)` → produces an `H x W` random field in `[0,1]`
- `rand_logitnorm(mu=0.0, sigma=1.0)` → produces an `H x W` random field in `[0,1]`

And a scalar reduction:

- `sum_layer(x)` → returns a scalar float (`sum` over all elements)

(Exact function availability is controlled in code and may evolve.)

### Syntax restrictions

Expressions are parsed via Python `ast` in `eval` mode and restricted:

- only a limited set of AST node types is allowed
- **no attribute access** (`a.b`) and no `np.where` style calls
- only direct calls to allowed functions are permitted
- identifiers must exist in the environment (layers, let-vars, True/False)

### Shapes and broadcasting

Most expressions evaluate to either:

- an `H x W` array
- a scalar, which is coerced/broadcast to an array

The engine coerces results via `np.asarray`.

---

## 7) Step types

### 7.1) `op` — write expression output into one or many target layers

Schema (common fields):

```json
{
  "type": "op",
  "enabled": true,
  "name": "optional",
  "group": "optional",
  "target": "layer_name_or_pattern",
  "expr": "expression"
}
```

#### Target selection (supports wildcards)

`target` supports:

- a single layer name: `gene_a`
- a comma-separated list: `gene_a,gene_b`
- a glob-like pattern: `gene_*`
- an explicit glob prefix: `glob:gene_*`

Behavior:

- The runtime expands this target spec into a concrete list of target layer names.
- The expression is evaluated and then written into each matched target.

**Important nuance**: for wildcard targets, the expression is evaluated once and then applied to each target. There is not currently a special identifier like “current_target” available in the expression.

If you need “apply the same transform to each layer while referencing that layer’s current values”, you typically want a `foreach` step (see below).

#### Kind-specific writeback rules

When writing to the target layer:

- If `kind == "counts"`:
  - the output is rounded to integer and clipped to be non-negative.
- If `kind == "categorical"`:
  - the output is rounded to integer
  - then snapped/clipped to the allowed category set.
  - if `allowed_values` is not provided in the step, it infers allowed values from the current layer values.
- Otherwise (`continuous`):
  - the float output is written directly.

#### Example: zero genes outside cells

Assuming `cell` is 1 for cell pixels and 0 otherwise:

```json
{ "type": "op", "target": "gene_*", "expr": "where(cell==1, gene_a, 0)" }
```

This is **not correct** because it references `gene_a`. For multi-target transforms, use `foreach`.

Correct version using `foreach` is shown later.

---

### 7.2) `let` — define a temporary variable

Schema:

```json
{
  "type": "let",
  "var": "name",
  "expr": "expression"
}
```

Behavior:

- Evaluates `expr` to an `H x W` array
- Stores it in the environment as `var`
- Later steps can reference `var`

Constraints:

- `var` must be a valid identifier
- It must not collide with a layer name
- It must not be already defined earlier in the tick

Example:

```json
{ "type": "let", "var": "is_cell", "expr": "cell==1" }
```

Then:

```json
{ "type": "op", "target": "molecule_atp", "expr": "where(is_cell, molecule_atp, 0)" }
```

---

### 7.3) `foreach` — expand a template over matching layers

`foreach` is the mechanism that lets you:

- match a set of layer names using a pattern
- expand a template step list into many concrete steps
- substitute capture groups into `target`, `expr`, `var`, `name`, `group`

Schema:

```json
{
  "type": "foreach",
  "match": "pattern",
  "require_match": false,
  "steps": [
    { "type": "op", "target": "gene_${1}", "expr": "where(cell==1, gene_${1}, 0)" }
  ]
}
```

#### Pattern matching

The `match` field is compiled as a regex in one of several ways:

- If it begins with `glob:` it is treated as a glob pattern and converted to a regex.
- If it begins with `re:` or `regex:` it is treated as a raw regex.
- If it looks like a simple glob (contains `*`, `?`, `[]` and not regex metacharacters), it is treated as a glob.
- Otherwise it is treated as a raw regex.

Glob-to-regex conversion uses capturing groups:

- `*` becomes `(.*?)`
- `?` becomes `(.)`
- character classes like `[abc]` are supported

This means that in a `foreach` expansion, you can reference capture groups as `${1}`, `${2}`, ...

#### Substitution

In each template step, any string fields among:

- `name`, `group`, `expr`, `target`, `var`

have `${N}` substituted with the corresponding capture group from the match.

#### `require_match`

If `require_match` is `true` and the pattern matches zero layers, Layer Ops raises an error.

#### Example: apply the same transform to every `gene_*` layer

This example matches `gene_*` layers and rewrites each one to be 0 outside cells:

```json
{
  "type": "foreach",
  "match": "glob:gene_*(.*)",
  "require_match": true,
  "steps": [
    { "type": "op", "target": "gene_${1}", "expr": "where(cell==1, gene_${1}, 0)" }
  ]
}
```

In practice, the UI typically generates the correct pattern/captures for you.

#### UI shorthand

The web UI offers a higher-level text shorthand like:

```text
for (i in "gene_*") {
  gene_{i} <- where(cell==1, gene_{i}, 0)
}
```

#### Placeholders: `{i}` vs `{j}` / `{k}` / ...

In the UI `foreach` mini-language there are two kinds of placeholders:

- **`{i}` (or generally `{<loopVar>}`)**
  - This is the **loop variable** from `for (i in "...")`.
  - It expands to the **full matched layer name**.
  - Example: if a matched layer is `gene_atp_maker`, then `{i}` becomes `gene_atp_maker`.

- **`{j}`, `{k}`, `{l}`, ...**
  - These are **capture-group placeholders** produced by the wildcard pattern.
  - When the loop pattern is treated as a glob, each `*` and `?` creates a capture group.
  - The capture groups are exposed as:
    - `{j}` = capture group 1
    - `{k}` = capture group 2
    - `{l}` = capture group 3
    - ...
  - Example: pattern `gene_*` with matched layer `gene_atp_maker` captures `atp_maker` as group 1, so `{j}` becomes `atp_maker`.

Notes:

- You can choose any valid identifier as the loop variable (not just `i`).
- Under the hood, the UI compiles the mini-language into a JSON `foreach` step whose template strings use `${1}`, `${2}`, ... for capture groups (and `${0}`-like behavior for the full match), and those templates are expanded at runtime.

Additional UI mini-language details:

- The UI shorthand currently only supports **glob-style patterns** (examples: `gene_*`, `glob:gene_*`).
  - If you want a true regex match, you must edit the underlying JSON-style `foreach.match` (using `re:` / `regex:`) rather than the shorthand.
- The loop body is parsed as a list of statements, one per line:
  - comment lines starting with `#` are ignored
  - an optional trailing `;` is allowed
  - each statement must be `LHS <- RHS` (the `<-` arrow is required)
- The UI tries to infer whether a statement is an `op` or a `let`:
  - if `LHS` has no `{...}` placeholders and is a valid identifier:
    - if it matches an existing layer name, it becomes an `op`
    - otherwise it becomes a `let`
  - if `LHS` contains placeholders, the UI samples a few expansions to decide whether they look like layer targets vs let-vars.
- Capture placeholders are limited to a small set of convenience names:
  - `{j}`..`{p}` map to capture groups 1..7 (skipping any name that conflicts with your chosen loop variable)
  - if your glob pattern contains more capture groups than that, only the first few are directly addressable via these built-in placeholder names.

Internally, the UI compiles this into a `foreach` step with captures and `${N}` substitution.

---

### 7.4) `diffusion` — diffuse molecule count layers through non-cell space

This is a built-in step designed for “molecule layers” of kind `counts`.

Key fields:

- `molecules`: string glob (like `molecule_*`) or list of layer names
- `cell_layer`: typically `cell`
- `cell_value`: typically `1`
- `rate` or `rate_layer`:
  - `rate`: scalar in `[0,1]`
  - `rate_layer`: name of a layer that scales diffusion rate per-pixel
- `seed`: optional
- `dt`: optional (clamped internally)

Behavior summary:

- Determines which pixels are “cells” via `cell_layer == cell_value`
- Diffusion occurs only through **non-cell** pixels
- For each molecule layer, the step redistributes integer counts stochastically using binomial draws and neighbor conductances.

If a molecule layer is not kind `counts`, the step errors.

---

### 7.5) `transport` — directional transport of molecule counts via exporter/importer proteins

Transport moves molecule counts across edges where “exporter” and “importer” capacity exists.

Key fields:

- `molecules`: string glob or list
- `protein_prefix`: default `protein_`
- `molecule_prefix`: default `molecule_`
- `cell_layer`, `cell_value`
- `dirs`: list of directions (`north/south/east/west` or `n/s/e/w`)
- `per_pair_rate`: scalar in `[0,1]`
- `seed`: optional

Naming convention:

For each molecule layer (e.g. `molecule_glucose`), transport looks for capacity layers like:

- exporters: `protein_north_exporter_glucose` (for north movement)
- importers: `protein_south_importer_glucose` (opposite side)

(Exact suffix extraction uses the configured prefixes.)

The transport algorithm computes edge capacities and then uses hypergeometric draws to allocate integer moves.

---

### 7.6) `divide_cells` — stochastic cell division into nearby empty spaces

This step identifies cells that should divide (based on a trigger layer/variable), then attempts to place daughter cells into nearby empty cells.

Key fields:

- `cell_layer`, `cell_value`, `empty_value`
- `trigger_layer`: layer name or let-var name
- `threshold`: division triggers where `trigger_layer > threshold`
- `split_fraction`: fraction of material moved into daughter cell
- `max_radius`: search radius for empty target
- `layer_prefixes`: which layers should be split/copied on division
- optional target mask (`target_mask_layer`, `target_mask_value`)
- `seed`: optional

Behavior summary:

- Cells eligible for division: `cell_layer == cell_value` and `trigger_layer > threshold`
- Daughter targets must currently equal `empty_value` (often `0`)
- For layers in `layer_prefixes`:
  - continuous-like layers: split amounts by `split_fraction`
  - `counts`: split integer counts by `split_fraction`
  - `damage*` layers are copied to daughter
  - categorical layers are not split

This step also increments event counters (e.g. `divisions`).

---

### 7.7) `pathway` — “pathway network” transformation between input and output molecule layers

This step creates a deterministic pseudo-random DAG topology derived from `pathway_name` and then transforms input layers into output layers with some efficiency.

Key fields:

- `pathway_name`
- `inputs`: list or comma-separated string
- `outputs`: list or comma-separated string
- `num_enzymes`
- `cell_layer`, `cell_value`
- `efficiency`
- `seed`: optional

The implementation is more complex; treat it as a black-box transformation that depends on name/topology.

---

## 8) Wildcards/patterns: where they work (and where they don’t)

### Works in

- `op.target`:
  - `gene_*`
  - `glob:gene_*`
  - `gene_a,gene_b,gene_*`

- `foreach.match`:
  - glob-like patterns or regex

- `sum_layers("gene_*")` and similar helper functions inside expressions.

### Does NOT work in

- Expressions as an identifier.

Example that will **not** work:

```text
expr = where(cell==0, 0, gene_*)
```

Because `gene_*` is not a valid identifier. Identifiers must correspond to a real layer name or `let` variable.

### The recommended way to apply the same transform to every matched layer

Use `foreach` so the expression can reference each concrete layer name:

```text
for (i in "gene_*") {
  gene_{i} <- where(cell==1, gene_{i}, 0)
}
```

---

## 9) Performance and profiling knobs

Layer Ops supports optional profiling and caching controls in `layer_ops_config` and/or payload flags:

- `_profile_layer_ops`, `profile_layer_ops`, `profile`, `profile_deep`
- `_profile_expr`, `profile_expr`
- `profile_step_names`

And some optimization toggles:

- `opt_env_cache` / `optimize_env_cache`:
  - caches the expression environment across steps and updates it incrementally
- `opt_expr_cache` / `optimize_expr_cache`:
  - caches compiled expressions
- `opt_diffusion_buffers`:
  - reuses diffusion buffers to reduce allocations

Profiling results are written into:

- `payload.event_counters.layer_ops_perf`

---

## 10) Practical examples

### Example A: Clamp a continuous layer

```json
{ "type": "op", "target": "gene_a", "expr": "clip(gene_a, 0, 1)" }
```

### Example B: Create a mask with `let`

```json
{ "type": "let", "var": "in_cell", "expr": "cell==1" }
```

Then:

```json
{ "type": "op", "target": "molecule_atp", "expr": "where(in_cell, molecule_atp, 0)" }
```

### Example C: Update multiple explicit targets

```json
{ "type": "op", "target": "gene_a,gene_b", "expr": "where(cell==1, 1, 0)" }
```

This writes the same computed field into both `gene_a` and `gene_b`.

### Example D: Reduce many layers inside an expression

```json
{ "type": "op", "target": "gene_total", "expr": "sum_layers('gene_*')" }
```

### Example E: Apply a per-layer transform to all `gene_*` layers (foreach)

UI form:

```text
for (i in "gene_*") {
  gene_{i} <- where(cell==1, gene_{i}, 0)
}
```

Underlying conceptual form:

- match gene layers
- expand into one `op` per matched layer, each referencing itself

---

## 11) Common failure modes

- **Unknown identifier**: you referenced a layer name that does not exist (or a `let` var not yet defined).
- **Invalid expr syntax**: Python expression parsing failed.
- **Disallowed syntax**: attribute access, unsupported AST nodes, etc.
- **Pattern matched no layers**:
  - `op.target` pattern matched nothing (runtime error)
  - or `foreach.require_match = true` and no layers matched
- **Kind mismatch**:
  - `diffusion`/`transport` expects molecule layers to be `counts`

---

## 12) Runtime gotchas / subtle semantics

These are details that matter when you’re generating or reasoning about Layer Ops programmatically.

### Multi-target `op`: expression evaluation is not “per target”

If an `op` uses a wildcard or comma-separated target spec (e.g. `target: "gene_*"` or `target: "gene_a,gene_b"`):

- The engine expands that spec into a list of concrete target layer names.
- The `expr` is evaluated to produce an output array.
- That same computed output is then written into each target.

There is no built-in “current target layer” identifier inside the expression.

If you want “apply the same transform to every matched layer while referencing that layer’s current values”, use a `foreach` expansion instead.

### `counts` kind writeback: integer, non-negative

When writing into a layer whose kind is `counts`:

- The result is rounded to integer (`rint`).
- It is clipped to be non-negative (`>= 0`).
- It is then stored back as float32.

This means negative values in an expression will become `0`, and non-integers will be rounded.

### `categorical` kind writeback: snap to allowed set

When writing into a `categorical` layer:

- The result is rounded to integer.
- Then it is snapped/clipped to an allowed set of integer values.

How the allowed set is chosen:

- If the step provides `allowed_values`, those values are used.
- Otherwise, the engine infers allowed values from the current values present in the target layer (unique integers).

This inference means that “introducing a brand-new category value” via an expression will typically not work unless you include it in `allowed_values`.

### `molecule_atp` reference guard

Before running steps, the engine checks whether the config references `molecule_atp` anywhere.

- If it does and the payload does not contain a `molecule_atp` layer, Layer Ops errors.

This check is recursive over the full step structure (including nested `foreach` templates) and looks for either:

- a literal layer name match, or
- a word-boundary match inside strings (e.g. inside expressions).

---

## 13) Notes for LLM-driven authoring

If you are generating Layer Ops configs programmatically:

- Prefer `version: 2`.
- Use `foreach` for “do the same thing to each matched layer while referencing itself”.
- Use wildcard targets (`op.target = gene_*`) when the expression does not need to reference the current target.
- When using `foreach`, use `${N}` placeholders in the underlying JSON templates.
- Keep expressions simple and limited to allowed functions and identifiers.

---

## 14) Key source references

- `apply_layer_ops.py`
  - `apply_layer_ops_inplace`
  - `_expand_foreach_steps`
  - expression validation/eval (`_compile_expr_cached`, `_eval_expr_fast`)

- `web_editor/app.js`
  - validation helpers and UI rendering for Layer Ops
