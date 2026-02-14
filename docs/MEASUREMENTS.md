# Measurements Manual

This document describes the **Measurements** system (the “Functions” tab) and how measurement expressions are validated and evaluated.

The goal of measurements is to compute **scalar (single-number) summaries** from 2D layer fields.

---

## 1) Where measurements live

Measurements are stored in the simulation payload under:

- `payload.measurements_config`

In the editor UI, the “Measurements” / “Functions” tab edits the measurement list, and when exporting a payload / `gridstate.json`, the editor embeds:

- `measurements_config: buildFunctionsConfigJson()`

---

## 2) Config schema (current)

### Version 3 (current)

`measurements_config` is expected to be:

- `version: 3`
- `measurements: [ { name, expr }, ... ]`

Example:

```json
{
  "version": 3,
  "measurements": [
    {"name": "inflammation", "expr": "mean(cytokine, where=(circulation==1))"},
    {"name": "glucose_per_circ_cell", "expr": "sum(glucose, where=(circulation==1)) / count(where=(circulation==1))"}
  ]
}
```

Notes:

- The editor trims `name` and preserves `expr` as-entered.
- Empty names or empty expressions are dropped when building the JSON.

### Version 1 / 2 (legacy)

Legacy measurement configs can exist in older saved payloads.

- The editor can parse `version: 1` / `version: 2` embedded configs and convert them into a v3 list.
- The runtime server only evaluates measurements when `measurements_config.version == 3`.

Legacy configs were “mask + calculations” based:

- v2: `{ version: 2, mask: {...}, calculations: [...] }`
- v1: `{ version: 1, measurements: [...] }` (treated as a v2-like config with a default `circulation==1` mask)

The editor’s conversion rewrites these into v3 expressions of the form:

- `masked_mean` -> `mean(layer, where=<maskExpr>)`
- `masked_sum` -> `sum(layer, where=<maskExpr>)`
- `mask_count` -> `count(where=<maskExpr>)`
- `masked_sum_over_count` -> `sum(layer, where=<maskExpr>) / count(where=<maskExpr>)`
- `masked_sum_ratio` -> `sum(layerA, where=<maskExpr>) / sum(layerB, where=<maskExpr>)`

---

## 3) Expression language (what actually runs)

At runtime, measurement expressions are evaluated by a small, restricted evaluator (`output_calc._ExprEval`) that:

- Parses the expression as Python `ast.parse(..., mode="eval")`
- Evaluates only a limited set of AST node types
- Requires the final result to be a **scalar number** (float)

If the expression evaluates to an array, runtime raises an error.

### 3.1) Allowed identifiers

You can reference:

- Layer names (must exist in `payload.data`)
- `True`, `False`

You can also call a small set of reduction functions (see below).

### 3.2) Allowed operators

Supported operators include:

- **Unary**
  - `+x`, `-x` (scalars)
  - `~mask` (boolean NOT for arrays)
  - `not x` (works for booleans and boolean arrays)

- **Binary**
  - `+ - * / **` (scalar arithmetic only)
  - `&` and `|` (mask composition; both sides must be arrays)

- **Comparisons**
  - `== != > >= < <=`

Comparison rules:

- array-vs-scalar is allowed (broadcast scalar)
- array-vs-array is allowed only if shapes match
- scalar-vs-scalar is allowed
- chained comparisons like `a < b < c` are not supported

Important:

- Python boolean operators `and` / `or` are not supported; use `&` / `|` with parentheses for masks.

---

## 4) Reduction functions

Measurements are centered around reductions of a 2D array to a scalar.

Supported functions:

- `mean(layer, where=mask)`
- `sum(layer, where=mask)`
- `min(layer, where=mask)`
- `max(layer, where=mask)`
- `std(layer, where=mask)`
- `var(layer, where=mask)`
- `median(layer, where=mask)`
- `quantile(layer, q, where=mask)`
- `count(where=mask)`

Where:

- `layer` must evaluate to an array (usually a layer name)
- `where=` is optional and must evaluate to a boolean array mask

### 4.1) `where=` is a keyword argument, not a function

`where` is not a callable.

Correct:

- `mean(glucose, where=(circulation==1))`

Incorrect:

- `mean(glucose, where(circulation==1))`
- `where(circulation==1)`

### 4.2) `count()` special rules

- `count(where=mask)` counts the number of `True` cells in the mask
- `count()` does **not** take positional arguments
- `count()` with no `where` returns `H*W` (the full grid size)

### 4.3) `quantile()`

`quantile()` requires `q`:

- positional: `quantile(glucose, 0.9, where=(circulation==1))`
- or keyword: `quantile(glucose, q=0.9, where=(circulation==1))`

---

## 5) Masks: how to write them

Masks are boolean arrays. Typical pattern:

- `(layer == 1)`
- `(layer > 0.5)`

You can combine masks:

- `(circulation==1) & (cell==1)`
- `~(cell==1)`

Be careful with precedence; use parentheses.

---

## 6) Missing / empty-mask behavior

This is a common “tiny detail” that matters for downstream fitness.

### 6.1) Empty masks return `None`

For reductions, if `where` selects zero cells:

- the reduction returns `None`

That means an expression like:

- `mean(glucose, where=(circulation==2))`

will yield `None` if there are no `circulation==2` cells.

### 6.2) Runtime server stores failures as `null`

When the runtime server evaluates measurements from a payload:

- Any error evaluating a measurement expression yields `None` / `null` for that measurement.

---

## 7) UI validation vs runtime reality

The editor’s `validateMeasurementExpr()` is intentionally lightweight:

- It tokenizes identifiers and checks that each identifier is either:
  - a known layer name
  - an allowed function name
  - `where`, `True`, `False`

It does **not** fully validate:

- Python syntax correctness
- operator restrictions (`and`/`or`)
- function arity
- whether the expression returns a scalar

So you can see `OK` in the UI but still get runtime `null`.

---

## 8) How measurements are used in evolution

Evolution uses measurements in two places:

- **metrics reporting** (for inspection)
- **fitness computation** (weighted sum)

### 8.1) Fitness weights

Evolution config uses:

- `fitness_weights.measurements = { <measurement_name>: <weight>, ... }`

Fitness contribution is:

- `fitness = Σ weight[name] * measurements[name]`

If a measurement is missing / `null`, it effectively contributes `0.0` in the current implementation.

### 8.2) Aggregation across ticks (`measurement_aggs`)

By default, measurements are computed on the **final tick state**.

Optionally, you can ask evolution to aggregate certain measurements over time:

- `fitness_weights.measurement_aggs = { <measurement_name>: "mean"|"median", ... }`

Behavior:

- For listed measurement names, evolution re-evaluates them each tick.
- It stores the per-tick time series and then replaces the final measurement with:
  - mean of per-tick values, or
  - median of per-tick values

Subtle detail:

- Per-tick missing values are treated as `0.0` when aggregating.

---

## 9) Examples

### 9.1) Mean inflammation in circulation

```txt
mean(cytokine, where=(circulation==1))
```

### 9.2) Per-cell averages (safe denominator)

```txt
sum(glucose, where=(circulation==1)) / maximum(count(where=(circulation==1)), 1)
```

If you don’t guard the denominator, you can get runtime errors when the count is zero.

### 9.3) Mask composition

```txt
mean(glucose, where=(circulation==1) & (cell==1))
```

---

## 10) Key source references

- `backend/runtime_server.py`
  - parser and validator for `payload["measurements"]`
  - measurement evaluation and time series construction

- `apps/editor/app.js`
  - `buildFunctionsConfigJson()`
  - `validateMeasurementExpr()`
  - `_parseMeasurementsConfigObject()` and `_convertV2ToV3Measurements()`

  - `_compute_selected_measurements_from_layers()`
  - evolution usage of `fitness_weights.measurements` and `fitness_weights.measurement_aggs`

- `backend/digital_tissue/output_calc.py`
  - `_ExprEval` (authoritative measurement expression evaluator)
