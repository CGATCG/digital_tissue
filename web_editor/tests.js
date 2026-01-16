(() => {
  const ui = {
    testsStatus: document.getElementById("testsStatus"),
    moneyScore: document.getElementById("moneyScore"),
    resetMoneyBtn: document.getElementById("resetMoneyBtn"),
    backToEditorBtn: document.getElementById("backToEditorBtn"),

    stepModelPanel: document.getElementById("stepModelPanel"),
    stepExperimentPanel: document.getElementById("stepExperimentPanel"),
    stepParamsPanel: document.getElementById("stepParamsPanel"),
    stepInterventionsPanel: document.getElementById("stepInterventionsPanel"),
    stepRunPanel: document.getElementById("stepRunPanel"),

    modelSel: document.getElementById("modelSel"),
    modelMeta: document.getElementById("modelMeta"),
    experimentSel: document.getElementById("experimentSel"),
    experimentMeta: document.getElementById("experimentMeta"),

    ticksInput: document.getElementById("ticksInput"),
    repsInput: document.getElementById("repsInput"),

    ivLayerSel: document.getElementById("ivLayerSel"),
    ivDirSel: document.getElementById("ivDirSel"),
    ivDoseInput: document.getElementById("ivDoseInput"),
    ivAddBtn: document.getElementById("ivAddBtn"),
    ivList: document.getElementById("ivList"),
    ivClearBtn: document.getElementById("ivClearBtn"),

    paramsSpatialPanel: document.getElementById("paramsSpatialPanel"),
    paramsBulkPanel: document.getElementById("paramsBulkPanel"),
    paramsCharPanel: document.getElementById("paramsCharPanel"),
    paramsScreenPanel: document.getElementById("paramsScreenPanel"),
    paramsClaimPanel: document.getElementById("paramsClaimPanel"),

    geneSetSel: document.getElementById("geneSetSel"),
    omicsSetSel: document.getElementById("omicsSetSel"),
    screenDir: document.getElementById("screenDir"),
    screenDose: document.getElementById("screenDose"),

    runBtn: document.getElementById("runBtn"),
    runCost: document.getElementById("runCost"),
    clearBtn: document.getElementById("clearBtn"),

    outSummary: document.getElementById("outSummary"),
    outText: document.getElementById("outText"),
    downloadJsonBtn: document.getElementById("downloadJsonBtn"),
    downloadNoisyMatrixCsvBtn: document.getElementById("downloadNoisyMatrixCsvBtn"),
    downloadMetadataCsvBtn: document.getElementById("downloadMetadataCsvBtn"),
  };

  const state = {
    playerId: "",
    experiment: "",
    interventions: [],
    result: null,
    lastEstimate: null,
    modelsByKey: {},
    estimateTimer: null,
    estimateSeq: 0,
  };

  const _HARD_SEED = 1;

  const _EXPERIMENTS = {
    spatial_tx: {
      label: "Spatial omics",
      estimateKey: "spatial",
      runPath: "/api/tests/cancer/spatial_tx",
    },
    bulk_omics: {
      label: "Bulk omics",
      estimateKey: "bulk",
      runPath: "/api/tests/cancer/bulk_omics",
    },
    characterization: {
      label: "Characterization",
      estimateKey: "characterization",
      runPath: "/api/tests/cancer/characterization",
    },
    protein_screen: {
      label: "Drug screen",
      estimateKey: "protein_screen",
      runPath: "/api/tests/cancer/protein_screen",
    },
    claim_cure: {
      label: "Claim cure",
      estimateKey: "claim_cure",
      runPath: "/api/tests/cancer/claim_cure",
    },
  };

  function _isNonEmptyString(s) {
    return typeof s === "string" && s.trim().length > 0;
  }

  function _safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch (e) {
      return null;
    }
  }

  function _setStatus(msg) {
    if (!ui.testsStatus) return;
    ui.testsStatus.textContent = String(msg || "");
  }

  function _newPlayerId() {
    const raw = globalThis.crypto && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}_${Math.random()}`;
    return `p_${String(raw).replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 40)}`;
  }

  function _getPlayerId() {
    try {
      const prev = String(localStorage.getItem("dt_player_id") || "").trim();
      if (prev) return prev;
    } catch (e) {
    }
    const pid = _newPlayerId();
    try {
      localStorage.setItem("dt_player_id", pid);
    } catch (e) {
    }
    return pid;
  }

  function _formatUsd(cents) {
    const c = Number.isFinite(Number(cents)) ? Number(cents) : 0;
    return `$${(c / 100).toFixed(2)}`;
  }

  async function _postJson(path, obj) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obj),
    });
    const text = await resp.text();
    const data = _safeJsonParse(text) || { ok: false, error: text };
    if (!resp.ok || !data || data.ok !== true) {
      const msg = data && data.error ? String(data.error) : `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.data = data;
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  function _downloadText(filename, text, mime) {
    const blob = new Blob([text], { type: mime || "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2500);
  }

  function _countCsvRows(csvText) {
    if (!csvText || typeof csvText !== "string") return 0;
    const lines = csvText.split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lines.length <= 1) return 0;
    return lines.length - 1;
  }

  function _fillSelectOptions(sel, items, prev) {
    if (!sel) return;
    const p = prev != null ? String(prev) : String(sel.value || "");
    sel.innerHTML = "";
    for (const nm of items) {
      const opt = document.createElement("option");
      opt.value = nm;
      opt.textContent = nm;
      if (nm === p) opt.selected = true;
      sel.appendChild(opt);
    }
    if (!sel.value && sel.options.length > 0) sel.options[0].selected = true;
  }

  function _svgEscape(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function _updateMoneyUi(game) {
    if (!ui.moneyScore) return;
    const cents = game && typeof game.money_spent_cents === "number" ? game.money_spent_cents : 0;
    ui.moneyScore.textContent = `Money spent: ${_formatUsd(cents)}`;
  }

  async function _refreshGameState() {
    const pid = state.playerId;
    if (!pid) return;
    try {
      const resp = await fetch(`/api/game/state?player_id=${encodeURIComponent(pid)}`, { method: "GET" });
      const text = await resp.text();
      const out = _safeJsonParse(text);
      if (!resp.ok || !out || out.ok !== true) return;
      _updateMoneyUi(out.game);
    } catch (e) {
    }
  }

  function _interventionLabel(iv) {
    const layer = _isNonEmptyString(iv && iv.layer) ? iv.layer.trim() : "";
    const dir = String(iv && iv.direction ? iv.direction : "").toLowerCase() === "down" ? "down" : "up";
    const dose = Number.isFinite(Number(iv && iv.dose)) ? Number(iv.dose) : 0;
    const pct = (dir === "up" ? 10 : -10) * dose;
    const sign = pct >= 0 ? "+" : "";
    const drugType = dir === "up" ? "activator" : "inhibitor";
    return `${layer} ${drugType} @ ${dose} nM (${sign}${pct.toFixed(1)}% activity)`;
  }

  function _renderInterventionsList() {
    if (!ui.ivList) return;
    const arr = Array.isArray(state.interventions) ? state.interventions : [];
    if (!arr.length) {
      ui.ivList.textContent = "None";
      return;
    }
    let html = "";
    for (let i = 0; i < arr.length; i++) {
      const iv = arr[i];
      const label = _svgEscape(_interventionLabel(iv));
      html += `<div style="display:flex; gap:10px; align-items:center; margin: 6px 0;">`;
      html += `<div style="flex:1;">${label}</div>`;
      html += `<button class="btn btn--danger btn--tiny" data-iv-remove="${i}">Remove</button>`;
      html += `</div>`;
    }
    ui.ivList.innerHTML = html;
  }

  function _setPanelActive(el, active) {
    if (!el) return;
    el.classList.toggle("tabPanel--active", !!active);
  }

  function _modelKeyForLayers() {
    const exp = String(state.experiment || "");
    if (exp === "claim_cure") return "cancer";
    return String((ui.modelSel && ui.modelSel.value) || "").trim();
  }

  function _syncModelLockForExperiment() {
    if (!ui.modelSel) return;
    const exp = String(state.experiment || "");
    if (exp === "claim_cure") {
      ui.modelSel.value = "cancer";
      ui.modelSel.disabled = true;
    } else {
      ui.modelSel.disabled = false;
    }
  }

  function _updateParamsUi() {
    const exp = String(state.experiment || "");
    _setPanelActive(ui.paramsSpatialPanel, exp === "spatial_tx");
    _setPanelActive(ui.paramsBulkPanel, exp === "bulk_omics");
    _setPanelActive(ui.paramsCharPanel, exp === "characterization");
    _setPanelActive(ui.paramsScreenPanel, exp === "protein_screen");
    _setPanelActive(ui.paramsClaimPanel, exp === "claim_cure");
  }

  function _updateStepUi() {
    const modelKey = String((ui.modelSel && ui.modelSel.value) || "").trim();
    const exp = String(state.experiment || "");
    const hasExperiment = !!exp;

    if (ui.stepExperimentPanel) ui.stepExperimentPanel.style.display = modelKey ? "" : "none";
    if (ui.stepParamsPanel) ui.stepParamsPanel.style.display = modelKey && hasExperiment ? "" : "none";
    if (ui.stepInterventionsPanel) ui.stepInterventionsPanel.style.display = modelKey && hasExperiment ? "" : "none";
    if (ui.stepRunPanel) ui.stepRunPanel.style.display = modelKey && hasExperiment ? "" : "none";

    if (ui.experimentSel) ui.experimentSel.disabled = !modelKey;

    const expInfo = _EXPERIMENTS[exp];
    if (ui.experimentMeta) {
      ui.experimentMeta.textContent = expInfo ? expInfo.label : "";
    }

    const md = modelKey && state.modelsByKey ? state.modelsByKey[modelKey] : null;
    if (ui.modelMeta) {
      const domain = md && md.domain ? String(md.domain) : "";
      ui.modelMeta.textContent = domain ? `domain: ${domain}` : "";
    }
  }

  function _addInterventionFromUi() {
    const layer = String((ui.ivLayerSel && ui.ivLayerSel.value) || "").trim();
    if (!layer) return;
    const direction = String((ui.ivDirSel && ui.ivDirSel.value) || "up").trim().toLowerCase() === "down" ? "down" : "up";
    const dose = Math.max(0, Number(ui.ivDoseInput && ui.ivDoseInput.value ? ui.ivDoseInput.value : 0) || 0);
    state.interventions = Array.isArray(state.interventions) ? state.interventions : [];
    state.interventions.push({ layer, direction, dose });
    _renderInterventionsList();
  }

  function _clearInterventions() {
    state.interventions = [];
    _renderInterventionsList();
  }

  function _setExperiment(exp) {
    const e = _EXPERIMENTS[String(exp || "")] ? String(exp || "") : "";
    state.experiment = e;
    if (ui.experimentSel) ui.experimentSel.value = e;
    _syncModelLockForExperiment();
    _updateParamsUi();
    state.lastEstimate = null;
    _updateStepUi();
    _scheduleEstimate();
  }

  async function _updateModels() {
    if (!ui.modelSel) return;
    let out;
    try {
      const resp = await fetch("/api/tests/cancer/models", { method: "GET" });
      const text = await resp.text();
      out = _safeJsonParse(text);
      if (!resp.ok || !out || out.ok !== true) {
        throw new Error((out && out.error) || text || `HTTP ${resp.status}`);
      }
    } catch (e) {
      _setStatus(`Models unavailable: ${e && e.message ? e.message : String(e)}`);
      ui.modelSel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(unavailable)";
      ui.modelSel.appendChild(opt);
      return;
    }

    const models = Array.isArray(out.models) ? out.models : [];
    const byKey = {};
    const prev = String(ui.modelSel.value || "");
    ui.modelSel.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a model…";
    if (!prev) placeholder.selected = true;
    ui.modelSel.appendChild(placeholder);

    for (const m of models) {
      if (m && typeof m === "object") {
        const key = String(m.key || "").trim();
        if (!key) continue;
        byKey[key] = m;
        const label0 = String(m.label || "").trim();
        const label = label0 ? label0 : key;
        const domain = String(m.domain || "").trim();
        const text = domain ? `${label} (${domain})` : label;
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = text;
        if (key === prev) opt.selected = true;
        ui.modelSel.appendChild(opt);
        continue;
      }
      if (typeof m === "string" && m.trim()) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === prev) opt.selected = true;
        ui.modelSel.appendChild(opt);
      }
    }
    state.modelsByKey = byKey;
    _syncModelLockForExperiment();
    _updateStepUi();
  }

  async function _updateProteinLayers() {
    if (!ui.ivLayerSel) return;
    const model = _modelKeyForLayers();
    if (!model) {
      _fillSelectOptions(ui.ivLayerSel, []);
      return;
    }

    let out;
    try {
      const resp = await fetch(`/api/tests/cancer/protein_layers?model=${encodeURIComponent(model)}`, { method: "GET" });
      const text = await resp.text();
      out = _safeJsonParse(text);
      if (!resp.ok || !out || out.ok !== true) {
        throw new Error((out && out.error) || text || `HTTP ${resp.status}`);
      }
    } catch (e) {
      _setStatus(`Protein layers unavailable: ${e && e.message ? e.message : String(e)}`);
      _fillSelectOptions(ui.ivLayerSel, []);
      return;
    }

    const layers = Array.isArray(out.protein_layers) ? out.protein_layers : [];
    _fillSelectOptions(ui.ivLayerSel, layers);
  }

  async function _updateGeneSets() {
    const sel = ui.geneSetSel;
    if (!sel) return;
    let out;
    try {
      const resp = await fetch("/api/spatial_tx/gene_sets", { method: "GET" });
      const text = await resp.text();
      out = _safeJsonParse(text);
      if (!resp.ok || !out || out.ok !== true) {
        throw new Error((out && out.error) || text || `HTTP ${resp.status}`);
      }
    } catch (e) {
      sel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "default";
      sel.appendChild(opt);
      return;
    }

    const sets = Array.isArray(out.gene_sets) ? out.gene_sets : [];
    const prev = String(sel.value || "");
    sel.innerHTML = "";
    if (sets.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "default";
      sel.appendChild(opt);
      return;
    }

    for (const nm of sets) {
      if (typeof nm !== "string" || !nm.trim()) continue;
      const opt = document.createElement("option");
      opt.value = nm;
      opt.textContent = nm;
      if (nm === prev) opt.selected = true;
      sel.appendChild(opt);
    }
    if (!sel.value && sel.options.length > 0) sel.options[0].selected = true;
  }

  async function _updateOmicsSets() {
    const sel = ui.omicsSetSel;
    if (!sel) return;
    let out;
    try {
      const resp = await fetch("/api/bulk_omics/sets", { method: "GET" });
      const text = await resp.text();
      out = _safeJsonParse(text);
      if (!resp.ok || !out || out.ok !== true) {
        throw new Error((out && out.error) || text || `HTTP ${resp.status}`);
      }
    } catch (e) {
      sel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "default";
      sel.appendChild(opt);
      return;
    }

    const sets = Array.isArray(out.sets) ? out.sets : [];
    const prev = String(sel.value || "");
    sel.innerHTML = "";
    if (sets.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "default";
      sel.appendChild(opt);
      return;
    }

    for (const nm of sets) {
      if (typeof nm !== "string" || !nm.trim()) continue;
      const opt = document.createElement("option");
      opt.value = nm;
      opt.textContent = nm;
      if (nm === prev) opt.selected = true;
      sel.appendChild(opt);
    }
    if (!sel.value && sel.options.length > 0) sel.options[0].selected = true;
  }

  function _estimateExperimentKey() {
    const exp = String(state.experiment || "");
    const ent = _EXPERIMENTS[exp];
    return ent ? ent.estimateKey : "spatial";
  }

  function _selectedModelKeyForEstimate() {
    if (String(state.experiment || "") === "claim_cure") return "cancer";
    return String((ui.modelSel && ui.modelSel.value) || "").trim();
  }

  function _selectedModelKeyForRun() {
    if (String(state.experiment || "") === "claim_cure") return "";
    return String((ui.modelSel && ui.modelSel.value) || "").trim();
  }

  function _readTicks() {
    return Math.max(0, Number(ui.ticksInput && ui.ticksInput.value ? ui.ticksInput.value : 0) || 0);
  }

  function _readReps() {
    return Math.max(1, Number(ui.repsInput && ui.repsInput.value ? ui.repsInput.value : 1) || 1);
  }

  function _readSeed() {
    return _HARD_SEED;
  }

  function _updateRunCostUi() {
    if (!ui.runCost) return;
    const charge = state.lastEstimate && state.lastEstimate.charge ? state.lastEstimate.charge : null;
    const total = charge && Number.isFinite(Number(charge.total_cost_cents)) ? Number(charge.total_cost_cents) : null;
    if (total == null) {
      ui.runCost.textContent = "Estimated cost: —";
      return;
    }
    ui.runCost.textContent = `Estimated cost: ${_formatUsd(total)}`;
  }

  function _scheduleEstimate() {
    const model = _selectedModelKeyForEstimate();
    const exp = String(state.experiment || "");
    if (!model) {
      state.lastEstimate = null;
      _updateRunCostUi();
      return;
    }
    if (!exp) {
      state.lastEstimate = null;
      _updateRunCostUi();
      return;
    }
    if (state.estimateTimer) {
      try {
        clearTimeout(state.estimateTimer);
      } catch (e) {
      }
      state.estimateTimer = null;
    }
    if (ui.runCost) ui.runCost.textContent = "Estimated cost: estimating...";
    state.estimateTimer = setTimeout(() => {
      state.estimateTimer = null;
      _estimateCost();
    }, 250);
  }

  function _getOutputHost() {
    if (!ui.outText || !ui.outText.parentElement) return null;
    let host = document.getElementById("testsVizHost");
    if (!host) {
      host = document.createElement("div");
      host.id = "testsVizHost";
      ui.outText.parentElement.appendChild(host);
    }
    return host;
  }

  function _renderClaimCure(result) {
    const r = result;
    if (!r || r.experiment !== "tests_cancer_claim_cure_v1") return "";
    const h = r.healthy && typeof r.healthy === "object" ? r.healthy : null;
    const s = r.sick && typeof r.sick === "object" ? r.sick : null;
    const ch = h && h.curve && typeof h.curve === "object" ? h.curve : null;
    const cs = s && s.curve && typeof s.curve === "object" ? s.curve : null;
    const ticks = Number(r.ticks || 0);
    if (!ch || !cs || !ticks || ticks <= 1) return "";

    const timesH = Array.isArray(ch.times) ? ch.times : [];
    const survH = Array.isArray(ch.survival) ? ch.survival : [];
    const timesS = Array.isArray(cs.times) ? cs.times : [];
    const survS = Array.isArray(cs.survival) ? cs.survival : [];
    if (!timesH.length || !timesS.length || timesH.length !== survH.length || timesS.length !== survS.length) return "";

    const W = 760;
    const H = 170;
    const padL = 44;
    const padR = 10;
    const padT = 18;
    const padB = 28;
    const x0 = padL;
    const x1 = W - padR;
    const y0 = padT;
    const y1 = H - padB;
    const sx = (t) => x0 + (Math.max(0, Math.min(ticks, t)) / ticks) * (x1 - x0);
    const sy = (v) => y1 - Math.max(0, Math.min(1, v)) * (y1 - y0);

    const stepPath = (times, survival) => {
      let d = "";
      let prev = null;
      for (let i = 0; i < times.length; i++) {
        const tt = Number(times[i]);
        const vv = Number(survival[i]);
        if (!Number.isFinite(tt) || !Number.isFinite(vv)) continue;
        if (i === 0) {
          d += `M${sx(tt).toFixed(2)},${sy(vv).toFixed(2)} `;
          prev = vv;
          continue;
        }
        d += `L${sx(tt).toFixed(2)},${sy(prev).toFixed(2)} `;
        d += `L${sx(tt).toFixed(2)},${sy(vv).toFixed(2)} `;
        prev = vv;
      }
      return d.trim();
    };

    const dH = stepPath(timesH, survH);
    const dS = stepPath(timesS, survS);

    const win = !!r.win;
    const delta = Number(r.delta_median_ticks);
    const deltaTxt = Number.isFinite(delta) ? delta.toFixed(2) : "0.00";
    const title = win ? "WIN" : "Not cured";
    const color = win ? "rgba(50,215,75,.95)" : "rgba(255,69,58,.95)";

    let out = "";
    out += `<div style="margin: 8px 0 12px 0; padding: 10px 12px; border: 1px solid rgba(255,255,255,.10); border-radius: 12px; background: rgba(255,255,255,.02);">`;
    out += `<div class="meta" style="font-weight: 700; color: ${color}; margin-bottom: 4px;">${_svgEscape(title)}</div>`;
    out += `<div class="meta">delta median ticks (treated cancer - healthy): ${_svgEscape(deltaTxt)}</div>`;
    out += `</div>`;

    out += `<div style="margin: 10px 0 18px 0;">`;
    out += `<div class="meta" style="margin-bottom: 6px;">Survival</div>`;
    out += `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" style="border:1px solid rgba(255,255,255,.10); border-radius:12px; background: rgba(255,255,255,.02);">`;
    out += `<line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
    out += `<line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
    out += `<path d="${dH}" fill="none" stroke="rgba(50,215,75,.95)" stroke-width="2" />`;
    out += `<path d="${dS}" fill="none" stroke="rgba(255,69,58,.95)" stroke-width="2" />`;
    out += `<text x="${x1 - 170}" y="${y0 - 6}" fill="rgba(50,215,75,.95)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">healthy</text>`;
    out += `<text x="${x1 - 90}" y="${y0 - 6}" fill="rgba(255,69,58,.95)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">treated cancer</text>`;
    out += `</svg>`;
    out += `</div>`;

    return out;
  }

  function _renderProteinScreen(result) {
    const r = result;
    if (!r || r.experiment !== "tests_cancer_protein_screen_v1") return "";
    const items0 = Array.isArray(r.results) ? r.results : [];
    const items = [];
    for (const it of items0) {
      if (!it || typeof it !== "object") continue;
      if (!_isNonEmptyString(it.layer)) continue;
      items.push(it);
    }
    if (!items.length) return "";

    let out = "";
    const dir0 = _isNonEmptyString(r.direction) ? r.direction : "";
    const drugType = String(dir0).toLowerCase() === "down" ? "inhibitor" : "activator";
    const dose = Number.isFinite(Number(r.dose)) ? Number(r.dose) : 0;
    out += `<div style="margin: 8px 0 12px 0; padding: 10px 12px; border: 1px solid rgba(255,255,255,.10); border-radius: 12px; background: rgba(255,255,255,.02);">`;
    out += `<div class="meta" style="font-weight: 700; margin-bottom: 4px;">Drug screen</div>`;
    out += `<div class="meta">drug_type=${_svgEscape(drugType)} dose_nM=${_svgEscape(dose.toFixed(2))} reps=${_svgEscape(String(r.replicates || ""))} ticks=${_svgEscape(String(r.ticks || ""))}</div>`;
    const baseline = r.baseline && typeof r.baseline === "object" ? r.baseline : null;
    const baseSamp = baseline && Array.isArray(baseline.measurements_end_sample) ? baseline.measurements_end_sample : null;
    const baseN = baseSamp ? baseSamp.length : 0;
    out += `<div class="meta" style="margin-top: 6px;">baseline replicates: ${_svgEscape(String(baseN))}</div>`;
    out += `</div>`;

    const measNames = Array.isArray(r.measurements) ? r.measurements : [];
    const previewN = Math.min(8, items.length);
    out += `<div style="margin: 8px 0 8px 0;" class="meta">screened_layers=${_svgEscape(String(items.length))} measurements=${_svgEscape(String(measNames.length))}</div>`;
    out += `<div style="max-height: 560px; overflow:auto; border: 1px solid rgba(255,255,255,.10); border-radius: 12px;">`;
    for (let i = 0; i < previewN; i++) {
      const it = items[i];
      const layer = String(it.layer || "");
      const samp = Array.isArray(it.measurements_end_sample) ? it.measurements_end_sample : [];
      const first = samp.length ? samp[0] : null;
      const m0 = first && typeof first === "object" ? first.measurements_end : null;
      const m0txt = m0 && typeof m0 === "object" ? JSON.stringify(m0) : "";
      const bg = i % 2 === 0 ? "rgba(255,255,255,.02)" : "rgba(255,255,255,.00)";
      out += `<div style="padding: 8px 10px; background:${bg};">`;
      out += `<div class="meta" style="font-weight:600;">${_svgEscape(layer)}</div>`;
      out += `<div class="meta" style="opacity:.8;">replicates=${_svgEscape(String(samp.length))}</div>`;
      if (m0txt) out += `<div class="meta" style="opacity:.85; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">rep0=${_svgEscape(m0txt)}</div>`;
      out += `</div>`;
    }
    if (items.length > previewN) {
      out += `<div class="meta" style="padding: 8px 10px; opacity:.8;">(showing first ${_svgEscape(String(previewN))} layers)</div>`;
    }
    out += `</div>`;

    return out;
  }

  function _renderCharacterization(result) {
    const r = result;
    if (!r || r.experiment !== "tests_cancer_characterization_v1") return "";
    const series = r.series && typeof r.series === "object" ? r.series : null;
    const s = series && series.sample && typeof series.sample === "object" ? series.sample : null;
    const names = Array.isArray(r.measurements) ? r.measurements : (s ? Object.keys(s) : []);
    const ticks = Number(r.ticks || 0);
    if (!s || !ticks || ticks <= 1 || !names.length) return "";

    const repsWrap = r.series_replicates && typeof r.series_replicates === "object" ? r.series_replicates : null;
    const reps = repsWrap && Array.isArray(repsWrap.sample) ? repsWrap.sample : null;

    const W = 760;
    const H = 170;
    const padL = 44;
    const padR = 10;
    const padT = 18;
    const padB = 28;
    const x0 = padL;
    const x1 = W - padR;
    const y0 = padT;
    const y1 = H - padB;
    const sx = (i) => x0 + (i / (ticks - 1)) * (x1 - x0);

    const pathOf = (arr, vmin, vmax) => {
      let d = "";
      let penDown = false;
      for (let i = 0; i < ticks; i++) {
        const raw = arr[i];
        if (raw == null) {
          penDown = false;
          continue;
        }
        const v = Number(raw);
        if (!Number.isFinite(v)) {
          penDown = false;
          continue;
        }
        const yy = y1 - ((v - vmin) / (vmax - vmin)) * (y1 - y0);
        const cmd = penDown ? "L" : "M";
        d += `${cmd}${sx(i).toFixed(2)},${yy.toFixed(2)} `;
        penDown = true;
      }
      return d.trim();
    };

    let out = "";
    out += `<div style="margin: 8px 0 12px 0; padding: 10px 12px; border: 1px solid rgba(255,255,255,.10); border-radius: 12px; background: rgba(255,255,255,.02);">`;
    out += `<div class="meta" style="font-weight: 700; margin-bottom: 4px;">Characterization</div>`;
    out += `<div class="meta">replicates=${_svgEscape(String(r.replicates || ""))} ticks=${_svgEscape(String(r.ticks || ""))}</div>`;
    out += `</div>`;

    for (const nm of names) {
      const meanArr = Array.isArray(s[nm]) ? s[nm] : null;
      if (!meanArr || meanArr.length !== ticks) continue;

      const vals = [];
      for (let i = 0; i < ticks; i++) {
        const v = Number(meanArr[i]);
        if (Number.isFinite(v)) vals.push(v);
      }
      if (reps) {
        const k = Math.min(32, reps.length);
        for (let ri = 0; ri < k; ri++) {
          const rr = reps[ri];
          if (!rr || typeof rr !== "object") continue;
          const arr = Array.isArray(rr[nm]) ? rr[nm] : null;
          if (!arr || arr.length !== ticks) continue;
          for (let i = 0; i < ticks; i++) {
            const v = Number(arr[i]);
            if (Number.isFinite(v)) vals.push(v);
          }
        }
      }

      let vmin = 0;
      let vmax = 1;
      if (vals.length) {
        vmin = Math.min(...vals);
        vmax = Math.max(...vals);
        if (!Number.isFinite(vmin)) vmin = 0;
        if (!Number.isFinite(vmax)) vmax = 1;
      }
      if (vmax - vmin < 1e-9) vmax = vmin + 1;

      const dMean = pathOf(meanArr, vmin, vmax);
      const repPaths = [];
      if (reps) {
        const k = Math.min(32, reps.length);
        for (let ri = 0; ri < k; ri++) {
          const rr = reps[ri];
          if (!rr || typeof rr !== "object") continue;
          const arr = Array.isArray(rr[nm]) ? rr[nm] : null;
          if (!arr || arr.length !== ticks) continue;
          const d = pathOf(arr, vmin, vmax);
          if (d) repPaths.push(d);
        }
      }

      out += `<div style="margin: 10px 0 18px 0;">`;
      out += `<div class="meta" style="margin-bottom: 6px;">${_svgEscape(nm)}</div>`;
      out += `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" style="border:1px solid rgba(255,255,255,.10); border-radius:12px; background: rgba(255,255,255,.02);">`;
      out += `<line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
      out += `<line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
      for (const d of repPaths) {
        out += `<path d="${d}" fill="none" stroke="rgba(10,132,255,.20)" stroke-width="1" />`;
      }
      out += `<path d="${dMean}" fill="none" stroke="rgba(10,132,255,.95)" stroke-width="2" />`;
      out += `</svg>`;
      out += `</div>`;
    }

    return out;
  }

  function _updateOutput() {
    const r = state.result;
    if (!r) {
      if (ui.outSummary) ui.outSummary.textContent = "";
      if (ui.outText) ui.outText.value = "";
      if (ui.outText) ui.outText.style.display = "";
      if (ui.downloadJsonBtn) ui.downloadJsonBtn.disabled = true;
      if (ui.downloadNoisyMatrixCsvBtn) ui.downloadNoisyMatrixCsvBtn.disabled = true;
      if (ui.downloadMetadataCsvBtn) ui.downloadMetadataCsvBtn.disabled = true;
      const host = document.getElementById("testsVizHost");
      if (host) host.innerHTML = "";
      return;
    }

    const noisyCsv = typeof r.matrix_noisy_csv === "string" ? r.matrix_noisy_csv : (typeof r.matrix_csv === "string" ? r.matrix_csv : "");
    const metaCsv = typeof r.metadata_csv === "string" ? r.metadata_csv : "";

    const noisyRows = _countCsvRows(noisyCsv);
    const metaRows = _countCsvRows(metaCsv);

    const exp = _isNonEmptyString(r.experiment) ? r.experiment : "";
    const model = _isNonEmptyString(r.model) ? r.model : "";
    const ticks = Number.isFinite(Number(r.ticks)) ? Number(r.ticks) : null;
    const reps = Number.isFinite(Number(r.replicates)) ? Number(r.replicates) : null;

    if (ui.outSummary) {
      ui.outSummary.textContent = `experiment=${exp}${model ? ` model=${model}` : ""}${ticks != null ? ` ticks=${ticks}` : ""}${reps != null ? ` reps=${reps}` : ""} matrix_rows=${noisyRows} meta_rows=${metaRows}`;
    }

    if (ui.downloadJsonBtn) ui.downloadJsonBtn.disabled = false;
    if (ui.downloadNoisyMatrixCsvBtn) ui.downloadNoisyMatrixCsvBtn.disabled = !(noisyCsv && noisyRows > 0);
    if (ui.downloadMetadataCsvBtn) ui.downloadMetadataCsvBtn.disabled = !(metaCsv && metaRows > 0);

    const viz = _renderProteinScreen(r) || _renderClaimCure(r) || _renderCharacterization(r);
    const host = _getOutputHost();
    if (viz && host && ui.outText) {
      ui.outText.style.display = "none";
      host.innerHTML = viz;
    } else {
      if (ui.outText) ui.outText.style.display = "";
      if (host) host.innerHTML = "";
    }

    if (ui.outText) ui.outText.value = JSON.stringify(r, null, 2);
  }

  function _clearAll() {
    state.result = null;
    state.lastEstimate = null;
    _updateRunCostUi();
    _updateOutput();
  }

  async function _estimateCost() {
    const model = _selectedModelKeyForEstimate();
    if (!model) return;

    const ticks = _readTicks();
    const reps = _readReps();

    const payload = {
      player_id: state.playerId,
      model,
      experiment: _estimateExperimentKey(),
      ticks,
      replicates: reps,
      interventions: Array.isArray(state.interventions) ? state.interventions : [],
    };

    if (String(state.experiment || "") === "bulk_omics") {
      payload.omics_set = String((ui.omicsSetSel && ui.omicsSetSel.value) || "");
    }
    if (String(state.experiment || "") === "spatial_tx") {
      payload.gene_set = String((ui.geneSetSel && ui.geneSetSel.value) || "");
    }

    try {
      const seq = (state.estimateSeq = (state.estimateSeq || 0) + 1);
      const out = await _postJson("/api/tests/cancer/estimate_cost", payload);
      if (seq !== state.estimateSeq) return;
      state.lastEstimate = out;
      const charge = out.charge;
      const total = charge && Number.isFinite(Number(charge.total_cost_cents)) ? Number(charge.total_cost_cents) : 0;
      _setStatus("Ready");
      _updateRunCostUi();
    } catch (e) {
      state.lastEstimate = null;
      _updateRunCostUi();
      _setStatus(`Error: ${e && e.message ? e.message : String(e)}`);
    }
  }

  async function _runExperiment() {
    const model = _selectedModelKeyForRun();
    if (!model && String(state.experiment || "") !== "claim_cure") {
      _setStatus("Select a model");
      return;
    }

    const ticks = _readTicks();
    const reps = _readReps();
    const seed = _HARD_SEED;

    const base = {
      player_id: state.playerId,
      ticks,
      replicates: reps,
      seed,
      interventions: Array.isArray(state.interventions) ? state.interventions : [],
    };

    let path = "";
    let payload = {};

    if (String(state.experiment || "") === "bulk_omics") {
      path = _EXPERIMENTS.bulk_omics.runPath;
      payload = {
        ...base,
        model,
        omics_set: String((ui.omicsSetSel && ui.omicsSetSel.value) || ""),
      };
    } else if (String(state.experiment || "") === "characterization") {
      path = _EXPERIMENTS.characterization.runPath;
      payload = {
        ...base,
        model,
        include_replicates: true,
      };
    } else if (String(state.experiment || "") === "protein_screen") {
      path = _EXPERIMENTS.protein_screen.runPath;
      const workers = Math.min(35, Math.max(1, Number(reps) || 1));
      payload = {
        ...base,
        model,
        workers,
        worker_mode: "process",
        direction: String((ui.screenDir && ui.screenDir.value) || "up"),
        dose: Number(ui.screenDose && ui.screenDose.value ? ui.screenDose.value : 1) || 1,
      };
    } else if (String(state.experiment || "") === "claim_cure") {
      path = _EXPERIMENTS.claim_cure.runPath;
      const workers = Math.min(35, Math.max(1, Number(reps) || 1));
      payload = {
        ...base,
        ticks,
        replicates: reps,
        workers,
        worker_mode: "process",
      };
    } else {
      path = _EXPERIMENTS.spatial_tx.runPath;
      payload = {
        ...base,
        model,
        gene_set: String((ui.geneSetSel && ui.geneSetSel.value) || ""),
      };
    }

    try {
      _setStatus("Running...");
      const out = await _postJson(path, payload);
      state.result = out;
      _updateOutput();
      _updateMoneyUi(out.game);
      _setStatus("Done");
    } catch (e) {
      _setStatus(`Error: ${e && e.message ? e.message : String(e)}`);
    }
  }

  if (ui.backToEditorBtn) {
    ui.backToEditorBtn.addEventListener("click", () => {
      window.location.href = "index.html";
    });
  }

  if (ui.resetMoneyBtn) {
    ui.resetMoneyBtn.addEventListener("click", async () => {
      try {
        ui.resetMoneyBtn.disabled = true;
        await _postJson("/api/game/reset", { player_id: state.playerId });
        await _refreshGameState();
        _setStatus("Reset money spent");
      } catch (e) {
        _setStatus(`Error: ${e && e.message ? e.message : String(e)}`);
      } finally {
        ui.resetMoneyBtn.disabled = false;
      }
    });
  }

  if (ui.modelSel) {
    ui.modelSel.addEventListener("change", async () => {
      _clearInterventions();
      await _updateProteinLayers();
      state.lastEstimate = null;
      _updateRunCostUi();
      _updateStepUi();
      _scheduleEstimate();
    });
  }

  if (ui.experimentSel) {
    ui.experimentSel.addEventListener("change", async () => {
      _setExperiment(String(ui.experimentSel.value || ""));
      _clearInterventions();
      await _updateProteinLayers();
      _scheduleEstimate();
    });
  }

  if (ui.ivAddBtn) {
    ui.ivAddBtn.addEventListener("click", () => {
      _addInterventionFromUi();
      state.lastEstimate = null;
      _updateRunCostUi();
      _scheduleEstimate();
    });
  }

  if (ui.ivList) {
    ui.ivList.addEventListener("click", (e) => {
      const t = e && e.target ? e.target : null;
      const idx = t && t.getAttribute ? t.getAttribute("data-iv-remove") : null;
      if (idx == null) return;
      const i = Number(idx);
      if (!Number.isFinite(i) || i < 0) return;
      if (!Array.isArray(state.interventions)) state.interventions = [];
      if (i >= state.interventions.length) return;
      state.interventions.splice(i, 1);
      _renderInterventionsList();
      state.lastEstimate = null;
      _updateRunCostUi();
      _scheduleEstimate();
    });
  }

  if (ui.ivClearBtn) {
    ui.ivClearBtn.addEventListener("click", () => {
      _clearInterventions();
      state.lastEstimate = null;
      _updateRunCostUi();
      _scheduleEstimate();
    });
  }

  if (ui.runBtn) ui.runBtn.addEventListener("click", () => _runExperiment());
  if (ui.clearBtn) ui.clearBtn.addEventListener("click", () => _clearAll());

  if (ui.downloadJsonBtn) {
    ui.downloadJsonBtn.addEventListener("click", () => {
      if (!state.result) return;
      _downloadText(`tests_${String(state.result.experiment || "result")}.json`, JSON.stringify(state.result, null, 2), "application/json");
    });
  }

  if (ui.downloadNoisyMatrixCsvBtn) {
    ui.downloadNoisyMatrixCsvBtn.addEventListener("click", () => {
      if (!state.result) return;
      const csv = typeof state.result.matrix_noisy_csv === "string" ? state.result.matrix_noisy_csv : (typeof state.result.matrix_csv === "string" ? state.result.matrix_csv : "");
      if (!csv) return;
      _downloadText(`tests_${String(state.result.experiment || "matrix")}_matrix.csv`, csv, "text/csv");
    });
  }

  if (ui.downloadMetadataCsvBtn) {
    ui.downloadMetadataCsvBtn.addEventListener("click", () => {
      if (!state.result) return;
      const csv = typeof state.result.metadata_csv === "string" ? state.result.metadata_csv : "";
      if (!csv) return;
      _downloadText(`tests_${String(state.result.experiment || "metadata")}_metadata.csv`, csv, "text/csv");
    });
  }

  if (ui.ticksInput) ui.ticksInput.addEventListener("input", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });
  if (ui.repsInput) ui.repsInput.addEventListener("input", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });
  if (ui.geneSetSel) ui.geneSetSel.addEventListener("change", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });
  if (ui.omicsSetSel) ui.omicsSetSel.addEventListener("change", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });
  if (ui.screenDir) ui.screenDir.addEventListener("change", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });
  if (ui.screenDose) ui.screenDose.addEventListener("input", () => { state.lastEstimate = null; _updateRunCostUi(); _scheduleEstimate(); });

  async function _init() {
    state.playerId = _getPlayerId();
    _renderInterventionsList();
    _setExperiment(String((ui.experimentSel && ui.experimentSel.value) || ""));
    await _updateModels();
    await _updateProteinLayers();
    await _updateGeneSets();
    await _updateOmicsSets();
    await _refreshGameState();
    _updateParamsUi();
    _updateStepUi();
    _setStatus("Ready");
  }

  _init();
})();
