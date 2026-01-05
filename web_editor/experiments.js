(() => {
  const ui = {
    expStatus: document.getElementById("expStatus"),
    moneyScore: document.getElementById("moneyScore"),
    resetMoneyBtn: document.getElementById("resetMoneyBtn"),
    backToEditorBtn: document.getElementById("backToEditorBtn"),

    healthyFile: document.getElementById("healthyFile"),
    sickFile: document.getElementById("sickFile"),
    healthyName: document.getElementById("healthyName"),
    sickName: document.getElementById("sickName"),
    ticksInput: document.getElementById("ticksInput"),
    repsInput: document.getElementById("repsInput"),
    seedInput: document.getElementById("seedInput"),

    tabSpatialBtn: document.getElementById("tabSpatialBtn"),
    tabBulkBtn: document.getElementById("tabBulkBtn"),
    tabInVivoBtn: document.getElementById("tabInVivoBtn"),
    tabSpatialPanel: document.getElementById("tabSpatialPanel"),
    tabBulkPanel: document.getElementById("tabBulkPanel"),
    tabInVivoPanel: document.getElementById("tabInVivoPanel"),

    geneSetSel: document.getElementById("geneSetSel"),
    omicsSetSel: document.getElementById("omicsSetSel"),

    runBtn: document.getElementById("runBtn"),
    runCost: document.getElementById("runCost"),
    clearBtn: document.getElementById("clearBtn"),

    ivLayerSel: document.getElementById("ivLayerSel"),
    ivDirSel: document.getElementById("ivDirSel"),
    ivDoseInput: document.getElementById("ivDoseInput"),
    ivAddBtn: document.getElementById("ivAddBtn"),
    ivList: document.getElementById("ivList"),

    outSummary: document.getElementById("outSummary"),
    outText: document.getElementById("outText"),
    downloadJsonBtn: document.getElementById("downloadJsonBtn"),
    downloadTruthMatrixCsvBtn: document.getElementById("downloadTruthMatrixCsvBtn"),
    downloadNoisyMatrixCsvBtn: document.getElementById("downloadNoisyMatrixCsvBtn"),
    downloadMetadataCsvBtn: document.getElementById("downloadMetadataCsvBtn"),
  };

  const state = {
    healthy: null,
    sick: null,
    result: null,
    tab: "spatial",
    playerId: "",
    interventions: [],
  };

  function _isNonEmptyString(s) {
    return typeof s === "string" && s.trim().length > 0;
  }

  function _layerNamesFromPayload(payload) {
    const out = [];
    if (!payload || typeof payload !== "object") return out;
    const data = payload.data;
    if (!data || typeof data !== "object") return out;
    for (const k of Object.keys(data)) {
      const ent = data[k];
      if (!_isNonEmptyString(k) || !ent || typeof ent !== "object") continue;
      if (ent.dtype !== "float32") continue;
      // b64 may be huge; just ensure it exists
      if (typeof ent.b64 !== "string") continue;
      out.push(k);
    }
    out.sort((a, b) => a.localeCompare(b));
    return out;
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

  function _updateInterventionLayerOptions() {
    const layers = _layerNamesFromPayload(state.sick);
    _fillSelectOptions(ui.ivLayerSel, layers);
  }

  function _interventionLabel(iv) {
    const layer = _isNonEmptyString(iv && iv.layer) ? iv.layer.trim() : "";
    const dir = String(iv && iv.direction ? iv.direction : "").toLowerCase() === "down" ? "down" : "up";
    const dose = Number.isFinite(Number(iv && iv.dose)) ? Number(iv.dose) : 0;
    const pct = (dir === "up" ? 10 : -10) * dose;
    const sign = pct >= 0 ? "+" : "";
    return `${layer} ${dir} @ dose ${dose} (${sign}${pct.toFixed(1)}%)`;
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

  function _setStatus(msg) {
    ui.expStatus.textContent = String(msg || "");
  }

  function _newPlayerId() {
    const raw = (globalThis.crypto && crypto.randomUUID) ? crypto.randomUUID() : `${Date.now()}_${Math.random()}`;
    return `p_${String(raw).replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 40)}`;
  }

  function _getPlayerId() {
    try {
      const prev = String(localStorage.getItem("dt_player_id") || "").trim();
      if (prev) return prev;
    } catch (e) {
      // ignore
    }
    const pid = _newPlayerId();
    try {
      localStorage.setItem("dt_player_id", pid);
    } catch (e) {
      // ignore
    }
    return pid;
  }

  function _formatUsd(cents) {
    const c = Number.isFinite(Number(cents)) ? Number(cents) : 0;
    const usd = c / 100;
    return `$${usd.toFixed(2)}`;
  }

  function _estimateRunCostCents() {
    const reps = Math.max(1, Number(ui.repsInput && ui.repsInput.value ? ui.repsInput.value : 1) || 1);

    // Need both conditions loaded for all current experiments.
    const haveHealthy = !!state.healthy;
    const haveSick = !!state.sick;
    if (!haveHealthy || !haveSick) return null;

    // samples = replicates per condition
    const samples = 2 * reps;

    if (state.tab === "invivo") {
      return 500000 * samples;
    }

    if (state.tab === "bulk") {
      const setName = String((ui.omicsSetSel && ui.omicsSetSel.value) || "").trim().replace(/\\/g, "/");
      let unit = 20000; // bulk_rnaseq default
      if (setName.startsWith("protein/")) unit = 80000;
      else if (setName.startsWith("metabolite/") || setName.startsWith("metabolomics/")) unit = 50000;
      else if (setName.startsWith("rna/")) unit = 20000;
      return unit * samples;
    }

    // spatial
    return 250000 * samples;
  }

  function _updateRunCostUi() {
    if (!ui.runCost) return;
    const cents = _estimateRunCostCents();
    if (cents == null) {
      ui.runCost.textContent = "Cost: —";
      return;
    }
    ui.runCost.textContent = `Cost: ${_formatUsd(cents)}`;
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
      if (!resp.ok || !out || out.ok !== true) {
        return;
      }
      _updateMoneyUi(out.game);
    } catch (e) {
      // ignore
    }
  }

  function _pick(obj, key, fallback) {
    const v = obj && Object.prototype.hasOwnProperty.call(obj, key) ? obj[key] : undefined;
    return v == null ? fallback : v;
  }

  function _safeJsonParse(text) {
    try {
      const obj = JSON.parse(text);
      return obj;
    } catch (e) {
      return null;
    }
  }

  async function _readFileJson(file) {
    const txt = await file.text();
    return _safeJsonParse(txt);
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
      _setStatus(`Gene sets unavailable: ${e && e.message ? e.message : String(e)}`);
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
      _setStatus(`Bulk omics sets unavailable: ${e && e.message ? e.message : String(e)}`);
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

  function _setTab(tab) {
    const t = tab === "bulk" ? "bulk" : tab === "invivo" ? "invivo" : "spatial";
    state.tab = t;

    if (ui.tabSpatialBtn) ui.tabSpatialBtn.classList.toggle("tabBtn--active", t === "spatial");
    if (ui.tabBulkBtn) ui.tabBulkBtn.classList.toggle("tabBtn--active", t === "bulk");
    if (ui.tabInVivoBtn) ui.tabInVivoBtn.classList.toggle("tabBtn--active", t === "invivo");
    if (ui.tabSpatialPanel) ui.tabSpatialPanel.classList.toggle("tabPanel--active", t === "spatial");
    if (ui.tabBulkPanel) ui.tabBulkPanel.classList.toggle("tabPanel--active", t === "bulk");
    if (ui.tabInVivoPanel) ui.tabInVivoPanel.classList.toggle("tabPanel--active", t === "invivo");

    _updateRunCostUi();
  }

  function _svgEscape(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function _renderInVivoPlots(result) {
    const r = result;
    const series = r && r.series;
    if (!series || typeof series !== "object") return "";
    const h = series.healthy;
    const s = series.sick;
    if (!h || !s || typeof h !== "object" || typeof s !== "object") return "";
    const names = Array.isArray(r.measurements) ? r.measurements : Object.keys(h);
    const ticks = Number(r.ticks || 0);
    if (!ticks || ticks <= 1) return "";

    const cure = r && r.cure && typeof r.cure === "object" ? r.cure : null;
    let cureHtml = "";
    if (cure) {
      const scorePct = Number(cure.score_pct);
      const win = !!cure.win;
      const dist = Number(cure.distance);
      const pctTxt = Number.isFinite(scorePct) ? scorePct.toFixed(1) : "0.0";
      const distTxt = Number.isFinite(dist) ? dist.toFixed(4) : "0.0000";
      const title = win ? "WIN" : "Cure progress";
      const color = win ? "rgba(50,215,75,.95)" : "rgba(245,246,247,.85)";
      cureHtml += `<div style="margin: 8px 0 12px 0; padding: 10px 12px; border: 1px solid rgba(255,255,255,.10); border-radius: 12px; background: rgba(255,255,255,.02);">`;
      cureHtml += `<div class="meta" style="font-weight: 600; color: ${color}; margin-bottom: 4px;">${_svgEscape(title)}</div>`;
      cureHtml += `<div class="meta">Score: ${_svgEscape(pctTxt)}% &nbsp;&nbsp; Distance: ${_svgEscape(distTxt)}</div>`;
      cureHtml += `</div>`;
    }

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

    let out = cureHtml;
    for (const nm of names) {
      const hv = Array.isArray(h[nm]) ? h[nm] : null;
      const sv = Array.isArray(s[nm]) ? s[nm] : null;
      if (!hv || !sv || hv.length !== ticks || sv.length !== ticks) continue;
      const vals = [];
      for (let i = 0; i < ticks; i++) {
        const a = Number(hv[i]);
        const b = Number(sv[i]);
        if (Number.isFinite(a)) vals.push(a);
        if (Number.isFinite(b)) vals.push(b);
      }
      let vmin = 0;
      let vmax = 1;
      if (vals.length) {
        vmin = Math.min(...vals);
        vmax = Math.max(...vals);
        if (!Number.isFinite(vmin)) vmin = 0;
        if (!Number.isFinite(vmax)) vmax = 1;
      }
      if (vmax - vmin < 1e-9) {
        vmax = vmin + 1;
      }

      const sx = (i) => x0 + (i / (ticks - 1)) * (x1 - x0);
      const sy = (v) => y1 - ((v - vmin) / (vmax - vmin)) * (y1 - y0);

      const pathOf = (arr) => {
        let d = "";
        for (let i = 0; i < ticks; i++) {
          const v = Number(arr[i]);
          const vv = Number.isFinite(v) ? v : 0;
          const cmd = i === 0 ? "M" : "L";
          d += `${cmd}${sx(i).toFixed(2)},${sy(vv).toFixed(2)} `;
        }
        return d.trim();
      };

      const dH = pathOf(hv);
      const dS = pathOf(sv);
      const title = _svgEscape(nm);
      const labelMin = _svgEscape(vmin.toFixed(3));
      const labelMax = _svgEscape(vmax.toFixed(3));

      out += `\n<div style="margin: 10px 0 18px 0;">`;
      out += `\n  <div class="meta" style="margin-bottom: 6px;">${title}</div>`;
      out += `\n  <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" style="border:1px solid rgba(255,255,255,.10); border-radius:12px; background: rgba(255,255,255,.02);">`;
      out += `\n    <line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
      out += `\n    <line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" stroke="rgba(255,255,255,.18)" />`;
      out += `\n    <text x="${x0}" y="${y0 - 6}" fill="rgba(245,246,247,.62)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">${labelMax}</text>`;
      out += `\n    <text x="${x0}" y="${y1 + 16}" fill="rgba(245,246,247,.62)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">${labelMin}</text>`;
      out += `\n    <path d="${dH}" fill="none" stroke="rgba(50,215,75,.95)" stroke-width="2" />`;
      out += `\n    <path d="${dS}" fill="none" stroke="rgba(255,69,58,.95)" stroke-width="2" />`;
      out += `\n    <text x="${x1 - 170}" y="${y0 - 6}" fill="rgba(50,215,75,.95)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">healthy</text>`;
      out += `\n    <text x="${x1 - 90}" y="${y0 - 6}" fill="rgba(255,69,58,.95)" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace">sick</text>`;
      out += `\n  </svg>`;
      out += `\n</div>`;
    }
    return out.trim();
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
      throw new Error(msg);
    }
    return data;
  }

  function _updateOutput() {
    const r = state.result;
    if (!r) {
      ui.outSummary.textContent = "";
      ui.outText.value = "";
      ui.downloadJsonBtn.disabled = true;
      ui.downloadTruthMatrixCsvBtn.disabled = true;
      ui.downloadNoisyMatrixCsvBtn.disabled = true;
      ui.downloadMetadataCsvBtn.disabled = true;
      return;
    }

    const runs = Array.isArray(r.runs) ? r.runs : [];
    const truthRows = _countCsvRows(r.matrix_truth_csv);
    const noisyRows = _countCsvRows(r.matrix_noisy_csv || r.matrix_csv);
    const metaRows = _countCsvRows(r.metadata_csv);
    ui.outSummary.textContent = `experiment=${r.experiment || ""} runs=${runs.length} cells=${metaRows} genes=${(r.genes || []).length} truth_rows=${truthRows} noisy_rows=${noisyRows}`;
    ui.outText.value = JSON.stringify(r, null, 2);
    ui.downloadJsonBtn.disabled = false;
    ui.downloadTruthMatrixCsvBtn.disabled = !(typeof r.matrix_truth_csv === "string" && r.matrix_truth_csv.length > 0 && truthRows > 0);
    ui.downloadNoisyMatrixCsvBtn.disabled = !(
      typeof (r.matrix_noisy_csv || r.matrix_csv) === "string" &&
      (r.matrix_noisy_csv || r.matrix_csv).length > 0 &&
      noisyRows > 0
    );
    ui.downloadMetadataCsvBtn.disabled = !(typeof r.metadata_csv === "string" && r.metadata_csv.length > 0 && metaRows > 0);
  }

  function _clearAll() {
    state.result = null;
    ui.outText.value = "";
    ui.outSummary.textContent = "";
    ui.outText.style.display = "";
    const host = document.getElementById("invivoPlotsHost");
    if (host) host.innerHTML = "";
    _setStatus("");
    ui.downloadJsonBtn.disabled = true;
    ui.downloadTruthMatrixCsvBtn.disabled = true;
    ui.downloadNoisyMatrixCsvBtn.disabled = true;
    ui.downloadMetadataCsvBtn.disabled = true;
  }

  ui.backToEditorBtn.addEventListener("click", () => {
    window.location.href = "index.html";
  });

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

  if (ui.tabSpatialBtn) {
    ui.tabSpatialBtn.addEventListener("click", () => _setTab("spatial"));
  }
  if (ui.tabBulkBtn) {
    ui.tabBulkBtn.addEventListener("click", () => _setTab("bulk"));
  }
  if (ui.tabInVivoBtn) {
    ui.tabInVivoBtn.addEventListener("click", () => _setTab("invivo"));
  }

  ui.healthyFile.addEventListener("change", async () => {
    const f = ui.healthyFile.files && ui.healthyFile.files[0];
    if (!f) return;
    _setStatus("Loading healthy...");
    const p = await _readFileJson(f);
    if (!p) {
      _setStatus("Failed to parse healthy JSON");
      return;
    }
    state.healthy = p;
    ui.healthyName.textContent = f.name;
    _setStatus("Loaded healthy");
    _updateRunCostUi();
  });

  ui.sickFile.addEventListener("change", async () => {
    const f = ui.sickFile.files && ui.sickFile.files[0];
    if (!f) return;
    _setStatus("Loading sick...");
    const p = await _readFileJson(f);
    if (!p) {
      _setStatus("Failed to parse sick JSON");
      return;
    }
    state.sick = p;
    ui.sickName.textContent = f.name;
    _setStatus("Loaded sick");
    _clearInterventions();
    _updateInterventionLayerOptions();
    _updateRunCostUi();
  });

  if (ui.ivAddBtn) {
    ui.ivAddBtn.addEventListener("click", () => {
      _addInterventionFromUi();
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
    });
  }

  if (ui.repsInput) {
    ui.repsInput.addEventListener("input", () => _updateRunCostUi());
  }

  if (ui.omicsSetSel) {
    ui.omicsSetSel.addEventListener("change", () => _updateRunCostUi());
  }

  ui.clearBtn.addEventListener("click", () => {
    _clearAll();
    _clearInterventions();
  });

  ui.downloadJsonBtn.addEventListener("click", () => {
    if (!state.result) return;
    _downloadText("spatial_tx.json", JSON.stringify(state.result, null, 2), "application/json");
  });

  ui.downloadTruthMatrixCsvBtn.addEventListener("click", () => {
    if (!state.result) return;
    const csv = state.result.matrix_truth_csv;
    if (typeof csv !== "string") return;
    _downloadText("spatial_tx_matrix_truth.csv", csv, "text/csv");
  });

  ui.downloadNoisyMatrixCsvBtn.addEventListener("click", () => {
    if (!state.result) return;
    const csv = state.result.matrix_noisy_csv || state.result.matrix_csv;
    if (typeof csv !== "string") return;
    _downloadText("spatial_tx_matrix_noisy.csv", csv, "text/csv");
  });

  ui.downloadMetadataCsvBtn.addEventListener("click", () => {
    if (!state.result) return;
    const csv = state.result.metadata_csv;
    if (typeof csv !== "string") return;
    _downloadText("spatial_tx_metadata.csv", csv, "text/csv");
  });

  ui.runBtn.addEventListener("click", async () => {
    try {
      if (!state.healthy || !state.sick) {
        _setStatus("Upload both healthy and sick first");
        return;
      }

      const baseReq = {
        player_id: state.playerId,
        ticks: Number(ui.ticksInput.value || 0),
        replicates: Number(ui.repsInput.value || 1),
        seed: Number(ui.seedInput.value || 1),
        healthy: state.healthy,
        sick: state.sick,
        interventions: Array.isArray(state.interventions) ? state.interventions : [],
      };

      const isBulk = state.tab === "bulk";
      const isInVivo = state.tab === "invivo";
      const path = isInVivo
        ? "/api/experiments/in_vivo_trial"
        : (isBulk ? "/api/experiments/bulk_omics" : "/api/experiments/spatial_tx");
      const req = {
        ...baseReq,
        ...(isInVivo
          ? {}
          : isBulk
          ? { omics_set: String((ui.omicsSetSel && ui.omicsSetSel.value) || "").trim() }
          : { gene_set: String((ui.geneSetSel && ui.geneSetSel.value) || "").trim() }),
      };

      ui.runBtn.disabled = true;
      _setStatus("Running...");
      const out = await _postJson(path, req);
      state.result = out;
      _updateOutput();
      if (out && out.experiment === "in_vivo_trial_v1") {
        const html = _renderInVivoPlots(out);
        if (html) {
          ui.outText.style.display = "none";
          let host = document.getElementById("invivoPlotsHost");
          if (!host) {
            host = document.createElement("div");
            host.id = "invivoPlotsHost";
            ui.outText.parentElement.appendChild(host);
          }
          host.innerHTML = html;
        } else {
          ui.outText.style.display = "";
          const host = document.getElementById("invivoPlotsHost");
          if (host) host.innerHTML = "";
        }
      } else {
        ui.outText.style.display = "";
        const host = document.getElementById("invivoPlotsHost");
        if (host) host.innerHTML = "";
      }
      if (out && out.game) {
        _updateMoneyUi(out.game);
      } else {
        await _refreshGameState();
      }
      _setStatus("Done");
    } catch (e) {
      _setStatus(`Error: ${e && e.message ? e.message : String(e)}`);
    } finally {
      ui.runBtn.disabled = false;
    }
  });

  // init
  _clearAll();
  _setTab("spatial");
  state.playerId = _getPlayerId();
  _updateMoneyUi({ money_spent_cents: 0 });
  _refreshGameState();
  _updateRunCostUi();
  _renderInterventionsList();
  _updateGeneSets();
  _updateOmicsSets();
})();
