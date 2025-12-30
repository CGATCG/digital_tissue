const $ = (id) => document.getElementById(id);

const ui = {
  currentFileInfo: $("currentFileInfo"),
  fileInput: $("fileInput"),
  newBtn: $("newBtn"),
  saveBtn: $("saveBtn"),
  demoBtn: $("demoBtn"),
  HInput: $("HInput"),
  WInput: $("WInput"),
  zoomInput: $("zoomInput"),
  autoFitZoom: $("autoFitZoom"),
  gridLinesInput: $("gridLinesInput"),
  layerSelect: $("layerSelect"),
  layersTable: $("layersTable"),
  layerSearchInput: $("layerSearchInput"),
  groupByPrefix: $("groupByPrefix"),
  collapseAllBtn: $("collapseAllBtn"),
  expandAllBtn: $("expandAllBtn"),
  activeLayerTitle: $("activeLayerTitle"),
  activeLayerStats: $("activeLayerStats"),
  activeLayerColor: $("activeLayerColor"),
  newLayerName: $("newLayerName"),
  newLayerKind: $("newLayerKind"),
  newLayerInit: $("newLayerInit"),
  newLayerValue: $("newLayerValue"),
  newLayerSeed: $("newLayerSeed"),
  addLayerBtn: $("addLayerBtn"),
  renameToInput: $("renameToInput"),
  renameBtn: $("renameBtn"),
  removeBtn: $("removeBtn"),

  derivedSourcePrefix: $("derivedSourcePrefix"),
  derivedTargetTemplate: $("derivedTargetTemplate"),
  derivedMetaFrom: $("derivedMetaFrom"),
  derivedPrototypeLayer: $("derivedPrototypeLayer"),
  derivedDataInit: $("derivedDataInit"),
  derivedSkipExisting: $("derivedSkipExisting"),
  derivedPreview: $("derivedPreview"),
  derivedApplyBtn: $("derivedApplyBtn"),
  derivedPresetGeneFromMolecule: $("derivedPresetGeneFromMolecule"),

  bulkDeleteBtn: $("bulkDeleteBtn"),
  bulkDeleteInfo: $("bulkDeleteInfo"),
  bulkAddPreview: $("bulkAddPreview"),

  bulkPrefix: $("bulkPrefix"),
  bulkStart: $("bulkStart"),
  bulkCount: $("bulkCount"),
  bulkCollision: $("bulkCollision"),
  bulkKind: $("bulkKind"),
  bulkInit: $("bulkInit"),
  bulkValue: $("bulkValue"),
  bulkSeed: $("bulkSeed"),
  maskLayer: $("maskLayer"),
  maskOp: $("maskOp"),
  maskValue: $("maskValue"),
  maskInvert: $("maskInvert"),
  bulkAddBtn: $("bulkAddBtn"),

  opTargetLayer: $("opTargetLayer"),
  opUseSelected: $("opUseSelected"),
  opType: $("opType"),
  opValue: $("opValue"),
  opMin: $("opMin"),
  opMax: $("opMax"),
  opSeed: $("opSeed"),
  opMaskLayer: $("opMaskLayer"),
  opMaskOp: $("opMaskOp"),
  opMaskValue: $("opMaskValue"),
  opMaskInvert: $("opMaskInvert"),
  opMaskPreview: $("opMaskPreview"),
  opApplyBtn: $("opApplyBtn"),

  opTargetFilter: $("opTargetFilter"),
  opTargetsSelectAll: $("opTargetsSelectAll"),
  opTargetsClear: $("opTargetsClear"),
  opTargetsList: $("opTargetsList"),

  fnInsertLayer: $("fnInsertLayer"),
  fnInsertLayerBtn: $("fnInsertLayerBtn"),
  fnInsertFn: $("fnInsertFn"),
  fnInsertFnBtn: $("fnInsertFnBtn"),
  fnAddRowBtn: $("fnAddRowBtn"),
  fnResetBtn: $("fnResetBtn"),
  fnSpecsTable: $("fnSpecsTable"),

  opsInsertLayer: $("opsInsertLayer"),
  opsInsertLayerBtn: $("opsInsertLayerBtn"),
  opsInsertFn: $("opsInsertFn"),
  opsInsertFnBtn: $("opsInsertFnBtn"),
  opsAddRowBtn: $("opsAddRowBtn"),
  opsAddVarBtn: $("opsAddVarBtn"),
  opsAddForEachBtn: $("opsAddForEachBtn"),
  opsAddTransportBtn: $("opsAddTransportBtn"),
  opsAddDiffusionBtn: $("opsAddDiffusionBtn"),
  opsAddDivisionBtn: $("opsAddDivisionBtn"),
  opsAddPathwayBtn: $("opsAddPathwayBtn"),
  opsResetBtn: $("opsResetBtn"),
  opsTestBtn: $("opsTestBtn"),
  opsSpecsTable: $("opsSpecsTable"),

  opsDupGroupFrom: $("opsDupGroupFrom"),
  opsDupGroupTo: $("opsDupGroupTo"),
  opsDupGroupBtn: $("opsDupGroupBtn"),

  opsHelpBtn: $("opsHelpBtn"),
  helpModal: $("helpModal"),
  helpModalOverlay: $("helpModalOverlay"),
  helpModalClose: $("helpModalClose"),
  helpModalTitle: $("helpModalTitle"),
  helpModalBody: $("helpModalBody"),

  pathwayModal: $("pathwayModal"),
  pathwayModalOverlay: $("pathwayModalOverlay"),
  pathwayModalClose: $("pathwayModalClose"),
  pathwayModalCreate: $("pathwayModalCreate"),
  pathwayModalCancel: $("pathwayModalCancel"),
  pathwayName: $("pathwayName"),
  pathwayNumEnzymes: $("pathwayNumEnzymes"),
  pathwayInputsDropdown: $("pathwayInputsDropdown"),
  pathwayInputsSelected: $("pathwayInputsSelected"),
  pathwayOutputsDropdown: $("pathwayOutputsDropdown"),
  pathwayOutputsSelected: $("pathwayOutputsSelected"),
  pathwayCellLayer: $("pathwayCellLayer"),
  pathwayCellValue: $("pathwayCellValue"),
  pathwayEfficiency: $("pathwayEfficiency"),

  rtIntervalMs: $("rtIntervalMs"),
  rtStartStopBtn: $("rtStartStopBtn"),
  rtStepBtn: $("rtStepBtn"),
  rtResetBtn: $("rtResetBtn"),
  rtDownloadBtn: $("rtDownloadBtn"),
  rtStatus: $("rtStatus"),
  rtEventsList: $("rtEventsList"),
  rtScalarsWindow: $("rtScalarsWindow"),
  rtScalarsLog: $("rtScalarsLog"),
  rtScalarsList: $("rtScalarsList"),
  rtMeasWindow: $("rtMeasWindow"),
  rtMeasNormalize: $("rtMeasNormalize"),
  rtMeasLog: $("rtMeasLog"),
  rtMeasCanvas: $("rtMeasCanvas"),
  rtMeasList: $("rtMeasList"),
  rtSurvFocus: $("rtSurvFocus"),
  rtSurvTopK: $("rtSurvTopK"),
  rtSurvLog1p: $("rtSurvLog1p"),
  rtSurvList: $("rtSurvList"),
  rtAddLayerSelect: $("rtAddLayerSelect"),
  rtAddLayerBtn: $("rtAddLayerBtn"),
  rtWatchList: $("rtWatchList"),
  rtOverlayCanvas: $("rtOverlayCanvas"),
  rtHistLayer: $("rtHistLayer"),
  rtHistBins: $("rtHistBins"),
  rtHistLogY: $("rtHistLogY"),
  rtHistMaskLayer: $("rtHistMaskLayer"),
  rtHistMaskOp: $("rtHistMaskOp"),
  rtHistMaskValue: $("rtHistMaskValue"),
  rtHistCanvas: $("rtHistCanvas"),
  rtVizCols: $("rtVizCols"),
  rtHeatMaskEnabled: $("rtHeatMaskEnabled"),
  rtVizGrid: $("rtVizGrid"),

  evoBase: $("evoBase"),
  evoAlgo: $("evoAlgo"),
  evoAffineParams: $("evoAffineParams"),
  evoCemParams: $("evoCemParams"),
  evoVariants: $("evoVariants"),
  evoTicks: $("evoTicks"),
  evoGenerations: $("evoGenerations"),
  evoElites: $("evoElites"),
  evoReplicates: $("evoReplicates"),
  evoWorkers: $("evoWorkers"),
  evoSeed: $("evoSeed"),
  evoMutationRate: $("evoMutationRate"),
  evoSigmaScale: $("evoSigmaScale"),
  evoSigmaBias: $("evoSigmaBias"),
  evoCemSigma: $("evoCemSigma"),
  evoCemAlpha: $("evoCemAlpha"),
  evoCemSigmaFloor: $("evoCemSigmaFloor"),
  evoCemMask: $("evoCemMask"),
  evoHuge: $("evoHuge"),
  evoTargetFilter: $("evoTargetFilter"),
  evoTargetAddGene: $("evoTargetAddGene"),
  evoTargetAddRna: $("evoTargetAddRna"),
  evoTargetAddProtein: $("evoTargetAddProtein"),
  evoTargetAddMolecule: $("evoTargetAddMolecule"),
  evoTargetClear: $("evoTargetClear"),
  evoTargetCustomPattern: $("evoTargetCustomPattern"),
  evoTargetAddCustom: $("evoTargetAddCustom"),
  evoTargetPatterns: $("evoTargetPatterns"),
  evoTargetInfo: $("evoTargetInfo"),
  evoTargetList: $("evoTargetList"),
  evoMeasList: $("evoMeasList"),
  evoRefreshMeasBtn: $("evoRefreshMeasBtn"),
  evoStartBtn: $("evoStartBtn"),
  evoStopBtn: $("evoStopBtn"),
  evoStatus: $("evoStatus"),
  evoCanvas: $("evoCanvas"),
  evoTopList: $("evoTopList"),
  editMode: $("editMode"),
  paintValue: $("paintValue"),
  brushRadius: $("brushRadius"),
  toggleEraser: $("toggleEraser"),
  cursorInfo: $("cursorInfo"),
  inspectTable: $("inspectTable"),
  inspectMode: $("inspectMode"),
  inspectHistMaskLayer: $("inspectHistMaskLayer"),
  inspectHistMaskOp: $("inspectHistMaskOp"),
  inspectHistMaskValue: $("inspectHistMaskValue"),
  inspectCursorValue: $("inspectCursorValue"),
  inspectSummaryStats: $("inspectSummaryStats"),
  inspectCanvasHist: $("inspectCanvasHist"),
  canvasWrap: $("canvasWrap"),
  canvas: $("canvas"),
  overlay: $("overlay"),
};

function _rtClearEvents() {
  rtLastEvents = null;
  rtEventRows.clear();
  if (ui.rtEventsList) ui.rtEventsList.innerHTML = "";
}

if (ui.evoAlgo) {
  ui.evoAlgo.addEventListener("change", () => {
    _evoUpdateAlgoUi();
  });
}

_evoUpdateAlgoUi();

let evoAvailableMeasurements = [];
let evoMeasurementWeights = {};
let evoMeasurementAggs = {};

function _evoResetMeasurementWeights() {
  evoAvailableMeasurements = [];
  evoMeasurementWeights = {};
  evoMeasurementAggs = {};
  if (ui.evoMeasList) ui.evoMeasList.innerHTML = "";
}

// Target layers for evolution (glob patterns)
let evoTargetPatterns = ["gene_*", "rna_*", "protein_*"];
let evoTargetFilterText = "";
const EVO_TARGET_PATTERNS_KEY = "grid_layer_editor_evo_target_patterns_v1";

function _evoLoadTargetPatterns() {
  try {
    const stored = localStorage.getItem(EVO_TARGET_PATTERNS_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        evoTargetPatterns = parsed.filter(p => typeof p === "string" && p.trim());
      }
    }
  } catch {}
}

function _evoSaveTargetPatterns() {
  try {
    localStorage.setItem(EVO_TARGET_PATTERNS_KEY, JSON.stringify(evoTargetPatterns));
  } catch {}
}

function _evoAddTargetPattern(pattern) {
  const p = String(pattern || "").trim();
  if (!p) return;
  if (!evoTargetPatterns.includes(p)) {
    evoTargetPatterns.push(p);
    _evoSaveTargetPatterns();
    _evoRenderTargetLayersUI();
  }
}

function _evoRemoveTargetPattern(pattern) {
  const idx = evoTargetPatterns.indexOf(pattern);
  if (idx >= 0) {
    evoTargetPatterns.splice(idx, 1);
    _evoSaveTargetPatterns();
    _evoRenderTargetLayersUI();
  }
}

function _evoClearTargetPatterns() {
  evoTargetPatterns = [];
  _evoSaveTargetPatterns();
  _evoRenderTargetLayersUI();
}

function _evoMatchesTargetPatterns(layerName) {
  if (!evoTargetPatterns.length) return false;
  for (const pat of evoTargetPatterns) {
    // Simple glob matching: * matches any characters
    const regex = new RegExp("^" + pat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.') + "$");
    if (regex.test(layerName)) return true;
  }
  return false;
}

function _evoGetMatchingLayers() {
  if (!state || !Array.isArray(state.layers)) return [];
  return state.layers.filter(l => _evoMatchesTargetPatterns(l.name)).map(l => l.name);
}

function _evoRenderTargetLayersUI() {
  // Render patterns list
  if (ui.evoTargetPatterns) {
    if (evoTargetPatterns.length === 0) {
      ui.evoTargetPatterns.innerHTML = '<span style="color: var(--muted); font-style: italic;">No patterns (defaults to gene_*, rna_*, protein_*)</span>';
    } else {
      ui.evoTargetPatterns.innerHTML = "";
      for (const pat of evoTargetPatterns) {
        const tag = document.createElement("span");
        tag.style.cssText = "display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; margin: 2px; background: rgba(10,132,255,0.15); border-radius: 4px; font-size: 12px;";
        tag.textContent = pat;
        const removeBtn = document.createElement("button");
        removeBtn.textContent = "×";
        removeBtn.style.cssText = "border: none; background: none; color: var(--muted); cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1;";
        removeBtn.addEventListener("click", () => _evoRemoveTargetPattern(pat));
        tag.appendChild(removeBtn);
        ui.evoTargetPatterns.appendChild(tag);
      }
    }
  }

  // Render matching count
  const matching = _evoGetMatchingLayers();
  if (ui.evoTargetInfo) {
    if (evoTargetPatterns.length === 0) {
      ui.evoTargetInfo.textContent = "Using defaults: gene_*, rna_*, protein_*";
    } else {
      ui.evoTargetInfo.textContent = `${matching.length} layer${matching.length !== 1 ? "s" : ""} will be evolvable`;
    }
  }

  // Render layer list
  if (ui.evoTargetList) {
    ui.evoTargetList.innerHTML = "";
    if (!state || !Array.isArray(state.layers) || state.layers.length === 0) {
      const msg = document.createElement("div");
      msg.className = "meta";
      msg.style.padding = "8px";
      msg.textContent = "No layers available";
      ui.evoTargetList.appendChild(msg);
      return;
    }

    const q = String(evoTargetFilterText || "").trim().toLowerCase();
    let layers = state.layers;
    if (q) {
      layers = layers.filter(l => String(l.name).toLowerCase().includes(q));
    }

    // Group by prefix
    const groups = new Map();
    for (const l of layers) {
      const prefix = String(l.name).split("_")[0] || "other";
      if (!groups.has(prefix)) groups.set(prefix, []);
      groups.get(prefix).push(l);
    }

    const sortedKeys = [...groups.keys()].sort();
    for (const prefix of sortedKeys) {
      const groupLayers = groups.get(prefix);
      const allMatch = groupLayers.every(l => _evoMatchesTargetPatterns(l.name));
      const someMatch = groupLayers.some(l => _evoMatchesTargetPatterns(l.name));

      // Group header
      const header = document.createElement("div");
      header.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border); font-weight: 600; font-size: 11px; text-transform: uppercase;";
      
      const groupCb = document.createElement("input");
      groupCb.type = "checkbox";
      groupCb.checked = allMatch;
      groupCb.indeterminate = someMatch && !allMatch;
      groupCb.addEventListener("change", () => {
        if (groupCb.checked) {
          _evoAddTargetPattern(prefix + "_*");
        } else {
          _evoRemoveTargetPattern(prefix + "_*");
        }
      });
      header.appendChild(groupCb);
      
      const headerLabel = document.createElement("span");
      headerLabel.textContent = `${prefix}_ (${groupLayers.length})`;
      header.appendChild(headerLabel);
      
      ui.evoTargetList.appendChild(header);

      // Layer rows
      for (const l of groupLayers) {
        const row = document.createElement("div");
        row.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 4px 8px 4px 20px; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 12px;";
        
        const isMatch = _evoMatchesTargetPatterns(l.name);
        if (isMatch) {
          row.style.background = "rgba(10,132,255,0.08)";
        }
        
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = isMatch;
        cb.addEventListener("change", () => {
          if (cb.checked) {
            // Add exact layer name as a pattern
            _evoAddTargetPattern(l.name);
          } else {
            // Try to remove exact match or leave as is (patterns might still match)
            _evoRemoveTargetPattern(l.name);
          }
        });
        row.appendChild(cb);
        
        const nameSpan = document.createElement("span");
        nameSpan.textContent = l.name;
        nameSpan.style.flex = "1";
        row.appendChild(nameSpan);
        
        const kindSpan = document.createElement("span");
        kindSpan.textContent = l.kind || "";
        kindSpan.style.cssText = "color: var(--muted); font-size: 10px;";
        row.appendChild(kindSpan);
        
        ui.evoTargetList.appendChild(row);
      }
    }
  }
}

function _evoResetTargetPatterns() {
  evoTargetPatterns = ["gene_*", "rna_*", "protein_*"];
  evoTargetFilterText = "";
  _evoSaveTargetPatterns();
  _evoRenderTargetLayersUI();
}

// Initialize target patterns
_evoLoadTargetPatterns();

function _evoUpdateMeasurementsUI() {
  if (!ui.evoMeasList) return;
  ui.evoMeasList.innerHTML = "";
  
  if (!evoAvailableMeasurements.length) {
    const msg = document.createElement("div");
    msg.className = "meta";
    msg.style.padding = "12px";
    msg.style.backgroundColor = "rgba(255, 200, 100, 0.1)";
    msg.style.borderRadius = "4px";
    msg.style.marginBottom = "8px";
    msg.innerHTML = "<strong>No measurements found.</strong><br>Add measurements in the Measurements tab, then click Refresh.";
    ui.evoMeasList.appendChild(msg);
    return;
  }
  
  for (const meas of evoAvailableMeasurements) {
    const row = document.createElement("div");
    row.className = "evoWeightRow";
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "12px";
    row.style.marginBottom = "10px";
    row.style.padding = "8px";
    row.style.backgroundColor = "rgba(255, 255, 255, 0.02)";
    row.style.borderRadius = "4px";
    
    const labelContainer = document.createElement("div");
    labelContainer.style.flex = "1";
    labelContainer.style.minWidth = "0";
    
    const label = document.createElement("div");
    label.className = "meta";
    label.textContent = meas.name;
    label.style.fontWeight = "500";
    label.style.marginBottom = "2px";
    
    const expr = document.createElement("div");
    expr.className = "meta";
    expr.textContent = meas.expr;
    expr.style.fontSize = "10px";
    expr.style.opacity = "0.6";
    expr.style.overflow = "hidden";
    expr.style.textOverflow = "ellipsis";
    expr.style.whiteSpace = "nowrap";
    expr.title = meas.expr;
    
    labelContainer.appendChild(label);
    labelContainer.appendChild(expr);
    
    const weightLabel = document.createElement("div");
    weightLabel.className = "meta";
    weightLabel.textContent = "Weight";
    weightLabel.style.fontSize = "10px";
    weightLabel.style.opacity = "0.6";
    weightLabel.style.marginRight = "4px";
    
    const weightInput = document.createElement("input");
    weightInput.className = "input input--tiny";
    weightInput.type = "number";
    weightInput.step = "any";
    weightInput.value = "1.0";
    weightInput.placeholder = "0";
    weightInput.id = `evoMeas_${meas.name}`;
    weightInput.style.width = "70px";
    weightInput.style.minWidth = "60px";
    weightInput.style.maxWidth = "70px";
    weightInput.addEventListener("input", () => {
      const w = parseFloat(weightInput.value);
      if (!isNaN(w)) {
        // Store the weight value including 0 (0 means exclude from fitness)
        evoMeasurementWeights[meas.name] = w;
      }
    });

    const aggLabel = document.createElement("div");
    aggLabel.className = "meta";
    aggLabel.textContent = "Over run";
    aggLabel.style.fontSize = "10px";
    aggLabel.style.opacity = "0.6";
    aggLabel.style.marginRight = "4px";

    const aggSelect = document.createElement("select");
    aggSelect.className = "input input--tiny";
    aggSelect.id = `evoMeasAgg_${meas.name}`;
    aggSelect.style.width = "110px";
    aggSelect.style.minWidth = "100px";
    aggSelect.style.maxWidth = "130px";
    aggSelect.innerHTML = "";
    {
      const optLast = document.createElement("option");
      optLast.value = "last";
      optLast.textContent = "Last tick";
      const optMean = document.createElement("option");
      optMean.value = "mean";
      optMean.textContent = "Mean";
      const optMed = document.createElement("option");
      optMed.value = "median";
      optMed.textContent = "Median";
      aggSelect.appendChild(optLast);
      aggSelect.appendChild(optMean);
      aggSelect.appendChild(optMed);
    }
    aggSelect.addEventListener("change", () => {
      const v = String(aggSelect.value || "last");
      evoMeasurementAggs[meas.name] = v;
    });
    
    // Initialize all measurements with weight 1.0
    if (!evoMeasurementWeights.hasOwnProperty(meas.name)) {
      evoMeasurementWeights[meas.name] = 1.0;
    }

    if (!evoMeasurementAggs.hasOwnProperty(meas.name)) {
      evoMeasurementAggs[meas.name] = "last";
    }
    
    // Set input value from current weights
    weightInput.value = evoMeasurementWeights[meas.name];

    aggSelect.value = evoMeasurementAggs[meas.name] || "last";
    
    row.appendChild(labelContainer);
    row.appendChild(weightLabel);
    row.appendChild(weightInput);
    row.appendChild(aggLabel);
    row.appendChild(aggSelect);
    ui.evoMeasList.appendChild(row);
  }
}

async function _evoRefreshMeasurements() {
  try {
    const payload = _evoBasePayloadFromUi();
    const res = await _rtPostJson("/api/evolution/fitness-config", { payload });
    if (res.ok && Array.isArray(res.measurements)) {
      evoAvailableMeasurements = res.measurements;
      _evoUpdateMeasurementsUI();
    }
  } catch (err) {
    console.error("Failed to fetch measurements:", err);
  }
}

if (ui.evoRefreshMeasBtn) {
  ui.evoRefreshMeasBtn.addEventListener("click", () => {
    _evoRefreshMeasurements();
  });
}

// Target layers UI event listeners
if (ui.evoTargetFilter) {
  ui.evoTargetFilter.addEventListener("input", () => {
    evoTargetFilterText = String(ui.evoTargetFilter.value || "");
    _evoRenderTargetLayersUI();
  });
}

if (ui.evoTargetAddGene) {
  ui.evoTargetAddGene.addEventListener("click", () => _evoAddTargetPattern("gene_*"));
}
if (ui.evoTargetAddRna) {
  ui.evoTargetAddRna.addEventListener("click", () => _evoAddTargetPattern("rna_*"));
}
if (ui.evoTargetAddProtein) {
  ui.evoTargetAddProtein.addEventListener("click", () => _evoAddTargetPattern("protein_*"));
}
if (ui.evoTargetAddMolecule) {
  ui.evoTargetAddMolecule.addEventListener("click", () => _evoAddTargetPattern("molecule_*"));
}
if (ui.evoTargetClear) {
  ui.evoTargetClear.addEventListener("click", () => _evoClearTargetPatterns());
}
if (ui.evoTargetAddCustom) {
  ui.evoTargetAddCustom.addEventListener("click", () => {
    const pat = String(ui.evoTargetCustomPattern?.value || "").trim();
    if (pat) {
      _evoAddTargetPattern(pat);
      if (ui.evoTargetCustomPattern) ui.evoTargetCustomPattern.value = "";
    }
  });
}
if (ui.evoTargetCustomPattern) {
  ui.evoTargetCustomPattern.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const pat = String(ui.evoTargetCustomPattern.value || "").trim();
      if (pat) {
        _evoAddTargetPattern(pat);
        ui.evoTargetCustomPattern.value = "";
      }
    }
  });
}

function _rtEnsureEventRow(key) {
  if (!ui.rtEventsList) return null;
  const k = String(key || "");
  if (!k) return null;
  const existing = rtEventRows.get(k);
  if (existing) return existing;

  const row = document.createElement("div");
  row.className = "runtimeMeasRow";

  const sw = document.createElement("div");
  sw.className = "runtimeMeasSwatch";
  sw.style.background = "rgba(255,255,255,.25)";

  const nm = document.createElement("div");
  nm.className = "runtimeMeasName";
  nm.textContent = k;

  const val = document.createElement("div");
  val.className = "runtimeMeasVal";
  val.textContent = "–";

  row.appendChild(sw);
  row.appendChild(nm);
  row.appendChild(val);
  ui.rtEventsList.appendChild(row);

  const obj = { row, sw, nm, val };
  rtEventRows.set(k, obj);
  return obj;
}

function _rtApplyEvents(events) {
  if (!ui.rtEventsList) return;
  if (!events || typeof events !== "object") return;
  rtLastEvents = events;

  const last = events.last && typeof events.last === "object" ? events.last : {};
  const totals = events.totals && typeof events.totals === "object" ? events.totals : {};

  const keys = ["starvation_deaths", "damage_deaths", "divisions"];
  for (const k of keys) {
    const dv = Number(last[k] ?? 0);
    const tv = Number(totals[k] ?? 0);
    const row = _rtEnsureEventRow(k);
    if (!row) continue;
    const dvTxt = Number.isFinite(dv) ? String(Math.round(dv)) : "0";
    const tvTxt = Number.isFinite(tv) ? String(Math.round(tv)) : "0";
    row.val.textContent = `+${dvTxt}  (total ${tvTxt})`;
  }
}

function _openHelpModal(title, html) {
  if (!ui.helpModal || !ui.helpModalBody || !ui.helpModalTitle) return;
  ui.helpModalTitle.textContent = String(title || "Help");
  ui.helpModalBody.innerHTML = String(html || "");
  ui.helpModal.classList.add("modal--open");
  ui.helpModal.setAttribute("aria-hidden", "false");
}

let _datalistCounter = 0;

function makeSearchableSelect(options, currentValue = "", placeholder = "Select...", onChange = null) {
  const id = `searchable_${_datalistCounter++}`;
  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  
  const input = document.createElement("input");
  input.className = "input";
  input.type = "text";
  input.placeholder = placeholder;
  input.value = currentValue;
  input.setAttribute("list", id);
  input.autocomplete = "off";
  
  const datalist = document.createElement("datalist");
  datalist.id = id;
  
  for (const opt of options) {
    const option = document.createElement("option");
    option.value = typeof opt === "string" ? opt : opt.value;
    if (opt.label) option.label = opt.label;
    datalist.appendChild(option);
  }
  
  if (onChange) {
    input.addEventListener("input", () => onChange(input.value));
    input.addEventListener("change", () => onChange(input.value));
  }
  
  wrapper.appendChild(input);
  wrapper.appendChild(datalist);
  
  return { wrapper, input, datalist };
}

// Runtime (local server)
let rtLoaded = false;
let rtRunning = false;
let rtTimer = null;
let rtTick = 0;
let rtMeta = null; // {H,W,layers:[{name,kind}]}
let rtPayloadObj = null;
let rtKinds = new Map();
let rtColors = new Map();
let rtWatch = []; // [{name, alpha}]
const rtCanvases = new Map();
const rtLastArrays = new Map();
let rtOverlayScratch = null;
let rtLastScalars = null;
const rtScalarHist = new Map();
const rtScalarRows = new Map();
let rtLastMeasurements = null;
const rtMeasHist = new Map();
const rtMeasRows = new Map();
let rtLastEvents = null;
const rtEventRows = new Map();

let rtHistLayer = "";
let rtHistBins = 60;
let rtHistLogY = false;
let rtHistMaskLayer = "";
let rtHistMaskOp = "==";
let rtHistMaskValue = 1;
let rtHeatMaskEnabled = false;

let rtLastSyncedStateTxt = "";

async function _rtEnsureSyncedFromEditor(force = false, sourceLabel = "editor") {
  const txt = serializeState(state);
  if (!force && rtLoaded && rtLastSyncedStateTxt === txt) return;
  const payload = JSON.parse(txt);
  await _rtResetWithPayload(payload, sourceLabel);
  rtLastSyncedStateTxt = txt;
}

const RT_HIST_LAYER_KEY = "rt_hist_layer";
const RT_HIST_MASK_LAYER_KEY = "rt_hist_mask_layer";
const RT_HIST_MASK_OP_KEY = "rt_hist_mask_op";
const RT_HIST_MASK_VALUE_KEY = "rt_hist_mask_value";
const RT_HIST_BINS_KEY = "rt_hist_bins";
const RT_HIST_LOGY_KEY = "rt_hist_logy";
const RT_HEAT_MASK_ENABLED_KEY = "rt_heat_mask_enabled";

const RT_SURV_TOPK_KEY = "rt_surv_topk";
const RT_SURV_LOG1P_KEY = "rt_surv_log1p";
const RT_SURV_FOCUS_KEY = "rt_surv_focus";

let rtSurvTopK = 12;
let rtSurvLog1p = true;
let rtSurvFocus = "gene";

let rtBaseline = null; // { H, W, tick0, names: string[], kinds: Map, cell0: Float32Array, layers: Map<string, Float32Array> }

function _rtSetStatus(s) {
  if (ui.rtStatus) ui.rtStatus.textContent = String(s || "");
}

function _evoSetStatus(s) {
  if (ui.evoStatus) ui.evoStatus.textContent = String(s || "");
}

function _rtMeasWindowN() {
  const n = Math.floor(Number(ui.rtMeasWindow?.value ?? 300));
  if (!Number.isFinite(n)) return 300;
  return Math.max(10, Math.min(5000, n));
}

function _rtClearMeasurements() {
  rtLastMeasurements = null;
  rtMeasHist.clear();
  rtMeasRows.clear();
  if (ui.rtMeasList) ui.rtMeasList.innerHTML = "";
  if (ui.rtMeasCanvas) {
    const p = _stepsPrepPlotCanvas(ui.rtMeasCanvas, 900, 180);
    if (p) p.ctx.clearRect(0, 0, p.W, p.H);
  }
}

function _rtMeasColor(i) {
  const palette = ["#4caf50", "#2196f3", "#ff9800", "#e91e63", "#9c27b0", "#00bcd4", "#ffc107", "#8bc34a"];
  return palette[i % palette.length];
}

function _rtEnsureMeasRow(name) {
  if (!ui.rtMeasList) return null;
  const key = String(name);
  const existing = rtMeasRows.get(key);
  if (existing) return existing;
  const row = document.createElement("div");
  row.className = "runtimeMeasRow";

  const sw = document.createElement("div");
  sw.className = "runtimeMeasSwatch";
  row.appendChild(sw);

  const nm = document.createElement("div");
  nm.className = "runtimeMeasName";
  nm.textContent = key;
  const val = document.createElement("div");
  val.className = "runtimeMeasVal";
  val.textContent = "–";
  row.appendChild(nm);
  row.appendChild(val);
  ui.rtMeasList.appendChild(row);
  const obj = { row, sw, nm, val };
  rtMeasRows.set(key, obj);
  return obj;
}

function _rtDrawMeasPlot(names) {
  if (!ui.rtMeasCanvas) return;
  const p = _stepsPrepPlotCanvas(ui.rtMeasCanvas, 900, 180);
  if (!p) return;
  const { ctx, W, H } = p;
  ctx.clearRect(0, 0, W, H);

  const padL = 30;
  const padR = 10;
  const padT = 10;
  const padB = 16;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  ctx.strokeStyle = "rgba(255,255,255,.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(padL, padT, plotW, plotH);
  ctx.stroke();

  const winN = _rtMeasWindowN();
  const normalize = !!ui.rtMeasNormalize?.checked;
  const useLog = !!ui.rtMeasLog?.checked;

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.font = "11px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("tick", padL + plotW - 22, padT + plotH + 14);

  const yLabel = normalize ? "normalized" : "value";
  const yScale = useLog ? "log" : "linear";
  ctx.save();
  ctx.translate(10, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${yLabel} (${yScale})`, 0, 0);
  ctx.restore();

  let maxLen = 0;
  for (const nm of names) {
    const hist = rtMeasHist.get(nm) || [];
    if (hist.length > maxLen) maxLen = hist.length;
  }
  const nPts = Math.max(2, Math.min(winN, maxLen));

  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    const hist0 = rtMeasHist.get(name) || [];
    if (!hist0.length) continue;
    const hist = hist0.slice(Math.max(0, hist0.length - nPts));

    let mn = Infinity;
    let mx = -Infinity;
    for (let j = 0; j < hist.length; j++) {
      const raw = hist[j];
      const v = useLog ? Math.log1p(Math.max(0, raw)) : raw;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) continue;
    const denom = normalize ? (mx - mn === 0 ? 1e-6 : mx - mn) : 1;

    ctx.strokeStyle = _rtMeasColor(i);
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let j = 0; j < hist.length; j++) {
      const raw = hist[j];
      const v0 = useLog ? Math.log1p(Math.max(0, raw)) : raw;
      const t = normalize ? (v0 - mn) / denom : v0;
      const u = normalize ? Math.max(0, Math.min(1, t)) : Math.max(0, Math.min(1, t));
      const x = padL + (hist.length <= 1 ? 0 : (j / (hist.length - 1)) * plotW);
      const y = padT + (1 - u) * plotH;
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawHLine(y, color, dash) {
    if (!Number.isFinite(y)) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash(Array.isArray(dash) ? dash : []);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function _rtApplyMeasurements(measurements) {
  if (!measurements || typeof measurements !== "object") return;
  rtLastMeasurements = measurements;
  const winN = _rtMeasWindowN();

  const names = Object.keys(measurements).sort();
  for (const name of names) {
    const v = measurements[name];
    const vv = typeof v === "number" && Number.isFinite(v) ? v : null;
    const hist = rtMeasHist.get(name) || [];
    hist.push(vv == null ? NaN : vv);
    while (hist.length > winN) hist.shift();
    rtMeasHist.set(name, hist);

    const rowObj = _rtEnsureMeasRow(name);
    if (rowObj) {
      rowObj.val.textContent = vv == null ? "–" : _stepsFmt(vv);
      const idx = names.indexOf(name);
      rowObj.sw.style.background = _rtMeasColor(idx);
    }
  }
  _rtDrawMeasPlot(names);
}

function _rtIntervalValueMs() {
  const v = Number(ui.rtIntervalMs?.value);
  if (!Number.isFinite(v)) return 500;
  return Math.max(50, Math.floor(v));
}

async function _rtPostJson(path, bodyObj) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj || {}),
  });
  const txt = await res.text();
  const contentType = String(res.headers.get("content-type") || "");

  // Common dev footgun: serving web_editor with a static server (e.g. python -m http.server)
  // which returns an HTML 501 for POST. Make this actionable.
  const looksLikeHtml = /^\s*<!doctype\s+html/i.test(txt) || /^\s*<html/i.test(txt);
  if (res.status === 501 || (looksLikeHtml && /Unsupported method \('POST'\)/i.test(txt))) {
    throw new Error(
      "Runtime API not available at this URL. You are likely serving the editor with a static server (e.g. `python -m http.server`).\n\nRun: `python3 runtime_server.py`\nThen open: http://127.0.0.1:8000/"
    );
  }

  let out = null;
  try {
    out = JSON.parse(txt);
  } catch {
    // Avoid dumping large HTML into the UI.
    const preview = looksLikeHtml ? "(HTML error response)" : String(txt || "").slice(0, 400);
    out = { ok: false, error: preview };
  }
  if (!res.ok || !out || out.ok === false) {
    const msg = out?.error || out?.message || (contentType.includes("text/html") ? "(HTML error response)" : `HTTP ${res.status}`);
    throw new Error(msg);
  }
  return out;
}

function _rtPopulateLayerSelect() {
  if (!ui.rtAddLayerSelect || !ui.rtAddLayerSelect.parentNode) return;
  const names = rtMeta?.layers ? rtMeta.layers.map((x) => x.name) : [];
  if (!names.length) return;
  
  const searchable = makeSearchableSelect(names, "", "(pick layer)");
  searchable.input.className = "input";
  ui.rtAddLayerSelect.replaceWith(searchable.wrapper);
  ui.rtAddLayerSelect = searchable.input;
}

function _rtEnsureVizItem(layerName) {
  if (!ui.rtVizGrid) return null;
  let item = rtCanvases.get(layerName) || null;
  if (item && item.wrap && item.canvas) return item;

  const wrap = document.createElement("div");
  wrap.className = "runtimeVizItem";

  const title = document.createElement("div");
  title.className = "runtimeVizTitle";
  const left = document.createElement("div");
  left.className = "meta";
  left.textContent = layerName;
  title.appendChild(left);
  wrap.appendChild(title);

  const canvas = document.createElement("canvas");
  canvas.className = "stepsCanvas";
  wrap.appendChild(canvas);

  ui.rtVizGrid.appendChild(wrap);
  item = { wrap, canvas };
  rtCanvases.set(layerName, item);
  return item;
}

function _rtDropVizItem(layerName) {
  const item = rtCanvases.get(layerName) || null;
  if (item?.wrap && item.wrap.parentElement) item.wrap.parentElement.removeChild(item.wrap);
  rtCanvases.delete(layerName);
}

function _rtRenderWatchList() {
  if (!ui.rtWatchList) return;
  ui.rtWatchList.innerHTML = "";
  for (let i = 0; i < rtWatch.length; i++) {
    const w = rtWatch[i];
    const row = document.createElement("div");
    row.className = "runtimeWatchRow";

    const nm = document.createElement("div");
    nm.className = "meta";
    nm.textContent = w.name;
    row.appendChild(nm);

    const alpha = document.createElement("input");
    alpha.className = "input input--tiny";
    alpha.type = "number";
    alpha.min = "0";
    alpha.max = "1";
    alpha.step = "0.05";
    alpha.value = String(w.alpha ?? 1);
    alpha.title = "Overlay alpha";
    alpha.addEventListener("input", () => {
      const v = Number(alpha.value);
      w.alpha = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 1;
      _rtRenderOverlay();
    });
    row.appendChild(alpha);

    const up = document.createElement("button");
    up.className = "btn btn--secondary btn--tiny";
    up.textContent = "↑";
    up.disabled = i === 0;
    up.addEventListener("click", () => {
      if (i <= 0) return;
      const tmp = rtWatch[i - 1];
      rtWatch[i - 1] = rtWatch[i];
      rtWatch[i] = tmp;
      _rtRenderWatchList();
      _rtRenderOverlay();
    });
    row.appendChild(up);

    const del = document.createElement("button");
    del.className = "btn btn--danger btn--tiny";
    del.textContent = "Remove";
    del.addEventListener("click", () => {
      rtWatch = rtWatch.filter((x) => x.name !== w.name);
      _rtDropVizItem(w.name);
      _rtRenderWatchList();
      _rtRenderOverlay();
    });
    row.appendChild(del);

    ui.rtWatchList.appendChild(row);
  }
}

function _rtEnsureCanvasSizes() {
  if (!rtMeta) return;
  const H = rtMeta.H;
  const W = rtMeta.W;
  if (ui.rtOverlayCanvas) _stepsEnsureCanvasSize(ui.rtOverlayCanvas, H, W);
  for (const w of rtWatch) {
    const item = _rtEnsureVizItem(w.name);
    if (item?.canvas) _stepsEnsureCanvasSize(item.canvas, H, W);
  }
}

function _rtRenderOverlay() {
  if (!ui.rtOverlayCanvas || !rtMeta) return;
  _rtEnsureCanvasSizes();
  const ctx = ui.rtOverlayCanvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, ui.rtOverlayCanvas.width, ui.rtOverlayCanvas.height);
  const H = rtMeta.H;
  const W = rtMeta.W;
  const cw = ui.rtOverlayCanvas.width;
  const ch = ui.rtOverlayCanvas.height;

  if (!rtOverlayScratch) rtOverlayScratch = document.createElement("canvas");
  if (rtOverlayScratch.width !== cw) rtOverlayScratch.width = cw;
  if (rtOverlayScratch.height !== ch) rtOverlayScratch.height = ch;
  const sctx = rtOverlayScratch.getContext("2d");
  if (!sctx) return;

  const sx = W / cw;
  const sy = H / ch;

  for (const w of rtWatch) {
    const name = String(w.name);
    const arr = rtLastArrays.get(name);
    if (!arr || !rtMeta) continue;
    const kind = rtKinds.get(name) || "continuous";
    const col = hexToRgb(rtColors.get(name) || DEFAULT_LAYER_COLOR);

    let mn = Infinity;
    let mx = -Infinity;
    if (kind !== "categorical") {
      for (let i = 0; i < arr.length; i++) {
        const v = kind === "counts" ? clampCounts(arr[i]) : arr[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === mx) {
        mn = mn === mx ? mn - 1 : 0;
        mx = mx === mn ? mx + 1 : 1;
      }
    }
    const denom = mx - mn === 0 ? 1e-9 : mx - mn;

    const img = sctx.createImageData(cw, ch);
    const px = img.data;
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const srcY = Math.min(H - 1, Math.floor(y * sy));
        const srcX = Math.min(W - 1, Math.floor(x * sx));
        const raw = arr[srcY * W + srcX];

        let r = 0,
          g = 0,
          b = 0,
          a = 0;

        if (kind === "categorical") {
          const vi = Math.round(raw);
          if (vi !== 0) {
            const c = palette[((vi % palette.length) + palette.length) % palette.length];
            const alpha = 0.70;
            r = Math.round((1 - alpha) * c[0] + alpha * col.r);
            g = Math.round((1 - alpha) * c[1] + alpha * col.g);
            b = Math.round((1 - alpha) * c[2] + alpha * col.b);
            a = 255;
          }
        } else {
          const v = kind === "counts" ? clampCounts(raw) : raw;
          if (v !== 0) {
            const t = (v - mn) / denom;
            const u = Math.max(0, Math.min(1, t));
            r = col.r;
            g = col.g;
            b = col.b;
            a = Math.round(255 * u);
          }
        }

        const i = (y * cw + x) * 4;
        px[i + 0] = r;
        px[i + 1] = g;
        px[i + 2] = b;
        px[i + 3] = a;
      }
    }

    sctx.putImageData(img, 0, 0);
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, Number(w.alpha ?? 1)));
    ctx.drawImage(rtOverlayScratch, 0, 0);
    ctx.restore();
  }
}

function _rtGetSelectedLayerNames() {
  return rtWatch.map((w) => w.name);
}

function _rtGetRequestedLayerNames() {
  const out = new Set(_rtGetSelectedLayerNames());
  const hl = String(rtHistLayer || "").trim();
  if (hl) out.add(hl);
  const hm = String(rtHistMaskLayer || "").trim();
  if (hm) out.add(hm);
  if (rtMeta && rtMeta.layers && rtMeta.layers.some((m) => String(m?.name || "") === "cell")) out.add("cell");
  return [...out];
}

function _rtSurvTopKN() {
  const n = Math.floor(Number(ui.rtSurvTopK?.value ?? rtSurvTopK));
  if (!Number.isFinite(n)) return 12;
  return Math.max(5, Math.min(50, n));
}

function _rtClearSurvival() {
  rtBaseline = null;
  if (ui.rtSurvList) ui.rtSurvList.innerHTML = "";
}

function _rtInitSurvivalControls() {
  try {
    const rawK = localStorage.getItem(RT_SURV_TOPK_KEY);
    const n = Math.floor(Number(rawK ?? "12"));
    if (Number.isFinite(n)) rtSurvTopK = Math.max(5, Math.min(50, n));
  } catch {}
  try {
    rtSurvLog1p = (localStorage.getItem(RT_SURV_LOG1P_KEY) || "1") !== "0";
  } catch {}
  try {
    const f = String(localStorage.getItem(RT_SURV_FOCUS_KEY) || "gene");
    rtSurvFocus = f || "gene";
  } catch {}

  if (ui.rtSurvFocus) ui.rtSurvFocus.value = String(rtSurvFocus || "gene");
  if (ui.rtSurvTopK) ui.rtSurvTopK.value = String(rtSurvTopK);
  if (ui.rtSurvLog1p) ui.rtSurvLog1p.checked = !!rtSurvLog1p;
}

function _rtSurvMatchesFocus(name) {
  const f = String(ui.rtSurvFocus?.value ?? rtSurvFocus ?? "gene");
  if (f === "all") return true;
  if (f === "gene") return name.startsWith("gene_");
  if (f === "protein") return name.startsWith("protein_");
  if (f === "rna") return name.startsWith("rna_");
  if (f === "molecule") return name.startsWith("molecule_");
  if (f === "damage") return name.startsWith("damage");
  return true;
}

async function _rtCaptureBaseline() {
  if (!rtMeta || !Array.isArray(rtMeta.layers)) return;
  const allNames = rtMeta.layers.map((m) => String(m?.name || "")).filter((x) => x);
  if (!allNames.length) return;

  const frame = await _rtPostJson("/api/runtime/frame", { layers: allNames });
  const data = frame?.data || {};

  const layers = new Map();
  let cell0 = null;
  for (const name of Object.keys(data)) {
    const entry = data[name];
    if (!entry || typeof entry !== "object") continue;
    if (entry.dtype !== "float32" || typeof entry.b64 !== "string") continue;
    const arr = decodeFloat32Base64(entry.b64);
    if (arr.length !== rtMeta.H * rtMeta.W) continue;
    layers.set(name, arr);
    if (name === "cell") cell0 = arr;
  }

  if (!cell0) {
    rtBaseline = null;
    return;
  }

  rtBaseline = {
    H: rtMeta.H,
    W: rtMeta.W,
    tick0: typeof frame?.tick === "number" ? frame.tick : 0,
    names: allNames,
    kinds: rtKinds,
    cell0,
    layers,
  };
}

function _rtRenderSurvival() {
  if (!ui.rtSurvList) return;
  if (!rtBaseline) {
    ui.rtSurvList.innerHTML = '<div class="meta">Baseline not captured yet.</div>';
    return;
  }

  const cellNow = rtLastArrays.get("cell") || null;
  if (!cellNow || cellNow.length !== rtBaseline.H * rtBaseline.W) {
    ui.rtSurvList.innerHTML = '<div class="meta">Need current cell layer to compute survival split.</div>';
    return;
  }

  const topK = _rtSurvTopKN();
  const useLog1p = !!ui.rtSurvLog1p?.checked;

  const N = rtBaseline.H * rtBaseline.W;
  const cell0 = rtBaseline.cell0;
  let n0 = 0;
  let nAlive = 0;
  let nDead = 0;

  // Sample for speed if the initial population is huge.
  const maxSamples = 20000;
  let initCells = 0;
  for (let i = 0; i < N; i++) if (cell0[i] >= 0.5) initCells++;
  const step = Math.max(1, Math.floor(initCells / maxSamples));
  let seen = 0;

  for (let i = 0; i < N; i++) {
    if (cell0[i] < 0.5) continue;
    if (seen % step !== 0) {
      seen++;
      continue;
    }
    seen++;
    n0++;
    if (cellNow[i] >= 0.5) nAlive++;
    else nDead++;
  }

  if (n0 <= 0) {
    ui.rtSurvList.innerHTML = '<div class="meta">No initial cells at t=0.</div>';
    return;
  }
  if (nAlive <= 0 || nDead <= 0) {
    ui.rtSurvList.innerHTML = `<div class="meta">tick=${rtTick}  initial=${n0}  alive=${nAlive}  dead=${nDead}  (need both alive and dead to rank drivers)</div>`;
    return;
  }

  const rows = [];
  const names = rtBaseline.names;
  for (const name of names) {
    if (name === "cell") continue;
    if (!_rtSurvMatchesFocus(name)) continue;
    const kind = rtKinds.get(name) || "continuous";
    if (kind === "categorical") continue;

    const a0 = rtBaseline.layers.get(name) || null;
    if (!a0 || a0.length !== N) continue;

    let sSumT = 0,
      sSumT2 = 0,
      sSumRaw = 0,
      sN = 0;
    let dSumT = 0,
      dSumT2 = 0,
      dSumRaw = 0,
      dN = 0;

    let seen2 = 0;
    for (let i = 0; i < N; i++) {
      if (cell0[i] < 0.5) continue;
      if (seen2 % step !== 0) {
        seen2++;
        continue;
      }
      seen2++;
      let v = a0[i];
      if (!Number.isFinite(v)) continue;
      const vRaw = v;
      const vT = useLog1p ? Math.log1p(Math.max(0, vRaw)) : vRaw;
      if (cellNow[i] >= 0.5) {
        sSumRaw += vRaw;
        sSumT += vT;
        sSumT2 += vT * vT;
        sN++;
      } else {
        dSumRaw += vRaw;
        dSumT += vT;
        dSumT2 += vT * vT;
        dN++;
      }
    }
    if (sN < 30 || dN < 30) continue;

    const sMeanRaw = sSumRaw / sN;
    const dMeanRaw = dSumRaw / dN;

    const sMeanT = sSumT / sN;
    const dMeanT = dSumT / dN;
    const sVarT = Math.max(0, sSumT2 / sN - sMeanT * sMeanT);
    const dVarT = Math.max(0, dSumT2 / dN - dMeanT * dMeanT);
    const denomT = Math.sqrt((sVarT + dVarT) / 2 + 1e-12);
    const dEff = (sMeanT - dMeanT) / denomT;
    if (!Number.isFinite(dEff)) continue;

    rows.push({ name, kind, d: dEff, sMeanRaw, dMeanRaw, sN, dN });
  }

  rows.sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
  const top = rows.slice(0, topK);

  const parts = [];
  parts.push(`<div class="meta">tick=${rtTick}  initial=${n0}  alive=${nAlive}  dead=${nDead}  (ranking uses t=0 values)</div>`);
  if (!top.length) {
    parts.push('<div class="meta">No stable drivers found (try increasing steps or disabling log1p).</div>');
    ui.rtSurvList.innerHTML = parts.join("");
    return;
  }

  for (const r of top) {
    const eps = 1e-12;
    const ratio = (r.sMeanRaw + eps) / (r.dMeanRaw + eps);
    const pct = ((r.sMeanRaw - r.dMeanRaw) / Math.max(eps, Math.abs(r.dMeanRaw))) * 100;
    const dir = ratio >= 1 ? "higher" : "lower";
    const ratioTxt = Number.isFinite(ratio) ? `${_stepsFmt(ratio)}×` : "–";
    const pctTxt = Number.isFinite(pct) ? `${_stepsFmt(pct)}%` : "–";
    const why = `${r.name}: survivors started ${dir} at t=0 (${ratioTxt}, ${pctTxt})`;
    parts.push(
      `<div class="runtimeMeasRow"><div class="runtimeMeasSwatch" style="background: rgba(200,200,200,.25)"></div><div class="runtimeMeasName">${r.name}</div><div class="runtimeMeasVal">${why}  d=${_stepsFmt(r.d)}  μ0_alive=${_stepsFmt(r.sMeanRaw)}  μ0_dead=${_stepsFmt(r.dMeanRaw)}</div></div>`
    );
  }
  ui.rtSurvList.innerHTML = parts.join("");
}

function _rtPopulateHistLayerSelect() {
  if (!ui.rtHistLayer || !ui.rtHistLayer.parentNode) return;
  if (!rtMeta || !Array.isArray(rtMeta.layers)) return;
  
  const names = rtMeta.layers.map(m => String(m?.name || "")).filter(nm => nm);
  const searchable = makeSearchableSelect(names, rtHistLayer || "", "(none)");
  searchable.input.className = "input input--tiny";
  ui.rtHistLayer.replaceWith(searchable.wrapper);
  ui.rtHistLayer = searchable.input;
}

function _rtPopulateHistMaskLayerSelect() {
  if (!ui.rtHistMaskLayer || !ui.rtHistMaskLayer.parentNode) return;
  if (!rtMeta || !Array.isArray(rtMeta.layers)) return;
  
  const names = rtMeta.layers.map(m => String(m?.name || "")).filter(nm => nm);
  const searchable = makeSearchableSelect(names, rtHistMaskLayer || "", "(none)");
  searchable.input.className = "input input--tiny";
  ui.rtHistMaskLayer.replaceWith(searchable.wrapper);
  ui.rtHistMaskLayer = searchable.input;
}

function _rtInitHistogramControls() {
  try {
    rtHistLayer = String(localStorage.getItem(RT_HIST_LAYER_KEY) || "");
  } catch {}
  try {
    const n = Math.floor(Number(localStorage.getItem(RT_HIST_BINS_KEY) || "60"));
    if (Number.isFinite(n)) rtHistBins = Math.max(10, Math.min(400, n));
  } catch {}
  try {
    rtHistLogY = (localStorage.getItem(RT_HIST_LOGY_KEY) || "") === "1";
  } catch {}
  try {
    rtHistMaskLayer = String(localStorage.getItem(RT_HIST_MASK_LAYER_KEY) || "");
  } catch {}
  try {
    rtHistMaskOp = String(localStorage.getItem(RT_HIST_MASK_OP_KEY) || "==");
  } catch {}
  try {
    const v = Number(localStorage.getItem(RT_HIST_MASK_VALUE_KEY));
    if (Number.isFinite(v)) rtHistMaskValue = v;
  } catch {}
  try {
    rtHeatMaskEnabled = (localStorage.getItem(RT_HEAT_MASK_ENABLED_KEY) || "") === "1";
  } catch {}

  if (ui.rtHistBins) ui.rtHistBins.value = String(rtHistBins);
  if (ui.rtHistLogY) ui.rtHistLogY.checked = !!rtHistLogY;
  if (ui.rtHistMaskOp) ui.rtHistMaskOp.value = String(rtHistMaskOp);
  if (ui.rtHistMaskValue) ui.rtHistMaskValue.value = String(rtHistMaskValue);
  if (ui.rtHeatMaskEnabled) ui.rtHeatMaskEnabled.checked = !!rtHeatMaskEnabled;
}

function _rtRenderHeatmaps() {
  if (!rtMeta) return;
  const dataH = rtMeta.H;
  const dataW = rtMeta.W;

  const watchSet = new Set(rtWatch.map((w) => String(w?.name || "")));

  const maskLayerName = rtHeatMaskEnabled ? String(rtHistMaskLayer || "").trim() : "";
  const maskArr = maskLayerName ? (rtLastArrays.get(maskLayerName) || null) : null;

  let drawn = 0;
  for (const name of _rtGetSelectedLayerNames()) {
    if (!watchSet.has(name)) continue;
    const arr = rtLastArrays.get(name) || null;
    if (!arr || arr.length !== dataH * dataW) continue;
    const kind = rtKinds.get(name) || "continuous";
    const color = rtColors.get(name) || DEFAULT_LAYER_COLOR;
    const item = _rtEnsureVizItem(name);
    if (item?.canvas) {
      _stepsEnsureCanvasSize(item.canvas, dataH, dataW);
      _stepsDrawHeatmap(item.canvas, arr, dataH, dataW, kind, "value", null, color, maskArr, rtHistMaskOp, rtHistMaskValue);
      drawn++;
    }
  }
  return drawn;
}

function _rtDrawHistogram(canvas, arr, bins, xLabel, logY, maskArr, maskOp, maskValue) {
  if (!canvas || !arr || !arr.length) return;
  const p = _stepsPrepPlotCanvas(canvas, 520, 180);
  if (!p) return;
  const { ctx, W, H } = p;

  const padL = 52;
  const padR = 14;
  const padT = 10;
  const padB = 28;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  let filteredArr = arr;
  if (maskArr && maskArr.length === arr.length) {
    const mv = Number(maskValue);
    const filtered = [];
    for (let i = 0; i < arr.length; i++) {
      const m = maskArr[i];
      let pass = false;
      if (maskOp === "==") pass = m === mv;
      else if (maskOp === "!=") pass = m !== mv;
      else if (maskOp === ">") pass = m > mv;
      else if (maskOp === ">=") pass = m >= mv;
      else if (maskOp === "<") pass = m < mv;
      else if (maskOp === "<=") pass = m <= mv;
      if (pass) filtered.push(arr[i]);
    }
    filteredArr = filtered;
  }

  if (!filteredArr.length) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("no values match mask", 12, 20);
    return;
  }

  const r = _stepsComputeRange(filteredArr);
  let lo = r.mn;
  let hi = r.mx;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
    lo = 0;
    hi = 1;
  }

  const B = Math.max(10, Math.min(400, Math.floor(Number(bins) || 60)));
  const counts = new Array(B).fill(0);
  const denom = hi - lo;
  if (denom <= 0) return;

  for (let i = 0; i < filteredArr.length; i++) {
    const v = filteredArr[i];
    if (!Number.isFinite(v)) continue;
    const t = (v - lo) / denom;
    if (t < 0 || t > 1) continue;
    const bi = Math.max(0, Math.min(B - 1, Math.floor(t * B)));
    counts[bi]++;
  }

  let maxC = 1;
  for (const c of counts) if (c > maxC) maxC = c;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,.04)";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,.06)";
  ctx.lineWidth = 1;
  for (let k = 1; k <= 3; k++) {
    const yy = padT + (plotH * k) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, yy + 0.5);
    ctx.lineTo(padL + plotW, yy + 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  ctx.fillText(String(maxC), padL - 6, padT + 6);
  ctx.fillText("0", padL - 6, padT + plotH);

  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  ctx.fillText(_stepsFmt(lo), padL, padT + plotH + 6);
  ctx.fillText(_stepsFmt((lo + hi) / 2), padL + plotW / 2, padT + plotH + 6);
  ctx.fillText(_stepsFmt(hi), padL + plotW, padT + plotH + 6);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.fillText(xLabel, padL + plotW / 2, H - 4);
  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(logY ? "count (log)" : "count", 0, 0);
  ctx.restore();

  const barW = plotW / B;
  const logMax = Math.log10(1 + maxC);
  ctx.fillStyle = "rgba(140,200,255,.75)";
  for (let i = 0; i < B; i++) {
    const c = counts[i];
    const frac = logY ? Math.log10(1 + c) / Math.max(1e-9, logMax) : c / maxC;
    const h = Math.max(0, Math.min(1, frac)) * (plotH - 6);
    const x = padL + i * barW;
    const y = padT + plotH - h;
    ctx.fillRect(x, y, Math.max(1, barW - 1.5), h);
  }
}

function _rtRenderHistogram() {
  if (!ui.rtHistCanvas) return;
  const nm = String(rtHistLayer || "").trim();
  if (!nm) {
    const p = _stepsPrepPlotCanvas(ui.rtHistCanvas, 520, 180);
    if (!p) return;
    const { ctx, W, H } = p;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("histogram disabled", 12, 20);
    return;
  }

  const arr = rtLastArrays.get(nm) || null;
  if (!arr) return;
  
  const maskLayerName = String(rtHistMaskLayer || "").trim();
  const maskArr = maskLayerName ? (rtLastArrays.get(maskLayerName) || null) : null;
  
  _rtDrawHistogram(ui.rtHistCanvas, arr, rtHistBins, nm, rtHistLogY, maskArr, rtHistMaskOp, rtHistMaskValue);
}

function _rtScalarWindowN() {
  const n = Math.floor(Number(ui.rtScalarsWindow?.value ?? 200));
  if (!Number.isFinite(n)) return 200;
  return Math.max(10, Math.min(2000, n));
}

function _rtClearScalarHistory() {
  rtScalarHist.clear();
}

function _rtScalarMetricForLayer(name) {
  const kind = rtKinds.get(name) || "continuous";
  if (kind === "categorical") return "count1";
  if (kind === "counts") return "sum";
  return "mean";
}

function _rtScalarValueFromEntry(name, ent) {
  const metric = _rtScalarMetricForLayer(name);
  if (!ent || typeof ent !== "object") return { metric, value: 0 };
  if (metric === "count1") {
    const v = Number(ent.eq1 ?? ent.nonzero ?? 0);
    return { metric, value: Number.isFinite(v) ? v : 0 };
  }
  if (metric === "sum") {
    const v = Number(ent.sum ?? 0);
    return { metric, value: Number.isFinite(v) ? v : 0 };
  }
  const v = Number(ent.mean ?? 0);
  return { metric, value: Number.isFinite(v) ? v : 0 };
}

function _rtEnsureScalarRow(name, metric) {
  if (!ui.rtScalarsList) return null;
  const key = String(name);
  const existing = rtScalarRows.get(key);
  if (existing) return existing;

  const row = document.createElement("div");
  row.className = "runtimeScalarRow";

  const nm = document.createElement("div");
  nm.className = "runtimeScalarName";
  nm.textContent = `${key} (${metric})`;
  row.appendChild(nm);

  const val = document.createElement("div");
  val.className = "runtimeScalarValueTxt";
  val.textContent = "0";
  row.appendChild(val);

  const spark = document.createElement("canvas");
  spark.className = "runtimeScalarSpark";
  row.appendChild(spark);

  ui.rtScalarsList.appendChild(row);

  const obj = { row, nm, val, spark, metric };
  rtScalarRows.set(key, obj);
  return obj;
}

function _rtDrawScalarSparkline(canvas, hist, color) {
  if (!canvas) return;
  if (!hist || hist.length < 2) return;
  const p = _stepsPrepPlotCanvas(canvas, 140, 18);
  if (!p) return;
  const { ctx, W, H } = p;
  ctx.clearRect(0, 0, W, H);

  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < hist.length; i++) {
    const v = hist[i];
    if (!Number.isFinite(v)) continue;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return;
  const denom = mx - mn <= 0 ? 1e-6 : mx - mn;

  ctx.strokeStyle = "rgba(255,255,255,.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(0.5, 0.5, W - 1, H - 1);
  ctx.stroke();

  ctx.strokeStyle = color || "rgba(255,255,255,.6)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const v = hist[i];
    if (!Number.isFinite(v)) continue;
    const t = (v - mn) / denom;
    const u = Math.max(0, Math.min(1, t));
    const x = (i / (hist.length - 1)) * (W - 1);
    const y = (1 - u) * (H - 1);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function _rtApplyScalars(scalars) {
  if (!ui.rtScalarsList) return;
  if (!rtMeta) return;
  if (!scalars || typeof scalars !== "object") return;

  rtLastScalars = scalars;

  const winN = _rtScalarWindowN();
  const useLog = !!ui.rtScalarsLog?.checked;
  const layerNames = rtMeta.layers ? rtMeta.layers.map((m) => String(m.name)) : Object.keys(scalars);

  const prefer = ["cell", "damage", "molecule_atp", "molecule_glucose"];
  const seen = new Set();
  const ordered = [];
  for (const p of prefer) {
    if (layerNames.includes(p)) {
      ordered.push(p);
      seen.add(p);
    }
  }
  for (const nm of layerNames) {
    if (seen.has(nm)) continue;
    ordered.push(nm);
  }

  for (const name of ordered) {
    const ent = scalars[name];
    const { metric, value } = _rtScalarValueFromEntry(name, ent);
    const rowObj = _rtEnsureScalarRow(name, metric);
    if (!rowObj) continue;
    if (rowObj.metric !== metric) {
      rowObj.metric = metric;
      rowObj.nm.textContent = `${name} (${metric})`;
    }

    const vPlot = useLog ? Math.log1p(Math.max(0, value)) : value;
    const hist = rtScalarHist.get(name) || [];
    hist.push(vPlot);
    while (hist.length > winN) hist.shift();
    rtScalarHist.set(name, hist);

    rowObj.val.textContent = metric === "count1" ? String(Math.round(value)) : _stepsFmt(value);
    const c = rtColors.get(name) || DEFAULT_LAYER_COLOR;
    _rtDrawScalarSparkline(rowObj.spark, hist, c);
  }
}

function _rtApplyFrame(frame) {
  if (!frame) return;
  if (!rtMeta) return;
  if (typeof frame.tick === "number") rtTick = frame.tick;
  const data = frame.data || {};
  if (frame.scalars) _rtApplyScalars(frame.scalars);
  if (frame.measurements) _rtApplyMeasurements(frame.measurements);
  if (frame.events) _rtApplyEvents(frame.events);
  const H = rtMeta.H;
  const W = rtMeta.W;

  const watchSet = new Set(rtWatch.map((w) => String(w?.name || "")));

  _rtEnsureCanvasSizes();

  let drawn = 0;
  let firstDbg = null;

  for (const name of Object.keys(data)) {
    const entry = data[name];
    if (!entry || typeof entry !== "object") continue;
    if (entry.dtype !== "float32" || typeof entry.b64 !== "string") continue;
    const arr = decodeFloat32Base64(entry.b64);
    if (arr.length !== H * W) continue;
    rtLastArrays.set(name, arr);
  }

  drawn = _rtRenderHeatmaps() || 0;
  _rtRenderOverlay();
  _rtRenderHistogram();
  _rtRenderSurvival();
  if (!drawn) {
    _rtSetStatus(`tick=${rtTick}  drawn=0/${rtWatch.length} (no layer buffers drawn)`);
  } else if (firstDbg) {
    _rtSetStatus(
      `tick=${rtTick}  drawn=${drawn}/${rtWatch.length}  ${firstDbg.name} range=${_stepsFmt(firstDbg.mn)}…${_stepsFmt(firstDbg.mx)}  canvas=${firstDbg.cw}x${firstDbg.ch}`
    );
  } else {
    _rtSetStatus(`tick=${rtTick}  drawn=${drawn}/${rtWatch.length}`);
  }
}

async function _rtResetWithPayload(payloadObj, sourceLabel = "") {
  _rtSetStatus("Resetting…");
  try {
    rtPayloadObj = payloadObj ? JSON.parse(JSON.stringify(payloadObj)) : null;
  } catch {
    rtPayloadObj = payloadObj || null;
  }
  const res = await _rtPostJson("/api/runtime/reset", { payload: payloadObj });
  rtMeta = { H: Number(res.H), W: Number(res.W), layers: Array.isArray(res.layers) ? res.layers : [] };
  rtKinds = new Map();
  for (const m of rtMeta.layers) rtKinds.set(String(m.name), String(m.kind || "continuous"));

  rtColors = new Map();
  if (payloadObj && Array.isArray(payloadObj.layers)) {
    for (const m of payloadObj.layers) {
      if (!m || typeof m !== "object") continue;
      const nm = String(m.name || "");
      if (!nm) continue;
      const c = typeof m.color === "string" && m.color.trim() ? m.color.trim() : DEFAULT_LAYER_COLOR;
      rtColors.set(nm, c);
    }
  }

  rtLastArrays.clear();
  rtLastScalars = null;
  _rtClearScalarHistory();
  rtScalarRows.clear();
  if (ui.rtScalarsList) ui.rtScalarsList.innerHTML = "";
  _rtClearMeasurements();
  _rtClearEvents();
  _rtClearSurvival();
  rtTick = Number(res.tick || 0);
  rtLoaded = true;
  // File tracking is now unified - no separate runtime source tracking
  _rtPopulateLayerSelect();
  _rtPopulateHistLayerSelect();
  _rtPopulateHistMaskLayerSelect();

  const avail = new Set(rtMeta.layers.map((m) => String(m.name)));

  // Default watch: if empty, pick a few common layers.
  if (!rtWatch.length) {
    const preferred = ["cell", "circulation", "atp", "glucose", "molecule_atp", "molecule_glucose"];
    const picked = [];
    for (const p of preferred) if (avail.has(p)) picked.push(p);
    if (!picked.length && rtMeta.layers.length) picked.push(rtMeta.layers[0].name);
    rtWatch = picked.map((nm) => ({ name: nm, alpha: 1 }));
  } else {
    // If user had an existing watchlist from another payload, drop missing layers.
    rtWatch = rtWatch.filter((w) => avail.has(String(w.name)));
    if (!rtWatch.length && rtMeta.layers.length) {
      rtWatch = [{ name: rtMeta.layers[0].name, alpha: 1 }];
    }
  }

  // Histogram selection: validate or choose a sensible default.
  if (rtHistLayer && !avail.has(String(rtHistLayer))) rtHistLayer = "";
  if (!rtHistLayer) {
    const preferHist = ["damage", "molecule_atp", "molecule_glucose", "cell"];
    rtHistLayer = preferHist.find((p) => avail.has(p)) || "";
  }
  if (ui.rtHistLayer) ui.rtHistLayer.value = String(rtHistLayer || "");
  try {
    localStorage.setItem(RT_HIST_LAYER_KEY, String(rtHistLayer || ""));
  } catch {}

  if (ui.rtVizGrid) ui.rtVizGrid.innerHTML = "";
  rtCanvases.clear();
  for (const w of rtWatch) _rtEnsureVizItem(w.name);
  _rtRenderWatchList();
  _rtEnsureCanvasSizes();
  _rtSetStatus(`Ready. H=${rtMeta.H} W=${rtMeta.W}`);

  try {
    await _rtCaptureBaseline();
  } catch (e) {
    _rtSetStatus(String(e?.message || e));
  }

  // Draw initial frame (no step) so canvases are not blank on load.
  try {
    const frame = await _rtPostJson("/api/runtime/frame", { layers: _rtGetRequestedLayerNames() });
    _rtApplyFrame(frame);
  } catch (e) {
    _rtSetStatus(String(e?.message || e));
  }
}

async function _rtStepOnce() {
  if (!rtLoaded) {
    _rtSetStatus("Load a gridstate first");
    return;
  }
  const layers = _rtGetRequestedLayerNames();
  const frame = await _rtPostJson("/api/runtime/step", { layers });
  _rtApplyFrame(frame);
}

async function _rtDownloadTick() {
  if (!rtLoaded) await _rtEnsureSyncedFromEditor(true, "editor");
  const res = await _rtPostJson("/api/runtime/export", {});
  const payload = res?.payload;
  const tick = Number(res?.tick);
  if (!payload || typeof payload !== "object") throw new Error("export payload missing");
  const t = Number.isFinite(tick) ? Math.max(0, Math.floor(tick)) : 0;
  downloadJsonObject(payload, `gridstate.tick.${t}.json`);
}

function _rtStop() {
  rtRunning = false;
  if (rtTimer) {
    clearTimeout(rtTimer);
    rtTimer = null;
  }
  if (ui.rtStartStopBtn) ui.rtStartStopBtn.textContent = "Start";
}

async function _rtLoopTick() {
  if (!rtRunning) return;
  try {
    await _rtStepOnce();
  } catch (e) {
    _rtSetStatus(String(e?.message || e));
    _rtStop();
    return;
  }
  const ms = _rtIntervalValueMs();
  rtTimer = setTimeout(_rtLoopTick, ms);
}

function _rtStart() {
  if (!rtLoaded) {
    _rtSetStatus("Load a gridstate first");
    return;
  }
  if (rtRunning) return;
  rtRunning = true;
  if (ui.rtStartStopBtn) ui.rtStartStopBtn.textContent = "Stop";
  rtTimer = setTimeout(_rtLoopTick, 0);
}

function _rtToggleRun() {
  if (rtRunning) _rtStop();
  else _rtStart();
}

let evoRunning = false;
let evoTimer = null;
let evoJobId = "";
let evoLastStatus = null;

function _evoUpdateAlgoUi() {
  const algo = String(ui.evoAlgo?.value || "cem_delta");
  if (ui.evoAffineParams) ui.evoAffineParams.style.display = algo === "affine" ? "block" : "none";
  if (ui.evoCemParams) ui.evoCemParams.style.display = algo === "cem_delta" ? "block" : "none";
}

function _evoConfigFromUi() {
  const algo = String(ui.evoAlgo?.value || "cem_delta");
  const variants = Math.max(1, Math.floor(Number(ui.evoVariants?.value ?? 75)));
  const ticks = Math.max(1, Math.floor(Number(ui.evoTicks?.value ?? 100)));
  const generations = Math.max(1, Math.floor(Number(ui.evoGenerations?.value ?? 50)));
  const elites = Math.max(1, Math.floor(Number(ui.evoElites?.value ?? 5)));
  const replicates = Math.max(1, Math.floor(Number(ui.evoReplicates?.value ?? 1)));
  const workers = Math.max(1, Math.floor(Number(ui.evoWorkers?.value ?? 35)));
  const seed = Math.floor(Number(ui.evoSeed?.value ?? 1));
  const mutationRate = Number(ui.evoMutationRate?.value ?? 0.15);
  const sigmaScale = Number(ui.evoSigmaScale?.value ?? 0.25);
  const sigmaBias = Number(ui.evoSigmaBias?.value ?? 0.25);
  const cemSigma = Number(ui.evoCemSigma?.value ?? 0.5);
  const cemAlpha = Number(ui.evoCemAlpha?.value ?? 0.7);
  const cemSigmaFloor = Number(ui.evoCemSigmaFloor?.value ?? 0.05);
  const cemMask = String(ui.evoCemMask?.value || "cell");
  const huge = Number(ui.evoHuge?.value ?? 1e9);

  const fitnessWeights = {};
  
  // Ensure all available measurements have weights
  const measurementWeights = {};
  const measurementAggs = {};
  
  if (evoAvailableMeasurements && evoAvailableMeasurements.length > 0) {
    evoAvailableMeasurements.forEach(meas => {
      // Use weight from UI if available, otherwise default to 1.0
      measurementWeights[meas.name] = evoMeasurementWeights[meas.name] !== undefined 
        ? evoMeasurementWeights[meas.name] 
        : 1.0;

      const agg = evoMeasurementAggs?.[meas.name];
      if (agg && agg !== "last") {
        measurementAggs[meas.name] = String(agg);
      }
    });
  } else if (Object.keys(evoMeasurementWeights).length > 0) {
    // No measurements retrieved yet, but we have weights
    Object.assign(measurementWeights, evoMeasurementWeights);

    // Include any non-default aggregations we already have.
    for (const [k, v] of Object.entries(evoMeasurementAggs || {})) {
      if (v && String(v) !== "last") {
        measurementAggs[String(k)] = String(v);
      }
    }
  }
  
  fitnessWeights.measurements = measurementWeights;
  if (Object.keys(measurementAggs).length > 0) {
    fitnessWeights.measurement_aggs = measurementAggs;
  }

  const out = {
    algo,
    variants,
    ticks,
    generations,
    elites,
    replicates,
    workers,
    seed: Number.isFinite(seed) ? seed : 1,
    huge: Number.isFinite(huge) && huge > 0 ? huge : 1e9,
    fitness_weights: fitnessWeights,
  };

  if (algo === "affine") {
    out.mutation_rate = Number.isFinite(mutationRate) ? mutationRate : 0.15;
    out.sigma_scale = Number.isFinite(sigmaScale) ? sigmaScale : 0.25;
    out.sigma_bias = Number.isFinite(sigmaBias) ? sigmaBias : 0.25;
  }
  if (algo === "cem_delta") {
    out.cem_sigma_init = Number.isFinite(cemSigma) ? cemSigma : 0.5;
    out.cem_alpha = Number.isFinite(cemAlpha) ? cemAlpha : 0.7;
    out.cem_sigma_floor = Number.isFinite(cemSigmaFloor) ? cemSigmaFloor : 0.05;
    out.cem_mask = cemMask;
  }
  
  // Include target layers patterns (if specified)
  if (evoTargetPatterns && evoTargetPatterns.length > 0) {
    out.target_layers = [...evoTargetPatterns];
  }
  
  return out;
}

function _evoBasePayloadFromUi() {
  const txt = serializeState(state);
  return JSON.parse(txt);
}

function _applyPayloadToEditor(payloadObj, sourceLabel) {
  try {
    const text = JSON.stringify(payloadObj);
    const parsed = JSON.parse(text);
    state = parseState(text);
    ui.HInput.value = String(state.H);
    ui.WInput.value = String(state.W);
    selectedLayer = state.layers[0]?.name || "";

    // Apply embedded functions from payload (if present)
    _tryApplyEmbeddedMeasurementsConfig(parsed);
    _tryApplyEmbeddedLayerOpsConfig(parsed);

    applyAutoFitZoom();
    syncLayerSelect();
    _setCurrentFile(sourceLabel || "loaded");
    saveToLocalStorage();
  } catch (e) {
    alert(String(e?.message || e));
  }
}

function _evoStopLocal() {
  evoRunning = false;
  if (evoTimer) {
    clearTimeout(evoTimer);
    evoTimer = null;
  }
}

async function _evoStart() {
  const payload = _evoBasePayloadFromUi();
  const config = _evoConfigFromUi();
  
  // Debug: log the weights being sent to backend
  console.log("DEBUG: Starting evolution with config:", config);
  console.log("DEBUG: Measurement weights being sent:", config.fitness_weights?.measurements);
  
  _evoSetStatus("Starting…");
  const res = await _rtPostJson("/api/evolution/start", { payload, config });
  evoJobId = String(res.job_id || "");
  evoRunning = true;
  await _evoPollOnce();
}

async function _evoStop() {
  _evoStopLocal();
  try {
    await _rtPostJson("/api/evolution/stop", {});
  } catch {
  }
  _evoSetStatus("Stopping…");
}

function _evoDrawPlot(st) {
  if (!ui.evoCanvas) return;
  const p = _stepsPrepPlotCanvas(ui.evoCanvas, 900, 240);
  if (!p) return;
  const { ctx, W, H } = p;
  ctx.clearRect(0, 0, W, H);

  const history = st?.history || {};
  const series = st?.series || {};
  const baseline = st?.baseline || {};
  const baseFit = Number(baseline?.fitness);

  const sFitness = Array.isArray(series?.fitness) ? series.fitness : [];
  const sBest = Array.isArray(series?.best) ? series.best : [];
  const sMean = Array.isArray(series?.mean) ? series.mean : [];
  const sOff = Number(series?.offset) || 0;

  const hBest = Array.isArray(history?.best) ? history.best : [];
  const hMean = Array.isArray(history?.mean) ? history.mean : [];
  const hP10 = Array.isArray(history?.p10) ? history.p10 : [];
  const hP90 = Array.isArray(history?.p90) ? history.p90 : [];

  const useSeries = sFitness.length || sBest.length || sMean.length;

  const best = useSeries ? sBest : hBest;
  const mean = useSeries ? sMean : hMean;
  const p10 = useSeries ? [] : hP10;
  const p90 = useSeries ? [] : hP90;

  const n = Math.max(best.length, mean.length, p10.length, p90.length);
  if (!n) {
    ctx.fillStyle = "rgba(255,255,255,.65)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("no evolution data yet", 10, 18);
    return;
  }

  let mn = Infinity;
  let mx = -Infinity;
  const scan = [best, mean, p10, p90];
  if (Number.isFinite(baseFit)) scan.push([baseFit]);
  if (useSeries && sFitness.length) scan.push(sFitness);
  for (const s of scan) {
    for (const v of s) {
      const vv = Number(v);
      if (!Number.isFinite(vv)) continue;
      if (vv < mn) mn = vv;
      if (vv > mx) mx = vv;
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
    mn = 0;
    mx = 1;
  }
  if (mx - mn < 1e-6) {
    mx = mn + 1;
  }

  const padL = 40;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  ctx.strokeStyle = "rgba(255,255,255,.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(padL, padT, plotW, plotH);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.font = "11px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
  const xLabel = useSeries ? "eval" : "generation";
  ctx.fillText(xLabel, padL + plotW - (useSeries ? 28 : 64), padT + plotH + 16);
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("fitness", 0, 0);
  ctx.restore();

  ctx.fillStyle = "rgba(255,255,255,.35)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText(_stepsFmt(mx), padL + 6, padT + 10);
  ctx.fillText(_stepsFmt(mn), padL + 6, padT + plotH);
  ctx.fillText(useSeries ? String(sOff) : "0", padL, padT + plotH + 16);
  ctx.fillText(useSeries ? String(sOff + Math.max(0, n - 1)) : String(Math.max(0, n - 1)), padL + plotW - 18, padT + plotH + 16);

  function yOf(v) {
    const t = (Number(v) - mn) / (mx - mn);
    const u = Math.max(0, Math.min(1, t));
    return padT + (1 - u) * plotH;
  }
  function xOf(i) {
    return padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  }
  function drawLine(arr, color, lw) {
    if (!arr || !arr.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    const m = Math.min(n, arr.length);
    for (let i = 0; i < m; i++) {
      const v = Number(arr[i]);
      if (!Number.isFinite(v)) continue;
      const x = xOf(i);
      const y = yOf(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawHLine(y, color, dash) {
    if (!Number.isFinite(y)) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash(Array.isArray(dash) ? dash : []);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (p10.length && p90.length) {
    const m = Math.min(n, p10.length, p90.length);
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.beginPath();
    for (let i = 0; i < m; i++) {
      const v = Number(p90[i]);
      if (!Number.isFinite(v)) continue;
      const x = xOf(i);
      const y = yOf(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = m - 1; i >= 0; i--) {
      const v = Number(p10[i]);
      if (!Number.isFinite(v)) continue;
      const x = xOf(i);
      const y = yOf(v);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  if (Number.isFinite(baseFit)) {
    drawHLine(yOf(baseFit), "rgba(255,255,255,.30)", [4, 3]);
  }

  if (useSeries && sFitness.length) {
    drawLine(sFitness, "rgba(255,255,255,.18)", 1);
  }

  drawLine(mean, "rgba(255,255,255,.65)", 2);
  drawLine(best, "#0a84ff", 2);
  drawLine(p10, "rgba(255,255,255,.25)", 1);
  drawLine(p90, "rgba(255,255,255,.25)", 1);

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  const curBest = Number(best[best.length - 1] ?? NaN);
  const curMean = Number(mean[mean.length - 1] ?? NaN);
  const dBest = Number.isFinite(baseFit) && Number.isFinite(curBest) ? curBest - baseFit : NaN;
  const baseTxt = Number.isFinite(baseFit) ? `  base=${_stepsFmt(baseFit)}` : "";
  const dTxt = Number.isFinite(dBest) ? `  Δbest=${_stepsFmt(dBest)}` : "";
  ctx.fillText(`best=${_stepsFmt(curBest)}  mean=${_stepsFmt(curMean)}${baseTxt}${dTxt}`, padL + 6, padT + 14);
}

function _evoRenderTop(top) {
  if (!ui.evoTopList) return;
  const rows = Array.isArray(top) ? top : [];

  // Use all available measurements for table columns, not just ones with weights
  // First collect all available measurement names
  let allMeasNames = new Set();
  
  // Add names from available measurements list
  if (evoAvailableMeasurements && evoAvailableMeasurements.length > 0) {
    evoAvailableMeasurements.forEach(m => allMeasNames.add(m.name));
  }
  
  // Add names from weight configuration
  const measWeights = evoMeasurementWeights || {};
  Object.keys(measWeights).forEach(name => allMeasNames.add(name));
  
  // Add names from candidate metrics if any
  rows.forEach(r => {
    const mm = r?.metrics?.measurements;
    if (mm && typeof mm === "object") {
      Object.keys(mm).forEach(name => allMeasNames.add(name));
    }
  });
  
  // Convert to array of measurement column objects with weights
  const measCols = Array.from(allMeasNames)
    .map(name => ({
      name,
      w: measWeights[name] || 1.0 // Default weight 1.0 if not specified
    }))
    .sort((a, b) => Math.abs(b.w) - Math.abs(a.w)); // Sort by absolute weight value

  const body = rows
    .map((r, idx) => {
      const id = String(r?.id || "");
      const fit = r?.fitness;
      const m = r?.metrics || {};
      const alive = m?.alive;
      const div = m?.divisions;
      const sd = m?.starvation_deaths;
      const dd = m?.damage_deaths;
      const mm = m?.measurements && typeof m.measurements === "object" ? m.measurements : null;
      const gen = r?.gen;

      const measTds = measCols
        .map((c) => {
          const v = mm && Object.prototype.hasOwnProperty.call(mm, c.name) ? mm[c.name] : null;
          const vv = typeof v === "number" && Number.isFinite(v) ? v : null;
          return `<td>${vv == null ? "–" : _stepsFmt(Number(vv))}</td>`;
        })
        .join("");
      return `<tr>
        <td>${idx + 1}</td>
        <td class="mono">${id.slice(0, 8)}</td>
        <td>${gen == null ? "–" : String(gen)}</td>
        <td>${fit == null ? "–" : _stepsFmt(Number(fit))}</td>
        ${measTds}
        <td style="text-align:right; white-space:nowrap;">
          <button class="btn btn--secondary btn--tiny" data-evo-act="load" data-evo-id="${id}">Load</button>
          <button class="btn btn--secondary btn--tiny" data-evo-act="download" data-evo-id="${id}">Download</button>
        </td>
      </tr>`;
    })
    .join("");

  const measTh = measCols
    .map((c) => `<th title="w=${String(c.w)}">${c.name}</th>`)
    .join("");
  ui.evoTopList.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>id</th>
          <th>gen</th>
          <th>fitness</th>
          ${measTh}
          <th></th>
        </tr>
      </thead>
      <tbody>${body || ""}</tbody>
    </table>
  `;
}

function _evoApplyStatus(st) {
  evoLastStatus = st;
  const prog = st?.progress || {};
  const gen = prog?.generation;
  const v = prog?.variant;
  const done = prog?.evaluations_done;
  const total = prog?.evaluations_total;
  const running = !!st?.running;
  const err = st?.error;
  const baseFit = Number(st?.baseline?.fitness);
  const curBest = Number(st?.series?.best?.[st?.series?.best?.length - 1] ?? st?.history?.best?.[st?.history?.best?.length - 1] ?? NaN);
  const pct = total ? Math.max(0, Math.min(100, (100 * Number(done || 0)) / Number(total))) : 0;
  
  // Debug log for baseline and top candidate fitness
  if (st?.baseline) {
    console.log("Baseline fitness:", baseFit);
    console.log("Baseline metrics:", st.baseline.metrics);
  }
  if (st?.top && st.top.length > 0) {
    console.log("Top candidate fitness:", st.top[0].fitness);
    console.log("Top candidate metrics:", st.top[0].metrics);
  }
  const baseTxt = Number.isFinite(baseFit) ? `  base=${_stepsFmt(baseFit)}` : "";
  const bestTxt = Number.isFinite(curBest) ? `  best=${_stepsFmt(curBest)}` : "";
  const msg = err
    ? `error: ${err}`
    : running
      ? `running gen=${gen} variant=${v}  eval=${done}/${total} (${_stepsFmt(pct)}%)${bestTxt}${baseTxt}`
      : "idle";
  _evoSetStatus(msg);
  _evoDrawPlot(st || {});
  _evoRenderTop(st?.top || []);
}

async function _evoPollOnce() {
  const st = await _rtPostJson("/api/evolution/status", {});
  _evoApplyStatus(st);
  evoRunning = !!st?.running;
  if (evoRunning) {
    evoTimer = setTimeout(_evoPollOnce, 400);
  } else {
    _evoStopLocal();
  }
}

function _evoRenderNow() {
  if (evoRunning) return;
  if (evoLastStatus) {
    _evoApplyStatus(evoLastStatus);
    return;
  }
  _evoSetStatus("ready");
  _evoDrawPlot({});
  _evoRenderTop([]);
}

function _rtRenderNow() {
  _rtPopulateLayerSelect();
  _rtRenderWatchList();
  _rtEnsureCanvasSizes();
  _rtRenderOverlay();
  _rtRenderHeatmaps();
  _rtRenderHistogram();
}

function _rtSetVizCols(n) {
  const k = Math.max(2, Math.min(4, Math.floor(Number(n) || 2)));
  if (ui.rtVizGrid) ui.rtVizGrid.style.gridTemplateColumns = `repeat(${k}, minmax(0, 1fr))`;
  if (ui.rtVizCols) ui.rtVizCols.value = String(k);
  try {
    localStorage.setItem(RT_VIZ_COLS_KEY, String(k));
  } catch {}
}

function _rtInitVizCols() {
  let v = 2;
  try {
    const raw = localStorage.getItem(RT_VIZ_COLS_KEY);
    if (raw != null) v = Math.floor(Number(raw) || 2);
  } catch {}
  _rtSetVizCols(v);
}

function _closeHelpModal() {
  if (!ui.helpModal) return;
  ui.helpModal.classList.remove("modal--open");
  ui.helpModal.setAttribute("aria-hidden", "true");
}

let pathwayModalInputs = [];
let pathwayModalOutputs = [];

function _openPathwayModal() {
  if (!ui.pathwayModal) return;
  
  pathwayModalInputs = [];
  pathwayModalOutputs = [];
  
  if (ui.pathwayName) ui.pathwayName.value = "glycolysis";
  if (ui.pathwayNumEnzymes) ui.pathwayNumEnzymes.value = "3";
  if (ui.pathwayCellValue) ui.pathwayCellValue.value = "1";
  if (ui.pathwayEfficiency) ui.pathwayEfficiency.value = "1.0";
  
  _populatePathwayLayerDropdowns();
  _updatePathwaySelectedItems();
  
  ui.pathwayModal.classList.add("modal--open");
  ui.pathwayModal.setAttribute("aria-hidden", "false");
}

function _closePathwayModal() {
  if (!ui.pathwayModal) return;
  ui.pathwayModal.classList.remove("modal--open");
  ui.pathwayModal.setAttribute("aria-hidden", "true");
}

let pathwayInputsSearchable = null;
let pathwayOutputsSearchable = null;
let pathwayCellLayerSearchable = null;

function _populatePathwayLayerDropdowns() {
  if (!state || !state.layers) return;
  
  const layers = state.layers.map(l => l.name).sort();
  
  const defaultCellLayer = state?.layers?.some((l) => l.name === "cell")
    ? "cell"
    : state?.layers?.some((l) => l.name === "cell_type")
      ? "cell_type"
      : layers[0] || "cell";
  
  if (ui.pathwayInputsDropdown) {
    ui.pathwayInputsDropdown.innerHTML = "";
    pathwayInputsSearchable = makeSearchableSelect(
      layers,
      "",
      "+ Add input layer...",
      (val) => {
        if (val && layers.includes(val) && !pathwayModalInputs.includes(val)) {
          pathwayModalInputs.push(val);
          _updatePathwaySelectedItems();
          if (pathwayInputsSearchable) pathwayInputsSearchable.input.value = "";
        }
      }
    );
    ui.pathwayInputsDropdown.appendChild(pathwayInputsSearchable.wrapper);
  }
  
  if (ui.pathwayOutputsDropdown) {
    ui.pathwayOutputsDropdown.innerHTML = "";
    pathwayOutputsSearchable = makeSearchableSelect(
      layers,
      "",
      "+ Add output layer...",
      (val) => {
        if (val && layers.includes(val) && !pathwayModalOutputs.includes(val)) {
          pathwayModalOutputs.push(val);
          _updatePathwaySelectedItems();
          if (pathwayOutputsSearchable) pathwayOutputsSearchable.input.value = "";
        }
      }
    );
    ui.pathwayOutputsDropdown.appendChild(pathwayOutputsSearchable.wrapper);
  }
  
  if (ui.pathwayCellLayer) {
    ui.pathwayCellLayer.innerHTML = "";
    pathwayCellLayerSearchable = makeSearchableSelect(
      layers,
      defaultCellLayer,
      "Select cell layer..."
    );
    ui.pathwayCellLayer.appendChild(pathwayCellLayerSearchable.wrapper);
  }
}

function _updatePathwaySelectedItems() {
  if (ui.pathwayInputsSelected) {
    ui.pathwayInputsSelected.innerHTML = "";
    pathwayModalInputs.forEach((name, idx) => {
      const item = document.createElement("div");
      item.className = "pathwayForm__selectedItem";
      item.innerHTML = `${name}<button type="button" data-idx="${idx}" data-type="input">×</button>`;
      ui.pathwayInputsSelected.appendChild(item);
    });
  }
  
  if (ui.pathwayOutputsSelected) {
    ui.pathwayOutputsSelected.innerHTML = "";
    pathwayModalOutputs.forEach((name, idx) => {
      const item = document.createElement("div");
      item.className = "pathwayForm__selectedItem";
      item.innerHTML = `${name}<button type="button" data-idx="${idx}" data-type="output">×</button>`;
      ui.pathwayOutputsSelected.appendChild(item);
    });
  }
}

const palette = [
  [0, 0, 0],
  [76, 175, 80],
  [33, 150, 243],
  [244, 67, 54],
  [156, 39, 176],
  [255, 193, 7],
  [0, 188, 212],
  [255, 152, 0],
];

const DEFAULT_LAYER_COLOR = "#4caf50";

const STORAGE_KEY = "grid_layer_editor_state_v1";
const FUNCTIONS_CFG_KEY = "grid_layer_editor_functions_cfg_v1";
const OPS_GROUPS_COLLAPSED_KEY = "grid_layer_editor_ops_groups_collapsed_v1";
const RT_VIZ_COLS_KEY = "grid_layer_editor_rt_viz_cols_v1";
let dirtySinceLastSave = false;

let currentFileName = "";

function _setCurrentFile(label) {
  currentFileName = String(label || "");
  _updateCurrentFileInfo();
}

function _updateCurrentFileInfo() {
  if (!ui.currentFileInfo) return;
  const name = currentFileName || "untitled";
  const star = dirtySinceLastSave ? " *" : "";
  ui.currentFileInfo.textContent = `${name}${star}`;
}

function _resetAllForNewFile() {
  // Stop any running processes
  _rtStop();
  _evoStopLocal();
  
  // Reset runtime state
  rtLoaded = false;
  rtTick = 0;
  rtMeta = null;
  rtPayloadObj = null;
  rtLastSyncedStateTxt = "";
  rtKinds.clear();
  rtColors.clear();
  rtWatch = [];
  rtCanvases.clear();
  rtLastArrays.clear();
  rtBaseline = null;
  _rtClearEvents();
  _rtClearMeasurements();
  _rtClearScalarHistory();
  rtScalarRows.clear();
  if (ui.rtScalarsList) ui.rtScalarsList.innerHTML = "";
  _rtClearSurvival();
  if (ui.rtVizGrid) ui.rtVizGrid.innerHTML = "";
  if (ui.rtWatchList) ui.rtWatchList.innerHTML = "";
  if (ui.rtStatus) ui.rtStatus.textContent = "";
  
  // Reset evolution state
  evoRunning = false;
  evoJobId = "";
  evoLastStatus = null;
  if (ui.evoStatus) ui.evoStatus.textContent = "";
  if (ui.evoTopList) ui.evoTopList.innerHTML = "";
  if (ui.evoCanvas) {
    const ctx = ui.evoCanvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, ui.evoCanvas.width, ui.evoCanvas.height);
  }
  _evoResetMeasurementWeights();
}

let inspectSummaryLastLayer = "";
let inspectSummaryDirty = true;

let inspectHistMaskLayer = "";
let inspectHistMaskOp = "==";
let inspectHistMaskValue = 1;

const INSPECT_HIST_MASK_LAYER_KEY = "grid_layer_editor_inspect_hist_mask_layer_v1";
const INSPECT_HIST_MASK_OP_KEY = "grid_layer_editor_inspect_hist_mask_op_v1";
const INSPECT_HIST_MASK_VALUE_KEY = "grid_layer_editor_inspect_hist_mask_value_v1";

const bulkSelectedLayers = new Set();

const opTargetsSelected = new Set();

const DEFAULT_MEASUREMENTS = [
  { name: "inflammation", expr: "mean(cytokine, where=(circulation==1))" },
  { name: "glucose_per_circ_cell", expr: "sum(glucose, where=(circulation==1)) / count(where=(circulation==1))" },
  { name: "protein_per_circ_cell", expr: "sum(amino_acids, where=(circulation==1)) / count(where=(circulation==1))" },
  { name: "toxins_per_circ_cell", expr: "sum(toxins, where=(circulation==1)) / count(where=(circulation==1))" },
  { name: "bacterial_infection", expr: "mean(bacterial_antigen, where=(circulation==1))" },
];

const DEFAULT_LAYER_OPS = [];

let fnMeasurements = DEFAULT_MEASUREMENTS.map((x) => ({ ...x }));
let fnLastFocusedExprInput = null;

let layerOps = DEFAULT_LAYER_OPS.map((x) => ({ ...x }));
let opsLastFocusedExprInput = null;
let opsLastFocusedRowIndex = -1;

function _opsInsertAtFocused(newOp) {
  if (opsLastFocusedRowIndex >= 0 && opsLastFocusedRowIndex < layerOps.length) {
    // Insert after the focused row
    layerOps.splice(opsLastFocusedRowIndex + 1, 0, newOp);
    opsLastFocusedRowIndex = opsLastFocusedRowIndex + 1;
  } else {
    // No focus, add at the end
    layerOps.push(newOp);
    opsLastFocusedRowIndex = layerOps.length - 1;
  }
}

const FN_INSERTER_FUNCS = [
  { value: "mean", snippet: "mean(LAYER)" },
  { value: "sum", snippet: "sum(LAYER)" },
  { value: "min", snippet: "min(LAYER)" },
  { value: "max", snippet: "max(LAYER)" },
  { value: "std", snippet: "std(LAYER)" },
  { value: "var", snippet: "var(LAYER)" },
  { value: "median", snippet: "median(LAYER)" },
  { value: "quantile", snippet: "quantile(LAYER, 0.5)" },
  { value: "count", snippet: "count()" },
  { value: "where", snippet: "where=(LAYER==1)" },
];

const OPS_INSERTER_FUNCS = [
  { value: "where", snippet: "where(LAYER==1, LAYER, LAYER)" },
  { value: "clip", snippet: "clip(LAYER, 0, 1)" },
  { value: "abs", snippet: "abs(LAYER)" },
  { value: "sqrt", snippet: "sqrt(LAYER)" },
  { value: "exp", snippet: "exp(LAYER)" },
  { value: "log", snippet: "log(LAYER)" },
  { value: "minimum", snippet: "minimum(LAYER, LAYER)" },
  { value: "maximum", snippet: "maximum(LAYER, LAYER)" },
  { value: "sum_layer", snippet: "sum_layer(LAYER)" },
  { value: "rand_beta", snippet: "rand_beta(alpha, beta)" },
  { value: "rand_logitnorm", snippet: "rand_logitnorm(mu, sigma)" },
  { value: "sum_layers", snippet: "sum_layers(\"*\")" },
  { value: "mean_layers", snippet: "mean_layers(\"*\")" },
  { value: "min_layers", snippet: "min_layers(\"*\")" },
  { value: "max_layers", snippet: "max_layers(\"*\")" },
];
function loadFunctionsCfg() {
  try {
    const raw = localStorage.getItem(FUNCTIONS_CFG_KEY);
    if (!raw) return;
    const o = JSON.parse(raw);
    if (o && typeof o === "object") {
      if (Array.isArray(o.measurements)) {
        fnMeasurements = o.measurements
          .filter((m) => m && typeof m === "object")
          .map((m) => ({ name: String(m.name || ""), expr: String(m.expr || "") }))
          .filter((m) => m.name);
      }
      if (Array.isArray(o.layer_ops)) {
        layerOps = o.layer_ops
          .filter((x) => x && typeof x === "object")
          .map((x) => {
            const rawType = String(x.type || "op").trim();
            const knownTypes = ["let", "foreach", "transport", "diffusion", "divide_cells", "pathway"];
            const type = knownTypes.includes(rawType) ? rawType : "op";
            
            if (type === "foreach") {
              const steps = Array.isArray(x.steps) ? x.steps : [];
              const stepsTextRaw = typeof x.stepsText === "string" ? x.stepsText : "";
              const stepsText = stepsTextRaw.trim().startsWith("for")
                ? stepsTextRaw
                : _forEachStepToR({ match: String(x.match || "*"), steps });
              return {
                type,
                name: String(x.name || ""),
                group: String(x.group || ""),
                enabled: x.enabled !== false,
                match: String(x.match || ""),
                require_match: !!x.require_match,
                steps,
                stepsText,
              };
            }
            
            if (type === "transport" || type === "diffusion") {
              return {
                type,
                name: String(x.name || ""),
                group: String(x.group || ""),
                enabled: x.enabled !== false,
                molecules: x.molecules || "molecule_*",
                molecule_prefix: x.molecule_prefix || "molecule_",
                protein_prefix: x.protein_prefix || "protein_",
                cell_layer: x.cell_layer || "cell",
                cell_mode: x.cell_mode || "eq",
                cell_value: x.cell_value ?? 1,
                dirs: Array.isArray(x.dirs) ? x.dirs : ["north", "south", "east", "west"],
                per_pair_rate: x.per_pair_rate ?? 1.0,
                rate: x.rate ?? 0.2,
                rate_layer: x.rate_layer || null,
                seed: x.seed ?? 0,
              };
            }
            
            if (type === "divide_cells") {
              return {
                type,
                name: String(x.name || ""),
                group: String(x.group || ""),
                enabled: x.enabled !== false,
                cell_layer: x.cell_layer || "cell",
                cell_value: x.cell_value ?? 1,
                empty_value: x.empty_value ?? 0,
                trigger_layer: x.trigger_layer || "protein_divider",
                threshold: x.threshold ?? 50,
                split_fraction: x.split_fraction ?? 0.5,
                max_radius: x.max_radius ?? null,
                layer_prefixes: Array.isArray(x.layer_prefixes) ? x.layer_prefixes : ["molecule", "protein", "rna", "damage", "gene"],
                seed: x.seed ?? 0,
              };
            }
            
            if (type === "pathway") {
              return {
                type,
                name: String(x.name || ""),
                group: String(x.group || ""),
                enabled: x.enabled !== false,
                pathway_name: String(x.pathway_name || x.name || ""),
                inputs: Array.isArray(x.inputs) ? x.inputs : [],
                outputs: Array.isArray(x.outputs) ? x.outputs : [],
                num_enzymes: x.num_enzymes ?? 3,
                cell_layer: x.cell_layer || "cell",
                cell_value: x.cell_value ?? 1,
                efficiency: x.efficiency ?? 1.0,
                seed: x.seed ?? 0,
              };
            }
            
            const out = {
              type,
              name: String(x.name || ""),
              group: String(x.group || ""),
              enabled: x.enabled !== false,
              expr: String(x.expr || ""),
            };
            if (type === "let") return { ...out, var: String(x.var || "") };
            return { ...out, target: String(x.target || "") };
          })
          .filter((x) => {
            if (x.type === "foreach") return x.match && Array.isArray(x.steps);
            if (x.type === "transport" || x.type === "diffusion") return true;
            if (x.type === "divide_cells") return true;
            if (x.type === "pathway") return x.pathway_name && Array.isArray(x.inputs) && x.inputs.length > 0;
            return x.expr && (x.type === "let" ? x.var : x.target);
          });
      }
      if (!fnMeasurements.length) fnMeasurements = DEFAULT_MEASUREMENTS.map((x) => ({ ...x }));
      if (!layerOps.length) layerOps = DEFAULT_LAYER_OPS.map((x) => ({ ...x }));
    }
  } catch {
    // ignore
  }
}

function updateAssignOpUi() {
  if (!ui.opType) return;
  const opType = String(ui.opType.value || "");

  const valueRow = ui.opValue ? ui.opValue.closest(".row") : null;
  const minRow = ui.opMin ? ui.opMin.closest(".row") : null;
  const seedRow = ui.opSeed ? ui.opSeed.closest(".row") : null;

  const showValue = opType === "set_constant";
  const showMinMax = opType === "set_random_uniform" || opType === "add_random_uniform";
  const showSeed = showMinMax;

  if (valueRow) valueRow.style.display = showValue ? "" : "none";
  if (minRow) minRow.style.display = showMinMax ? "" : "none";
  if (seedRow) seedRow.style.display = showSeed ? "" : "none";

  if (isRandomAssignOpType(opType)) {
    let changed = false;
    for (const nm of [...opTargetsSelected]) {
      const meta = state.layers.find((l) => l.name === nm);
      if (meta && meta.kind === "categorical") {
        opTargetsSelected.delete(nm);
        changed = true;
      }
    }
    if (changed) {
      // fall through to redraw
    }
  }

  renderOpTargetsList();
  updateMaskedOpsPreview();
}

function isRandomAssignOpType(opType) {
  const t = String(opType || "");
  return t === "set_random_uniform" || t === "add_random_uniform";
}

function saveFunctionsCfg() {
  try {
    localStorage.setItem(
      FUNCTIONS_CFG_KEY,
      JSON.stringify({
        measurements: fnMeasurements,
        layer_ops: layerOps,
      })
    );
  } catch {
    // ignore
  }
}

function buildFunctionsConfigJson() {
  return {
    version: 3,
    measurements: fnMeasurements
      .map((m) => ({ name: String(m.name || "").trim(), expr: String(m.expr || "") }))
      .filter((m) => m.name && m.expr),
  };
}

function buildLayerOpsConfigJson() {
  return {
    version: 2,
    steps: layerOps
      .filter((x) => x && typeof x === "object")
      .map((x) => {
        const rawType = String(x.type || "op").trim();
        const type =
          rawType === "let"
            ? "let"
            : rawType === "foreach"
              ? "foreach"
              : rawType === "transport"
                ? "transport"
                : rawType === "diffusion"
                  ? "diffusion"
                  : rawType === "divide_cells"
                    ? "divide_cells"
                    : "op";
        if (type === "foreach") {
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            match: String(x.match || "").trim(),
            require_match: !!x.require_match,
            steps: Array.isArray(x.steps) ? x.steps : [],
          };
        }
        if (type === "transport") {
          const dirs = Array.isArray(x.dirs) ? x.dirs.map((d) => String(d || "").trim()).filter((d) => d) : null;
          const seed = x.seed == null || String(x.seed).trim() === "" ? null : Math.floor(Number(x.seed));
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            molecules: typeof x.molecules === "string" ? String(x.molecules || "").trim() : x.molecules,
            molecule_prefix: String(x.molecule_prefix ?? "molecule_"),
            protein_prefix: String(x.protein_prefix ?? "protein_"),
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_mode: String(x.cell_mode ?? "eq"),
            cell_value: Number(x.cell_value ?? 1),
            dirs: dirs && dirs.length ? dirs : ["north", "south", "east", "west"],
            per_pair_rate: Number(x.per_pair_rate ?? 1.0),
            seed: seed,
          };
        }
        if (type === "diffusion") {
          const seed = x.seed == null || String(x.seed).trim() === "" ? null : Math.floor(Number(x.seed));
          const rateLayer = x.rate_layer == null ? null : String(x.rate_layer || "").trim();
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            molecules: typeof x.molecules === "string" ? String(x.molecules || "").trim() : x.molecules,
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_mode: String(x.cell_mode ?? "eq"),
            cell_value: Number(x.cell_value ?? 1),
            rate: x.rate == null || String(x.rate).trim() === "" ? null : Number(x.rate),
            rate_layer: rateLayer && rateLayer.length ? rateLayer : null,
            seed: seed,
          };
        }
        if (type === "divide_cells") {
          const seed = x.seed == null || String(x.seed).trim() === "" ? null : Math.floor(Number(x.seed));
          const maxRadius = x.max_radius == null || String(x.max_radius).trim() === "" ? null : Math.floor(Number(x.max_radius));
          const layerPrefixes = Array.isArray(x.layer_prefixes)
            ? x.layer_prefixes
                .map((p) => String(p || "").trim())
                .filter((p) => p)
            : null;
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_value: Number(x.cell_value ?? 1),
            empty_value: Number(x.empty_value ?? 0),
            trigger_layer: String(x.trigger_layer ?? "protein_divider"),
            threshold: x.threshold == null || String(x.threshold).trim() === "" ? 50 : Number(x.threshold),
            split_fraction: x.split_fraction == null || String(x.split_fraction).trim() === "" ? 0.5 : Number(x.split_fraction),
            max_radius: maxRadius,
            layer_prefixes: layerPrefixes && layerPrefixes.length ? layerPrefixes : ["molecule", "protein", "rna", "damage", "gene"],
            seed: seed,
          };
        }
        if (rawType === "pathway") {
          const seed = x.seed == null || String(x.seed).trim() === "" ? null : Math.floor(Number(x.seed));
          return {
            type: "pathway",
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            pathway_name: String(x.pathway_name || x.name || "").trim(),
            inputs: Array.isArray(x.inputs) ? x.inputs.map((s) => String(s || "").trim()).filter((s) => s) : [],
            outputs: Array.isArray(x.outputs) ? x.outputs.map((s) => String(s || "").trim()).filter((s) => s) : [],
            num_enzymes: Number(x.num_enzymes ?? 3),
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_value: Number(x.cell_value ?? 1),
            efficiency: Number(x.efficiency ?? 1.0),
            seed: seed,
          };
        }
        const base = {
          type,
          name: String(x.name || "").trim(),
          group: String(x.group || "").trim(),
          enabled: x.enabled !== false,
          expr: String(x.expr || ""),
        };
        if (type === "let") return { ...base, var: String(x.var || "").trim() };
        return { ...base, target: String(x.target || "").trim() };
      })
      .filter((x) => {
        if (x.type === "foreach") return x.match && Array.isArray(x.steps) && x.steps.length;
        if (x.type === "transport") return !!x.molecules;
        if (x.type === "diffusion") return !!x.molecules && (x.rate != null || x.rate_layer);
        if (x.type === "divide_cells") return !!x.cell_layer && !!x.trigger_layer;
        if (x.type === "pathway") return !!x.pathway_name && Array.isArray(x.inputs) && x.inputs.length > 0;
        return x.expr && (x.type === "let" ? x.var : x.target);
      }),
  };
}

function _parseLayerOpsConfigObject(o) {
  if (!o || typeof o !== "object") return null;
  const v = Number(o.version);
  if (v === 1) {
    if (!Array.isArray(o.ops)) return null;
    const next = o.ops
      .filter((x) => x && typeof x === "object")
      .map((x) => ({
        type: "op",
        name: String(x.name || "").trim(),
        group: String(x.group || "").trim(),
        enabled: x.enabled !== false,
        target: String(x.target || "").trim(),
        expr: String(x.expr || ""),
      }))
      .filter((x) => x.target && x.expr);
    return next;
  }

  if (v === 2) {
    if (!Array.isArray(o.steps)) return null;
    const next = o.steps
      .filter((x) => x && typeof x === "object")
      .map((x) => {
        const rawType = String(x.type || "op").trim();
        const knownTypes = ["let", "foreach", "transport", "diffusion", "divide_cells", "pathway"];
        const type = knownTypes.includes(rawType) ? rawType : "op";
        if (type === "foreach") {
          const steps = Array.isArray(x.steps) ? x.steps : [];
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            match: String(x.match || "").trim(),
            require_match: !!x.require_match,
            steps,
            stepsText: _forEachStepToR({ match: String(x.match || "*").trim() || "*", steps }),
          };
        }
        if (type === "transport") {
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            molecules: typeof x.molecules === "string" ? String(x.molecules || "").trim() : x.molecules,
            molecule_prefix: String(x.molecule_prefix ?? "molecule_"),
            protein_prefix: String(x.protein_prefix ?? "protein_"),
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_mode: String(x.cell_mode ?? "eq"),
            cell_value: Number(x.cell_value ?? 1),
            dirs: Array.isArray(x.dirs) ? x.dirs.map((d) => String(d || "").trim()).filter((d) => d) : ["north", "south", "east", "west"],
            per_pair_rate: Number(x.per_pair_rate ?? 1.0),
            seed: x.seed == null ? null : Math.floor(Number(x.seed)),
          };
        }
        if (type === "diffusion") {
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            molecules: typeof x.molecules === "string" ? String(x.molecules || "").trim() : x.molecules,
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_mode: String(x.cell_mode ?? "eq"),
            cell_value: Number(x.cell_value ?? 1),
            rate: x.rate == null ? null : Number(x.rate),
            rate_layer: x.rate_layer == null ? null : String(x.rate_layer || "").trim(),
            seed: x.seed == null ? null : Math.floor(Number(x.seed)),
          };
        }
        if (type === "divide_cells") {
          const layerPrefixes = Array.isArray(x.layer_prefixes)
            ? x.layer_prefixes
                .map((p) => String(p || "").trim())
                .filter((p) => p)
            : null;
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_value: Number(x.cell_value ?? 1),
            empty_value: Number(x.empty_value ?? 0),
            trigger_layer: String(x.trigger_layer ?? "protein_divider"),
            threshold: x.threshold == null ? 50 : Number(x.threshold),
            split_fraction: x.split_fraction == null ? 0.5 : Number(x.split_fraction),
            max_radius: x.max_radius == null ? null : Math.floor(Number(x.max_radius)),
            layer_prefixes: layerPrefixes && layerPrefixes.length ? layerPrefixes : ["molecule", "protein", "rna", "damage", "gene"],
            seed: x.seed == null ? null : Math.floor(Number(x.seed)),
          };
        }
        if (type === "pathway") {
          return {
            type,
            name: String(x.name || "").trim(),
            group: String(x.group || "").trim(),
            enabled: x.enabled !== false,
            pathway_name: String(x.pathway_name || x.name || "").trim(),
            inputs: Array.isArray(x.inputs) ? x.inputs.map((s) => String(s || "").trim()).filter((s) => s) : [],
            outputs: Array.isArray(x.outputs) ? x.outputs.map((s) => String(s || "").trim()).filter((s) => s) : [],
            num_enzymes: Number(x.num_enzymes ?? 3),
            cell_layer: String(x.cell_layer ?? "cell"),
            cell_value: Number(x.cell_value ?? 1),
            efficiency: Number(x.efficiency ?? 1.0),
            seed: x.seed == null ? null : Math.floor(Number(x.seed)),
          };
        }
        const base = {
          type,
          name: String(x.name || "").trim(),
          group: String(x.group || "").trim(),
          enabled: x.enabled !== false,
          expr: String(x.expr || ""),
        };
        if (type === "let") return { ...base, var: String(x.var || "").trim() };
        return { ...base, target: String(x.target || "").trim() };
      })
      .filter((x) => {
        if (x.type === "foreach") return x.match && Array.isArray(x.steps);
        if (x.type === "transport") return !!x.molecules;
        if (x.type === "diffusion") return !!x.molecules;
        if (x.type === "divide_cells") return !!x.cell_layer && !!x.trigger_layer;
        if (x.type === "pathway") return !!x.pathway_name && Array.isArray(x.inputs) && x.inputs.length > 0;
        return x.expr && (x.type === "let" ? x.var : x.target);
      });
    return next;
  }

  return null;
}

function _isValidIdentifier(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(String(name || ""));
}

function _substTemplateStepsText(s, matchObj) {
  return String(s || "").replace(/\$\{(\d+)\}/g, (_, idxStr) => {
    const idx = Number(idxStr);
    const g = matchObj && matchObj[idx];
    return g == null ? "" : String(g);
  });
}

if (ui.opsDupGroupBtn) {
  ui.opsDupGroupBtn.addEventListener("click", () => {
    const from = String(ui.opsDupGroupFrom?.value || "").trim();
    const to = String(ui.opsDupGroupTo?.value || "").trim();
    if (!from) {
      alert("Pick a source group to duplicate");
      return;
    }
    if (!to) {
      alert("Enter a new group name");
      return;
    }
    if (to === from) {
      alert("New group name must be different from source group");
      return;
    }
    const existingGroups = new Set(layerOps.map((s) => String(s?.group || "").trim()).filter((g) => g));
    if (existingGroups.has(to)) {
      if (!confirm(`Group '${to}' already exists. Duplicate into it anyway?`)) return;
    }

    const indices = [];
    for (let i = 0; i < layerOps.length; i++) {
      if (String(layerOps[i]?.group || "").trim() === from) indices.push(i);
    }
    if (!indices.length) {
      alert(`No steps found in group '${from}'`);
      return;
    }

    const insertAt = indices[indices.length - 1] + 1;
    const copies = indices.map((idx) => {
      const src = layerOps[idx];
      const dst = JSON.parse(JSON.stringify(src || {}));
      dst.group = to;
      if (typeof dst.name === "string" && dst.name.trim()) dst.name = dst.name;
      return dst;
    });

    layerOps.splice(insertAt, 0, ...copies);
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsAddDivisionBtn) {
  ui.opsAddDivisionBtn.addEventListener("click", () => {
    const layerList = Array.isArray(state?.layers) ? state.layers : [];
    const defaultCellLayer = layerList.some((l) => l?.name === "cell") ? "cell" : layerList[0]?.name || "cell";
    const defaultTriggerLayer = layerList.some((l) => l?.name === "protein_divider") ? "protein_divider" : layerList[0]?.name || "protein_divider";
    _opsInsertAtFocused({
      type: "divide_cells",
      enabled: true,
      name: "",
      group: "",
      cell_layer: defaultCellLayer,
      cell_value: 1,
      empty_value: 0,
      trigger_layer: defaultTriggerLayer,
      threshold: 50,
      split_fraction: 0.5,
      max_radius: null,
      layer_prefixes: ["molecule", "protein", "rna", "damage", "gene"],
      seed: 0,
    });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

function _expandForEachStep(step, layerNames) {
  const raw = String(step?.match || "").trim();
  if (!raw) return { ok: false, text: "Missing match", steps: [] };
  let rx;
  try {
    if (raw.startsWith("glob:")) {
      const glob = raw.slice("glob:".length).trim();
      rx = new RegExp(_globToRegexSourceWithGroups(glob));
    } else if (raw.startsWith("re:") || raw.startsWith("regex:")) {
      const reSrc = raw.startsWith("re:") ? raw.slice(3).trim() : raw.slice("regex:".length).trim();
      rx = new RegExp(reSrc);
    } else if (_isGlobLikePattern(raw)) {
      rx = new RegExp(_globToRegexSourceWithGroups(raw));
    } else {
      rx = new RegExp(raw);
    }
  } catch {
    return { ok: false, text: "Bad pattern", steps: [] };
  }
  const templateSteps = Array.isArray(step?.steps) ? step.steps : [];
  if (!templateSteps.length) return { ok: false, text: "Missing steps", steps: [] };

  const expanded = [];
  let nLayers = 0;
  for (const nm of layerNames) {
    const m = rx.exec(nm);
    if (!m) continue;
    nLayers++;
    for (const t of templateSteps) {
      if (!t || typeof t !== "object") continue;
      const out = { ...t };
      for (const k of ["name", "group", "expr", "target", "var"]) {
        if (typeof out[k] === "string") out[k] = _substTemplateStepsText(out[k], m);
      }
      if (out.enabled == null) out.enabled = true;
      if (!out.group && step.group) out.group = step.group;
      expanded.push(out);
    }
  }
  if (!nLayers && step.require_match) return { ok: false, text: "No matches", steps: [] };
  return { ok: true, text: `expands: ${nLayers} layer(s), ${expanded.length} step(s)`, steps: expanded };
}

function validateLayerOpStep(step, knownVars) {
  const layerList = Array.isArray(state?.layers) ? state.layers : [];
  const rawType = String(step?.type || "op").trim();
  const type =
    rawType === "let"
      ? "let"
      : rawType === "foreach"
        ? "foreach"
        : rawType === "transport"
          ? "transport"
          : rawType === "diffusion"
            ? "diffusion"
            : rawType === "divide_cells"
              ? "divide_cells"
              : "op";

  if (type === "foreach") {
    const layerNames = state?.layers ? state.layers.map((l) => l.name) : [];
    let compiled = null;
    const src = String(step.stepsText || step.expr || "").trim();
    if (src.startsWith("for")) {
      const c = _compileForEachR(src, layerNames);
      if (!c.ok) return { ok: false, text: `foreach: ${c.text}` };
      compiled = { match: c.match, steps: c.steps };
    } else if (!step.match || !Array.isArray(step.steps) || !step.steps.length) {
      return { ok: false, text: "foreach: Missing loop" };
    }
    const ex = _expandForEachStep(compiled ? { ...step, ...compiled } : step, layerNames);
    if (!ex.ok) return { ok: false, text: `foreach: ${ex.text}` };

    let tmpKnown = new Set(Array.from(knownVars || []));
    const N = Math.min(ex.steps.length, 40);
    for (let i = 0; i < N; i++) {
      const s = ex.steps[i];
      const v = validateLayerOpStep(s, tmpKnown);
      if (!v.ok) return { ok: false, text: `foreach: ${v.text}` };
      if (String(s?.type || "op") === "let") {
        const nm = String(s?.var || "").trim();
        if (_isValidIdentifier(nm)) tmpKnown.add(nm);
      }
    }
    return { ok: true, text: `OK (${ex.text}${ex.steps.length > N ? ", sample validated" : ""})` };
  }

  if (type === "transport") {
    const mol = step?.molecules;
    const molOk =
      (typeof mol === "string" && String(mol).trim().length > 0) ||
      (Array.isArray(mol) && mol.some((x) => typeof x === "string" && String(x).trim().length > 0));
    if (!molOk) return { ok: false, text: "transport: Missing molecules" };

    const cellLayer = String(step?.cell_layer || "cell").trim();
    if (!cellLayer) return { ok: false, text: "transport: Missing cell_layer" };
    const meta = layerList.find((l) => l.name === cellLayer);
    if (!meta) return { ok: false, text: `transport: Unknown cell_layer: ${cellLayer}` };

    const cellMode = String(step?.cell_mode || "eq").trim().toLowerCase();
    if (cellMode !== "eq") return { ok: false, text: "transport: cell_mode must be 'eq'" };

    const r = Number(step?.per_pair_rate ?? 1);
    if (!Number.isFinite(r) || r < 0 || r > 1) return { ok: false, text: "transport: per_pair_rate must be in [0,1]" };

    const dirs = Array.isArray(step?.dirs) ? step.dirs : ["north", "south", "east", "west"];
    const allowed = new Set(["north", "south", "east", "west", "n", "s", "e", "w"]);
    for (const d of dirs) {
      const dd = String(d || "").trim().toLowerCase();
      if (!allowed.has(dd)) return { ok: false, text: `transport: bad dir: ${String(d)}` };
    }

    if (step?.seed != null && String(step.seed).trim() !== "") {
      const s = Number(step.seed);
      if (!Number.isFinite(s)) return { ok: false, text: "transport: seed must be a number" };
    }

    return { ok: true, text: "OK" };
  }

  if (type === "divide_cells") {
    const cellLayer = String(step?.cell_layer || "cell").trim();
    if (!cellLayer) return { ok: false, text: "divide_cells: Missing cell_layer" };
    const meta = layerList.find((l) => l.name === cellLayer);
    if (!meta) return { ok: false, text: `divide_cells: Unknown cell_layer: ${cellLayer}` };

    const triggerLayer = String(step?.trigger_layer || "protein_divider").trim();
    if (!triggerLayer) return { ok: false, text: "divide_cells: Missing trigger_layer" };
    const meta2 = layerList.find((l) => l.name === triggerLayer);
    if (!meta2) {
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(triggerLayer)) {
        return { ok: false, text: `divide_cells: Unknown trigger_layer: ${triggerLayer}` };
      }
    }

    const threshold = Number(step?.threshold ?? 50);
    if (!Number.isFinite(threshold)) return { ok: false, text: "divide_cells: threshold must be a number" };

    const f = Number(step?.split_fraction ?? 0.5);
    if (!Number.isFinite(f) || f <= 0 || f >= 1) return { ok: false, text: "divide_cells: split_fraction must be in (0,1)" };

    if (step?.max_radius != null && String(step.max_radius).trim() !== "") {
      const r = Number(step.max_radius);
      if (!Number.isFinite(r) || r < 1) return { ok: false, text: "divide_cells: max_radius must be >= 1" };
    }

    if (step?.seed != null && String(step.seed).trim() !== "") {
      const s = Number(step.seed);
      if (!Number.isFinite(s)) return { ok: false, text: "divide_cells: seed must be a number" };
    }

    const lp = step?.layer_prefixes;
    if (lp != null && !(Array.isArray(lp) && lp.every((x) => String(x || "").trim().length > 0))) {
      return { ok: false, text: "divide_cells: layer_prefixes must be an array of non-empty strings" };
    }

    return { ok: true, text: "OK" };
  }

  if (type === "diffusion") {
    const mol = step?.molecules;
    const molOk =
      (typeof mol === "string" && String(mol).trim().length > 0) ||
      (Array.isArray(mol) && mol.some((x) => typeof x === "string" && String(x).trim().length > 0));
    if (!molOk) return { ok: false, text: "diffusion: Missing molecules" };

    const cellLayer = String(step?.cell_layer || "cell").trim();
    if (!cellLayer) return { ok: false, text: "diffusion: Missing cell_layer" };
    const meta = layerList.find((l) => l.name === cellLayer);
    if (!meta) return { ok: false, text: `diffusion: Unknown cell_layer: ${cellLayer}` };

    const cellMode = String(step?.cell_mode || "eq").trim().toLowerCase();
    if (cellMode !== "eq") return { ok: false, text: "diffusion: cell_mode must be 'eq'" };

    const haveRate = step?.rate != null && String(step.rate).trim() !== "";
    const rateLayer = String(step?.rate_layer || "").trim();
    if (!haveRate && !rateLayer) return { ok: false, text: "diffusion: need rate or rate_layer" };

    if (haveRate) {
      const r = Number(step.rate);
      if (!Number.isFinite(r) || r < 0 || r > 1) return { ok: false, text: "diffusion: rate must be in [0,1]" };
    }
    if (rateLayer) {
      const m2 = layerList.find((l) => l.name === rateLayer);
      if (!m2) return { ok: false, text: `diffusion: Unknown rate_layer: ${rateLayer}` };
    }
    if (step?.seed != null && String(step.seed).trim() !== "") {
      const s = Number(step.seed);
      if (!Number.isFinite(s)) return { ok: false, text: "diffusion: seed must be a number" };
    }

    return { ok: true, text: "OK" };
  }

  if (rawType === "pathway") {
    const pathwayName = String(step?.pathway_name || "").trim();
    if (!pathwayName) return { ok: false, text: "pathway: Missing pathway_name" };

    const inputs = step?.inputs;
    if (!Array.isArray(inputs) || inputs.length === 0) {
      return { ok: false, text: "pathway: Missing inputs" };
    }

    const outputs = step?.outputs;
    if (!Array.isArray(outputs) || outputs.length === 0) {
      return { ok: false, text: "pathway: Missing outputs" };
    }

    const cellLayer = String(step?.cell_layer || "cell").trim();
    if (!cellLayer) return { ok: false, text: "pathway: Missing cell_layer" };
    const meta = layerList.find((l) => l.name === cellLayer);
    if (!meta) return { ok: false, text: `pathway: Unknown cell_layer: ${cellLayer}` };

    const numEnzymes = Number(step?.num_enzymes ?? 3);
    if (!Number.isFinite(numEnzymes) || numEnzymes < 1) {
      return { ok: false, text: "pathway: num_enzymes must be >= 1" };
    }

    const efficiency = Number(step?.efficiency ?? 1.0);
    if (!Number.isFinite(efficiency) || efficiency < 0) {
      return { ok: false, text: "pathway: efficiency must be >= 0" };
    }

    return { ok: true, text: `OK (${inputs.length} in → ${outputs.length} out, ${numEnzymes} enzymes)` };
  }

  const e = String(step?.expr || "").trim();
  if (!e) return { ok: false, text: "Missing expr" };

  if (type === "op") {
    const t = String(step?.target || "").trim();
    if (!t) return { ok: false, text: "Missing target" };
    const meta = layerList.find((l) => l.name === t);
    if (!meta) return { ok: false, text: `Unknown target: ${t}` };
  } else {
    const v = String(step?.var || "").trim();
    if (!v) return { ok: false, text: "Missing var" };
    if (!_isValidIdentifier(v)) return { ok: false, text: "Var must be an identifier" };
    const layerSet = new Set(layerList.map((l) => l.name));
    if (layerSet.has(v)) return { ok: false, text: "Var conflicts with a layer name" };
    if (OPS_ALLOWED_FUNCS.has(v) || v === "True" || v === "False") return { ok: false, text: "Var name not allowed" };
    if (knownVars.has(v)) return { ok: false, text: "Var already defined" };
  }

  const bal = _checkBalancedDelimiters(e);
  if (bal) return { ok: false, text: bal };
  if (_hasDisallowedAttributeAccess(e)) return { ok: false, text: "Attribute access not allowed" };
  if (_hasDisallowedKeywordArgsOrAssignment(e)) return { ok: false, text: "Keyword args / assignment not allowed" };
  const ar = _checkAllowedFuncArity(e);
  if (ar) return { ok: false, text: ar };

  const ids = _tokenizeIdentifiers(e);
  const layerSet = new Set(layerList.map((l) => l.name));
  const unknown = [];
  for (const id of ids) {
    if (OPS_ALLOWED_FUNCS.has(id)) continue;
    if (id === "True" || id === "False") continue;
    if (layerSet.has(id)) continue;
    if (knownVars.has(id)) continue;
    unknown.push(id);
  }
  if (unknown.length) return { ok: false, text: `Unknown (${unknown.length}): ${unknown.join(", ")}` };
  return { ok: true, text: "OK" };
}

function _parseMeasurementsConfigObject(o) {
  if (!o || typeof o !== "object") return null;
  const v = Number(o.version);
  if (v !== 1 && v !== 2 && v !== 3) return null;

  if (v === 3) {
    if (!Array.isArray(o.measurements)) return null;
    const next = o.measurements
      .filter((m) => m && typeof m === "object")
      .map((m) => ({ name: String(m.name || "").trim(), expr: String(m.expr || "") }))
      .filter((m) => m.name && m.expr);
    if (!next.length) return null;
    return next;
  }

  if (v === 2) {
    try {
      const next = _convertV2ToV3Measurements(o);
      return next && next.length ? next : null;
    } catch {
      return null;
    }
  }

  // v1: treat as v2-like with default circulation==1 mask
  try {
    const next = _convertV2ToV3Measurements({
      version: 2,
      mask: { layer: "circulation", op: "==", value: 1, invert: false },
      calculations: (Array.isArray(o.measurements) ? o.measurements : []).map((m) => ({
        output: String(m?.output || "").trim(),
        op: "masked_mean",
        layer: String(m?.layer || ""),
      })),
    });
    return next && next.length ? next : null;
  } catch {
    return null;
  }
}

function _tryApplyEmbeddedMeasurementsConfig(rootObj) {
  const embedded = rootObj && typeof rootObj === "object" ? rootObj.measurements_config : null;
  const next = _parseMeasurementsConfigObject(embedded);
  if (!next) return false;
  fnMeasurements = next;
  saveFunctionsCfg();
  renderFunctionsSpecsTable();
  return true;
}

function _tryApplyEmbeddedLayerOpsConfig(rootObj) {
  const embedded = rootObj && typeof rootObj === "object" ? rootObj.layer_ops_config : null;
  const next = _parseLayerOpsConfigObject(embedded);
  if (!next) return false;
  layerOps = next;
  saveFunctionsCfg();
  renderLayerOpsTable();
  return true;
}

function insertTextIntoInput(input, text) {
  if (!input) return;
  const start = Number(input.selectionStart ?? input.value.length);
  const end = Number(input.selectionEnd ?? input.value.length);
  const v = String(input.value || "");
  input.value = v.slice(0, start) + text + v.slice(end);
  const pos = start + text.length;
  input.selectionStart = pos;
  input.selectionEnd = pos;
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function _maskExprFromV2(mask) {
  const base = `(${String(mask.layer)}${String(mask.op)}${String(mask.value)})`;
  if (mask.invert) return `~${base}`;
  return base;
}

function _convertV2ToV3Measurements(o) {
  const mask = o.mask;
  if (!mask || typeof mask !== "object") throw new Error("Missing mask");
  const whereExpr = _maskExprFromV2({
    layer: String(mask.layer || ""),
    op: String(mask.op || "=="),
    value: Number(mask.value ?? 1),
    invert: !!mask.invert,
  });

  const calcs = Array.isArray(o.calculations) ? o.calculations : [];
  return calcs
    .filter((c) => c && typeof c === "object")
    .map((c) => {
      const name = String(c.output || "").trim();
      const op = String(c.op || "");
      const a = typeof c.layer === "string" ? c.layer : "";
      const b = typeof c.layer_b === "string" ? c.layer_b : "";

      if (!name) return null;
      if (op === "masked_mean") return { name, expr: `mean(${a}, where=${whereExpr})` };
      if (op === "masked_sum") return { name, expr: `sum(${a}, where=${whereExpr})` };
      if (op === "mask_count") return { name, expr: `count(where=${whereExpr})` };
      if (op === "masked_sum_over_count") return { name, expr: `sum(${a}, where=${whereExpr}) / count(where=${whereExpr})` };
      if (op === "masked_sum_ratio") return { name, expr: `sum(${a}, where=${whereExpr}) / sum(${b}, where=${whereExpr})` };
      return { name, expr: "" };
    })
    .filter((x) => x && x.name && x.expr);
}

function _tokenizeIdentifiers(expr) {
  const s = String(expr || "");
  const out = [];
  let tok = "";
  let inSingle = false;
  let inDouble = false;
  let esc = false;

  const flush = () => {
    if (tok) out.push(tok);
    tok = "";
  };

  const isStart = (ch) => /[A-Za-z_]/.test(ch);
  const isPart = (ch) => /[A-Za-z0-9_]/.test(ch);

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];

    if (esc) {
      esc = false;
      continue;
    }

    if (ch === "\\") {
      esc = true;
      continue;
    }

    if (!inDouble && ch === "'") {
      inSingle = !inSingle;
      flush();
      continue;
    }
    if (!inSingle && ch === '"') {
      inDouble = !inDouble;
      flush();
      continue;
    }

    if (inSingle || inDouble) {
      continue;
    }

    if (!tok) {
      if (isStart(ch)) tok = ch;
      continue;
    }

    if (isPart(ch)) {
      tok += ch;
      continue;
    }
    flush();
  }
  flush();
  return out;
}

const FN_ALLOWED_FUNCS = new Set([
  "mean",
  "sum",
  "min",
  "max",
  "std",
  "var",
  "median",
  "quantile",
  "count",
]);

function validateMeasurementExpr(expr) {
  const ids = _tokenizeIdentifiers(expr);
  const layerList = Array.isArray(state?.layers) ? state.layers : [];
  const layerSet = new Set(layerList.map((l) => l.name));
  const unknown = [];
  for (const id of ids) {
    if (FN_ALLOWED_FUNCS.has(id)) continue;
    if (id === "where" || id === "True" || id === "False") continue;
    if (layerSet.has(id)) continue;
    unknown.push(id);
  }
  if (unknown.length) return { ok: false, text: `Unknown (${unknown.length}): ${unknown.join(", ")}` };
  return { ok: true, text: "OK" };
}

function downloadJsonObject(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function renderFunctionsSpecsTable() {
  if (!ui.fnSpecsTable) return;
  const wrap = document.createElement("div");
  wrap.className = "table";
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["Name", "Formula", "Status", ""]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  t.appendChild(thead);

  const tbody = document.createElement("tbody");

  for (let i = 0; i < fnMeasurements.length; i++) {
    const m = fnMeasurements[i];
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    const name = document.createElement("input");
    name.className = "input input--tiny";
    name.value = m.name;
    name.placeholder = "measurement_name";
    name.addEventListener("input", () => {
      fnMeasurements[i].name = name.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
    });
    tdName.appendChild(name);

    const tdExpr = document.createElement("td");
    const expr = document.createElement("textarea");
    expr.className = "input input--tiny input--formula";
    expr.value = m.expr;
    expr.placeholder = "e.g. sum(glucose, where=(circulation==1)) / count(where=(circulation==1))";
    expr.spellcheck = false;
    expr.addEventListener("focus", () => {
      fnLastFocusedExprInput = expr;
    });
    tdExpr.appendChild(expr);

    const tdStatus = document.createElement("td");
    const st = document.createElement("div");
    const updateStatus = () => {
      const v = validateMeasurementExpr(expr.value);
      st.className = v.ok ? "meta" : "meta";
      st.textContent = v.text;
    };
    updateStatus();
    tdStatus.appendChild(st);

    expr.addEventListener("input", () => {
      fnMeasurements[i].expr = expr.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      updateStatus();
    });

    const tdDel = document.createElement("td");
    const del = document.createElement("button");
    del.className = "btn btn--danger btn--tiny";
    del.textContent = "Remove";
    del.addEventListener("click", () => {
      fnMeasurements.splice(i, 1);
      if (!fnMeasurements.length) fnMeasurements = DEFAULT_MEASUREMENTS.map((x) => ({ ...x }));
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderFunctionsSpecsTable();
    });
    tdDel.appendChild(del);

    tr.appendChild(tdName);
    tr.appendChild(tdExpr);
    tr.appendChild(tdStatus);
    tr.appendChild(tdDel);
    tbody.appendChild(tr);
  }

  t.appendChild(tbody);
  wrap.appendChild(t);
  ui.fnSpecsTable.innerHTML = "";
  ui.fnSpecsTable.appendChild(wrap);
}

const OPS_ALLOWED_FUNCS = new Set([
  "where",
  "clip",
  "abs",
  "sqrt",
  "exp",
  "log",
  "minimum",
  "maximum",
  "sum_layer",
  "rand_beta",
  "rand_logitnorm",
  "sum_layers",
  "mean_layers",
  "min_layers",
  "max_layers",
]);

const OPS_FUNC_ARITY = {
  where: 3,
  clip: 3,
  abs: 1,
  sqrt: 1,
  exp: 1,
  log: 1,
  minimum: 2,
  maximum: 2,
  sum_layer: 1,
  rand_beta: [0, 2],
  rand_logitnorm: [0, 2],
  sum_layers: 1,
  mean_layers: 1,
  min_layers: 1,
  max_layers: 1,
};

let __randPlotModal = null;
let __randPlotState = {
  kind: "rand_beta",
  alpha: 1.0,
  beta: 1.0,
  mu: 0.0,
  sigma: 1.0,
  n: 5000,
  bins: 40,
};

function _randPlotEnsureModal() {
  if (__randPlotModal) return __randPlotModal;
  const modal = document.createElement("div");
  modal.id = "randPlotModal";
  modal.className = "modal";
  modal.setAttribute("aria-hidden", "true");

  const overlay = document.createElement("div");
  overlay.className = "modal__overlay";
  const panel = document.createElement("div");
  panel.className = "modal__panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");

  const header = document.createElement("div");
  header.className = "modal__header";
  const title = document.createElement("div");
  title.className = "modal__title";
  title.textContent = "Random distribution";
  const closeBtn = document.createElement("button");
  closeBtn.className = "btn btn--secondary btn--tiny";
  closeBtn.type = "button";
  closeBtn.textContent = "Close";
  header.appendChild(title);
  header.appendChild(closeBtn);

  const body = document.createElement("div");
  body.className = "modal__body";
  const wrap = document.createElement("div");
  wrap.className = "randPlot";

  const top = document.createElement("div");
  top.className = "randPlot__top";
  const kindSel = document.createElement("select");
  kindSel.className = "input input--tiny";
  for (const k of [
    { v: "rand_beta", t: "beta" },
    { v: "rand_logitnorm", t: "logit-normal" },
  ]) {
    const opt = document.createElement("option");
    opt.value = k.v;
    opt.textContent = k.t;
    kindSel.appendChild(opt);
  }
  top.appendChild(kindSel);
  wrap.appendChild(top);

  const params = document.createElement("div");
  params.className = "randPlot__params";

  const mkNum = (lab, step, init) => {
    const row = document.createElement("div");
    row.className = "randPlot__param";
    const l = document.createElement("div");
    l.className = "label";
    l.textContent = lab;
    const inp = document.createElement("input");
    inp.className = "input input--tiny";
    inp.type = "number";
    inp.step = String(step);
    inp.value = String(init);
    row.appendChild(l);
    row.appendChild(inp);
    return { row, inp };
  };

  const a = mkNum("alpha", 0.1, __randPlotState.alpha);
  const b = mkNum("beta", 0.1, __randPlotState.beta);
  const mu = mkNum("mu", 0.1, __randPlotState.mu);
  const sig = mkNum("sigma", 0.1, __randPlotState.sigma);
  const n = mkNum("samples", 100, __randPlotState.n);
  const bins = mkNum("bins", 1, __randPlotState.bins);
  params.appendChild(a.row);
  params.appendChild(b.row);
  params.appendChild(mu.row);
  params.appendChild(sig.row);
  params.appendChild(n.row);
  params.appendChild(bins.row);
  wrap.appendChild(params);

  const viz = document.createElement("div");
  viz.className = "randPlot__viz";
  const canvas = document.createElement("canvas");
  canvas.className = "stepsCanvas randPlot__canvas";
  viz.appendChild(canvas);
  wrap.appendChild(viz);
  body.appendChild(wrap);
  panel.appendChild(header);
  panel.appendChild(body);

  const close = () => {
    modal.classList.remove("modal--open");
    modal.setAttribute("aria-hidden", "true");
  };
  overlay.addEventListener("click", close);
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", (ev) => {
    if (!modal.classList.contains("modal--open")) return;
    if (ev.key === "Escape") close();
  });

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  const randn = () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const randGamma = (k) => {
    const kk = Number(k);
    if (!(kk > 0)) return 0;
    if (kk < 1) {
      const u = Math.random();
      return randGamma(kk + 1) * Math.pow(u, 1 / kk);
    }
    const d = kk - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      const x = randn();
      let v = 1 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };

  const sample = () => {
    const st = __randPlotState;
    const out = new Float32Array(Math.max(1, Math.floor(st.n)));
    if (st.kind === "rand_beta") {
      const aa = Math.max(1e-6, Number(st.alpha));
      const bb = Math.max(1e-6, Number(st.beta));
      for (let i = 0; i < out.length; i++) {
        const x = randGamma(aa);
        const y = randGamma(bb);
        const z = x + y;
        out[i] = z > 0 ? x / z : 0;
      }
      return out;
    }
    const mm = Number(st.mu);
    const ss = Math.max(0, Number(st.sigma));
    for (let i = 0; i < out.length; i++) {
      const z = mm + ss * randn();
      const x = 1 / (1 + Math.exp(-z));
      out[i] = clamp(x, 0, 1);
    }
    return out;
  };

  const draw = () => {
    const st = __randPlotState;
    const ctx = canvas.getContext("2d");
    const w = 760;
    const h = 260;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    ctx.clearRect(0, 0, w, h);
    const binsN = Math.max(5, Math.min(120, Math.floor(Number(st.bins) || 40)));
    const xs = sample();
    const counts = new Float32Array(binsN);
    for (let i = 0; i < xs.length; i++) {
      const x = clamp(xs[i], 0, 1);
      let b = Math.floor(x * binsN);
      if (b >= binsN) b = binsN - 1;
      counts[b] += 1;
    }
    let maxC = 1;
    for (let i = 0; i < binsN; i++) maxC = Math.max(maxC, counts[i]);
    const pad = 20;
    const innerW = w - pad * 2;
    const innerH = h - pad * 2;
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.fillRect(pad, pad, innerW, innerH);
    const bw = innerW / binsN;
    ctx.fillStyle = "rgba(10,132,255,.65)";
    for (let i = 0; i < binsN; i++) {
      const bh = (counts[i] / maxC) * innerH;
      const x0 = pad + i * bw;
      const y0 = pad + innerH - bh;
      ctx.fillRect(x0, y0, Math.max(1, bw - 1), bh);
    }
    ctx.strokeStyle = "rgba(255,255,255,.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, innerW, innerH);
    ctx.fillStyle = "rgba(245,246,247,.62)";
    ctx.font = "12px ui-sans-serif, system-ui";
    const label =
      st.kind === "rand_beta"
        ? `beta(alpha=${Number(st.alpha)}, beta=${Number(st.beta)})`
        : `logit-normal(mu=${Number(st.mu)}, sigma=${Number(st.sigma)})`;
    ctx.fillText(label, pad, 14);
  };

  const syncVis = () => {
    const st = __randPlotState;
    kindSel.value = st.kind;
    a.row.style.display = st.kind === "rand_beta" ? "block" : "none";
    b.row.style.display = st.kind === "rand_beta" ? "block" : "none";
    mu.row.style.display = st.kind === "rand_logitnorm" ? "block" : "none";
    sig.row.style.display = st.kind === "rand_logitnorm" ? "block" : "none";
  };

  const update = () => {
    __randPlotState.kind = String(kindSel.value || "rand_beta");
    __randPlotState.alpha = Number(a.inp.value);
    __randPlotState.beta = Number(b.inp.value);
    __randPlotState.mu = Number(mu.inp.value);
    __randPlotState.sigma = Number(sig.inp.value);
    __randPlotState.n = Math.max(200, Math.floor(Number(n.inp.value) || 5000));
    __randPlotState.bins = Math.max(5, Math.floor(Number(bins.inp.value) || 40));
    syncVis();
    draw();
  };

  kindSel.addEventListener("change", update);
  for (const x of [a.inp, b.inp, mu.inp, sig.inp, n.inp, bins.inp]) x.addEventListener("input", update);

  modal.__open = (kindHint) => {
    if (kindHint === "rand_logitnorm" || kindHint === "rand_beta") __randPlotState.kind = kindHint;
    syncVis();
    draw();
    modal.classList.add("modal--open");
    modal.setAttribute("aria-hidden", "false");
  };

  modal.appendChild(overlay);
  modal.appendChild(panel);
  document.body.appendChild(modal);
  __randPlotModal = modal;
  return modal;
}

function _exprRandKindHint(exprText) {
  const s = String(exprText || "");
  const hasBeta = /\brand_beta\s*\(/.test(s);
  const hasLogit = /\brand_logitnorm\s*\(/.test(s);
  if (hasLogit) return "rand_logitnorm";
  if (hasBeta) return "rand_beta";
  return null;
}

function _escapeRegexLiteral(s) {
  return String(s || "").replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}

function _globToRegexSourceWithGroups(glob) {
  const s = String(glob || "");
  let out = "^";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];

    if (ch === "\\") {
      if (i + 1 < s.length) {
        out += _escapeRegexLiteral(s[i + 1]);
        i++;
        continue;
      }
      out += "\\\\";
      continue;
    }

    if (ch === "*") {
      out += "(.*?)";
      continue;
    }
    if (ch === "?") {
      out += "(.)";
      continue;
    }

    if (ch === "[") {
      let j = i + 1;
      while (j < s.length && s[j] !== "]") j++;
      if (j < s.length) {
        let cls = s.slice(i + 1, j);
        if (cls.startsWith("!")) cls = "^" + cls.slice(1);
        out += "[" + cls + "]";
        i = j;
        continue;
      }
      out += "\\[";
      continue;
    }

    out += _escapeRegexLiteral(ch);
  }
  out += "$";
  return out;
}

function _isGlobLikePattern(pat) {
  const p = String(pat || "").trim();
  if (!p) return false;
  if (p.startsWith("glob:")) return true;
  if (p.startsWith("re:") || p.startsWith("regex:")) return false;
  if (!/[\*\?\[]/.test(p)) return false;
  if (p.includes(".*")) return false;
  return true;
}

function _countGlobCaptures(glob) {
  const s = String(glob || "");
  let n = 0;
  let inClass = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "\\") {
      i++;
      continue;
    }
    if (inClass) {
      if (ch === "]") inClass = false;
      continue;
    }
    if (ch === "[") {
      inClass = true;
      continue;
    }
    if (ch === "*" || ch === "?") n++;
  }
  return n;
}

function _applyLoopVarSubst(text, varToGroupIdx) {
  let out = String(text || "");
  for (const [v, idx] of Object.entries(varToGroupIdx || {})) {
    out = out.replaceAll(`{${v}}`, "${" + String(idx) + "}");
  }
  return out;
}

function _forEachStepToR(step) {
  const raw = String(step?.match || "*").trim() || "*";
  const pat = raw;

  const isGlob = !pat.startsWith("re:") && !pat.startsWith("regex:");
  const glob = pat.startsWith("glob:") ? pat.slice("glob:".length).trim() : pat;
  const nCaps = isGlob ? _countGlobCaptures(glob) : 0;
  const varToGroupIdx = {};
  if (isGlob) {
    // R-style: loop var `i` is the full matched layer name.
    varToGroupIdx.i = 0;
    // Capture groups from * / ? are exposed as {j},{k},... (group 1..N)
    const extra = ["j", "k", "l", "m", "n", "o", "p"];
    let g = 1;
    for (const v of extra) {
      if (g > nCaps) break;
      varToGroupIdx[v] = g;
      g++;
    }
  }

  const lines = [];
  const templateSteps = Array.isArray(step?.steps) ? step.steps : [];
  for (const t of templateSteps) {
    if (!t || typeof t !== "object") continue;
    const tt = String(t.type || "op").trim();
    const lhs = tt === "let" ? String(t.var || "") : String(t.target || "");
    const rhs = String(t.expr || "");

    let lhs2 = lhs;
    let rhs2 = rhs;
    for (const [v, idx] of Object.entries(varToGroupIdx)) {
      const tok = "${" + String(idx) + "}";
      lhs2 = lhs2.replaceAll(tok, `{${v}}`);
      rhs2 = rhs2.replaceAll(tok, `{${v}}`);
    }
    if (!lhs2 || !rhs2) continue;
    lines.push(`  ${lhs2} <- ${rhs2}`);
  }

  const loopVar = "i";
  return `for (${loopVar} in "${pat}") {\n${lines.join("\n")}\n}`;
}

function _compileForEachR(text, layerNames) {
  const src = String(text || "").trim();
  if (!src) return { ok: false, text: "Missing loop" };
  const m = src.match(/^for\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(["'])(.*?)\2\s*\)\s*\{([\s\S]*?)\}\s*$/);
  if (!m) return { ok: false, text: "Expected: for (i in \"gene_*\") { target <- expr }" };

  const loopVar = m[1];
  const pat = String(m[3] || "").trim();
  const body = String(m[4] || "");
  if (!pat) return { ok: false, text: "Missing pattern" };

  if (pat.startsWith("re:") || pat.startsWith("regex:")) {
    return { ok: false, text: "Use a glob pattern (e.g. gene_*)" };
  }

  const glob = pat.startsWith("glob:") ? pat.slice("glob:".length).trim() : pat;
  const nCaps = _countGlobCaptures(glob);
  const varToGroupIdx = {};
  // R-style: loop var refers to full matched layer name (group 0)
  varToGroupIdx[loopVar] = 0;
  // Capture groups from * / ? are exposed as {j},{k},... (group 1..N)
  const extras = ["j", "k", "l", "m", "n", "o", "p"];
  let g = 1;
  for (const v of extras) {
    if (g > nCaps) break;
    if (v === loopVar) {
      g++;
      continue;
    }
    varToGroupIdx[v] = g;
    g++;
  }

  const layerSet = new Set((layerNames || []).map((x) => String(x)));
  let foreachRx = null;
  try {
    foreachRx = new RegExp(_globToRegexSourceWithGroups(glob));
  } catch {
    foreachRx = null;
  }
  const steps = [];
  const lines = body
    .split(/\r?\n/)
    .map((l) => String(l).trim())
    .filter((l) => l && !l.startsWith("#"));
  for (const line0 of lines) {
    const line = line0.endsWith(";") ? line0.slice(0, -1).trim() : line0;
    const mm = line.match(/^(.+?)\s*<-\s*(.+)$/);
    if (!mm) return { ok: false, text: `Bad line: ${line0}` };
    const lhsRaw = String(mm[1] || "").trim();
    const rhsRaw = String(mm[2] || "").trim();
    if (!lhsRaw || !rhsRaw) return { ok: false, text: `Bad line: ${line0}` };

    const lhs = _applyLoopVarSubst(lhsRaw, varToGroupIdx);
    const rhs = _applyLoopVarSubst(rhsRaw, varToGroupIdx);

    const hasPlaceholders = lhsRaw.includes("{");
    let asLet = false;
    if (!hasPlaceholders) {
      const isId = _isValidIdentifier(lhsRaw);
      const isLayerTarget = isId && layerSet.has(lhsRaw);
      if (isId && !isLayerTarget) asLet = true;
    } else if (foreachRx) {
      const candidates = [];
      for (const nm of layerNames || []) {
        const mm2 = foreachRx.exec(String(nm));
        if (!mm2) continue;
        const cand = _substTemplateStepsText(lhs, mm2);
        if (cand) candidates.push(cand);
        if (candidates.length >= 50) break;
      }
      if (candidates.length) {
        const allIds = candidates.every((c) => _isValidIdentifier(c));
        const allLayers = candidates.every((c) => layerSet.has(c));
        if (allIds && !allLayers) asLet = true;
      }
    }
    if (asLet) steps.push({ type: "let", var: lhs, expr: rhs });
    else steps.push({ type: "op", target: lhs, expr: rhs });
  }

  if (!steps.length) return { ok: false, text: "No statements" };
  return { ok: true, text: "OK", match: pat, steps };
}

function _checkBalancedDelimiters(s) {
  const text = String(s || "");
  const stack = [];
  let inSingle = false;
  let inDouble = false;
  let esc = false;

  const openToClose = { "(": ")", "[": "]", "{": "}" };
  const closeToOpen = { ")": "(", "]": "[", "}": "{" };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (esc) {
      esc = false;
      continue;
    }
    if (ch === "\\") {
      esc = true;
      continue;
    }

    if (!inDouble && ch === "'") {
      inSingle = !inSingle;
      continue;
    }
    if (!inSingle && ch === '"') {
      inDouble = !inDouble;
      continue;
    }
    if (inSingle || inDouble) continue;

    if (openToClose[ch]) {
      stack.push(ch);
      continue;
    }
    if (closeToOpen[ch]) {
      if (!stack.length) return `Unmatched '${ch}'`;
      const top = stack.pop();
      if (top !== closeToOpen[ch]) return `Mismatched '${top}' and '${ch}'`;
    }
  }

  if (inSingle || inDouble) return "Unterminated string";
  if (stack.length) return `Unmatched '${stack[stack.length - 1]}'`;
  return null;
}

function _hasDisallowedAttributeAccess(expr) {
  const e = String(expr || "");
  return /\b[A-Za-z_]\w*\s*\./.test(e);
}

function _hasDisallowedKeywordArgsOrAssignment(expr) {
  const e = String(expr || "");
  for (let i = 0; i < e.length; i++) {
    if (e[i] !== "=") continue;
    const prev = e[i - 1] || "";
    const next = e[i + 1] || "";
    if (next === "=") continue;
    if (prev === "=" || prev === "!" || prev === "<" || prev === ">") continue;
    return true;
  }
  return false;
}

function _findMatchingParen(text, openIdx) {
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  let esc = false;

  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];

    if (esc) {
      esc = false;
      continue;
    }
    if (ch === "\\") {
      esc = true;
      continue;
    }

    if (!inDouble && ch === "'") {
      inSingle = !inSingle;
      continue;
    }
    if (!inSingle && ch === '"') {
      inDouble = !inDouble;
      continue;
    }
    if (inSingle || inDouble) continue;

    if (ch === "(") depth++;
    else if (ch === ")") {
      depth--;
      if (depth === 0) return i;
      if (depth < 0) return -1;
    }
  }
  return -1;
}

function _splitTopLevelArgs(text) {
  const s = String(text || "");
  const out = [];
  let cur = "";
  let depthP = 0;
  let depthB = 0;
  let depthC = 0;
  let inSingle = false;
  let inDouble = false;
  let esc = false;

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];

    if (esc) {
      cur += ch;
      esc = false;
      continue;
    }
    if (ch === "\\") {
      cur += ch;
      esc = true;
      continue;
    }

    if (!inDouble && ch === "'") {
      cur += ch;
      inSingle = !inSingle;
      continue;
    }
    if (!inSingle && ch === '"') {
      cur += ch;
      inDouble = !inDouble;
      continue;
    }
    if (inSingle || inDouble) {
      cur += ch;
      continue;
    }

    if (ch === "(") depthP++;
    else if (ch === ")") depthP = Math.max(0, depthP - 1);
    else if (ch === "[") depthB++;
    else if (ch === "]") depthB = Math.max(0, depthB - 1);
    else if (ch === "{") depthC++;
    else if (ch === "}") depthC = Math.max(0, depthC - 1);

    if (ch === "," && depthP === 0 && depthB === 0 && depthC === 0) {
      out.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }

  if (cur.trim() || out.length) out.push(cur.trim());
  return out.filter((x) => x.length > 0);
}

function _checkAllowedFuncArity(expr) {
  const e = String(expr || "");
  for (const fn of Object.keys(OPS_FUNC_ARITY)) {
    const re = new RegExp(`\\b${fn}\\s*\\(`, "g");
    let m;
    while ((m = re.exec(e))) {
      const openIdx = e.indexOf("(", m.index);
      if (openIdx < 0) continue;
      const closeIdx = _findMatchingParen(e, openIdx);
      if (closeIdx < 0) return `Unmatched '(' in ${fn}(...)`;
      const inside = e.slice(openIdx + 1, closeIdx);
      const args = _splitTopLevelArgs(inside);
      const want = OPS_FUNC_ARITY[fn];
      if (Array.isArray(want)) {
        if (!want.includes(args.length)) return `${fn}(...) expects ${want.join(" or ")} arg(s)`;
      } else {
        if (args.length !== want) return `${fn}(...) expects ${want} arg(s)`;
      }
      re.lastIndex = openIdx + 1;
    }
  }
  return null;
}

function validateLayerOpExpr(target, expr) {
  const t = String(target || "").trim();
  if (!t) return { ok: false, text: "Missing target" };
  const layerList = Array.isArray(state?.layers) ? state.layers : [];
  const meta = layerList.find((l) => l.name === t);
  if (!meta) return { ok: false, text: `Unknown target: ${t}` };

  const e = String(expr || "").trim();
  if (!e) return { ok: false, text: "Missing expr" };

  const bal = _checkBalancedDelimiters(e);
  if (bal) return { ok: false, text: bal };
  if (_hasDisallowedAttributeAccess(e)) return { ok: false, text: "Attribute access not allowed" };
  if (_hasDisallowedKeywordArgsOrAssignment(e)) return { ok: false, text: "Keyword args / assignment not allowed" };

  const ar = _checkAllowedFuncArity(e);
  if (ar) return { ok: false, text: ar };

  const ids = _tokenizeIdentifiers(e);
  const layerSet = new Set(layerList.map((l) => l.name));
  const unknown = [];
  for (const id of ids) {
    if (OPS_ALLOWED_FUNCS.has(id)) continue;
    if (id === "True" || id === "False") continue;
    if (layerSet.has(id)) continue;
    unknown.push(id);
  }
  if (unknown.length) return { ok: false, text: `Unknown (${unknown.length}): ${unknown.join(", ")}` };
  return { ok: true, text: "OK" };
}

function renderLayerOpsTable() {
  const layerList = Array.isArray(state?.layers) ? state.layers : [];
  if (!ui.opsSpecsTable) return;

  if (ui.opsDupGroupFrom) {
    const cur = ui.opsDupGroupFrom.value;
    ui.opsDupGroupFrom.innerHTML = "";
    const groups = Array.from(
      new Set(layerOps.map((s) => String(s?.group || "").trim()).filter((g) => g.length > 0))
    ).sort((a, b) => a.localeCompare(b));
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = "(pick group)";
    ui.opsDupGroupFrom.appendChild(opt0);
    for (const g of groups) {
      const opt = document.createElement("option");
      opt.value = g;
      opt.textContent = g;
      ui.opsDupGroupFrom.appendChild(opt);
    }
    ui.opsDupGroupFrom.value = groups.includes(cur) ? cur : "";
  }

  const wrap = document.createElement("div");
  wrap.className = "table";
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["On", "Type", "Name", "Group", "Var/Target", "Expr", "Status", "Move", ""]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  t.appendChild(thead);

  const opsCollapsedGroups = window.__opsCollapsedGroups || new Set();
  window.__opsCollapsedGroups = opsCollapsedGroups;
  try {
    if (!opsCollapsedGroups.__loaded) {
      opsCollapsedGroups.__loaded = true;
      const raw = localStorage.getItem(OPS_GROUPS_COLLAPSED_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      if (Array.isArray(arr)) for (const g of arr) opsCollapsedGroups.add(String(g || ""));
    }
  } catch {
    // ignore
  }

  const groupCounts = new Map();
  for (const s of layerOps) {
    const g = String(s?.group || "").trim();
    if (!g) continue;
    groupCounts.set(g, (groupCounts.get(g) || 0) + 1);
  }

  const tbody = document.createElement("tbody");
  let lastGroup = null;
  for (let i = 0; i < layerOps.length; i++) {
    const op = layerOps[i];

    const group = String(op.group || "").trim();
    if (group && group !== lastGroup) {
      lastGroup = group;
      const gtr = document.createElement("tr");
      gtr.className = "opsGroupRow";
      const gtd = document.createElement("td");
      gtd.colSpan = 9;
      const btn = document.createElement("button");
      btn.className = "btn btn--secondary btn--tiny";
      const isCollapsed = opsCollapsedGroups.has(group);
      btn.textContent = isCollapsed ? "▸" : "▾";
      btn.addEventListener("click", () => {
        const nextCollapsed = !opsCollapsedGroups.has(group);
        if (nextCollapsed) opsCollapsedGroups.add(group);
        else opsCollapsedGroups.delete(group);
        try {
          localStorage.setItem(OPS_GROUPS_COLLAPSED_KEY, JSON.stringify(Array.from(opsCollapsedGroups)));
        } catch {
          // ignore
        }
        renderLayerOpsTable();
      });
      const label = document.createElement("span");
      label.className = "meta";
      label.textContent = `${group} (${groupCounts.get(group) || 0})`;
      const box = document.createElement("div");
      box.className = "two two--btnLeft";
      box.appendChild(btn);
      box.appendChild(label);
      gtd.appendChild(box);
      gtr.appendChild(gtd);
      tbody.appendChild(gtr);
    }

    if (group && opsCollapsedGroups.has(group)) {
      continue;
    }

    const tr = document.createElement("tr");
    
    // Track focus on this row
    const rowIndex = i;
    tr.addEventListener("focusin", () => {
      opsLastFocusedRowIndex = rowIndex;
    });

    const tdOn = document.createElement("td");
    const on = document.createElement("input");
    on.type = "checkbox";
    on.checked = op.enabled !== false;
    on.addEventListener("change", () => {
      layerOps[i].enabled = on.checked;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
    });
    tdOn.appendChild(on);

    const tdType = document.createElement("td");
    const typeSel = document.createElement("select");
    typeSel.className = "input input--tiny";
    for (const k of ["op", "let", "foreach", "transport", "diffusion", "divide_cells", "pathway"]) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k === "divide_cells" ? "divide" : k;
      typeSel.appendChild(opt);
    }
    typeSel.value =
      op.type === "let"
        ? "let"
        : op.type === "foreach"
          ? "foreach"
          : op.type === "transport"
            ? "transport"
            : op.type === "diffusion"
              ? "diffusion"
              : op.type === "divide_cells"
                ? "divide_cells"
                : op.type === "pathway"
                  ? "pathway"
                : "op";
    typeSel.addEventListener("change", () => {
      const nextType =
        typeSel.value === "let"
          ? "let"
          : typeSel.value === "foreach"
            ? "foreach"
            : typeSel.value === "transport"
              ? "transport"
              : typeSel.value === "diffusion"
                ? "diffusion"
                : typeSel.value === "divide_cells"
                  ? "divide_cells"
                  : typeSel.value === "pathway"
                    ? "pathway"
                : "op";
      layerOps[i].type = nextType;
      if (nextType === "op") {
        delete layerOps[i].molecules;
        delete layerOps[i].molecule_prefix;
        delete layerOps[i].protein_prefix;
        delete layerOps[i].cell_layer;
        delete layerOps[i].cell_mode;
        delete layerOps[i].cell_value;
        delete layerOps[i].dirs;
        delete layerOps[i].per_pair_rate;
        delete layerOps[i].seed;
        delete layerOps[i].rate;
        delete layerOps[i].rate_layer;
        layerOps[i].target = String(layerOps[i].target || layerList[0]?.name || "layer").trim();
        delete layerOps[i].var;
        delete layerOps[i].match;
        delete layerOps[i].steps;
        delete layerOps[i].stepsText;
      } else if (nextType === "let") {
        delete layerOps[i].molecules;
        delete layerOps[i].molecule_prefix;
        delete layerOps[i].protein_prefix;
        delete layerOps[i].cell_layer;
        delete layerOps[i].cell_mode;
        delete layerOps[i].cell_value;
        delete layerOps[i].dirs;
        delete layerOps[i].per_pair_rate;
        delete layerOps[i].seed;
        delete layerOps[i].rate;
        delete layerOps[i].rate_layer;
        layerOps[i].var = String(layerOps[i].var || "tmp").trim() || "tmp";
        delete layerOps[i].target;
        delete layerOps[i].match;
        delete layerOps[i].steps;
        delete layerOps[i].stepsText;
        delete layerOps[i].require_match;
      } else {
        if (nextType === "foreach") {
          delete layerOps[i].molecules;
          delete layerOps[i].molecule_prefix;
          delete layerOps[i].protein_prefix;
          delete layerOps[i].cell_layer;
          delete layerOps[i].cell_mode;
          delete layerOps[i].cell_value;
          delete layerOps[i].dirs;
          delete layerOps[i].per_pair_rate;
          delete layerOps[i].seed;
          delete layerOps[i].rate;
          delete layerOps[i].rate_layer;
          layerOps[i].match = String(layerOps[i].match || "gene_*").trim();
          layerOps[i].require_match = !!layerOps[i].require_match;
          if (!String(layerOps[i].stepsText || "").trim()) {
            layerOps[i].stepsText = 'for (i in "gene_*") {\n  {i} <- {i}\n}';
          }
          const layerNames = state?.layers ? state.layers.map((l) => l.name) : [];
          const c = _compileForEachR(layerOps[i].stepsText, layerNames);
          if (c.ok) {
            layerOps[i].match = c.match;
            layerOps[i].steps = c.steps;
          } else {
            layerOps[i].steps = [];
          }
          delete layerOps[i].target;
          delete layerOps[i].var;
          delete layerOps[i].expr;
        } else {
          if (nextType === "transport") {
            delete layerOps[i].target;
            delete layerOps[i].var;
            delete layerOps[i].expr;
            delete layerOps[i].match;
            delete layerOps[i].steps;
            delete layerOps[i].stepsText;
            delete layerOps[i].require_match;
            layerOps[i].molecules = String(layerOps[i].molecules || "molecule_*").trim() || "molecule_*";
            layerOps[i].molecule_prefix = String(layerOps[i].molecule_prefix ?? "molecule_");
            layerOps[i].protein_prefix = String(layerOps[i].protein_prefix ?? "protein_");
            layerOps[i].cell_layer = String(layerOps[i].cell_layer || "cell").trim() || "cell";
            layerOps[i].cell_mode = "eq";
            layerOps[i].cell_value = Number(layerOps[i].cell_value ?? 1);
            layerOps[i].dirs = Array.isArray(layerOps[i].dirs) && layerOps[i].dirs.length ? layerOps[i].dirs : ["north", "south", "east", "west"];
            layerOps[i].per_pair_rate = Number(layerOps[i].per_pair_rate ?? 1.0);
            layerOps[i].seed = layerOps[i].seed == null ? 0 : Math.floor(Number(layerOps[i].seed));
            delete layerOps[i].rate;
            delete layerOps[i].rate_layer;
          } else if (nextType === "diffusion") {
            delete layerOps[i].target;
            delete layerOps[i].var;
            delete layerOps[i].expr;
            delete layerOps[i].match;
            delete layerOps[i].steps;
            delete layerOps[i].stepsText;
            delete layerOps[i].require_match;
            delete layerOps[i].molecule_prefix;
            delete layerOps[i].protein_prefix;
            delete layerOps[i].dirs;
            delete layerOps[i].per_pair_rate;
            layerOps[i].molecules = String(layerOps[i].molecules || "molecule_*").trim() || "molecule_*";
            layerOps[i].cell_layer = String(layerOps[i].cell_layer || "cell").trim() || "cell";
            layerOps[i].cell_mode = "eq";
            layerOps[i].cell_value = Number(layerOps[i].cell_value ?? 1);
            layerOps[i].rate = layerOps[i].rate == null ? 0.2 : Number(layerOps[i].rate);
            layerOps[i].rate_layer = layerOps[i].rate_layer == null ? null : String(layerOps[i].rate_layer || "").trim() || null;
            layerOps[i].seed = layerOps[i].seed == null ? 0 : Math.floor(Number(layerOps[i].seed));
          } else if (nextType === "divide_cells") {
            delete layerOps[i].target;
            delete layerOps[i].var;
            delete layerOps[i].expr;
            delete layerOps[i].match;
            delete layerOps[i].steps;
            delete layerOps[i].stepsText;
            delete layerOps[i].require_match;
            delete layerOps[i].molecules;
            delete layerOps[i].molecule_prefix;
            delete layerOps[i].protein_prefix;
            delete layerOps[i].dirs;
            delete layerOps[i].per_pair_rate;
            delete layerOps[i].rate;
            delete layerOps[i].rate_layer;
            layerOps[i].cell_layer = String(layerOps[i].cell_layer || "cell").trim() || "cell";
            layerOps[i].cell_value = Number(layerOps[i].cell_value ?? 1);
            layerOps[i].empty_value = Number(layerOps[i].empty_value ?? 0);
            layerOps[i].trigger_layer = String(layerOps[i].trigger_layer || "protein_divider").trim() || "protein_divider";
            layerOps[i].threshold = layerOps[i].threshold == null ? 50 : Number(layerOps[i].threshold);
            layerOps[i].split_fraction = layerOps[i].split_fraction == null ? 0.5 : Number(layerOps[i].split_fraction);
            layerOps[i].max_radius = layerOps[i].max_radius == null ? null : Math.floor(Number(layerOps[i].max_radius));
            if (!Array.isArray(layerOps[i].layer_prefixes) || !layerOps[i].layer_prefixes.length) {
              layerOps[i].layer_prefixes = ["molecule", "protein", "rna", "damage", "gene"];
            }
            layerOps[i].seed = layerOps[i].seed == null ? 0 : Math.floor(Number(layerOps[i].seed));
          } else {
            layerOps[i].target = String(layerOps[i].target || layerList[0]?.name || "layer").trim();
            delete layerOps[i].var;
            delete layerOps[i].match;
            delete layerOps[i].steps;
            delete layerOps[i].stepsText;
            delete layerOps[i].require_match;
          }
        }
      }
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
    });
    tdType.appendChild(typeSel);

    const tdName = document.createElement("td");
    const nm = document.createElement("input");
    nm.className = "input input--tiny";
    nm.value = String(op.name || "");
    nm.placeholder = "what this does";
    nm.addEventListener("input", () => {
      layerOps[i].name = nm.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
    });
    tdName.appendChild(nm);

    const tdGroup = document.createElement("td");
    const grp = document.createElement("input");
    grp.className = "input input--tiny";
    grp.value = String(op.group || "");
    grp.placeholder = "energy_generation";
    grp.addEventListener("input", () => {
      layerOps[i].group = grp.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
    });
    grp.addEventListener("blur", () => {
      renderLayerOpsTable();
    });
    tdGroup.appendChild(grp);

    const tdTarget = document.createElement("td");
    const isLet = op.type === "let";
    const isForEach = op.type === "foreach";
    const isTransport = op.type === "transport";
    const isDiffusion = op.type === "diffusion";
    const isDivide = op.type === "divide_cells";
    const isPathway = op.type === "pathway";
    const varInput = document.createElement("input");
    varInput.className = "input input--tiny";
    varInput.placeholder = "atp_generated";
    varInput.value = String(op.var || "");
    let varInputOldName = String(op.var || "");
    varInput.addEventListener("focus", () => {
      varInputOldName = String(layerOps[i]?.var || "");
    });
    varInput.addEventListener("input", () => {
      layerOps[i].var = varInput.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      updateStatus();
    });
    varInput.addEventListener("blur", () => {
      const newName = String(varInput.value || "").trim();
      // For 'let' operations, rename the actual layer if it exists
      if (isLet && varInputOldName && newName && varInputOldName !== newName) {
        const oldLayerExists = state.layers.some((l) => l.name === varInputOldName);
        const newLayerExists = state.layers.some((l) => l.name === newName);
        if (oldLayerExists && !newLayerExists) {
          try {
            renameLayer(state, varInputOldName, newName);
            if (selectedLayer === varInputOldName) selectedLayer = newName;
            markDirty();
            saveToLocalStorage();
          } catch (e) {
            console.error("Failed to rename layer:", e);
          }
        }
      }
      syncLayerSelect();
    });

    const target = document.createElement("select");
    target.className = "input input--tiny";
    for (const l of layerList) {
      const opt = document.createElement("option");
      opt.value = l.name;
      opt.textContent = l.name;
      target.appendChild(opt);
    }
    target.value = String(op.target || layerList[0]?.name || "");
    target.addEventListener("change", () => {
      layerOps[i].target = target.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
    });

    const moleculesInput = document.createElement("input");
    moleculesInput.className = "input input--tiny";
    moleculesInput.placeholder = "molecule_*";
    moleculesInput.title = "Which molecule count layers to move (glob). Example: molecule_*";
    moleculesInput.value = typeof op.molecules === "string" ? String(op.molecules || "") : "";
    moleculesInput.addEventListener("input", () => {
      layerOps[i].molecules = moleculesInput.value;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      updateStatus();
    });

    const moleculesWrap = document.createElement("div");
    moleculesWrap.className = "opsTransportMolecules";
    const moleculesLabel = document.createElement("div");
    moleculesLabel.className = "label";
    moleculesLabel.textContent = "Molecules (glob)";
    moleculesWrap.appendChild(moleculesLabel);
    moleculesWrap.appendChild(moleculesInput);

    if (isLet) tdTarget.appendChild(varInput);
    else if (isTransport || isDiffusion) tdTarget.appendChild(moleculesWrap);
    else if (isForEach) {
      const fx = document.createElement("div");
      fx.className = "meta";
      fx.textContent = "for loop";
      tdTarget.appendChild(fx);
    } else if (isDivide) {
      const fx = document.createElement("div");
      fx.className = "meta";
      fx.textContent = "cell division";
      tdTarget.appendChild(fx);
    } else if (isPathway) {
      const fx = document.createElement("div");
      fx.className = "meta";
      fx.style.fontSize = "10px";
      const inputs = Array.isArray(op.inputs) ? op.inputs.join(", ") : "";
      const outputs = Array.isArray(op.outputs) ? op.outputs.join(", ") : "";
      fx.innerHTML = `<strong>${op.pathway_name || "pathway"}</strong><br>${inputs} → ${outputs}<br>${op.num_enzymes || 1} enzymes`;
      tdTarget.appendChild(fx);
    } else tdTarget.appendChild(target);

    const tdExpr = document.createElement("td");
    const expr = isTransport || isDiffusion || isDivide || isPathway ? null : document.createElement("textarea");
    if (!(isTransport || isDiffusion || isDivide || isPathway)) {
      expr.className = "input input--tiny input--formula";
      expr.value = op.type === "foreach" ? String(op.stepsText || "") : String(op.expr || "");
      expr.placeholder =
        op.type === "foreach"
          ? "for (i in \"gene_*\") {\n  rna_{i} <- rna_{i} + gene_{i}\n}"
          : "e.g. where(cell==1, atp + glucose*atp_maker*damage, atp)";
      expr.spellcheck = false;
      expr.addEventListener("focus", () => {
        opsLastFocusedExprInput = expr;
      });

      const exprWrap = document.createElement("div");
      exprWrap.className = "opsExprWrap";

      const randBtn = document.createElement("button");
      randBtn.className = "btn btn--secondary btn--tiny";
      randBtn.type = "button";
      randBtn.textContent = "Rand";

      const syncRandBtn = () => {
        const kind = _exprRandKindHint(expr.value);
        randBtn.style.display = kind ? "inline-flex" : "none";
      };
      randBtn.addEventListener("click", () => {
        const kind = _exprRandKindHint(expr.value);
        if (!kind) return;
        const m = _randPlotEnsureModal();
        m.__open(kind);
      });

      syncRandBtn();
      expr.addEventListener("input", syncRandBtn);

      exprWrap.appendChild(expr);
      exprWrap.appendChild(randBtn);
      tdExpr.appendChild(exprWrap);
    } else if (isTransport) {
      const form = document.createElement("div");
      form.className = "opsTransport";

      const hint = document.createElement("div");
      hint.className = "opsTransportHint";
      hint.textContent = "Moves molecule counts between N/S/E/W neighbors using matching importer/exporter proteins.";
      form.appendChild(hint);

      const mkField = (labelText, controlEl) => {
        const row = document.createElement("div");
        row.className = "opsTransportField";
        const lab = document.createElement("div");
        lab.className = "label";
        lab.textContent = labelText;
        row.appendChild(lab);
        row.appendChild(controlEl);
        return row;
      };

      const cellSel = document.createElement("select");
      cellSel.className = "input input--tiny";
      cellSel.title = "Layer that defines cell vs non-cell. Transport uses cell==Cell value.";
      for (const l of layerList) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        cellSel.appendChild(opt);
      }
      cellSel.value = String(op.cell_layer || "cell");
      cellSel.addEventListener("change", () => {
        layerOps[i].cell_layer = cellSel.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell layer", cellSel));

      const cellValue = document.createElement("input");
      cellValue.className = "input input--tiny";
      cellValue.type = "number";
      cellValue.step = "1";
      cellValue.value = String(op.cell_value ?? 1);
      cellValue.title = "Value in Cell layer that means 'cell' (usually 1).";
      cellValue.addEventListener("input", () => {
        layerOps[i].cell_value = Number(cellValue.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell value", cellValue));

      const rate = document.createElement("input");
      rate.className = "input input--tiny";
      rate.type = "number";
      rate.min = "0";
      rate.max = "1";
      rate.step = "0.1";
      rate.value = String(op.per_pair_rate ?? 1.0);
      rate.title = "Probability each transporter-pair slot is active this step. 1 = deterministic max throughput.";
      rate.addEventListener("input", () => {
        layerOps[i].per_pair_rate = Number(rate.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Pair rate", rate));

      const seed = document.createElement("input");
      seed.className = "input input--tiny";
      seed.type = "number";
      seed.step = "1";
      seed.value = String(op.seed ?? 0);
      seed.title = "Random seed for transport allocation (for reproducible runs).";
      seed.addEventListener("input", () => {
        layerOps[i].seed = Number(seed.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Seed", seed));

      const dirsWrap = document.createElement("div");
      dirsWrap.className = "opsDirs";
      const DIRS = [
        { key: "north", label: "North" },
        { key: "south", label: "South" },
        { key: "east", label: "East" },
        { key: "west", label: "West" },
      ];
      const getDirs = () => {
        const d = Array.isArray(layerOps[i].dirs) ? layerOps[i].dirs : [];
        return new Set(d.map((x) => String(x || "").trim()).filter((x) => x));
      };
      const setDirs = (s) => {
        layerOps[i].dirs = Array.from(s);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      };
      for (const { key, label } of DIRS) {
        const lb = document.createElement("label");
        lb.className = "checkbox";
        lb.title = `Allow movement to the ${label.toLowerCase()} neighbor`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        const cur = getDirs();
        cb.checked = cur.size ? cur.has(key) : true;
        cb.addEventListener("change", () => {
          const next = getDirs();
          if (cb.checked) next.add(key);
          else next.delete(key);
          setDirs(next);
        });
        lb.appendChild(cb);
        lb.appendChild(document.createTextNode(` ${label}`));
        dirsWrap.appendChild(lb);
      }
      form.appendChild(mkField("Directions", dirsWrap));

      const det = document.createElement("details");
      det.className = "details";
      const sum = document.createElement("summary");
      sum.className = "details__summary";
      sum.textContent = "Advanced";
      det.appendChild(sum);

      const advBox = document.createElement("div");
      advBox.className = "opsTransport";

      const molPref = document.createElement("input");
      molPref.className = "input input--tiny";
      molPref.placeholder = "molecule_";
      molPref.value = String(op.molecule_prefix ?? "molecule_");
      molPref.title = "Prefix stripped from molecule layer names to form the transporter suffix.";
      molPref.addEventListener("input", () => {
        layerOps[i].molecule_prefix = molPref.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
      });
      advBox.appendChild(mkField("Mol prefix", molPref));

      const protPref = document.createElement("input");
      protPref.className = "input input--tiny";
      protPref.placeholder = "protein_";
      protPref.value = String(op.protein_prefix ?? "protein_");
      protPref.title = "Prefix used for transporter layers, e.g. protein_north_exporter_<suffix>.";
      protPref.addEventListener("input", () => {
        layerOps[i].protein_prefix = protPref.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
      });
      advBox.appendChild(mkField("Prot prefix", protPref));

      det.appendChild(advBox);
      form.appendChild(det);

      tdExpr.appendChild(form);
    } else if (isDivide) {
      const form = document.createElement("div");
      form.className = "opsTransport";

      const hint = document.createElement("div");
      hint.className = "opsTransportHint";
      hint.textContent = "If trigger_layer > threshold at a cell pixel, creates a new cell at the nearest empty pixel and splits selected layers.";
      form.appendChild(hint);

      const mkField = (labelText, controlEl) => {
        const row = document.createElement("div");
        row.className = "opsTransportField";
        const lab = document.createElement("div");
        lab.className = "label";
        lab.textContent = labelText;
        row.appendChild(lab);
        row.appendChild(controlEl);
        return row;
      };

      const cellSel = document.createElement("select");
      cellSel.className = "input input--tiny";
      for (const l of layerList) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        cellSel.appendChild(opt);
      }
      cellSel.value = String(op.cell_layer || "cell");
      cellSel.addEventListener("change", () => {
        layerOps[i].cell_layer = cellSel.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell layer", cellSel));

      const cellValue = document.createElement("input");
      cellValue.className = "input input--tiny";
      cellValue.type = "number";
      cellValue.step = "1";
      cellValue.value = String(op.cell_value ?? 1);
      cellValue.addEventListener("input", () => {
        layerOps[i].cell_value = Number(cellValue.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell value", cellValue));

      const emptyValue = document.createElement("input");
      emptyValue.className = "input input--tiny";
      emptyValue.type = "number";
      emptyValue.step = "1";
      emptyValue.value = String(op.empty_value ?? 0);
      emptyValue.addEventListener("input", () => {
        layerOps[i].empty_value = Number(emptyValue.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Empty value", emptyValue));

      const trigInput = document.createElement("input");
      trigInput.className = "input input--tiny";
      trigInput.placeholder = "protein_divider";
      trigInput.value = String(op.trigger_layer || "protein_divider");

      const trigListId = `divideTrigLayerList_${i}`;
      const trigDatalist = document.createElement("datalist");
      trigDatalist.id = trigListId;
      for (const l of layerList) {
        const opt = document.createElement("option");
        opt.value = l.name;
        trigDatalist.appendChild(opt);
      }
      trigInput.setAttribute("list", trigListId);

      trigInput.addEventListener("input", () => {
        layerOps[i].trigger_layer = trigInput.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      const trigWrap = document.createElement("div");
      trigWrap.appendChild(trigInput);
      trigWrap.appendChild(trigDatalist);
      form.appendChild(mkField("Trigger layer", trigWrap));

      const threshold = document.createElement("input");
      threshold.className = "input input--tiny";
      threshold.type = "number";
      threshold.step = "0.1";
      threshold.value = String(op.threshold ?? 50);
      threshold.addEventListener("input", () => {
        layerOps[i].threshold = Number(threshold.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Threshold", threshold));

      const frac = document.createElement("input");
      frac.className = "input input--tiny";
      frac.type = "number";
      frac.min = "0";
      frac.max = "1";
      frac.step = "0.05";
      frac.value = String(op.split_fraction ?? 0.5);
      frac.addEventListener("input", () => {
        layerOps[i].split_fraction = Number(frac.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Split fraction", frac));

      const maxRad = document.createElement("input");
      maxRad.className = "input input--tiny";
      maxRad.type = "number";
      maxRad.step = "1";
      maxRad.placeholder = "(auto)";
      maxRad.value = op.max_radius == null ? "" : String(op.max_radius);
      maxRad.addEventListener("input", () => {
        const v = String(maxRad.value || "").trim();
        layerOps[i].max_radius = v ? Math.floor(Number(v)) : null;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Max radius", maxRad));

      const prefixes = document.createElement("input");
      prefixes.className = "input input--tiny";
      prefixes.placeholder = "molecule, protein, rna, damage, gene";
      prefixes.value = Array.isArray(op.layer_prefixes) ? op.layer_prefixes.join(", ") : "";
      prefixes.addEventListener("input", () => {
        const raw = String(prefixes.value || "");
        const arr = raw
          .split(",")
          .map((s) => String(s || "").trim())
          .filter((s) => s);
        layerOps[i].layer_prefixes = arr.length ? arr : ["molecule", "protein", "rna", "damage", "gene"];
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Layer prefixes", prefixes));

      const seed = document.createElement("input");
      seed.className = "input input--tiny";
      seed.type = "number";
      seed.step = "1";
      seed.placeholder = "(none)";
      seed.value = op.seed == null ? "" : String(op.seed);
      seed.addEventListener("input", () => {
        const v = String(seed.value || "").trim();
        layerOps[i].seed = v ? Math.floor(Number(v)) : null;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Seed", seed));

      tdExpr.appendChild(form);
    } else if (isPathway) {
      const form = document.createElement("div");
      form.className = "opsTransport";

      const hint = document.createElement("div");
      hint.className = "opsTransportHint";
      hint.innerHTML = `<strong>Pathway:</strong> Converts inputs to outputs through enzyme chain. Each enzyme modulates throughput.`;
      form.appendChild(hint);

      const mkField = (labelText, controlEl) => {
        const row = document.createElement("div");
        row.className = "opsTransportField";
        const lab = document.createElement("div");
        lab.className = "label";
        lab.textContent = labelText;
        row.appendChild(lab);
        row.appendChild(controlEl);
        return row;
      };

      const cellSel = document.createElement("select");
      cellSel.className = "input input--tiny";
      cellSel.title = "Layer that defines cell vs non-cell. Pathway only operates in cells.";
      for (const l of state.layers) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        cellSel.appendChild(opt);
      }
      cellSel.value = String(op.cell_layer || "cell");
      cellSel.addEventListener("change", () => {
        layerOps[i].cell_layer = cellSel.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell layer", cellSel));

      const cellValue = document.createElement("input");
      cellValue.className = "input input--tiny";
      cellValue.type = "number";
      cellValue.step = "1";
      cellValue.value = String(op.cell_value ?? 1);
      cellValue.title = "Value in Cell layer that means 'cell' (usually 1).";
      cellValue.addEventListener("input", () => {
        layerOps[i].cell_value = Number(cellValue.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell value", cellValue));

      const efficiency = document.createElement("input");
      efficiency.className = "input input--tiny";
      efficiency.type = "number";
      efficiency.step = "0.1";
      efficiency.min = "0";
      efficiency.max = "10";
      efficiency.value = String(op.efficiency ?? 1.0);
      efficiency.title = "Base efficiency multiplier for the pathway (1.0 = normal).";
      efficiency.addEventListener("input", () => {
        layerOps[i].efficiency = Number(efficiency.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Efficiency", efficiency));

      const seed = document.createElement("input");
      seed.className = "input input--tiny";
      seed.type = "number";
      seed.step = "1";
      seed.placeholder = "(none)";
      seed.value = op.seed == null ? "" : String(op.seed);
      seed.addEventListener("input", () => {
        const v = String(seed.value || "").trim();
        layerOps[i].seed = v ? Math.floor(Number(v)) : null;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Seed", seed));

      tdExpr.appendChild(form);
    } else {
      const form = document.createElement("div");
      form.className = "opsTransport";

      const hint = document.createElement("div");
      hint.className = "opsTransportHint";
      hint.textContent = "Diffuses molecule counts through non-cell space (no transporters). Cells block diffusion.";
      form.appendChild(hint);

      const mkField = (labelText, controlEl) => {
        const row = document.createElement("div");
        row.className = "opsTransportField";
        const lab = document.createElement("div");
        lab.className = "label";
        lab.textContent = labelText;
        row.appendChild(lab);
        row.appendChild(controlEl);
        return row;
      };

      const cellSel = document.createElement("select");
      cellSel.className = "input input--tiny";
      cellSel.title = "Layer that defines cell vs non-cell. Diffusion uses non-cell pixels only.";
      for (const l of state.layers) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        cellSel.appendChild(opt);
      }
      cellSel.value = String(op.cell_layer || "cell");
      cellSel.addEventListener("change", () => {
        layerOps[i].cell_layer = cellSel.value;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell layer", cellSel));

      const cellValue = document.createElement("input");
      cellValue.className = "input input--tiny";
      cellValue.type = "number";
      cellValue.step = "1";
      cellValue.value = String(op.cell_value ?? 1);
      cellValue.title = "Value in Cell layer that means 'cell' (usually 1).";
      cellValue.addEventListener("input", () => {
        layerOps[i].cell_value = Number(cellValue.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Cell value", cellValue));

      const rate = document.createElement("input");
      rate.className = "input input--tiny";
      rate.type = "number";
      rate.min = "0";
      rate.max = "1";
      rate.step = "0.05";
      rate.value = String(op.rate ?? 0.2);
      rate.title = "Per-step diffusion rate in [0,1]. You can also provide a rate layer to vary this per pixel.";
      rate.addEventListener("input", () => {
        layerOps[i].rate = Number(rate.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Rate", rate));

      const rateLayerSel = document.createElement("select");
      rateLayerSel.className = "input input--tiny";
      rateLayerSel.title = "Optional layer that scales diffusion rate per pixel (e.g. circulation vs lumen).";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = "(none)";
      rateLayerSel.appendChild(opt0);
      for (const l of layerList) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        rateLayerSel.appendChild(opt);
      }
      rateLayerSel.value = String(op.rate_layer || "");
      rateLayerSel.addEventListener("change", () => {
        const v = String(rateLayerSel.value || "").trim();
        layerOps[i].rate_layer = v ? v : null;
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Rate layer", rateLayerSel));

      const seed = document.createElement("input");
      seed.className = "input input--tiny";
      seed.type = "number";
      seed.step = "1";
      seed.value = String(op.seed ?? 0);
      seed.title = "Random seed for diffusion allocation (for reproducible runs).";
      seed.addEventListener("input", () => {
        layerOps[i].seed = Number(seed.value);
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
      form.appendChild(mkField("Seed", seed));

      tdExpr.appendChild(form);
    }

    const tdStatus = document.createElement("td");
    const st = document.createElement("div");
    const updateStatus = () => {
      const knownVars = new Set();
      for (let j = 0; j < i; j++) {
        const prev = layerOps[j];
        if (!prev || prev.enabled === false) continue;
        if (prev.type === "let") {
          const nm = String(prev.var || "").trim();
          if (_isValidIdentifier(nm)) knownVars.add(nm);
        }
      }
      const exprValue = expr ? String(expr.value || "") : "";
      const baseStep = layerOps[i];
      const stepForValidation =
        baseStep.type === "foreach"
          ? {
              type: "foreach",
              match: baseStep.match,
              require_match: !!baseStep.require_match,
              steps: Array.isArray(baseStep.steps) ? baseStep.steps : [],
              stepsText: exprValue,
            }
          : baseStep.type === "transport"
            ? {
                type: "transport",
                molecules: baseStep.molecules,
                molecule_prefix: baseStep.molecule_prefix,
                protein_prefix: baseStep.protein_prefix,
                cell_layer: baseStep.cell_layer,
                cell_mode: baseStep.cell_mode,
                cell_value: baseStep.cell_value,
                dirs: baseStep.dirs,
                per_pair_rate: baseStep.per_pair_rate,
                seed: baseStep.seed,
              }
            : baseStep.type === "diffusion"
              ? {
                  type: "diffusion",
                  molecules: baseStep.molecules,
                  cell_layer: baseStep.cell_layer,
                  cell_mode: baseStep.cell_mode,
                  cell_value: baseStep.cell_value,
                  rate: baseStep.rate,
                  rate_layer: baseStep.rate_layer,
                  seed: baseStep.seed,
                }
            : baseStep.type === "divide_cells"
              ? {
                  type: "divide_cells",
                  cell_layer: baseStep.cell_layer,
                  cell_value: baseStep.cell_value,
                  empty_value: baseStep.empty_value,
                  trigger_layer: baseStep.trigger_layer,
                  threshold: baseStep.threshold,
                  split_fraction: baseStep.split_fraction,
                  max_radius: baseStep.max_radius,
                  layer_prefixes: baseStep.layer_prefixes,
                  seed: baseStep.seed,
                }
              : baseStep.type === "pathway"
                ? {
                    type: "pathway",
                    pathway_name: baseStep.pathway_name,
                    inputs: baseStep.inputs,
                    outputs: baseStep.outputs,
                    num_enzymes: baseStep.num_enzymes,
                    cell_layer: baseStep.cell_layer,
                    cell_value: baseStep.cell_value,
                    efficiency: baseStep.efficiency,
                    seed: baseStep.seed,
                  }
            : {
                type: baseStep.type,
                var: baseStep.var,
                target: baseStep.target,
                expr: exprValue,
              };
      const v = validateLayerOpStep(stepForValidation, knownVars);
      st.className = "meta opsStatus";
      st.textContent = v.text;
      st.title = v.text;

      if (baseStep.type === "foreach") {
        const src = String(exprValue || "").trim();
        if (src.startsWith("for")) {
          const layerNames = state?.layers ? state.layers.map((l) => l.name) : [];
          const c = _compileForEachR(src, layerNames);
          if (c.ok) {
            layerOps[i].match = c.match;
            layerOps[i].steps = c.steps;
          }
        }
      }
    };
    updateStatus();
    tdStatus.appendChild(st);

    if (!(isTransport || isDiffusion || isDivide || isPathway) && expr) {
      expr.addEventListener("input", () => {
        if (layerOps[i].type === "foreach") {
          layerOps[i].stepsText = expr.value;
          const layerNames = state?.layers ? state.layers.map((l) => l.name) : [];
          const c = _compileForEachR(expr.value, layerNames);
          if (c.ok) {
            layerOps[i].match = c.match;
            layerOps[i].steps = c.steps;
          } else {
            layerOps[i].steps = [];
          }
        } else {
          layerOps[i].expr = expr.value;
        }
        saveFunctionsCfg();
        markDirty();
        saveToLocalStorage();
        updateStatus();
      });
    }

    const tdMove = document.createElement("td");
    const moveWrap = document.createElement("div");
    moveWrap.className = "opsRowActions";

    const up = document.createElement("button");
    up.className = "btn btn--secondary btn--tiny";
    up.textContent = "↑";
    up.disabled = i === 0;
    up.addEventListener("click", () => {
      if (i <= 0) return;
      const tmp = layerOps[i - 1];
      layerOps[i - 1] = layerOps[i];
      layerOps[i] = tmp;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
    });

    const down = document.createElement("button");
    down.className = "btn btn--secondary btn--tiny";
    down.textContent = "↓";
    down.disabled = i === layerOps.length - 1;
    down.addEventListener("click", () => {
      if (i >= layerOps.length - 1) return;
      const tmp = layerOps[i + 1];
      layerOps[i + 1] = layerOps[i];
      layerOps[i] = tmp;
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
    });

    moveWrap.appendChild(up);
    moveWrap.appendChild(down);
    tdMove.appendChild(moveWrap);

    const tdDel = document.createElement("td");
    const del = document.createElement("button");
    del.className = "btn btn--danger btn--tiny";
    del.textContent = "Remove";
    del.addEventListener("click", () => {
      // For 'let' operations, also delete the corresponding layer
      if (op.type === "let" && op.var) {
        const layerName = String(op.var).trim();
        if (layerName && state.layers.some((l) => l.name === layerName)) {
          try {
            removeLayer(state, layerName);
            if (selectedLayer === layerName) {
              selectedLayer = state.layers[0]?.name || "";
            }
            bulkSelectedLayers.delete(layerName);
          } catch (e) {
            console.error("Failed to remove layer:", e);
          }
        }
      }
      layerOps.splice(i, 1);
      saveFunctionsCfg();
      markDirty();
      saveToLocalStorage();
      renderLayerOpsTable();
      syncLayerSelect();
    });
    tdDel.appendChild(del);

    tr.appendChild(tdOn);
    tr.appendChild(tdType);
    tr.appendChild(tdName);
    tr.appendChild(tdGroup);
    tr.appendChild(tdTarget);
    tr.appendChild(tdExpr);
    tr.appendChild(tdStatus);
    tr.appendChild(tdMove);
    tr.appendChild(tdDel);
    tbody.appendChild(tr);
  }

  t.appendChild(tbody);
  wrap.appendChild(t);
  ui.opsSpecsTable.innerHTML = "";
  ui.opsSpecsTable.appendChild(wrap);
}

let layersFilterText = "";
let layersGroupByPrefix = true;
const collapsedGroups = new Set();

function tryLoadFromLocalStorage() {
  try {
    const text = localStorage.getItem(STORAGE_KEY);
    if (!text) return null;
    return parseState(text);
  } catch {
    return null;
  }
}

function updateBulkAddPreview() {
  if (!ui.bulkAddPreview) return;
  try {
    const prefix = String(ui.bulkPrefix?.value || "");
    const startN = Math.floor(Number(ui.bulkStart?.value));
    const n = Math.max(1, Math.floor(Number(ui.bulkCount?.value)));
    const collisionMode = String(ui.bulkCollision?.value || "error");
    const existing = new Set(state.layers.map((l) => l.name));

    let collisions = 0;
    const names = [];
    for (let i = 0; i < n; i++) {
      const nm = `${prefix}${startN + i}`;
      names.push(nm);
      if (existing.has(nm)) collisions++;
    }

    let maskMatches = null;
    if (ui.maskLayer && ui.maskOp && ui.maskValue && ui.maskInvert) {
      const mask = makeMask(state, ui.maskLayer.value, ui.maskOp.value, ui.maskValue.value, ui.maskInvert.checked);
      let m = 0;
      for (let i = 0; i < mask.length; i++) m += mask[i] ? 1 : 0;
      maskMatches = m;
    }

    const sample = names.slice(0, 10).join(", ");
    const extra = names.length > 10 ? ` (+${names.length - 10} more)` : "";
    const maskStr = maskMatches == null ? "" : ` | mask matches: ${maskMatches.toLocaleString()} cells`;
    const collisionStr = collisions ? ` | already exist: ${collisions} (${collisionMode})` : "";
    ui.bulkAddPreview.textContent = `Will create: ${n} layer(s)${collisionStr}${maskStr}\n${sample}${extra}`;
  } catch {
    ui.bulkAddPreview.textContent = "Will create: –";
  }
}

function _pickPrototypeLayer(prefix) {
  const pfx = String(prefix || "");
  const m = state.layers.find((l) => String(l.name).startsWith(pfx));
  return m ? m.name : state.layers[0]?.name || "";
}

function applyDerivedLayerPlan(state) {
  const plan = computeDerivedLayerPlan(state);

  const dataSnapshots = new Map();
  for (const p of plan) {
    if (!p || (p.action !== "create" && p.action !== "overwrite")) continue;
    if (!p.dataRefName) continue;
    const src = state.data[p.dataRefName];
    if (!src) continue;
    dataSnapshots.set(`${p.targetName}::${p.dataRefName}`, new Float32Array(src));
  }

  const createdNames = [];
  let nCreate = 0;
  let nOverwrite = 0;
  let nSkip = 0;
  let nErr = 0;

  for (const p of plan) {
    if (!p) continue;
    if (p.action === "skip_exists") {
      nSkip++;
      continue;
    }
    if (p.action !== "create" && p.action !== "overwrite") {
      nErr++;
      continue;
    }

    if (p.action === "overwrite") {
      removeLayer(state, p.targetName);
      nOverwrite++;
    } else {
      nCreate++;
    }

    addLayer(state, {
      name: p.targetName,
      kind: p.kind,
      color: p.color,
      init: "zeros",
      value: 0,
      seed: 0,
    });

    if (p.dataInit === "copy_source" || p.dataInit === "copy_prototype") {
      const key = `${p.targetName}::${p.dataRefName}`;
      const snap = dataSnapshots.get(key);
      if (snap && snap.length === state.H * state.W) {
        state.data[p.targetName] = new Float32Array(snap);
      }
    }

    createdNames.push(p.targetName);
  }

  ensureGeneTriplets(state);
  return { nCreate, nOverwrite, nSkip, nErr, createdNames };
}

function computeFitZoom() {
  const wrap = ui.canvasWrap || document.querySelector(".canvasWrap");
  if (!wrap) return null;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (!w || !h || !state?.W || !state?.H) return null;
  const z = Math.floor(Math.min(w / state.W, h / state.H));
  const minZ = Number(ui.zoomInput.min || 1);
  const maxZ = Number(ui.zoomInput.max || 24);
  return clamp(z, minZ, maxZ);
}

function applyAutoFitZoom() {
  if (!ui.zoomInput) return;
  const enabled = !!ui.autoFitZoom?.checked;
  ui.zoomInput.disabled = enabled;
  const wrap = ui.canvasWrap || document.querySelector(".canvasWrap");
  if (wrap) wrap.style.overflow = enabled ? "hidden" : "auto";
  if (!enabled) return;
  const z = computeFitZoom();
  if (z == null) return;
  if (Number(ui.zoomInput.value) !== z) ui.zoomInput.value = String(z);
  if (wrap) {
    wrap.scrollLeft = 0;
    wrap.scrollTop = 0;
  }
}

function _applyTemplate(template, ctx) {
  let out = String(template);
  out = out.replaceAll("{suffix}", String(ctx.suffix ?? ""));
  out = out.replaceAll("{source}", String(ctx.source ?? ""));
  return out;
}

function computeDerivedLayerPlan(state) {
  const srcPrefix = String(ui.derivedSourcePrefix?.value || "");
  const tmpl = String(ui.derivedTargetTemplate?.value || "");
  const metaFrom = String(ui.derivedMetaFrom?.value || "source");
  const protoName = String(ui.derivedPrototypeLayer?.value || "");
  const dataInit = String(ui.derivedDataInit?.value || "zeros");
  const skipExisting = !!ui.derivedSkipExisting?.checked;

  const existing = new Set(state.layers.map((l) => l.name));
  const sources = state.layers.map((l) => l.name).filter((n) => n.startsWith(srcPrefix) && n.length > srcPrefix.length);

  const plan = [];
  for (const sourceName of sources) {
    const suffix = sourceName.slice(srcPrefix.length);
    const targetName = _applyTemplate(tmpl, { suffix, source: sourceName });
    if (!targetName) continue;

    const collision = existing.has(targetName);
    if (collision && skipExisting) {
      plan.push({ sourceName, suffix, targetName, action: "skip_exists" });
      continue;
    }

    const sourceMeta = state.layers.find((l) => l.name === sourceName);
    const protoMeta = state.layers.find((l) => l.name === protoName);
    const metaRef = metaFrom === "prototype" ? protoMeta : sourceMeta;
    if (!metaRef) {
      plan.push({ sourceName, suffix, targetName, action: "error_missing_meta" });
      continue;
    }

    let dataRefName = null;
    if (dataInit === "copy_source") dataRefName = sourceName;
    else if (dataInit === "copy_prototype") dataRefName = protoName;

    if (dataRefName) {
      const a = state.data[dataRefName];
      if (!a) {
        plan.push({ sourceName, suffix, targetName, action: "error_missing_data" });
        continue;
      }
    }

    plan.push({
      sourceName,
      suffix,
      targetName,
      action: collision ? "overwrite" : "create",
      kind: metaRef.kind,
      color: metaRef.color,
      dataInit,
      dataRefName,
    });
  }

  return plan;
}

function updateDerivedPreview() {
  if (!ui.derivedPreview) return;
  try {
    const plan = computeDerivedLayerPlan(state);
    const counts = { create: 0, overwrite: 0, skip_exists: 0, error: 0 };
    for (const p of plan) {
      if (p.action === "create") counts.create++;
      else if (p.action === "overwrite") counts.overwrite++;
      else if (p.action === "skip_exists") counts.skip_exists++;
      else counts.error++;
    }
    const sample = plan
      .filter((p) => p.action === "create" || p.action === "overwrite")
      .slice(0, 10)
      .map((p) => `${p.sourceName} -> ${p.targetName}`)
      .join(" | ");
    const extra = plan.length > 10 ? ` (+${plan.length - 10} more)` : "";
    ui.derivedPreview.textContent = `Planned: create=${counts.create} overwrite=${counts.overwrite} skip=${counts.skip_exists} errors=${counts.error}${sample ? `\n${sample}${extra}` : ""}`;
  } catch (e) {
    ui.derivedPreview.textContent = "Planned: –";
  }
}

function updateMaskedOpsPreview() {
  if (!ui.opMaskPreview) return;
  try {
    const mask = makeMask(state, ui.opMaskLayer.value, ui.opMaskOp.value, ui.opMaskValue.value, ui.opMaskInvert.checked);
    let n = 0;
    for (let i = 0; i < mask.length; i++) n += mask[i] ? 1 : 0;

    const t = computeBatchTargets();
    const tStr = `Targets: ${t.targets.length}`;
    ui.opMaskPreview.textContent = `Mask matches: ${n.toLocaleString()} cells | ${tStr}`;
  } catch (e) {
    ui.opMaskPreview.textContent = "Mask matches: –";
  }
}

function pruneOpTargetsSelected() {
  const existing = new Set(state.layers.map((l) => l.name));
  for (const nm of [...opTargetsSelected]) {
    if (!existing.has(nm)) opTargetsSelected.delete(nm);
  }
}

function computeBatchTargets() {
  pruneOpTargetsSelected();

  const names = new Set();

  for (const nm of opTargetsSelected) names.add(nm);

  const targets = [];
  const isRandom = isRandomAssignOpType(ui.opType?.value);
  for (const nm of names) {
    const meta = state.layers.find((l) => l.name === nm);
    if (!meta) continue;
    if (isRandom && meta.kind === "categorical") continue;
    targets.push(nm);
  }
  targets.sort((a, b) => a.localeCompare(b));
  return { targets };
}

function renderOpTargetsList() {
  if (!ui.opTargetsList) return;

  pruneOpTargetsSelected();
  const isRandom = isRandomAssignOpType(ui.opType?.value);
  const q = String(ui.opTargetFilter?.value || "").trim();
  
  // Get all layers that match the search filter
  let filteredLayers = [];
  if (q) {
    // Convert wildcard pattern to regex
    const pattern = q.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
    const regex = new RegExp(pattern, 'i'); // Removed ^ and $ to allow partial matches
    filteredLayers = state.layers.filter((l) => regex.test(String(l.name)));
  } else {
    filteredLayers = [...state.layers];
  }
  
  // Add any selected layers that didn't match the filter
  const selectedLayerNames = new Set(opTargetsSelected);
  const filteredLayerNames = new Set(filteredLayers.map(l => l.name));
  
  // Find selected layers that aren't already in the filtered list
  const missingSelectedLayers = state.layers.filter(l => 
    selectedLayerNames.has(l.name) && !filteredLayerNames.has(l.name)
  );
  
  // Combine filtered layers with missing selected layers
  const layers = [...filteredLayers, ...missingSelectedLayers];
  
  // Sort alphabetically for consistent display
  layers.sort((a, b) => a.name.localeCompare(b.name));

  ui.opTargetsList.innerHTML = "";
  for (const l of layers) {
    const row = document.createElement("div");
    row.className = "targetsRow";

    const left = document.createElement("div");
    left.className = "targetsRow__left";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = opTargetsSelected.has(l.name);
    cb.disabled = isRandom && l.kind === "categorical";
    cb.addEventListener("change", () => {
      if (cb.disabled) {
        cb.checked = false;
        opTargetsSelected.delete(l.name);
        updateMaskedOpsPreview();
        return;
      }
      if (cb.checked) opTargetsSelected.add(l.name);
      else opTargetsSelected.delete(l.name);
      updateMaskedOpsPreview();
    });

    const nm = document.createElement("div");
    nm.className = "targetsRow__name";
    nm.textContent = l.name;
    nm.title = l.name;

    left.appendChild(cb);
    left.appendChild(nm);

    const kind = document.createElement("div");
    kind.className = "targetsRow__kind";
    kind.textContent = l.kind;

    row.appendChild(left);
    row.appendChild(kind);
    ui.opTargetsList.appendChild(row);
  }
}

function saveToLocalStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, serializeState(state));
    dirtySinceLastSave = false;
    _updateCurrentFileInfo();
  } catch {
    // ignore
  }
}

function markDirty() {
  dirtySinceLastSave = true;
  inspectSummaryDirty = true;
  _updateCurrentFileInfo();
}

function _syncInspectModeUi() {
  if (!ui.inspectMode) return;
  const mode = ui.inspectMode.value || "cursor";
  const showSummary = mode === "summary";
  if (ui.inspectTable) ui.inspectTable.style.display = showSummary ? "none" : "block";
  if (ui.inspectSummaryStats) ui.inspectSummaryStats.style.display = showSummary ? "block" : "none";
  if (ui.inspectCanvasHist) ui.inspectCanvasHist.style.display = showSummary ? "block" : "none";
  if (ui.inspectCursorValue) ui.inspectCursorValue.style.display = showSummary ? "block" : "none";
  if (showSummary) {
    inspectSummaryDirty = true;
    renderInspectSummary(state, selectedLayer);
  } else {
    if (ui.inspectSummaryStats) ui.inspectSummaryStats.textContent = "";
    if (ui.inspectCursorValue) ui.inspectCursorValue.textContent = "";
  }
}

function _ensureInspectSummaryUpToDate() {
  if (!ui.inspectMode || ui.inspectMode.value !== "summary") return;
  if (!inspectSummaryDirty && inspectSummaryLastLayer === selectedLayer) return;
  renderInspectSummary(state, selectedLayer);
  inspectSummaryLastLayer = selectedLayer;
  inspectSummaryDirty = false;
}

function pruneBulkSelectedLayers() {
  const existing = new Set(state.layers.map((l) => l.name));
  for (const nm of [...bulkSelectedLayers]) {
    if (!existing.has(nm)) bulkSelectedLayers.delete(nm);
  }
}

function isManageScreenActive() {
  const p = document.querySelector('.screenPanel[data-screen="manage"]');
  return !!p && p.classList.contains("screenPanel--active");
}

function updateBulkDeleteInfo() {
  if (!ui.bulkDeleteInfo) return;
  if (!isManageScreenActive()) return;
  pruneBulkSelectedLayers();
  const n = bulkSelectedLayers.size;
  ui.bulkDeleteInfo.textContent = n ? `Selected: ${n}` : "";
  if (ui.bulkDeleteBtn) ui.bulkDeleteBtn.disabled = n === 0;
}

function bulkDeleteSelected() {
  pruneBulkSelectedLayers();
  const names = [...bulkSelectedLayers];
  if (!names.length) return;
  
  // Filter out layers linked to 'let' operations
  const letLayerNames = new Set(
    layerOps.filter((op) => op.type === "let" && op.var).map((op) => op.var)
  );
  const deletable = names.filter((nm) => !letLayerNames.has(nm));
  const skipped = names.filter((nm) => letLayerNames.has(nm));
  
  if (!deletable.length) {
    alert(`Cannot delete: all ${skipped.length} selected layer(s) are linked to Layer Ops.\nRemove them from Layer Ops first.`);
    return;
  }
  
  const sample = deletable.slice(0, 8).join(", ");
  const more = deletable.length > 8 ? ` (+${deletable.length - 8} more)` : "";
  let msg = `Delete ${deletable.length} layer(s)?\n${sample}${more}`;
  if (skipped.length > 0) {
    msg += `\n\n(Skipping ${skipped.length} layer(s) linked to Layer Ops)`;
  }
  if (!confirm(msg)) return;

  for (const nm of deletable) removeLayer(state, nm);
  bulkSelectedLayers.clear();
  if (!state.layers.some((l) => l.name === selectedLayer)) selectedLayer = state.layers[0]?.name || "";
  syncLayerSelect();
  markDirty();
  saveToLocalStorage();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function clampCounts(v) {
  const n = Math.round(Number(v) || 0);
  return Math.max(0, n);
}

function hexToRgb(hex) {
  const s = String(hex || "").trim();
  const m = /^#?([0-9a-fA-F]{6})$/.exec(s);
  const h = m ? m[1] : "4caf50";
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function encodeFloat32Base64(f32) {
  const bytes = new Uint8Array(f32.buffer.slice(f32.byteOffset, f32.byteOffset + f32.byteLength));
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

function decodeFloat32Base64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const n = Math.floor(bytes.byteLength / 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, n);
}

function makeEmptyState(H, W) {
  return {
    version: 1,
    H,
    W,
    layers: [],
    data: {},
  };
}

function addLayer(state, { name, kind, init, value, seed, color }) {
  if (!name || typeof name !== "string") throw new Error("Layer name required");
  if (state.layers.some((l) => l.name === name)) throw new Error("Layer name must be unique");
  if (kind !== "continuous" && kind !== "categorical" && kind !== "counts") throw new Error("Invalid kind");
  const N = state.H * state.W;
  const a = new Float32Array(N);

  if (init === "zeros") {
    // already zeros
  } else if (init === "constant") {
    if (kind === "counts") a.fill(clampCounts(value));
    else if (kind === "categorical") a.fill(Math.round(Number(value) || 0));
    else a.fill(Number(value) || 0);
  } else if (init === "gradient") {
    const scale = Number(value) || 1;
    for (let y = 0; y < state.H; y++) {
      for (let x = 0; x < state.W; x++) {
        const v = (x / Math.max(1, state.W - 1)) * scale;
        a[y * state.W + x] = kind === "counts" ? clampCounts(v) : v;
      }
    }
  } else if (init === "random") {
    const rng = mulberry32(Number(seed) || 0);
    if (kind === "continuous") {
      for (let i = 0; i < N; i++) a[i] = rng();
    } else {
      for (let i = 0; i < N; i++) a[i] = Math.floor(rng() * 4);
    }
  } else {
    throw new Error("Invalid init");
  }

  state.layers.push({ name, kind, color: typeof color === "string" && color ? color : DEFAULT_LAYER_COLOR });
  state.data[name] = a;
}

function ensureGeneTriplets(state) {
  const created = [];
  const existing = new Set(state.layers.map((l) => l.name));

  for (const l of state.layers) {
    if (!l || typeof l.name !== "string") continue;
    if (!l.name.startsWith("gene_")) continue;
    const suffix = l.name.slice("gene_".length);
    if (!suffix) continue;

    const want = [`rna_${suffix}`, `protein_${suffix}`];
    for (const nm of want) {
      if (existing.has(nm)) continue;
      addLayer(state, {
        name: nm,
        kind: l.kind,
        init: "zeros",
        value: 0,
        seed: 0,
        color: l.color,
      });
      existing.add(nm);
      created.push(nm);
    }
  }

  return created;
}

function _applyMaskToArray(outArray, maskBoolArray) {
  for (let i = 0; i < outArray.length; i++) {
    if (!maskBoolArray[i]) outArray[i] = 0;
  }
}

function makeMask(state, layerName, op, rhs, invert) {
  const meta = state.layers.find((l) => l.name === layerName);
  if (!meta) throw new Error("Unknown mask layer");
  const a = state.data[layerName];
  const v = Number(rhs);
  const m = new Uint8Array(a.length);

  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    let ok = false;
    if (op === "==") ok = x === v;
    else if (op === "!=") ok = x !== v;
    else if (op === ">") ok = x > v;
    else if (op === ">=") ok = x >= v;
    else if (op === "<") ok = x < v;
    else if (op === "<=") ok = x <= v;
    else ok = false;
    m[i] = invert ? (ok ? 0 : 1) : ok ? 1 : 0;
  }
  return m;
}

function applyMaskedOperation(state, targetLayer, maskCfg, opCfg) {
  const meta = state.layers.find((l) => l.name === targetLayer);
  if (!meta) throw new Error("Unknown target layer");
  const a = state.data[targetLayer];

  const mask = makeMask(state, maskCfg.layer, maskCfg.op, maskCfg.value, maskCfg.invert);
  let n = 0;
  for (let i = 0; i < mask.length; i++) n += mask[i] ? 1 : 0;
  if (n === 0) throw new Error("Mask matched 0 cells");

  const opType = String(opCfg.type);
  if (opType === "set_constant") {
    const v = meta.kind === "categorical" ? Math.round(Number(opCfg.value)) : meta.kind === "counts" ? clampCounts(opCfg.value) : Number(opCfg.value);
    for (let i = 0; i < a.length; i++) if (mask[i]) a[i] = v;
    return;
  }

  const mn = Number(opCfg.min);
  const mx = Number(opCfg.max);
  const lo = Math.min(mn, mx);
  const hi = Math.max(mn, mx);
  const rng = mulberry32(Math.floor(Number(opCfg.seed)) || 0);

  if (opType === "set_random_uniform") {
    if (meta.kind === "categorical" || meta.kind === "counts") {
      const ilo = Math.round(lo);
      const ihi = Math.round(hi);
      const a0 = meta.kind === "counts" ? Math.max(0, Math.min(ilo, ihi)) : Math.min(ilo, ihi);
      const b0 = meta.kind === "counts" ? Math.max(0, Math.max(ilo, ihi)) : Math.max(ilo, ihi);
      const span = Math.max(1, b0 - a0 + 1);
      for (let i = 0; i < a.length; i++) {
        if (!mask[i]) continue;
        a[i] = a0 + Math.floor(rng() * span);
      }
    } else {
      for (let i = 0; i < a.length; i++) {
        if (!mask[i]) continue;
        a[i] = lo + (hi - lo) * rng();
      }
    }
    return;
  }

  if (opType === "add_random_uniform") {
    if (meta.kind === "categorical" || meta.kind === "counts") {
      const ilo = Math.round(lo);
      const ihi = Math.round(hi);
      // For counts: allow negative deltas but clamp result to >=0.
      const a0 = meta.kind === "counts" ? Math.min(ilo, ihi) : Math.min(ilo, ihi);
      const b0 = meta.kind === "counts" ? Math.max(ilo, ihi) : Math.max(ilo, ihi);
      const span = Math.max(1, b0 - a0 + 1);
      for (let i = 0; i < a.length; i++) {
        if (!mask[i]) continue;
        const dv = a0 + Math.floor(rng() * span);
        if (meta.kind === "counts") a[i] = clampCounts(a[i] + dv);
        else a[i] = Math.round(a[i] + dv);
      }
    } else {
      for (let i = 0; i < a.length; i++) {
        if (!mask[i]) continue;
        a[i] = a[i] + (lo + (hi - lo) * rng());
      }
    }
    return;
  }

  throw new Error("Unknown operation");
}

function bulkAddLayersMasked(state, cfg) {
  const {
    prefix,
    start,
    count,
    kind,
    init,
    value,
    seed,
    collision,
    maskLayer,
    maskOp,
    maskValue,
    maskInvert,
  } = cfg;

  const n = Math.max(1, Math.floor(Number(count)));
  const startN = Math.floor(Number(start));
  const baseSeed = Math.floor(Number(seed));
  const mask = makeMask(state, String(maskLayer), String(maskOp), Number(maskValue), !!maskInvert);

  for (let i = 0; i < n; i++) {
    const name = `${String(prefix)}${startN + i}`;
    if (state.layers.some((l) => l.name === name)) {
      const mode = String(collision || "error");
      if (mode === "skip") continue;
      if (mode === "overwrite") removeLayer(state, name);
      else throw new Error(`Layer already exists: ${name}`);
    }
    addLayer(state, {
      name,
      kind: String(kind),
      init: String(init),
      value: Number(value),
      seed: baseSeed + i,
    });
    _applyMaskToArray(state.data[name], mask);
  }
}

function removeLayer(state, name) {
  state.layers = state.layers.filter((l) => l.name !== name);
  delete state.data[name];
}

function renameLayer(state, oldName, newName) {
  if (!newName || typeof newName !== "string") throw new Error("New name required");
  if (state.layers.some((l) => l.name === newName)) throw new Error("New name already exists");
  const meta = state.layers.find((l) => l.name === oldName);
  if (!meta) throw new Error("Unknown layer");
  meta.name = newName;
  state.data[newName] = state.data[oldName];
  delete state.data[oldName];
}

function makeDemoState(H, W, seed = 0) {
  const rng = mulberry32(seed);
  const s = makeEmptyState(H, W);

  addLayer(s, { name: "cell_type", kind: "categorical", init: "zeros", value: 0, seed });
  addLayer(s, { name: "mol_nutrient", kind: "counts", init: "zeros", value: 0, seed });
  addLayer(s, { name: "mol_toxin", kind: "counts", init: "zeros", value: 0, seed });
  addLayer(s, { name: "prot_tight_junction", kind: "counts", init: "zeros", value: 0, seed });

  const cell = s.data.cell_type;
  const nutrient = s.data.mol_nutrient;
  const toxin = s.data.mol_toxin;
  const tj = s.data.prot_tight_junction;

  const band0 = Math.floor(0.35 * H);
  const band1 = Math.floor(0.65 * H);
  for (let y = band0; y < band1; y++) {
    for (let x = 0; x < W; x++) cell[y * W + x] = 1;
  }

  function circle(cy, cx, r, val) {
    const r2 = r * r;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dy = y - cy;
        const dx = x - cx;
        if (dy * dy + dx * dx <= r2) cell[y * W + x] = val;
      }
    }
  }

  for (let k = 0; k < 2; k++) {
    const cy = (0.42 + rng() * 0.16) * H;
    const cx = (0.15 + rng() * 0.20) * W;
    const r = (0.06 + rng() * 0.04) * Math.min(H, W);
    circle(cy, cx, r, 2);
  }
  for (let k = 0; k < 8; k++) {
    const cy = (0.05 + rng() * 0.90) * H;
    const cx = (0.05 + rng() * 0.90) * W;
    const r = (0.015 + rng() * 0.015) * Math.min(H, W);
    circle(cy, cx, r, 3);
  }

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const g = x / Math.max(1, W - 1);
      nutrient[i] = clamp(g + 0.15 * rng(), 0, 1);
    }
  }

  function gaussianBlob(cy, cx, sigma) {
    const s2 = 2 * sigma * sigma;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dy = y - cy;
        const dx = x - cx;
        const v = Math.exp(-(dy * dy + dx * dx) / s2);
        toxin[y * W + x] = clamp(toxin[y * W + x] + 0.9 * v, 0, 1);
      }
    }
  }

  gaussianBlob(0.25 * H, 0.75 * W, 0.14 * Math.min(H, W));
  for (let i = 0; i < toxin.length; i++) toxin[i] = clamp(toxin[i] + 0.15 * rng(), 0, 1);

  for (let y = band0; y < band1; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      tj[i] = clamp(0.75 + 0.25 * rng(), 0, 1);
    }
  }

  ensureGeneTriplets(s);
  return s;
}

function serializeState(state) {
  const out = {
    version: 1,
    H: state.H,
    W: state.W,
    measurements_config: buildFunctionsConfigJson(),
    layer_ops_config: buildLayerOpsConfigJson(),
    layers: state.layers,
    data: {},
  };
  for (const layer of state.layers) {
    out.data[layer.name] = {
      dtype: "float32",
      b64: encodeFloat32Base64(state.data[layer.name]),
    };
  }
  return JSON.stringify(out, null, 2);
}

function parseState(jsonText) {
  const o = JSON.parse(jsonText);
  if (!o || typeof o !== "object") throw new Error("Invalid JSON");
  if (o.version !== 1) throw new Error("Unsupported version");

  _tryApplyEmbeddedMeasurementsConfig(o);
  _tryApplyEmbeddedLayerOpsConfig(o);

  const H = Number(o.H);
  const W = Number(o.W);
  if (!Number.isInteger(H) || !Number.isInteger(W) || H <= 0 || W <= 0) throw new Error("Invalid H/W");
  if (!Array.isArray(o.layers)) throw new Error("Invalid layers");
  if (!o.data || typeof o.data !== "object") throw new Error("Invalid data");

  const state = makeEmptyState(H, W);
  for (const meta of o.layers) {
    if (!meta || typeof meta !== "object") throw new Error("Invalid layer meta");
    const name = String(meta.name);
    const kind = String(meta.kind);
    if (kind !== "continuous" && kind !== "categorical" && kind !== "counts") throw new Error("Invalid kind");
    const color = typeof meta.color === "string" && meta.color.trim() ? meta.color.trim() : DEFAULT_LAYER_COLOR;
    const d = o.data[name];
    if (!d || typeof d !== "object") throw new Error(`Missing data for ${name}`);
    if (d.dtype !== "float32") throw new Error("Only float32 supported");
    const f32 = decodeFloat32Base64(String(d.b64));
    if (f32.length !== H * W) throw new Error(`Bad buffer length for ${name}`);
    if (kind === "counts") {
      for (let i = 0; i < f32.length; i++) f32[i] = clampCounts(f32[i]);
    }
    state.layers.push({ name, kind, color });
    state.data[name] = f32;
  }

  ensureGeneTriplets(state);
  return state;
}

function computeStats(state, layerName) {
  const meta = state.layers.find((l) => l.name === layerName);
  const a = state.data[layerName];
  if (!meta || !a) return { text: "" };

  if (meta.kind === "continuous") {
    let mn = Infinity,
      mx = -Infinity,
      sum = 0,
      n = a.length;
    for (let i = 0; i < n; i++) {
      const v = a[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    const mean = sum / Math.max(1, n);
    return { text: `min=${mn.toFixed(4)} max=${mx.toFixed(4)} mean=${mean.toFixed(4)}` };
  }

  const counts = new Map();
  for (let i = 0; i < a.length; i++) {
    const k = Math.round(a[i]);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const items = [...counts.entries()].sort((x, y) => x[0] - y[0]);
  const preview = items.slice(0, 10).map(([k, c]) => `${k}:${c}`).join(" ");
  const extra = items.length > 10 ? ` (+${items.length - 10} more)` : "";
  return { text: `unique=${items.length} ${preview}${extra}` };
}

function draw(state, layerName) {
  const ctx = ui.canvas.getContext("2d");
  const overlayCtx = ui.overlay.getContext("2d");
  const zoom = Number(ui.zoomInput.value);
  const showGrid = ui.gridLinesInput.checked;

  const cssW = state.W * zoom;
  const cssH = state.H * zoom;
  ui.canvas.width = cssW;
  ui.canvas.height = cssH;
  ui.overlay.width = cssW;
  ui.overlay.height = cssH;
  ui.canvas.style.width = cssW + "px";
  ui.canvas.style.height = cssH + "px";
  ui.overlay.style.width = cssW + "px";
  ui.overlay.style.height = cssH + "px";

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cssW, cssH);

  const meta = state.layers.find((l) => l.name === layerName);
  const a = state.data[layerName];
  if (!meta || !a) return;

  if (meta.kind === "categorical") {
    const tint = hexToRgb(meta.color);
    for (let y = 0; y < state.H; y++) {
      for (let x = 0; x < state.W; x++) {
        const v = Math.round(a[y * state.W + x]);
        if (v === 0) {
          ctx.fillStyle = "rgb(0,0,0)";
        } else {
          const c = palette[((v % palette.length) + palette.length) % palette.length];
          const alpha = 0.70;
          const r = Math.round((1 - alpha) * c[0] + alpha * tint.r);
          const g = Math.round((1 - alpha) * c[1] + alpha * tint.g);
          const b = Math.round((1 - alpha) * c[2] + alpha * tint.b);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
        }
        ctx.fillRect(x * zoom, y * zoom, zoom, zoom);
      }
    }
  } else {
    let mn = Infinity,
      mx = -Infinity;
    for (let i = 0; i < a.length; i++) {
      const v = meta.kind === "counts" ? clampCounts(a[i]) : a[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const denom = mx - mn === 0 ? 1e-6 : mx - mn;
    const col = hexToRgb(meta.color);
    for (let y = 0; y < state.H; y++) {
      for (let x = 0; x < state.W; x++) {
        const raw = meta.kind === "counts" ? clampCounts(a[y * state.W + x]) : a[y * state.W + x];
        const v = (raw - mn) / denom;
        const t = clamp(v, 0, 1);
        const r = Math.round(col.r * t);
        const g = Math.round(col.g * t);
        const b = Math.round(col.b * t);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * zoom, y * zoom, zoom, zoom);
      }
    }
  }

  if (showGrid && zoom >= 6) {
    ctx.strokeStyle = "rgba(0,0,0,.35)";
    ctx.lineWidth = 1;
    for (let y = 0; y <= state.H; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * zoom + 0.5);
      ctx.lineTo(cssW, y * zoom + 0.5);
      ctx.stroke();
    }
    for (let x = 0; x <= state.W; x++) {
      ctx.beginPath();
      ctx.moveTo(x * zoom + 0.5, 0);
      ctx.lineTo(x * zoom + 0.5, cssH);
      ctx.stroke();
    }
  }

  overlayCtx.clearRect(0, 0, cssW, cssH);
}

function renderInspectTable(state, y, x) {
  if (!ui.inspectTable) return;
  const rows = state.layers
    .map((l) => {
      const a = state.data[l.name];
      if (!a) return `<tr><td>${l.name}</td><td>${l.kind}</td><td>–</td></tr>`;
      const v = a[y * state.W + x];
      const vv = l.kind === "categorical" || l.kind === "counts" ? String(clampCounts(v)) : Number(v).toFixed(6);
      return `<tr><td>${l.name}</td><td>${l.kind}</td><td>${vv}</td></tr>`;
    })
    .join("");
  ui.inspectTable.innerHTML = `
    <table>
      <thead><tr><th>name</th><th>kind</th><th>value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderInspectSummary(state, layerName) {
  if (!ui.inspectSummaryStats || !ui.inspectCanvasHist) return;
  const meta = state.layers.find((l) => l.name === layerName) || null;
  const a = state.data[layerName];
  if (!meta || !a) {
    ui.inspectSummaryStats.textContent = "";
    const ctx = ui.inspectCanvasHist.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, ui.inspectCanvasHist.width, ui.inspectCanvasHist.height);
    return;
  }

  const kind = String(meta.kind || "continuous");
  const n = state.H * state.W;

  if (kind === "categorical") {
    const counts = new Map();
    for (let i = 0; i < a.length; i++) {
      const k = Math.round(a[i]);
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    const items = [...counts.entries()].sort((x, y) => y[1] - x[1]);
    const top = items
      .slice(0, 10)
      .map(([k, c]) => `${k}:${c} (${_stepsFmt((100 * c) / Math.max(1, n))}%)`)
      .join("  ");
    const extra = items.length > 10 ? ` (+${items.length - 10} more)` : "";
    ui.inspectSummaryStats.textContent = `layer=${layerName} kind=${kind} unique=${items.length}${extra}  ${top}`;

    const p = _stepsPrepPlotCanvas(ui.inspectCanvasHist, 520, 180);
    if (p) {
      const { ctx, W, H } = p;
      ctx.fillStyle = "rgba(255,255,255,.02)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "rgba(255,255,255,.65)";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillText("categorical (no histogram)", W / 2, H / 2);
    }
    return;
  }

  let mn = Infinity,
    mx = -Infinity,
    sum = 0,
    sum2 = 0,
    nz = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
    sum2 += v * v;
    if (v !== 0) nz++;
  }
  const mean = sum / Math.max(1, n);
  const var0 = sum2 / Math.max(1, n) - mean * mean;
  const std = Math.sqrt(Math.max(0, var0));
  const nzPct = (100 * nz) / Math.max(1, n);
  ui.inspectSummaryStats.textContent = `layer=${layerName} kind=${kind}  min=${_stepsFmt(mn)}  max=${_stepsFmt(mx)}  mean=${_stepsFmt(mean)}  std=${_stepsFmt(std)}  nonzero=${_stepsFmt(nzPct)}%`;

  const maskName = String(inspectHistMaskLayer || "").trim();
  const maskArr = maskName ? (state.data?.[maskName] || null) : null;
  _drawInspectHistogram(ui.inspectCanvasHist, a, mn, mx, maskName, maskArr, inspectHistMaskOp, inspectHistMaskValue);
}

function _drawInspectHistogram(canvas, values, mn, mx, maskLayerName, maskArr, maskOp, maskValue) {
  if (!canvas || !values || !values.length) return;
  const p = _stepsPrepPlotCanvas(canvas, 520, 180);
  if (!p) return;
  const { ctx, W, H } = p;

  ctx.fillStyle = "rgba(255,255,255,.02)";
  ctx.fillRect(0, 0, W, H);

  let lo = mn;
  let hi = mx;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
    lo = 0;
    hi = 1;
  }

  const useMask = !!(maskArr && maskArr.length === values.length && String(maskLayerName || "").trim());
  const mv = Number(maskValue);

  if (useMask) {
    let mn2 = Infinity;
    let mx2 = -Infinity;
    let nPass = 0;
    for (let i = 0; i < values.length; i++) {
      const m = maskArr[i];
      let pass = false;
      if (maskOp === "==") pass = m === mv;
      else if (maskOp === "!=") pass = m !== mv;
      else if (maskOp === ">") pass = m > mv;
      else if (maskOp === ">=") pass = m >= mv;
      else if (maskOp === "<") pass = m < mv;
      else if (maskOp === "<=") pass = m <= mv;
      if (!pass) continue;
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < mn2) mn2 = v;
      if (v > mx2) mx2 = v;
      nPass++;
    }
    if (!nPass) {
      ctx.fillStyle = "rgba(255,255,255,.02)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "rgba(255,255,255,.65)";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillText("no values match mask", W / 2, H / 2);
      return;
    }
    if (Number.isFinite(mn2) && Number.isFinite(mx2) && mn2 !== mx2) {
      lo = mn2;
      hi = mx2;
    }
  }

  const fontPx = 11;
  ctx.font = `${fontPx}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace`;

  const xLo = _stepsFmt(lo);
  const xHi = _stepsFmt(hi);
  const yLabel = "count";
  const xLabel = "value";

  // Dynamic padding so labels are never clipped.
  const yLabelW = ctx.measureText(yLabel).width;
  const yTickW = ctx.measureText("999999").width;
  const padL = Math.ceil(Math.max(34, yLabelW + yTickW + 18));
  const padR = 10;
  const padT = 10;
  const padB = Math.ceil(Math.max(28, fontPx * 2.6));

  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  const bins = Math.min(80, Math.max(20, Math.floor(plotW / 7)));
  const counts = new Array(bins).fill(0);
  const denom = hi - lo;
  if (!(denom > 0)) return;
  for (let i = 0; i < values.length; i++) {
    if (useMask) {
      const m = maskArr[i];
      let pass = false;
      if (maskOp === "==") pass = m === mv;
      else if (maskOp === "!=") pass = m !== mv;
      else if (maskOp === ">") pass = m > mv;
      else if (maskOp === ">=") pass = m >= mv;
      else if (maskOp === "<") pass = m < mv;
      else if (maskOp === "<=") pass = m <= mv;
      if (!pass) continue;
    }

    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const t = (v - lo) / denom;
    if (t < 0 || t > 1) continue;
    const bi = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    counts[bi]++;
  }

  let maxC = 1;
  for (const c of counts) if (c > maxC) maxC = c;

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,.06)";
  ctx.lineWidth = 1;
  for (let k = 1; k <= 3; k++) {
    const yy = padT + (plotH * k) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, yy + 0.5);
    ctx.lineTo(padL + plotW, yy + 0.5);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  // Bars
  const barW = plotW / bins;
  const gradient = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  gradient.addColorStop(0, "rgba(10, 132, 255, 0.70)");
  gradient.addColorStop(1, "rgba(10, 132, 255, 0.22)");
  ctx.fillStyle = gradient;
  for (let i = 0; i < bins; i++) {
    const h = (counts[i] / maxC) * plotH;
    const x = padL + i * barW;
    const y = padT + plotH - h;
    ctx.fillRect(x, y, Math.max(1, barW - 1), h);
  }

  // Optional smooth outline
  ctx.strokeStyle = "rgba(10, 132, 255, 0.85)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < bins; i++) {
    const h = (counts[i] / maxC) * plotH;
    const x = padL + (i + 0.5) * barW;
    const y = padT + plotH - h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = "rgba(255,255,255,.78)";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  ctx.fillText(String(maxC), padL - 8, padT + 8);
  ctx.fillText("0", padL - 8, padT + plotH);

  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  ctx.fillText(xLo, padL, padT + plotH + 6);
  ctx.fillText(xHi, padL + plotW, padT + plotH + 6);

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.fillText(xLabel, padL + plotW / 2, H - fontPx - 2);

  ctx.save();
  ctx.translate(fontPx + 2, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function paintCircle(state, layerName, cy, cx, radius, value) {
  const meta = state.layers.find((l) => l.name === layerName);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) return;
  const a = state.data[layerName];
  const v0 = meta.kind === "counts" ? clampCounts(value) : Math.round(Number(value) || 0);
  const r = Math.max(0, Math.floor(radius));
  const r2 = r * r;
  for (let y = cy - r; y <= cy + r; y++) {
    if (y < 0 || y >= state.H) continue;
    for (let x = cx - r; x <= cx + r; x++) {
      if (x < 0 || x >= state.W) continue;
      const dy = y - cy;
      const dx = x - cx;
      if (dy * dy + dx * dx <= r2) a[y * state.W + x] = v0;
    }
  }
}

function fillRect(state, layerName, y0, x0, y1, x1, value) {
  const meta = state.layers.find((l) => l.name === layerName);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) return;
  const a = state.data[layerName];
  const v0 = meta.kind === "counts" ? clampCounts(value) : Math.round(Number(value) || 0);
  const ya = clamp(Math.min(y0, y1), 0, state.H - 1);
  const yb = clamp(Math.max(y0, y1), 0, state.H - 1);
  const xa = clamp(Math.min(x0, x1), 0, state.W - 1);
  const xb = clamp(Math.max(x0, x1), 0, state.W - 1);
  for (let y = ya; y <= yb; y++) {
    for (let x = xa; x <= xb; x++) a[y * state.W + x] = v0;
  }
}

let state = makeDemoState(64, 96, 0);
let selectedLayer = "cell_type";

function _inspectPopulateHistMaskLayerSelect() {
  if (!ui.inspectHistMaskLayer || !ui.inspectHistMaskLayer.parentNode) return;
  
  const layerNames = state.layers.map(l => String(l?.name || "")).filter(nm => nm);
  const avail = new Set(layerNames);
  if (inspectHistMaskLayer && !avail.has(String(inspectHistMaskLayer))) inspectHistMaskLayer = "";
  
  const searchable = makeSearchableSelect(layerNames, inspectHistMaskLayer || "", "(none)");
  searchable.input.className = "input input--tiny";
  ui.inspectHistMaskLayer.replaceWith(searchable.wrapper);
  ui.inspectHistMaskLayer = searchable.input;
}

function _inspectInitHistMaskControls() {
  try {
    inspectHistMaskLayer = String(localStorage.getItem(INSPECT_HIST_MASK_LAYER_KEY) || "");
  } catch {}
  try {
    inspectHistMaskOp = String(localStorage.getItem(INSPECT_HIST_MASK_OP_KEY) || "==");
  } catch {}
  try {
    const v = Number(localStorage.getItem(INSPECT_HIST_MASK_VALUE_KEY));
    if (Number.isFinite(v)) inspectHistMaskValue = v;
  } catch {}

  if (ui.inspectHistMaskOp) ui.inspectHistMaskOp.value = String(inspectHistMaskOp || "==");
  if (ui.inspectHistMaskValue) ui.inspectHistMaskValue.value = String(inspectHistMaskValue);
}

function syncLayerSelect() {
  pruneBulkSelectedLayers();
  pruneOpTargetsSelected();
  
  const layerNames = state.layers.map(l => l.name);
  const currentLayer = ui.layerSelect?.value || selectedLayer;
  
  // Replace select with searchable input
  if (ui.layerSelect && ui.layerSelect.parentNode) {
    const searchable = makeSearchableSelect(
      layerNames,
      currentLayer,
      "Select layer...",
      (val) => {
        selectedLayer = val;
        switchLayer(val);
      }
    );
    searchable.input.className = "input input--tiny";
    ui.layerSelect.replaceWith(searchable.wrapper);
    ui.layerSelect = searchable.input;
  }

  // Bulk operations mask layer
  if (ui.maskLayer && ui.maskLayer.parentNode) {
    const searchable = makeSearchableSelect(layerNames, ui.maskLayer.value || "", "Select mask layer...");
    ui.maskLayer.replaceWith(searchable.wrapper);
    ui.maskLayer = searchable.input;
  }

  // Single operation target layer
  if (ui.opTargetLayer && ui.opTargetLayer.parentNode) {
    const searchable = makeSearchableSelect(layerNames, ui.opTargetLayer.value || "", "Select target...");
    ui.opTargetLayer.replaceWith(searchable.wrapper);
    ui.opTargetLayer = searchable.input;
  }

  // Single operation mask layer
  if (ui.opMaskLayer && ui.opMaskLayer.parentNode) {
    const searchable = makeSearchableSelect(layerNames, ui.opMaskLayer.value || "", "Select mask layer...");
    ui.opMaskLayer.replaceWith(searchable.wrapper);
    ui.opMaskLayer = searchable.input;
  }

  // Functions insert layer
  if (ui.fnInsertLayer && ui.fnInsertLayer.parentNode) {
    const searchable = makeSearchableSelect(layerNames, "", "Select layer...");
    ui.fnInsertLayer.replaceWith(searchable.wrapper);
    ui.fnInsertLayer = searchable.input;
  }

  if (ui.fnInsertFn) {
    ui.fnInsertFn.innerHTML = "";
    for (const f of FN_INSERTER_FUNCS) {
      const opt = document.createElement("option");
      opt.value = f.value;
      opt.textContent = f.value;
      ui.fnInsertFn.appendChild(opt);
    }
  }

  // Layer ops insert layer (includes let variables)
  if (ui.opsInsertLayer && ui.opsInsertLayer.parentNode) {
    const vars = layerOps
      .filter((s) => s && s.enabled !== false && s.type === "let")
      .map((s) => String(s.var || "").trim())
      .filter((nm) => _isValidIdentifier(nm));
    const allOptions = [...layerNames, ...vars];
    const searchable = makeSearchableSelect(allOptions, "", "Select layer or variable...");
    ui.opsInsertLayer.replaceWith(searchable.wrapper);
    ui.opsInsertLayer = searchable.input;
  }

  if (ui.opsInsertFn) {
    ui.opsInsertFn.innerHTML = "";
    for (const f of OPS_INSERTER_FUNCS) {
      const opt = document.createElement("option");
      opt.value = f.value;
      opt.textContent = f.value;
      ui.opsInsertFn.appendChild(opt);
    }
  }

  if (!state.layers.some((l) => l.name === selectedLayer)) selectedLayer = state.layers[0]?.name || "";
  ui.layerSelect.value = selectedLayer;

  _inspectPopulateHistMaskLayerSelect();

  if (!state.layers.some((l) => l.name === ui.maskLayer.value)) {
    ui.maskLayer.value = state.layers[0]?.name || "";
  }

  if (ui.opTargetLayer && !state.layers.some((l) => l.name === ui.opTargetLayer.value)) {
    ui.opTargetLayer.value = state.layers[0]?.name || "";
  }
  if (ui.opMaskLayer && !state.layers.some((l) => l.name === ui.opMaskLayer.value)) {
    ui.opMaskLayer.value = state.layers[0]?.name || "";
  }

  if (ui.derivedPrototypeLayer) {
    ui.derivedPrototypeLayer.innerHTML = "";
    for (const l of state.layers) {
      const opt = document.createElement("option");
      opt.value = l.name;
      opt.textContent = l.name;
      ui.derivedPrototypeLayer.appendChild(opt);
    }
    if (!state.layers.some((l) => l.name === ui.derivedPrototypeLayer.value)) {
      ui.derivedPrototypeLayer.value = state.layers[0]?.name || "";
    }
  }

  if (isManageScreenActive()) {
    renderLayersTable();
    updateBulkAddPreview();
  }
  renderOpTargetsList();
  renderFunctionsSpecsTable();
  renderLayerOpsTable();
  updateDerivedPreview();
  updateMaskedOpsPreview();
  updateBulkDeleteInfo();
}

function renderLayersTable() {
  if (!ui.layersTable) return;
  if (!isManageScreenActive()) return;

  pruneBulkSelectedLayers();

  const query = String(layersFilterText || "").trim().toLowerCase();
  const layers = query
    ? state.layers.filter((l) => String(l.name).toLowerCase().includes(query))
    : state.layers;

  const groups = new Map();
  for (const l of layers) {
    const key = layersGroupByPrefix ? String(l.name).split("_")[0] : "all";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(l);
  }

  const wrap = document.createElement("div");
  const t = document.createElement("table");

  const thead = document.createElement("thead");
  const hr = document.createElement("tr");

  // Sel column: select-all for visible rows
  {
    const th = document.createElement("th");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    const visible = layers.map((l) => l.name);
    const nSelected = visible.reduce((acc, nm) => acc + (bulkSelectedLayers.has(nm) ? 1 : 0), 0);
    cb.checked = visible.length > 0 && nSelected === visible.length;
    cb.indeterminate = nSelected > 0 && nSelected < visible.length;
    cb.addEventListener("change", () => {
      if (cb.checked) {
        for (const nm of visible) bulkSelectedLayers.add(nm);
      } else {
        for (const nm of visible) bulkSelectedLayers.delete(nm);
      }
      updateBulkDeleteInfo();
      renderLayersTable();
    });
    th.appendChild(cb);
    hr.appendChild(th);
  }
  for (const label of ["Name", "Kind", "Color", "Actions"]) {
    const th = document.createElement("th");
    th.textContent = label;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  t.appendChild(thead);

  const tbody = document.createElement("tbody");

  const groupKeys = [...groups.keys()].sort((a, b) => a.localeCompare(b));
  for (const gk of groupKeys) {
    const isCollapsed = collapsedGroups.has(gk);

    const trh = document.createElement("tr");
    trh.className = "layersTable__groupRow";
    const thSel = document.createElement("td");
    thSel.colSpan = 4;
    const btn = document.createElement("button");
    btn.className = "btn btn--secondary btn--tiny";
    btn.textContent = isCollapsed ? `▶ ${gk} (${groups.get(gk).length})` : `▼ ${gk} (${groups.get(gk).length})`;
    btn.addEventListener("click", () => {
      if (collapsedGroups.has(gk)) collapsedGroups.delete(gk);
      else collapsedGroups.add(gk);
      renderLayersTable();
    });
    thSel.appendChild(btn);
    trh.appendChild(thSel);
    tbody.appendChild(trh);

    if (isCollapsed) continue;

    for (const l of groups.get(gk)) {
    const tr = document.createElement("tr");
    const isActive = l.name === selectedLayer;
    const isBulk = bulkSelectedLayers.has(l.name);
    if (isActive && isBulk) tr.className = "layersTable__rowActive layersTable__rowSelected";
    else if (isActive) tr.className = "layersTable__rowActive";
    else if (isBulk) tr.className = "layersTable__rowSelected";

    const tdSel = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = bulkSelectedLayers.has(l.name);
    cb.addEventListener("change", () => {
      if (cb.checked) bulkSelectedLayers.add(l.name);
      else bulkSelectedLayers.delete(l.name);
      updateBulkDeleteInfo();
      renderLayersTable();
    });
    tdSel.appendChild(cb);
    tr.appendChild(tdSel);

    const tdName = document.createElement("td");
    const nameInput = document.createElement("input");
    nameInput.className = "input input--tiny layersTable__name";
    nameInput.value = l.name;
    nameInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") nameInput.blur();
      if (ev.key === "Escape") {
        nameInput.value = l.name;
        nameInput.blur();
      }
    });
    nameInput.addEventListener("blur", () => {
      const next = nameInput.value.trim();
      if (!next || next === l.name) {
        nameInput.value = l.name;
        return;
      }
      try {
        const wasSelected = selectedLayer === l.name;
        const wasBulk = bulkSelectedLayers.has(l.name);
        renameLayer(state, l.name, next);
        if (wasSelected) selectedLayer = next;
        if (wasBulk) {
          bulkSelectedLayers.delete(l.name);
          bulkSelectedLayers.add(next);
        }
        syncLayerSelect();
        markDirty();
        saveToLocalStorage();
      } catch (e) {
        alert(String(e?.message || e));
        nameInput.value = l.name;
      }
    });
    tdName.appendChild(nameInput);
    tr.appendChild(tdName);

    const tdKind = document.createElement("td");
    const kindSelect = document.createElement("select");
    kindSelect.className = "input input--tiny";
    for (const k of ["continuous", "categorical", "counts"]) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      if (k === l.kind) opt.selected = true;
      kindSelect.appendChild(opt);
    }
    kindSelect.addEventListener("change", () => {
      const newKind = kindSelect.value;
      if (newKind !== l.kind) {
        l.kind = newKind;
        markDirty();
        saveToLocalStorage();
        updatePanels();
      }
    });
    tdKind.appendChild(kindSelect);
    tr.appendChild(tdKind);

    const tdColor = document.createElement("td");
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.className = "input input--tiny";
    colorInput.value = typeof l.color === "string" && l.color ? l.color : DEFAULT_LAYER_COLOR;
    colorInput.addEventListener("change", () => {
      l.color = colorInput.value;
      markDirty();
      saveToLocalStorage();
    });
    tdColor.appendChild(colorInput);
    tr.appendChild(tdColor);

    const tdAct = document.createElement("td");
    const act = document.createElement("div");
    act.className = "layersTable__actions";

    const btnSel = document.createElement("button");
    btnSel.className = "btn btn--secondary btn--tiny";
    btnSel.textContent = l.name === selectedLayer ? "•" : "→";
    btnSel.addEventListener("click", () => {
      selectedLayer = l.name;
      ui.layerSelect.value = selectedLayer;
      renderLayersTable();
    });

    // Check if this layer is linked to a 'let' operation
    const isLetLayer = layerOps.some((op) => op.type === "let" && op.var === l.name);

    const btnDel = document.createElement("button");
    btnDel.className = "btn btn--danger btn--tiny";
    btnDel.textContent = "Del";
    if (isLetLayer) {
      btnDel.disabled = true;
      btnDel.title = "Remove from Layer Ops to delete";
      btnDel.style.opacity = "0.4";
      btnDel.style.cursor = "not-allowed";
    } else {
      btnDel.addEventListener("click", () => {
        if (!confirm(`Remove layer '${l.name}'?`)) return;
        removeLayer(state, l.name);
        bulkSelectedLayers.delete(l.name);
        if (selectedLayer === l.name) selectedLayer = state.layers[0]?.name || "";
        syncLayerSelect();
        markDirty();
        saveToLocalStorage();
      });
    }

    act.appendChild(btnSel);
    act.appendChild(btnDel);
    tdAct.appendChild(act);
    tr.appendChild(tdAct);

    tbody.appendChild(tr);
    }
  }

  // Add-row
  {
    const tr = document.createElement("tr");

    const tdSel = document.createElement("td");
    tdSel.textContent = "+";
    tr.appendChild(tdSel);

    const tdName = document.createElement("td");
    const nm = document.createElement("input");
    nm.className = "input input--tiny";
    nm.placeholder = "new_layer";
    tdName.appendChild(nm);
    tr.appendChild(tdName);

    const tdKind = document.createElement("td");
    const kindSel = document.createElement("select");
    kindSel.className = "input input--tiny";
    for (const k of ["continuous", "categorical", "counts"]) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      kindSel.appendChild(opt);
    }
    tdKind.appendChild(kindSel);
    tr.appendChild(tdKind);

    const tdColor = document.createElement("td");
    const newColor = document.createElement("input");
    newColor.type = "color";
    newColor.className = "input input--tiny";
    newColor.value = DEFAULT_LAYER_COLOR;
    tdColor.appendChild(newColor);
    tr.appendChild(tdColor);

    const tdAct = document.createElement("td");
    const act = document.createElement("div");
    act.className = "layersTable__actions";
    const btnAdd = document.createElement("button");
    btnAdd.className = "btn btn--primary btn--tiny";
    btnAdd.textContent = "Add";
    btnAdd.addEventListener("click", () => {
      try {
        const name = nm.value.trim();
        addLayer(state, {
          name,
          kind: kindSel.value,
          color: newColor.value,
          init: "zeros",
          value: 0,
          seed: 0,
        });

        ensureGeneTriplets(state);
        selectedLayer = name;
        syncLayerSelect();
        markDirty();
        saveToLocalStorage();
      } catch (e) {
        alert(String(e?.message || e));
      }
    });
    act.appendChild(btnAdd);
    tdAct.appendChild(act);
    tr.appendChild(tdAct);

    tbody.appendChild(tr);
  }

  t.appendChild(tbody);
  wrap.className = "table";
  wrap.appendChild(t);

  ui.layersTable.innerHTML = "";
  ui.layersTable.appendChild(wrap);
}

function updatePanels() {
  const meta = state.layers.find((l) => l.name === selectedLayer);
  if (!meta) return;
  if (ui.activeLayerTitle) ui.activeLayerTitle.textContent = `${selectedLayer} (${meta.kind})`;
  if (ui.activeLayerStats) ui.activeLayerStats.textContent = computeStats(state, selectedLayer).text;
  if (ui.activeLayerColor) {
    const c = typeof meta.color === "string" && meta.color ? meta.color : DEFAULT_LAYER_COLOR;
    if (ui.activeLayerColor.value !== c) ui.activeLayerColor.value = c;
  }

  const isPaintable = meta.kind === "categorical" || meta.kind === "counts";
  ui.editMode.disabled = !isPaintable;
  ui.paintValue.disabled = !isPaintable;
  ui.brushRadius.disabled = !isPaintable;
  ui.toggleEraser.disabled = !isPaintable;
  ui.editMode.parentElement.parentElement.style.opacity = isPaintable ? "1" : "0.55";

  _ensureInspectSummaryUpToDate();
}

function resizeIfNeeded(H, W) {
  if (H === state.H && W === state.W) return;
  const old = state;
  state = makeEmptyState(H, W);
  for (const l of old.layers) {
    addLayer(state, { name: l.name, kind: l.kind, color: l.color, init: "zeros", value: 0, seed: 0 });
  }

  ensureGeneTriplets(state);
  applyAutoFitZoom();
  selectedLayer = state.layers[0]?.name || "";
}

function download(filename, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function tick() {
  try {
    draw(state, selectedLayer);
    updatePanels();
  } catch (e) {
    console.error(e);
  }
  requestAnimationFrame(tick);
}

// UI wiring
ui.layerSelect.addEventListener("change", () => {
  selectedLayer = ui.layerSelect.value;
  if (isManageScreenActive()) renderLayersTable();
});

if (ui.autoFitZoom) {
  ui.autoFitZoom.addEventListener("change", () => {
    applyAutoFitZoom();
    markDirty();
    saveToLocalStorage();
  });
}

if (ui.zoomInput) {
  ui.zoomInput.addEventListener("input", () => {
    if (ui.autoFitZoom?.checked) return;
    markDirty();
  });
}

for (const el of [
  ui.derivedSourcePrefix,
  ui.derivedTargetTemplate,
  ui.derivedMetaFrom,
  ui.derivedPrototypeLayer,
  ui.derivedDataInit,
  ui.derivedSkipExisting,
]) {
  if (!el) continue;
  el.addEventListener("input", () => updateDerivedPreview());
  el.addEventListener("change", () => updateDerivedPreview());
}

for (const el of [ui.bulkPrefix, ui.bulkStart, ui.bulkCount, ui.maskLayer, ui.maskOp, ui.maskValue, ui.maskInvert]) {
  if (!el) continue;
  el.addEventListener("input", () => updateBulkAddPreview());
  el.addEventListener("change", () => updateBulkAddPreview());
}

if (ui.bulkCollision) {
  ui.bulkCollision.addEventListener("change", () => updateBulkAddPreview());
}

if (ui.derivedPresetGeneFromMolecule) {
  ui.derivedPresetGeneFromMolecule.addEventListener("click", () => {
    if (ui.derivedSourcePrefix) ui.derivedSourcePrefix.value = "molecule_";
    if (ui.derivedTargetTemplate) ui.derivedTargetTemplate.value = "gene_{suffix}";
    if (ui.derivedMetaFrom) ui.derivedMetaFrom.value = "prototype";
    if (ui.derivedDataInit) ui.derivedDataInit.value = "zeros";
    if (ui.derivedSkipExisting) ui.derivedSkipExisting.checked = true;
    if (ui.derivedPrototypeLayer) ui.derivedPrototypeLayer.value = _pickPrototypeLayer("gene_");
    updateDerivedPreview();
  });
}

if (ui.derivedApplyBtn) {
  ui.derivedApplyBtn.addEventListener("click", () => {
    try {
      const plan = computeDerivedLayerPlan(state);
      const creates = plan.filter((p) => p.action === "create").length;
      const overwrites = plan.filter((p) => p.action === "overwrite").length;
      const skips = plan.filter((p) => p.action === "skip_exists").length;
      const errs = plan.length - creates - overwrites - skips;

      if (!confirm(`Create derived layers?\ncreate=${creates} overwrite=${overwrites} skip=${skips} errors=${errs}`)) return;
      const res = applyDerivedLayerPlan(state);
      if (res.createdNames.length) selectedLayer = res.createdNames[0];
      syncLayerSelect();
      markDirty();
      saveToLocalStorage();
    } catch (e) {
      alert(String(e?.message || e));
    }
  });
}

if (ui.activeLayerColor) {
  ui.activeLayerColor.addEventListener("change", () => {
    const meta = state.layers.find((l) => l.name === selectedLayer);
    if (!meta) return;
    meta.color = ui.activeLayerColor.value;
    markDirty();
    saveToLocalStorage();
  });
  ui.activeLayerColor.addEventListener("input", () => {
    const meta = state.layers.find((l) => l.name === selectedLayer);
    if (!meta) return;
    meta.color = ui.activeLayerColor.value;
  });
}

// Tabs
function setActiveTab(rootEl, name) {
  if (!rootEl) return;
  for (const b of rootEl.querySelectorAll(".tabBtn")) {
    b.classList.toggle("tabBtn--active", b.dataset.tab === name);
  }
  for (const p of rootEl.querySelectorAll(".tabPanel")) {
    p.classList.toggle("tabPanel--active", p.dataset.tab === name);
  }
}

let stepsBefore = null;
let stepsAfter = null;
let stepsDiff = null;
let stepsSelectedLayer = "";
let stepsMaskLayer = "";
let stepsMaskMode = "nonzero";
let stepsMaskValue = 0;
let stepsShowLetLayers = false;
let stepsScatterEnabled = false;
let stepsScatterSource = "after";
let stepsScatterX = "";
let stepsScatterY = "";

function _stepsFmt(v) {
  if (v == null || !Number.isFinite(v)) return "–";
  const av = Math.abs(v);
  let s;
  if (av >= 1000) s = v.toFixed(2);
  else if (av >= 1) s = v.toFixed(3);
  else s = v.toFixed(6);
  s = s.replace(/\.0+$/, "").replace(/(\.[0-9]*?)0+$/, "$1").replace(/\.$/, "");
  return s;
}

function _stepsSetStatus(text) {
  if (!ui.stepsStatus) return;
  ui.stepsStatus.textContent = String(text || "");
}

function _stepsIsLetLayer(name) {
  return String(name || "").startsWith("__let__");
}

function _stepsRowVisible(row) {
  if (!row || !row.name) return false;
  if (stepsShowLetLayers) return true;
  return !_stepsIsLetLayer(row.name);
}

function _stepsVisibleRows() {
  if (!stepsDiff) return [];
  return stepsDiff.ranked.filter((r) => _stepsRowVisible(r));
}

function _stepsPrepPlotCanvas(canvas, fallbackW = 640, fallbackH = 200) {
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const cssW = Math.max(10, Math.floor(r.width || fallbackW));
  const cssH = Math.max(10, Math.floor(r.height || fallbackH));
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  return { ctx, W: cssW, H: cssH };
}

function _stepsMaskTest(v, mode, t) {
  if (mode === "nonzero") return v !== 0;
  if (mode === "gte") return v >= t;
  if (mode === "eq") return v === t;
  return true;
}

function _stepsGetMaskArray() {
  const name = (stepsMaskLayer || "").trim();
  if (!name) return null;
  const a = stepsAfter?.data?.[name];
  if (a) return a;
  const b = stepsBefore?.data?.[name];
  if (b) return b;
  return null;
}

function _stepsRenderScatterSelects() {
  if (!ui.stepsScatterX || !ui.stepsScatterY) return;
  ui.stepsScatterX.innerHTML = "";
  ui.stepsScatterY.innerHTML = "";
  if (!stepsDiff) return;

  const rows = _stepsVisibleRows();
  for (const row of rows) {
    const optX = document.createElement("option");
    optX.value = row.name;
    optX.textContent = row.name;
    ui.stepsScatterX.appendChild(optX);

    const optY = document.createElement("option");
    optY.value = row.name;
    optY.textContent = row.name;
    ui.stepsScatterY.appendChild(optY);
  }

  if (!stepsScatterX) stepsScatterX = stepsSelectedLayer || rows[0]?.name || "";
  if (!stepsScatterY) {
    const prefer = ["atp", "glucose", "damage", "cell", "circulation"];
    const names = new Set(rows.map((r) => r.name));
    stepsScatterY = prefer.find((p) => names.has(p) && p !== stepsScatterX) || stepsScatterX;
  }
  ui.stepsScatterX.value = stepsScatterX;
  ui.stepsScatterY.value = stepsScatterY;
}

function _stepsGetArrayBySource(bundleBefore, bundleAfter, layerName, source) {
  if (source === "before") return bundleBefore?.data?.[layerName] || null;
  if (source === "after") return bundleAfter?.data?.[layerName] || null;
  if (source === "diff") {
    const b = bundleBefore?.data?.[layerName];
    const a = bundleAfter?.data?.[layerName];
    if (!b || !a || b.length !== a.length) return null;
    const d = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i];
    return d;
  }
  return null;
}

function _stepsDrawScatter(canvas, xs, ys, xLabel, yLabel) {
  const p = _stepsPrepPlotCanvas(canvas, 900, 320);
  if (!p) return;
  const { ctx, W, H } = p;

  const padL = 54;
  const padR = 14;
  const padT = 10;
  const padB = 34;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  let xmn = Infinity,
    xmx = -Infinity,
    ymn = Infinity,
    ymx = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i],
      y = ys[i];
    if (x < xmn) xmn = x;
    if (x > xmx) xmx = x;
    if (y < ymn) ymn = y;
    if (y > ymx) ymx = y;
  }
  if (!Number.isFinite(xmn) || !Number.isFinite(xmx) || xmn === xmx) {
    xmn = 0;
    xmx = 1;
  }
  if (!Number.isFinite(ymn) || !Number.isFinite(ymx) || ymn === ymx) {
    ymn = 0;
    ymx = 1;
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,.04)";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,.06)";
  ctx.lineWidth = 1;
  for (let k = 1; k <= 3; k++) {
    const yy = padT + (plotH * k) / 4;
    const xx = padL + (plotW * k) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, yy + 0.5);
    ctx.lineTo(padL + plotW, yy + 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xx + 0.5, padT);
    ctx.lineTo(xx + 0.5, padT + plotH);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(_stepsFmt(ymx), padL - 6, padT + 6);
  ctx.fillText(_stepsFmt(ymn), padL - 6, padT + plotH);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(_stepsFmt(xmn), padL, padT + plotH + 6);
  ctx.fillText(_stepsFmt((xmn + xmx) / 2), padL + plotW / 2, padT + plotH + 6);
  ctx.fillText(_stepsFmt(xmx), padL + plotW, padT + plotH + 6);

  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(xLabel, padL + plotW / 2, H - 6);
  ctx.save();
  ctx.translate(14, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  const sx = plotW / (xmx - xmn);
  const sy = plotH / (ymx - ymn);
  ctx.fillStyle = "rgba(140,200,255,.30)";
  for (let i = 0; i < xs.length; i++) {
    const x = padL + (xs[i] - xmn) * sx;
    const y = padT + plotH - (ys[i] - ymn) * sy;
    ctx.fillRect(x, y, 1.2, 1.2);
  }
}

function _stepsRenderScatter() {
  if (!ui.stepsCanvasScatter) return;
  if (!stepsScatterEnabled) {
    const p = _stepsPrepPlotCanvas(ui.stepsCanvasScatter, 900, 320);
    if (!p) return;
    const { ctx, W, H } = p;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("scatter disabled", 12, 20);
    return;
  }
  if (!stepsBefore || !stepsAfter) return;
  if (!stepsScatterX || !stepsScatterY) return;

  const maskArr = _stepsGetMaskArray();
  const mMode = stepsMaskMode;
  const mVal = stepsMaskValue;

  const ax = _stepsGetArrayBySource(stepsBefore, stepsAfter, stepsScatterX, stepsScatterSource);
  const ay = _stepsGetArrayBySource(stepsBefore, stepsAfter, stepsScatterY, stepsScatterSource);
  if (!ax || !ay || ax.length !== ay.length) return;

  const maxPoints = 60000;
  const N = ax.length;
  const step = Math.max(1, Math.floor(N / maxPoints));
  const xs = [];
  const ys = [];
  for (let i = 0; i < N; i += step) {
    if (maskArr && !_stepsMaskTest(maskArr[i], mMode, mVal)) continue;
    const x = ax[i];
    const y = ay[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    xs.push(x);
    ys.push(y);
  }
  _stepsDrawScatter(ui.stepsCanvasScatter, xs, ys, `${stepsScatterSource}:${stepsScatterX}`, `${stepsScatterSource}:${stepsScatterY}`);
}

function _stepsReadFileAsText(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("Missing file"));
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = () => reject(new Error("Failed to read file"));
    r.readAsText(file);
  });
}

function _stepsParseBundle(jsonText) {
  const o = JSON.parse(jsonText);
  if (!o || typeof o !== "object") throw new Error("Invalid JSON");
  if (Number(o.version) !== 1) throw new Error("Unsupported gridstate version");
  const H = Number(o.H);
  const W = Number(o.W);
  if (!Number.isInteger(H) || !Number.isInteger(W) || H <= 0 || W <= 0) throw new Error("Invalid H/W");
  if (!Array.isArray(o.layers)) throw new Error("Invalid layers");
  if (!o.data || typeof o.data !== "object") throw new Error("Invalid data");

  const layers = o.layers
    .filter((m) => m && typeof m === "object")
    .map((m) => ({
      name: String(m.name || ""),
      kind: String(m.kind || "continuous"),
      color: typeof m.color === "string" && m.color.trim() ? m.color.trim() : DEFAULT_LAYER_COLOR,
    }))
    .filter((m) => m.name);

  const data = {};
  for (const meta of layers) {
    const d = o.data[meta.name];
    if (!d || typeof d !== "object") continue;
    if (d.dtype !== "float32") continue;
    const f32 = decodeFloat32Base64(String(d.b64));
    if (f32.length !== H * W) continue;
    data[meta.name] = f32;
  }

  return { H, W, layers, data };
}

function _stepsComputeDiff(before, after) {
  if (!before || !after) return null;
  if (before.H !== after.H || before.W !== after.W) throw new Error("H/W mismatch between files");

  const bMeta = new Map(before.layers.map((m) => [m.name, m]));
  const aMeta = new Map(after.layers.map((m) => [m.name, m]));
  const names = Array.from(new Set([...bMeta.keys(), ...aMeta.keys()])).sort();

  const rows = [];
  for (const name of names) {
    const bm = bMeta.get(name) || null;
    const am = aMeta.get(name) || null;
    const kind = (am?.kind || bm?.kind || "continuous").toString();
    const b = before.data[name] || null;
    const a = after.data[name] || null;

    let n = before.H * before.W;
    let meanAbs = null;
    let maxAbs = null;
    let cellsChangedPct = null;
    let meanBefore = null;
    let meanAfter = null;
    let meanDelta = null;
    let meanPctChange = null;
    if (b && a && b.length === n && a.length === n) {
      let sumAbs = 0;
      let mAbs = 0;
      let changed = 0;
      let sumB = 0;
      let sumA = 0;
      for (let i = 0; i < n; i++) {
        const d = a[i] - b[i];
        const ad = Math.abs(d);
        sumAbs += ad;
        if (ad > mAbs) mAbs = ad;
        if (ad !== 0) changed++;
        sumB += b[i];
        sumA += a[i];
      }
      meanAbs = sumAbs / Math.max(1, n);
      maxAbs = mAbs;
      cellsChangedPct = (100 * changed) / Math.max(1, n);
      meanBefore = sumB / Math.max(1, n);
      meanAfter = sumA / Math.max(1, n);
      meanDelta = meanAfter - meanBefore;
      const denom = Math.max(1e-9, Math.abs(meanBefore));
      meanPctChange = (100 * meanDelta) / denom;
    }

    rows.push({
      name,
      kind,
      present: { before: Boolean(b), after: Boolean(a) },
      meanAbs,
      maxAbs,
      cellsChangedPct,
      meanBefore,
      meanAfter,
      meanDelta,
      meanPctChange,
      color: (am?.color || bm?.color || DEFAULT_LAYER_COLOR).toString(),
    });
  }

  const ranked = [...rows].sort((r1, r2) => {
    const a1 = r1.meanAbs ?? -1;
    const a2 = r2.meanAbs ?? -1;
    if (a1 !== a2) return a2 - a1;
    return r1.name.localeCompare(r2.name);
  });

  return { H: before.H, W: before.W, rows, ranked };
}

function _stepsRenderMaskSelect() {
  if (!ui.stepsMaskLayerSelect) return;
  ui.stepsMaskLayerSelect.innerHTML = "";
  const optNone = document.createElement("option");
  optNone.value = "";
  optNone.textContent = "(no mask)";
  ui.stepsMaskLayerSelect.appendChild(optNone);

  if (!stepsDiff) return;
  for (const row of _stepsVisibleRows()) {
    const opt = document.createElement("option");
    opt.value = row.name;
    opt.textContent = row.name;
    ui.stepsMaskLayerSelect.appendChild(opt);
  }

  if (!stepsMaskLayer) {
    const prefer = ["cell", "cells", "circulation", "mask"];
    const names = new Set(stepsDiff.ranked.map((r) => r.name));
    for (const p of prefer) {
      if (names.has(p)) {
        stepsMaskLayer = p;
        break;
      }
    }
  }
  ui.stepsMaskLayerSelect.value = stepsMaskLayer;
  if (ui.stepsMaskMode) ui.stepsMaskMode.value = stepsMaskMode;
  if (ui.stepsMaskValue) ui.stepsMaskValue.value = String(stepsMaskValue);
}

function _stepsEnsureCanvasSize(canvas, H, W, maxPx = 320) {
  if (!canvas) return;
  const scale = Math.max(1, Math.floor(Math.max(H, W) / maxPx));
  const h = Math.max(1, Math.floor(H / scale));
  const w = Math.max(1, Math.floor(W / scale));
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
}

function _stepsComputeRange(arr) {
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return { mn: 0, mx: 0 };
  return { mn, mx };
}

function _stepsComputeMaxAbs(arr) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    const av = Math.abs(arr[i]);
    if (av > m) m = av;
  }
  if (m === 0) m = 1e-9;
  return m;
}

function _stepsSetLegendContinuous(textEl, barEl, mn, mx) {
  if (textEl) textEl.textContent = `${_stepsFmt(mn)} … ${_stepsFmt(mx)}`;
  if (barEl) barEl.style.background = "linear-gradient(90deg, #000, #fff)";
}

function _stepsSetLegendCategorical(textEl, barEl) {
  if (textEl) textEl.textContent = "categorical";
  if (barEl) {
    const n = Math.min(8, palette.length);
    const stops = [];
    for (let i = 0; i < n; i++) {
      const c = palette[i];
      const t0 = (i / n) * 100;
      const t1 = ((i + 1) / n) * 100;
      stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${t0}% ${t1}%`);
    }
    barEl.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  }
}

function _stepsSetLegendDiff(textEl, barEl, maxAbs) {
  if (textEl) textEl.textContent = `-${_stepsFmt(maxAbs)} … +${_stepsFmt(maxAbs)}`;
  if (barEl) barEl.style.background = "linear-gradient(90deg, rgba(80,120,255,.95), rgba(0,0,0,.35), rgba(255,80,80,.95))";
}

function _stepsDrawHeatmap(canvas, arr, H, W, kind, mode, fixedRange = null, tintHex = null) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const cw = canvas.width;
  const ch = canvas.height;
  const img = ctx.createImageData(cw, ch);
  const px = img.data;

  const sx = W / cw;
  const sy = H / ch;

  let mn = Infinity;
  let mx = -Infinity;
  let maxAbs = 0;
  if (mode === "diff") {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      const av = Math.abs(v);
      if (av > maxAbs) maxAbs = av;
    }
    if (maxAbs === 0) maxAbs = 1e-9;
  } else if (kind !== "categorical") {
    const maskArr = arguments.length >= 9 ? arguments[8] : null;
    const maskOp = arguments.length >= 10 ? arguments[9] : "==";
    const maskValue = arguments.length >= 11 ? arguments[10] : 1;
    const useMask = maskArr && maskArr.length === arr.length;
    const mv = Number(maskValue);

    if (fixedRange && Number.isFinite(fixedRange.mn) && Number.isFinite(fixedRange.mx)) {
      mn = fixedRange.mn;
      mx = fixedRange.mx;
    } else {
      // If masking is active, only use pixels that pass the mask to calculate the range
      if (useMask) {
        let validPixelCount = 0;
        for (let i = 0; i < arr.length; i++) {
          const m = maskArr[i];
          let pass = false;
          if (maskOp === "==") pass = m === mv;
          else if (maskOp === "!=") pass = m !== mv;
          else if (maskOp === ">") pass = m > mv;
          else if (maskOp === ">=") pass = m >= mv;
          else if (maskOp === "<") pass = m < mv;
          else if (maskOp === "<=") pass = m <= mv;
          
          if (pass) {
            const v = kind === "counts" ? clampCounts(arr[i]) : arr[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            validPixelCount++;
          }
        }
        
        // If no pixels passed the mask, fall back to full range
        if (validPixelCount === 0) {
          mn = Infinity;
          mx = -Infinity;
          for (let i = 0; i < arr.length; i++) {
            const v = kind === "counts" ? clampCounts(arr[i]) : arr[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        }
      } else {
        // No mask, use all pixels
        for (let i = 0; i < arr.length; i++) {
          const v = kind === "counts" ? clampCounts(arr[i]) : arr[i];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === mx) {
      mn = mn === mx ? mn - 1 : 0;
      mx = mx === mn ? mx + 1 : 1;
    }
  }

  const tint = tintHex ? hexToRgb(tintHex) : null;

  // These are now defined earlier for range calculation
  const maskArr = arguments.length >= 9 ? arguments[8] : null;
  const maskOp = arguments.length >= 10 ? arguments[9] : "==";
  const maskValue = arguments.length >= 11 ? arguments[10] : 1;
  const useMask = maskArr && maskArr.length === arr.length;
  const mv = Number(maskValue);

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const srcY = Math.min(H - 1, Math.floor(y * sy));
      const srcX = Math.min(W - 1, Math.floor(x * sx));
      const si = srcY * W + srcX;
      if (useMask) {
        const m = maskArr[si];
        let pass = false;
        if (maskOp === "==") pass = m === mv;
        else if (maskOp === "!=") pass = m !== mv;
        else if (maskOp === ">") pass = m > mv;
        else if (maskOp === ">=") pass = m >= mv;
        else if (maskOp === "<") pass = m < mv;
        else if (maskOp === "<=") pass = m <= mv;
        if (!pass) {
          const i = (y * cw + x) * 4;
          px[i + 0] = 0;
          px[i + 1] = 0;
          px[i + 2] = 0;
          px[i + 3] = 0;
          continue;
        }
      }

      const v0 = arr[si];
      const v = kind === "counts" ? clampCounts(v0) : v0;

      let r = 0,
        g = 0,
        b = 0,
        a = 255;

      if (kind === "categorical" && mode !== "diff") {
        const vi = Math.round(v);
        if (vi === 0) {
          r = g = b = 0;
        } else {
          const c = palette[((vi % palette.length) + palette.length) % palette.length];
          if (tint) {
            const alpha = 0.70;
            r = Math.round((1 - alpha) * c[0] + alpha * tint.r);
            g = Math.round((1 - alpha) * c[1] + alpha * tint.g);
            b = Math.round((1 - alpha) * c[2] + alpha * tint.b);
          } else {
            r = c[0];
            g = c[1];
            b = c[2];
          }
        }
      } else if (mode === "diff") {
        const t = Math.max(-1, Math.min(1, v / maxAbs));
        if (t >= 0) {
          r = Math.round(255 * t);
          g = Math.round(40 * (1 - t));
          b = Math.round(40 * (1 - t));
        } else {
          const tt = -t;
          r = Math.round(40 * (1 - tt));
          g = Math.round(40 * (1 - tt));
          b = Math.round(255 * tt);
        }
      } else {
        const t = (v - mn) / (mx - mn);
        const u = Math.max(0, Math.min(1, t));
        if (tint) {
          r = Math.round(tint.r * u);
          g = Math.round(tint.g * u);
          b = Math.round(tint.b * u);
        } else {
          r = Math.round(255 * u);
          g = Math.round(255 * u);
          b = Math.round(255 * u);
        }
      }

      const i = (y * cw + x) * 4;
      px[i + 0] = r;
      px[i + 1] = g;
      px[i + 2] = b;
      px[i + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function _stepsDrawHistogramFixed(canvas, values, mn, mx, bins = 60, xLabel = "value", maskArr = null, maskMode = "nonzero", maskValue = 0) {
  const p = _stepsPrepPlotCanvas(canvas, 640, 200);
  if (!p) return;
  const { ctx, W, H } = p;

  const padL = 46;
  const padR = 14;
  const padT = 10;
  const padB = 26;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  let lo = mn;
  let hi = mx;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
    lo = 0;
    hi = 1;
  }

  const counts = new Array(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    if (maskArr && !_stepsMaskTest(maskArr[i], maskMode, maskValue)) continue;
    const v = values[i];
    const t = (v - lo) / (hi - lo);
    if (t < 0 || t > 1) continue;
    const bi = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    counts[bi]++;
  }
  let maxC = 1;
  for (const c of counts) if (c > maxC) maxC = c;

  ctx.fillStyle = "rgba(255,255,255,.04)";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,.06)";
  ctx.lineWidth = 1;
  for (let k = 1; k <= 3; k++) {
    const yy = padT + (plotH * k) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, yy + 0.5);
    ctx.lineTo(padL + plotW, yy + 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  ctx.fillText(String(maxC), padL - 6, padT + 6);
  ctx.fillText("0", padL - 6, padT + plotH);

  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  ctx.fillText(_stepsFmt(lo), padL, padT + plotH + 6);
  ctx.fillText(_stepsFmt((lo + hi) / 2), padL + plotW / 2, padT + plotH + 6);
  ctx.fillText(_stepsFmt(hi), padL + plotW, padT + plotH + 6);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.fillText(xLabel, padL + plotW / 2, H - 4);
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("count", 0, 0);
  ctx.restore();

  ctx.fillStyle = "rgba(255,255,255,.75)";
  const barW = plotW / bins;
  for (let i = 0; i < bins; i++) {
    const h = (counts[i] / maxC) * (plotH - 6);
    const x = padL + i * barW;
    const y = padT + plotH - h;
    ctx.fillRect(x, y, Math.max(1, barW - 1.5), h);
  }
}

function _stepsDrawHistogramSigned(canvas, diffs, bins = 60, maskArr = null, maskMode = "nonzero", maskValue = 0) {
  const p = _stepsPrepPlotCanvas(canvas, 640, 200);
  if (!p) return;
  const { ctx, W, H } = p;

  const padL = 46;
  const padR = 14;
  const padT = 10;
  const padB = 26;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);

  let maxAbs = 0;
  for (let i = 0; i < diffs.length; i++) {
    if (maskArr && !_stepsMaskTest(maskArr[i], maskMode, maskValue)) continue;
    const av = Math.abs(diffs[i]);
    if (av > maxAbs) maxAbs = av;
  }
  if (maxAbs === 0) maxAbs = 1e-9;

  const counts = new Array(bins).fill(0);
  for (let i = 0; i < diffs.length; i++) {
    if (maskArr && !_stepsMaskTest(maskArr[i], maskMode, maskValue)) continue;
    const t = (diffs[i] / maxAbs + 1) / 2;
    const bi = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    counts[bi]++;
  }
  let maxC = 1;
  for (const c of counts) if (c > maxC) maxC = c;

  ctx.fillStyle = "rgba(255,255,255,.06)";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  ctx.fillText(String(maxC), padL - 6, padT + 6);
  ctx.fillText("0", padL - 6, padT + plotH);

  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  ctx.fillText(_stepsFmt(-maxAbs), padL, padT + plotH + 6);
  ctx.fillText("0", padL + plotW / 2, padT + plotH + 6);
  ctx.fillText(_stepsFmt(maxAbs), padL + plotW, padT + plotH + 6);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.fillText("Δ", padL + plotW / 2, H - 4);
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("count", 0, 0);
  ctx.restore();

  const barW = plotW / bins;
  for (let i = 0; i < bins; i++) {
    const h = (counts[i] / maxC) * (plotH - 6);
    const x = padL + i * barW;
    const y = padT + plotH - h;
    const mid = i / (bins - 1);
    const t = mid * 2 - 1;
    if (t >= 0) ctx.fillStyle = "rgba(255,80,80,.75)";
    else ctx.fillStyle = "rgba(80,120,255,.75)";
    ctx.fillRect(x, y, Math.max(1, barW - 1.5), h);
  }
}

function _stepsRenderDiffTable() {
  if (!ui.stepsDiffTable) return;
  if (!stepsDiff) {
    ui.stepsDiffTable.innerHTML = "";
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "table";
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["Layer", "Kind", "Changed%", "Mean|Δ|", "Max|Δ|"]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  t.appendChild(thead);

  const tbody = document.createElement("tbody");
  const top = _stepsVisibleRows().slice(0, 50);
  for (const row of top) {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    if (row.name === stepsSelectedLayer) tr.style.background = "rgba(255,255,255,.03)";
    tr.addEventListener("click", () => {
      stepsSelectedLayer = row.name;
      if (ui.stepsLayerSelect) ui.stepsLayerSelect.value = stepsSelectedLayer;
      _stepsRenderSelectedLayer();
      _stepsRenderDiffTable();
    });

    const tdN = document.createElement("td");
    tdN.textContent = row.name;
    const tdK = document.createElement("td");
    tdK.textContent = row.kind;
    const tdC = document.createElement("td");
    tdC.textContent = row.meanPctChange == null ? "–" : `${_stepsFmt(row.meanPctChange)}%`;
    const tdM = document.createElement("td");
    tdM.textContent = row.meanAbs == null ? "–" : _stepsFmt(row.meanAbs);
    const tdX = document.createElement("td");
    tdX.textContent = row.maxAbs == null ? "–" : _stepsFmt(row.maxAbs);

    tr.appendChild(tdN);
    tr.appendChild(tdK);
    tr.appendChild(tdC);
    tr.appendChild(tdM);
    tr.appendChild(tdX);
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);
  wrap.appendChild(t);
  ui.stepsDiffTable.innerHTML = "";
  ui.stepsDiffTable.appendChild(wrap);
}

function _stepsRenderLayerSelect() {
  if (!ui.stepsLayerSelect) return;
  ui.stepsLayerSelect.innerHTML = "";
  if (!stepsDiff) return;
  const rows = _stepsVisibleRows();
  for (const row of rows) {
    const opt = document.createElement("option");
    opt.value = row.name;
    opt.textContent = row.name;
    ui.stepsLayerSelect.appendChild(opt);
  }
  const names = new Set(rows.map((r) => r.name));
  if (stepsSelectedLayer && !names.has(stepsSelectedLayer)) stepsSelectedLayer = "";
  if (!stepsSelectedLayer) stepsSelectedLayer = rows[0]?.name || "";
  ui.stepsLayerSelect.value = stepsSelectedLayer;
}

function _stepsRenderSelectedLayer() {
  if (!stepsBefore || !stepsAfter || !stepsDiff) return;
  const name = stepsSelectedLayer;
  if (!name) return;
  const bm = stepsBefore.layers.find((m) => m.name === name) || null;
  const am = stepsAfter.layers.find((m) => m.name === name) || null;
  const kind = (am?.kind || bm?.kind || "continuous").toString();
  const n = stepsBefore.H * stepsBefore.W;
  const bRaw = stepsBefore.data[name] || null;
  const aRaw = stepsAfter.data[name] || null;
  const b = bRaw && bRaw.length === n ? bRaw : new Float32Array(n);
  const a = aRaw && aRaw.length === n ? aRaw : new Float32Array(n);

  const maskArr = _stepsGetMaskArray();
  const mMode = stepsMaskMode;
  const mVal = stepsMaskValue;

  const presentBefore = Boolean(bRaw && bRaw.length === n);
  const presentAfter = Boolean(aRaw && aRaw.length === n);
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a[i] - b[i];

  _stepsEnsureCanvasSize(ui.stepsCanvasBefore, stepsBefore.H, stepsBefore.W);
  _stepsEnsureCanvasSize(ui.stepsCanvasAfter, stepsAfter.H, stepsAfter.W);
  _stepsEnsureCanvasSize(ui.stepsCanvasDiff, stepsAfter.H, stepsAfter.W);

  if (ui.stepsCanvasHistBefore) {
    ui.stepsCanvasHistBefore.width = 640;
    ui.stepsCanvasHistBefore.height = 200;
  }
  if (ui.stepsCanvasHistAfter) {
    ui.stepsCanvasHistAfter.width = 640;
    ui.stepsCanvasHistAfter.height = 200;
  }
  if (ui.stepsCanvasHistDiff) {
    ui.stepsCanvasHistDiff.width = 640;
    ui.stepsCanvasHistDiff.height = 200;
  }

  let sharedRange = null;
  if (kind !== "categorical") {
    const rb = _stepsComputeRange(b);
    const ra = _stepsComputeRange(a);
    sharedRange = { mn: Math.min(rb.mn, ra.mn), mx: Math.max(rb.mx, ra.mx) };
  }

  _stepsDrawHeatmap(ui.stepsCanvasBefore, b, stepsBefore.H, stepsBefore.W, kind, "value", sharedRange);
  _stepsDrawHeatmap(ui.stepsCanvasAfter, a, stepsAfter.H, stepsAfter.W, kind, "value", sharedRange);
  _stepsDrawHeatmap(ui.stepsCanvasDiff, d, stepsAfter.H, stepsAfter.W, kind, "diff");

  if (kind === "categorical") {
    if (ui.stepsCanvasHistBefore) {
      const ctx = ui.stepsCanvasHistBefore.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ui.stepsCanvasHistBefore.width, ui.stepsCanvasHistBefore.height);
        ctx.fillStyle = "rgba(255,255,255,.6)";
        ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
        ctx.fillText("categorical", 10, 20);
      }
    }
    if (ui.stepsCanvasHistAfter) {
      const ctx = ui.stepsCanvasHistAfter.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ui.stepsCanvasHistAfter.width, ui.stepsCanvasHistAfter.height);
        ctx.fillStyle = "rgba(255,255,255,.6)";
        ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
        ctx.fillText("categorical", 10, 20);
      }
    }
  } else if (sharedRange) {
    _stepsDrawHistogramFixed(ui.stepsCanvasHistBefore, b, sharedRange.mn, sharedRange.mx, 60, "value", maskArr, mMode, mVal);
    _stepsDrawHistogramFixed(ui.stepsCanvasHistAfter, a, sharedRange.mn, sharedRange.mx, 60, "value", maskArr, mMode, mVal);
  }
  _stepsDrawHistogramSigned(ui.stepsCanvasHistDiff, d, 60, maskArr, mMode, mVal);

  if (ui.stepsAxes) ui.stepsAxes.textContent = `axes: x = 0…${stepsBefore.W - 1}, y = 0…${stepsBefore.H - 1}`;

  const row = stepsDiff.rows.find((r) => r.name === name) || null;
  if (ui.stepsLayerStats) {
    const cellsChanged = row?.cellsChangedPct == null ? "–" : `${_stepsFmt(row.cellsChangedPct)}%`;
    const meanAbs = row?.meanAbs == null ? "–" : _stepsFmt(row.meanAbs);
    const maxAbs2 = row?.maxAbs == null ? "–" : _stepsFmt(row.maxAbs);
    const meanDelta = row?.meanDelta == null ? "–" : _stepsFmt(row.meanDelta);
    const meanPct = row?.meanPctChange == null ? "–" : `${_stepsFmt(row.meanPctChange)}%`;
    ui.stepsLayerStats.textContent = `kind=${kind}   present_before=${presentBefore ? 1 : 0}   present_after=${presentAfter ? 1 : 0}   cells_changed=${cellsChanged}   Δmean=${meanDelta}   Δmean%=${meanPct}   mean|Δ|=${meanAbs}   max|Δ|=${maxAbs2}`;
  }

  if (kind === "categorical") {
    _stepsSetLegendCategorical(ui.stepsLegendBeforeText, ui.stepsLegendBeforeBar);
    _stepsSetLegendCategorical(ui.stepsLegendAfterText, ui.stepsLegendAfterBar);
  } else {
    if (sharedRange) {
      _stepsSetLegendContinuous(ui.stepsLegendBeforeText, ui.stepsLegendBeforeBar, sharedRange.mn, sharedRange.mx);
      _stepsSetLegendContinuous(ui.stepsLegendAfterText, ui.stepsLegendAfterBar, sharedRange.mn, sharedRange.mx);
    }
  }
  const maxAbs = _stepsComputeMaxAbs(d);
  _stepsSetLegendDiff(ui.stepsLegendDiffText, ui.stepsLegendDiffBar, maxAbs);

  _stepsRenderScatter();
}

async function _stepsTryLoad(which) {
  try {
    const beforeFile = ui.stepsBeforeFile?.files?.[0] || null;
    const afterFile = ui.stepsAfterFile?.files?.[0] || null;
    if (!beforeFile || !afterFile) {
      _stepsSetStatus("Pick both before and after JSON files");
      return;
    }
    _stepsSetStatus("Loading…");
    const [tBefore, tAfter] = await Promise.all([_stepsReadFileAsText(beforeFile), _stepsReadFileAsText(afterFile)]);
    stepsBefore = _stepsParseBundle(tBefore);
    stepsAfter = _stepsParseBundle(tAfter);
    stepsDiff = _stepsComputeDiff(stepsBefore, stepsAfter);
    stepsSelectedLayer = _stepsVisibleRows()?.[0]?.name || "";
    _stepsSetStatus(`Loaded. H=${stepsBefore.H} W=${stepsBefore.W} layers=${stepsDiff.rows.length}`);
    _stepsRenderLayerSelect();
    _stepsRenderMaskSelect();
    _stepsRenderScatterSelects();
    _stepsRenderDiffTable();
    _stepsRenderSelectedLayer();
  } catch (e) {
    _stepsSetStatus(String(e?.message || e));
  }
}

function setActiveScreen(name) {
  for (const b of document.querySelectorAll(".screenBtn")) {
    b.classList.toggle("screenBtn--active", b.dataset.screen === name);
  }
  for (const p of document.querySelectorAll(".screenPanel")) {
    p.classList.toggle("screenPanel--active", p.dataset.screen === name);
  }

  if (name === "runtime") {
    if (!rtRunning && !rtLoaded) {
      void _rtEnsureSyncedFromEditor(false, "editor").catch((e) => _rtSetStatus(String(e?.message || e)));
    }
  }
  if (name === "workspace") {
    applyAutoFitZoom();
    return;
  }
  if (name === "manage") {
    renderLayersTable();
    updateBulkDeleteInfo();
    updateBulkAddPreview();
    updateDerivedPreview();
    return;
  }
  if (name === "assign") {
    renderOpTargetsList();
    updateAssignOpUi();
    updateMaskedOpsPreview();
  }

  if (name === "functions") {
    renderFunctionsSpecsTable();
    renderLayerOpsTable();
  }

  if (name === "evolution") {
    _evoRenderTargetLayersUI();
    setTimeout(() => _evoRenderNow(), 0);
  }
}

for (const b of document.querySelectorAll(".screenBtn")) {
  b.addEventListener("click", () => setActiveScreen(b.dataset.screen));
}

// Wire tab groups within each screen independently
for (const screenPanel of document.querySelectorAll(".screenPanel")) {
  const btns = [...screenPanel.querySelectorAll(".tabBtn")];
  const activeBtn = btns.find((x) => x.classList.contains("tabBtn--active"));
  if (activeBtn) setActiveTab(screenPanel, activeBtn.dataset.tab);
  else if (btns[0]) setActiveTab(screenPanel, btns[0].dataset.tab);
  for (const b of btns) {
    b.addEventListener("click", () => {
      setActiveTab(screenPanel, b.dataset.tab);
    });
  }
}

if (ui.inspectMode) {
  ui.inspectMode.addEventListener("change", () => {
    _syncInspectModeUi();
  });
}

if (ui.inspectHistMaskLayer) {
  ui.inspectHistMaskLayer.addEventListener("change", () => {
    inspectHistMaskLayer = String(ui.inspectHistMaskLayer.value || "").trim();
    try {
      localStorage.setItem(INSPECT_HIST_MASK_LAYER_KEY, String(inspectHistMaskLayer || ""));
    } catch {}
    inspectSummaryDirty = true;
    _ensureInspectSummaryUpToDate();
  });
}

if (ui.inspectHistMaskOp) {
  ui.inspectHistMaskOp.addEventListener("change", () => {
    inspectHistMaskOp = String(ui.inspectHistMaskOp.value || "==");
    try {
      localStorage.setItem(INSPECT_HIST_MASK_OP_KEY, String(inspectHistMaskOp));
    } catch {}
    inspectSummaryDirty = true;
    _ensureInspectSummaryUpToDate();
  });
}

if (ui.inspectHistMaskValue) {
  ui.inspectHistMaskValue.addEventListener("input", () => {
    const v = Number(ui.inspectHistMaskValue.value);
    if (Number.isFinite(v)) {
      inspectHistMaskValue = v;
      try {
        localStorage.setItem(INSPECT_HIST_MASK_VALUE_KEY, String(v));
      } catch {}
      inspectSummaryDirty = true;
      _ensureInspectSummaryUpToDate();
    }
  });
}

if (ui.helpModalClose) ui.helpModalClose.addEventListener("click", () => _closeHelpModal());
if (ui.helpModalOverlay) ui.helpModalOverlay.addEventListener("click", () => _closeHelpModal());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") _closeHelpModal();
});

if (ui.opsHelpBtn) {
  ui.opsHelpBtn.addEventListener("click", () => {
    _openHelpModal(
      "Layer Ops: expressions",
      `
      <div class="help">
        Expressions are NumPy-style and evaluated on 2D arrays (H×W). Use <span class="mono">let</span> to define temporary variables,
        and <span class="mono">op</span> to write into an existing target layer.
      </div>
      <div class="hr"></div>
      <div class="help">
        <div class="mono">where(cond, a, b)</div>
        Choose per-cell values.
      </div>
      <div class="help">
        <div class="mono">sum_layer(x)</div>
        Scalar sum over the whole layer/array (broadcasts if used in an op).
      </div>
      <div class="help">
        <div class="mono">sum_layers("pat*")</div>
        Per-cell sum across all layers matching a glob pattern (e.g. <span class="mono">"gene_*"</span>).
      </div>
      <div class="help">
        <div class="mono">rand_beta(alpha, beta)</div>
        Per-cell random Beta(alpha, beta) field in <span class="mono">[0,1]</span>.
      </div>
      <div class="help">
        <div class="mono">rand_logitnorm(mu, sigma)</div>
        Per-cell random logit-normal field in <span class="mono">[0,1]</span> where <span class="mono">logit(x) ~ Normal(mu, sigma)</span>.
      </div>
      <div class="hr"></div>
      <div class="help">
        <div><b>foreach</b> mini-language:</div>
        <div class="mono">for (i in "gene_*") {\n  {i} <- {i}\n}</div>
        <div><span class="mono">{i}</span> is the full matched layer name (e.g. gene_atp_maker). <span class="mono">{j}</span>, <span class="mono">{k}</span>… are wildcard capture groups.</div>
      </div>
      `
    );
  });
}

// Runtime wiring
if (ui.evoStartBtn) {
  ui.evoStartBtn.addEventListener("click", async () => {
    try {
      if (evoRunning) return;
      await _evoStart();
    } catch (e) {
      _evoSetStatus(String(e?.message || e));
      _evoStopLocal();
    }
  });
}

if (ui.evoStopBtn) {
  ui.evoStopBtn.addEventListener("click", async () => {
    try {
      await _evoStop();
    } catch (e) {
      _evoSetStatus(String(e?.message || e));
    }
  });
}

if (ui.evoTopList) {
  ui.evoTopList.addEventListener("click", async (e) => {
    const t = e?.target;
    const btn = t && t.closest ? t.closest("button") : null;
    if (!btn) return;
    const act = btn.getAttribute("data-evo-act") || "";
    const id = btn.getAttribute("data-evo-id") || "";
    if (!id) return;
    try {
      const res = await _rtPostJson("/api/evolution/candidate", { id });
      const payload = res?.payload;
      if (!payload || typeof payload !== "object") throw new Error("candidate payload missing");
      if (act === "download") {
        download(`gridstate.evo.${id.slice(0, 8)}.json`, JSON.stringify(payload, null, 2));
        return;
      }
      if (act === "load") {
        setActiveScreen("runtime");
        _applyPayloadToEditor(payload, `evo:${id.slice(0, 8)}`);
        await _rtEnsureSyncedFromEditor(true, `evo:${id.slice(0, 8)}`);
        setTimeout(() => _rtRenderNow(), 0);
      }
    } catch (err) {
      _evoSetStatus(String(err?.message || err));
    }
  });
}

if (ui.rtStartStopBtn)
  ui.rtStartStopBtn.addEventListener("click", async () => {
    try {
      if (!rtRunning && !rtLoaded) await _rtEnsureSyncedFromEditor(true, "editor");
      _rtToggleRun();
    } catch (e) {
      _rtSetStatus(String(e?.message || e));
    }
  });

_rtInitVizCols();

_rtInitHistogramControls();

_rtInitSurvivalControls();

if (ui.rtVizCols) {
  ui.rtVizCols.addEventListener("change", () => {
    _rtSetVizCols(ui.rtVizCols.value);
  });
}

if (ui.rtHeatMaskEnabled) {
  ui.rtHeatMaskEnabled.addEventListener("change", () => {
    rtHeatMaskEnabled = !!ui.rtHeatMaskEnabled.checked;
    try {
      localStorage.setItem(RT_HEAT_MASK_ENABLED_KEY, rtHeatMaskEnabled ? "1" : "0");
    } catch {}
    _rtRenderHeatmaps();
  });
}

if (ui.rtHistLayer) {
  ui.rtHistLayer.addEventListener("change", async () => {
    rtHistLayer = String(ui.rtHistLayer.value || "").trim();
    try {
      localStorage.setItem(RT_HIST_LAYER_KEY, String(rtHistLayer || ""));
    } catch {}
    if (!rtLoaded) {
      _rtRenderHistogram();
      return;
    }
    try {
      const frame = await _rtPostJson("/api/runtime/frame", { layers: _rtGetRequestedLayerNames() });
      _rtApplyFrame(frame);
    } catch (e) {
      _rtSetStatus(String(e?.message || e));
    }
  });
}

if (ui.rtHistBins) {
  ui.rtHistBins.addEventListener("input", () => {
    const n = Math.floor(Number(ui.rtHistBins.value || "60"));
    rtHistBins = Number.isFinite(n) ? Math.max(10, Math.min(400, n)) : 60;
    try {
      localStorage.setItem(RT_HIST_BINS_KEY, String(rtHistBins));
    } catch {}
    _rtRenderHistogram();
  });
}

if (ui.rtHistLogY) {
  ui.rtHistLogY.addEventListener("change", () => {
    rtHistLogY = !!ui.rtHistLogY.checked;
    try {
      localStorage.setItem(RT_HIST_LOGY_KEY, rtHistLogY ? "1" : "0");
    } catch {}
    _rtRenderHistogram();
  });
}

if (ui.rtHistMaskLayer) {
  ui.rtHistMaskLayer.addEventListener("change", async () => {
    rtHistMaskLayer = String(ui.rtHistMaskLayer.value || "").trim();
    try {
      localStorage.setItem(RT_HIST_MASK_LAYER_KEY, String(rtHistMaskLayer || ""));
    } catch {}
    if (!rtLoaded) {
      _rtRenderHistogram();
      return;
    }
    try {
      const layers = _rtGetRequestedLayerNames();
      await _rtStep(0, layers);
    } catch (e) {
      console.error("rtHistMaskLayer change error:", e);
    }
  });
}

if (ui.rtHistMaskOp) {
  ui.rtHistMaskOp.addEventListener("change", () => {
    rtHistMaskOp = String(ui.rtHistMaskOp.value || "==");
    try {
      localStorage.setItem(RT_HIST_MASK_OP_KEY, String(rtHistMaskOp));
    } catch {}
    _rtRenderHistogram();
    if (rtHeatMaskEnabled) _rtRenderHeatmaps();
  });
}

if (ui.rtHistMaskValue) {
  ui.rtHistMaskValue.addEventListener("input", () => {
    const v = Number(ui.rtHistMaskValue.value);
    if (Number.isFinite(v)) {
      rtHistMaskValue = v;
      try {
        localStorage.setItem(RT_HIST_MASK_VALUE_KEY, String(v));
      } catch {}
      _rtRenderHistogram();
      if (rtHeatMaskEnabled) _rtRenderHeatmaps();
    }
  });
}

if (ui.rtSurvTopK) {
  ui.rtSurvTopK.addEventListener("input", () => {
    const n = Math.floor(Number(ui.rtSurvTopK.value || "12"));
    rtSurvTopK = Number.isFinite(n) ? Math.max(5, Math.min(50, n)) : 12;
    try {
      localStorage.setItem(RT_SURV_TOPK_KEY, String(rtSurvTopK));
    } catch {}
    _rtRenderSurvival();
  });
}

if (ui.rtSurvLog1p) {
  ui.rtSurvLog1p.addEventListener("change", () => {
    rtSurvLog1p = !!ui.rtSurvLog1p.checked;
    try {
      localStorage.setItem(RT_SURV_LOG1P_KEY, rtSurvLog1p ? "1" : "0");
    } catch {}
    _rtRenderSurvival();
  });
}

if (ui.rtSurvFocus) {
  ui.rtSurvFocus.addEventListener("change", () => {
    rtSurvFocus = String(ui.rtSurvFocus.value || "gene");
    try {
      localStorage.setItem(RT_SURV_FOCUS_KEY, String(rtSurvFocus || "gene"));
    } catch {}
    _rtRenderSurvival();
  });
}

if (ui.rtStepBtn) {
  ui.rtStepBtn.addEventListener("click", async () => {
    try {
      _rtStop();
      if (!rtLoaded) await _rtEnsureSyncedFromEditor(true, "editor");
      await _rtStepOnce();
    } catch (e) {
      _rtSetStatus(String(e?.message || e));
    }
  });
}

if (ui.rtResetBtn) {
  ui.rtResetBtn.addEventListener("click", async () => {
    try {
      _rtStop();
      await _rtEnsureSyncedFromEditor(true, "editor");
      setTimeout(() => _rtRenderNow(), 0);
    } catch (e) {
      _rtSetStatus(String(e?.message || e));
    }
  });
}

if (ui.rtDownloadBtn) {
  ui.rtDownloadBtn.addEventListener("click", async () => {
    try {
      await _rtDownloadTick();
    } catch (e) {
      _rtSetStatus(String(e?.message || e));
    }
  });
}

if (ui.rtAddLayerBtn) {
  ui.rtAddLayerBtn.addEventListener("click", async () => {
    const nm = String(ui.rtAddLayerSelect?.value || "").trim();
    if (!nm) return;
    if (rtWatch.some((x) => x.name === nm)) return;
    rtWatch.push({ name: nm, alpha: 1 });
    _rtEnsureVizItem(nm);
    _rtRenderWatchList();
    _rtEnsureCanvasSizes();
    _rtRenderOverlay();

    // Draw immediately (otherwise the new canvas stays blank until the next tick).
    if (rtLoaded) {
      try {
        const frame = await _rtPostJson("/api/runtime/frame", { layers: [nm] });
        _rtApplyFrame(frame);
      } catch (e) {
        _rtSetStatus(String(e?.message || e));
      }
    }
  });
}

// Layers search/grouping
if (ui.layerSearchInput) {
  ui.layerSearchInput.addEventListener("input", () => {
    layersFilterText = ui.layerSearchInput.value;
    renderLayersTable();
  });
}
if (ui.groupByPrefix) {
  ui.groupByPrefix.addEventListener("change", () => {
    layersGroupByPrefix = ui.groupByPrefix.checked;
    collapsedGroups.clear();
    renderLayersTable();
  });
  layersGroupByPrefix = ui.groupByPrefix.checked;
}

if (ui.collapseAllBtn) {
  ui.collapseAllBtn.addEventListener("click", () => {
    // Get all current group keys and add them to collapsed set
    const query = String(layersFilterText || "").trim().toLowerCase();
    const layers = query
      ? state.layers.filter((l) => String(l.name).toLowerCase().includes(query))
      : state.layers;
    
    const groups = new Map();
    for (const l of layers) {
      const key = layersGroupByPrefix ? String(l.name).split("_")[0] : "all";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(l);
    }
    
    const groupKeys = [...groups.keys()];
    for (const key of groupKeys) {
      collapsedGroups.add(key);
    }
    
    renderLayersTable();
  });
}

if (ui.expandAllBtn) {
  ui.expandAllBtn.addEventListener("click", () => {
    // Clear all collapsed groups
    collapsedGroups.clear();
    renderLayersTable();
  });
}

// Masked ops preview
for (const el of [ui.opMaskLayer, ui.opMaskOp, ui.opMaskValue, ui.opMaskInvert]) {
  if (!el) continue;
  el.addEventListener("change", () => updateMaskedOpsPreview());
  el.addEventListener("input", () => updateMaskedOpsPreview());
}

if (ui.opType) {
  ui.opType.addEventListener("change", () => {
    updateAssignOpUi();
  });
}

// Functions
loadFunctionsCfg();

if (ui.fnInsertLayerBtn) {
  ui.fnInsertLayerBtn.addEventListener("click", () => {
    const layer = ui.fnInsertLayer?.value || "";
    const target = fnLastFocusedExprInput || ui.fnSpecsTable?.querySelector("textarea, input") || null;
    if (!layer) return;
    insertTextIntoInput(target, layer);
  });
}

if (ui.fnInsertFnBtn) {
  ui.fnInsertFnBtn.addEventListener("click", () => {
    const key = ui.fnInsertFn?.value || "";
    const item = FN_INSERTER_FUNCS.find((x) => x.value === key);
    const layer = ui.fnInsertLayer?.value || "layer";
    const snippet = item ? item.snippet.replaceAll("LAYER", layer) : "";
    const target = fnLastFocusedExprInput || ui.fnSpecsTable?.querySelector("textarea, input") || null;
    if (!snippet) return;
    insertTextIntoInput(target, snippet);
  });
}

if (ui.opsInsertLayerBtn) {
  ui.opsInsertLayerBtn.addEventListener("click", () => {
    const layer = ui.opsInsertLayer?.value || "";
    const target = opsLastFocusedExprInput || ui.opsSpecsTable?.querySelector("textarea, input") || null;
    if (!layer) return;
    insertTextIntoInput(target, layer);
  });
}

if (ui.opsInsertFnBtn) {
  ui.opsInsertFnBtn.addEventListener("click", () => {
    const key = ui.opsInsertFn?.value || "";
    const item = OPS_INSERTER_FUNCS.find((x) => x.value === key);
    const layer = ui.opsInsertLayer?.value || "layer";
    const snippet = item ? item.snippet.replaceAll("LAYER", layer) : "";
    const target = opsLastFocusedExprInput || ui.opsSpecsTable?.querySelector("textarea, input") || null;
    if (!snippet) return;
    insertTextIntoInput(target, snippet);
  });
}


if (ui.fnAddRowBtn) {
  ui.fnAddRowBtn.addEventListener("click", () => {
    const nm = "new_measurement";
    const layer = state.layers[0]?.name || "layer";
    fnMeasurements.push({ name: nm, expr: `mean(${layer})` });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderFunctionsSpecsTable();
  });
}
if (ui.fnResetBtn) {
  ui.fnResetBtn.addEventListener("click", () => {
    fnMeasurements = DEFAULT_MEASUREMENTS.map((x) => ({ ...x }));
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderFunctionsSpecsTable();
  });
}

if (ui.opsAddRowBtn) {
  ui.opsAddRowBtn.addEventListener("click", () => {
    const target = state.layers[0]?.name || "layer";
    _opsInsertAtFocused({ type: "op", enabled: true, name: "", target, expr: target });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsAddVarBtn) {
  ui.opsAddVarBtn.addEventListener("click", () => {
    // Find a unique name for the new layer
    let varName = "computed_1";
    let counter = 1;
    while (state.layers.some((l) => l.name === varName)) {
      counter++;
      varName = `computed_${counter}`;
    }
    
    // Create the actual layer
    try {
      addLayer(state, {
        name: varName,
        kind: "continuous",
        init: "zeros",
        value: 0,
        seed: 0,
        color: "#8B5CF6",
      });
    } catch (e) {
      console.error("Failed to create layer for let:", e);
    }
    
    // Add the let operation pointing to this layer
    _opsInsertAtFocused({ type: "let", enabled: true, name: "", var: varName, expr: "0" });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
    syncLayerSelect();
  });
}

if (ui.opsAddForEachBtn) {
  ui.opsAddForEachBtn.addEventListener("click", () => {
    const defaultLayer =
      state?.layers?.find((l) => l && l.kind !== "categorical")?.name || state?.layers?.[0]?.name || "*";
    const stepsText = `for (i in "${defaultLayer}") {\n  {i} <- {i}\n}`;
    const layerNames = state?.layers ? state.layers.map((l) => l.name) : [];
    const c = _compileForEachR(stepsText, layerNames);
    _opsInsertAtFocused({
      type: "foreach",
      enabled: true,
      name: "",
      group: "",
      match: c.ok ? c.match : "*",
      require_match: false,
      steps: c.ok ? c.steps : [],
      stepsText,
    });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsAddTransportBtn) {
  ui.opsAddTransportBtn.addEventListener("click", () => {
    const defaultCellLayer = state?.layers?.some((l) => l.name === "cell") ? "cell" : state.layers[0]?.name || "cell";
    _opsInsertAtFocused({
      type: "transport",
      enabled: true,
      name: "",
      group: "",
      molecules: "molecule_*",
      molecule_prefix: "molecule_",
      protein_prefix: "protein_",
      cell_layer: defaultCellLayer,
      cell_mode: "eq",
      cell_value: 1,
      dirs: ["north", "south", "east", "west"],
      per_pair_rate: 1.0,
      seed: 0,
    });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsAddDiffusionBtn) {
  ui.opsAddDiffusionBtn.addEventListener("click", () => {
    const defaultCellLayer = state?.layers?.some((l) => l.name === "cell") ? "cell" : state.layers[0]?.name || "cell";
    _opsInsertAtFocused({
      type: "diffusion",
      enabled: true,
      name: "",
      group: "",
      molecules: "molecule_*",
      cell_layer: defaultCellLayer,
      cell_mode: "eq",
      cell_value: 1,
      rate: 0.2,
      rate_layer: null,
      seed: 0,
    });
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsAddPathwayBtn) {
  ui.opsAddPathwayBtn.addEventListener("click", () => {
    _openPathwayModal();
  });
}

if (ui.pathwayModalClose) {
  ui.pathwayModalClose.addEventListener("click", () => _closePathwayModal());
}
if (ui.pathwayModalCancel) {
  ui.pathwayModalCancel.addEventListener("click", () => _closePathwayModal());
}
if (ui.pathwayModalOverlay) {
  ui.pathwayModalOverlay.addEventListener("click", () => _closePathwayModal());
}


if (ui.pathwayInputsSelected) {
  ui.pathwayInputsSelected.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON" && e.target.dataset.type === "input") {
      const idx = parseInt(e.target.dataset.idx);
      pathwayModalInputs.splice(idx, 1);
      _updatePathwaySelectedItems();
    }
  });
}

if (ui.pathwayOutputsSelected) {
  ui.pathwayOutputsSelected.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON" && e.target.dataset.type === "output") {
      const idx = parseInt(e.target.dataset.idx);
      pathwayModalOutputs.splice(idx, 1);
      _updatePathwaySelectedItems();
    }
  });
}

if (ui.pathwayModalCreate) {
  ui.pathwayModalCreate.addEventListener("click", () => {
    const pathwayName = ui.pathwayName?.value?.trim() || "";
    if (!pathwayName) {
      alert("Pathway name is required");
      return;
    }
    const name = pathwayName.toLowerCase().replace(/[^a-z0-9_]/g, "_");
    
    const inputs = [...pathwayModalInputs];
    const outputs = [...pathwayModalOutputs];
    
    if (!inputs.length) {
      alert("At least one input is required");
      return;
    }
    if (!outputs.length) {
      alert("At least one output is required");
      return;
    }
    
    const numEnzymes = Math.max(1, Math.min(10, parseInt(ui.pathwayNumEnzymes?.value) || 3));
    const cellLayer = pathwayCellLayerSearchable?.input?.value?.trim() || "cell";
    const cellValue = parseInt(ui.pathwayCellValue?.value) || 1;
    const efficiency = parseFloat(ui.pathwayEfficiency?.value) || 1.0;
    
    const resolveMolName = (nm) => {
      const s = String(nm || "").trim();
      if (!s) return "";
      if (state.layers.some((l) => l.name === s)) return s;
      const cand1 = `molecule_${s}`;
      if (state.layers.some((l) => l.name === cand1)) return cand1;
      const cand2 = `mol_${s}`;
      if (state.layers.some((l) => l.name === cand2)) return cand2;
      return s;
    };

    const resolvedInputs = Array.from(new Set(inputs.map(resolveMolName).filter((s) => s)));
    const resolvedOutputs = Array.from(new Set(outputs.map(resolveMolName).filter((s) => s)));
    
    // Create the enzyme layers (protein, rna, gene triplets)
    const createdLayers = [];
    for (let e = 1; e <= numEnzymes; e++) {
      const proteinName = `protein_${name}_enzyme_${e}`;
      const rnaName = `rna_${name}_enzyme_${e}`;
      const geneName = `gene_${name}_enzyme_${e}`;
      
      for (const layerName of [proteinName, rnaName, geneName]) {
        if (!state.layers.some(l => l.name === layerName)) {
          try {
            addLayer(state, {
              name: layerName,
              kind: "counts",
              init: layerName.startsWith("gene_") ? "ones" : "zeros",
              value: layerName.startsWith("gene_") ? 1 : 0,
              seed: 0,
              color: layerName.startsWith("protein_") ? "#10B981" : 
                     layerName.startsWith("rna_") ? "#F59E0B" : "#6366F1",
            });
            createdLayers.push(layerName);
          } catch (e) {
            console.error(`Failed to create layer ${layerName}:`, e);
          }
        }
      }
    }
    
    // Create input/output layers if they don't exist
    for (const inputName of resolvedInputs) {
      if (!state.layers.some(l => l.name === inputName)) {
        try {
          addLayer(state, {
            name: inputName,
            kind: "counts",
            init: "zeros",
            value: 0,
            seed: 0,
            color: "#3B82F6",
          });
          createdLayers.push(inputName);
        } catch (e) {
          console.error(`Failed to create input layer ${inputName}:`, e);
        }
      }
    }
    
    for (const outputName of resolvedOutputs) {
      if (!state.layers.some(l => l.name === outputName)) {
        try {
          addLayer(state, {
            name: outputName,
            kind: "counts",
            init: "zeros",
            value: 0,
            seed: 0,
            color: "#EF4444",
          });
          createdLayers.push(outputName);
        } catch (e) {
          console.error(`Failed to create output layer ${outputName}:`, e);
        }
      }
    }
    
    _opsInsertAtFocused({
      type: "pathway",
      enabled: true,
      name: name,
      group: "",
      pathway_name: name,
      inputs: resolvedInputs,
      outputs: resolvedOutputs,
      num_enzymes: numEnzymes,
      cell_layer: cellLayer,
      cell_value: cellValue,
      efficiency: efficiency,
      seed: 0,
    });
    
    _closePathwayModal();
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
    syncLayerSelect();
    
    if (createdLayers.length > 0) {
      alert(`Created ${createdLayers.length} new layers for pathway "${name}":\n${createdLayers.slice(0, 10).join(", ")}${createdLayers.length > 10 ? "..." : ""}`);
    }
  });
}

if (ui.opsResetBtn) {
  ui.opsResetBtn.addEventListener("click", () => {
    layerOps = [];
    saveFunctionsCfg();
    markDirty();
    saveToLocalStorage();
    renderLayerOpsTable();
  });
}

if (ui.opsTestBtn) {
  ui.opsTestBtn.addEventListener("click", () => {
    const knownVars = new Set();
    for (let i = 0; i < layerOps.length; i++) {
      const step = layerOps[i];
      if (!step || step.enabled === false) continue;
      const v = validateLayerOpStep(step, knownVars);
      if (!v.ok) {
        alert(`Layer Op #${i + 1}: ${v.text}`);
        return;
      }
      if (step.type === "let") {
        const nm = String(step.var || "").trim();
        if (_isValidIdentifier(nm)) knownVars.add(nm);
      }
    }
    alert("OK");
  });
}

if (ui.renameBtn) {
  ui.renameBtn.addEventListener("click", () => {
    try {
      const oldName = selectedLayer;
      const nn = ui.renameToInput.value.trim();
      renameLayer(state, oldName, nn);
      selectedLayer = nn;

      ensureGeneTriplets(state);
      syncLayerSelect();
      markDirty();
    } catch (e) {
      alert(String(e?.message || e));
    }
  });
}

if (ui.removeBtn) {
  ui.removeBtn.addEventListener("click", () => {
    if (!selectedLayer) return;
    if (!confirm(`Remove layer '${selectedLayer}'?`)) return;
    removeLayer(state, selectedLayer);
    if (!state.layers.some((l) => l.name === selectedLayer)) selectedLayer = state.layers[0]?.name || "";
    syncLayerSelect();
    markDirty();
    saveToLocalStorage();
  });
}

ui.bulkAddBtn.addEventListener("click", () => {
  try {
    const cfg = {
      prefix: ui.bulkPrefix.value,
      start: ui.bulkStart.value,
      count: ui.bulkCount.value,
      collision: ui.bulkCollision?.value || "error",
      kind: ui.bulkKind.value,
      init: ui.bulkInit.value,
      value: ui.bulkValue.value,
      seed: ui.bulkSeed.value,
      maskLayer: ui.maskLayer.value,
      maskOp: ui.maskOp.value,
      maskValue: ui.maskValue.value,
      maskInvert: ui.maskInvert.checked,
    };
    if (!confirm(`Create ${cfg.count} layer(s) named like '${cfg.prefix}${cfg.start}...' ?\nIf exists: ${cfg.collision}`)) return;
    bulkAddLayersMasked(state, cfg);

    ensureGeneTriplets(state);
    selectedLayer = `${String(cfg.prefix)}${Math.floor(Number(cfg.start))}`;
    syncLayerSelect();
    markDirty();
    saveToLocalStorage();
  } catch (e) {
    alert(String(e?.message || e));
  }
});

ui.opApplyBtn.addEventListener("click", () => {
  try {
    const maskCfg = {
      layer: ui.opMaskLayer.value,
      op: ui.opMaskOp.value,
      value: Number(ui.opMaskValue.value),
      invert: ui.opMaskInvert.checked,
    };
    const opCfg = {
      type: ui.opType.value,
      value: Number(ui.opValue.value),
      min: Number(ui.opMin.value),
      max: Number(ui.opMax.value),
      seed: Number(ui.opSeed.value),
    };

    const mask = makeMask(state, maskCfg.layer, maskCfg.op, maskCfg.value, maskCfg.invert);
    let n = 0;
    for (let i = 0; i < mask.length; i++) n += mask[i] ? 1 : 0;
    if (n === 0) throw new Error("Mask matched 0 cells");

    const { targets } = computeBatchTargets();
    if (!targets.length) throw new Error("No target layers selected");
    if (!confirm(`Apply '${opCfg.type}' to ${targets.length} layer(s) where ${maskCfg.layer} ${maskCfg.op} ${maskCfg.value}?\nMatches: ${n.toLocaleString()} cells`)) return;

    const opType = String(opCfg.type);
    const baseSeed = Math.floor(Number(opCfg.seed)) || 0;
    for (let li = 0; li < targets.length; li++) {
      const target = targets[li];
      const meta = state.layers.find((l) => l.name === target);
      if (!meta) continue;
      const a = state.data[target];
      if (!a) continue;

      const isCat = meta.kind === "categorical";
      const isCounts = meta.kind === "counts";
      if (isCat && isRandomAssignOpType(opType)) continue;

      if (opType === "set_constant") {
        const v = isCat ? Math.round(Number(opCfg.value)) : isCounts ? clampCounts(opCfg.value) : Number(opCfg.value);
        for (let i = 0; i < a.length; i++) if (mask[i]) a[i] = v;
        continue;
      }

      const mn = Number(opCfg.min);
      const mx = Number(opCfg.max);
      const lo = Math.min(mn, mx);
      const hi = Math.max(mn, mx);
      const rng = mulberry32(baseSeed + li);

      if (opType === "set_random_uniform") {
        if (isCat || isCounts) {
          const ilo = Math.round(lo);
          const ihi = Math.round(hi);
          const a0 = isCounts ? Math.max(0, Math.min(ilo, ihi)) : Math.min(ilo, ihi);
          const b0 = isCounts ? Math.max(0, Math.max(ilo, ihi)) : Math.max(ilo, ihi);
          const span = Math.max(1, b0 - a0 + 1);
          for (let i = 0; i < a.length; i++) {
            if (!mask[i]) continue;
            a[i] = a0 + Math.floor(rng() * span);
          }
        } else {
          for (let i = 0; i < a.length; i++) {
            if (!mask[i]) continue;
            a[i] = lo + (hi - lo) * rng();
          }
        }
        continue;
      }
      if (opType === "add_random_uniform") {
        if (isCat || isCounts) {
          const ilo = Math.round(lo);
          const ihi = Math.round(hi);
          const a0 = Math.min(ilo, ihi);
          const b0 = Math.max(ilo, ihi);
          const span = Math.max(1, b0 - a0 + 1);
          for (let i = 0; i < a.length; i++) {
            if (!mask[i]) continue;
            const dv = a0 + Math.floor(rng() * span);
            if (isCounts) a[i] = clampCounts(a[i] + dv);
            else a[i] = Math.round(a[i] + dv);
          }
        } else {
          for (let i = 0; i < a.length; i++) {
            if (!mask[i]) continue;
            a[i] = a[i] + (lo + (hi - lo) * rng());
          }
        }
        continue;
      }

      throw new Error("Unknown operation");
    }

    markDirty();
    saveToLocalStorage();
  } catch (e) {
    alert(String(e?.message || e));
  }
});

if (ui.opTargetFilter) {
  ui.opTargetFilter.addEventListener("input", () => {
    renderOpTargetsList();
    updateMaskedOpsPreview();
  });
  ui.opTargetFilter.addEventListener("change", () => {
    renderOpTargetsList();
    updateMaskedOpsPreview();
  });
}

if (ui.opTargetsSelectAll) {
  ui.opTargetsSelectAll.addEventListener("click", () => {
    const q = String(ui.opTargetFilter?.value || "").trim();
    
    let layers = state.layers;
    if (q) {
      const pattern = q.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
      const regex = new RegExp(`^${pattern}$`, 'i');
      layers = state.layers.filter((l) => regex.test(String(l.name)));
    }
    const isRandom = isRandomAssignOpType(ui.opType?.value);
    for (const l of layers) {
      if (isRandom && l.kind === "categorical") continue;
      opTargetsSelected.add(l.name);
    }
    renderOpTargetsList();
    updateMaskedOpsPreview();
  });
}
if (ui.opTargetsClear) {
  ui.opTargetsClear.addEventListener("click", () => {
    opTargetsSelected.clear();
    renderOpTargetsList();
    updateMaskedOpsPreview();
  });
}

// Keyboard shortcuts
function _isTypingTarget(el) {
  if (!el) return false;
  const tag = String(el.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}
function _selectLayerByDelta(delta) {
  const idx = state.layers.findIndex((l) => l.name === selectedLayer);
  if (idx < 0) return;
  const j = clamp(idx + delta, 0, state.layers.length - 1);
  selectedLayer = state.layers[j].name;
  ui.layerSelect.value = selectedLayer;
  renderLayersTable();
}
document.addEventListener("keydown", (ev) => {
  if (_isTypingTarget(ev.target)) return;
  if (ev.key === "j") {
    ev.preventDefault();
    _selectLayerByDelta(1);
  } else if (ev.key === "k") {
    ev.preventDefault();
    _selectLayerByDelta(-1);
  } else if (ev.key === "b") {
    ui.editMode.value = "brush";
  } else if (ev.key === "r") {
    ui.editMode.value = "rectangle";
  } else if (ev.key === "e") {
    ui.toggleEraser.checked = !ui.toggleEraser.checked;
  } else if (ev.key === "[") {
    const v = clamp(Number(ui.brushRadius.value) - 1, 0, 25);
    ui.brushRadius.value = String(v);
  } else if (ev.key === "]") {
    const v = clamp(Number(ui.brushRadius.value) + 1, 0, 25);
    ui.brushRadius.value = String(v);
  } else if (/^[1-9]$/.test(ev.key)) {
    ui.paintValue.value = String(Number(ev.key));
  } else if (ev.key === "Delete" || ev.key === "Backspace") {
    if (bulkSelectedLayers.size > 0) {
      ev.preventDefault();
      bulkDeleteSelected();
    }
  }
});

if (ui.bulkDeleteBtn) {
  ui.bulkDeleteBtn.addEventListener("click", () => bulkDeleteSelected());
}

ui.saveBtn.addEventListener("click", () => {
  const text = serializeState(state);
  download("gridstate.json", text);
  dirtySinceLastSave = false;
  _updateCurrentFileInfo();
  saveToLocalStorage();
});

ui.demoBtn.addEventListener("click", () => {
  const H = Number(ui.HInput.value);
  const W = Number(ui.WInput.value);
  
  // Reset all state for new file
  _resetAllForNewFile();
  
  state = makeDemoState(H, W, 0);
  selectedLayer = "cell_type";
  _setCurrentFile("demo");
  
  // Reset functions to defaults
  fnMeasurements = [];
  layerOps = [];
  saveFunctionsCfg();
  renderFunctionsSpecsTable();
  renderLayerOpsTable();
  
  applyAutoFitZoom();
  syncLayerSelect();
  markDirty();
  saveToLocalStorage();
});

ui.newBtn.addEventListener("click", () => {
  const H = Number(ui.HInput.value);
  const W = Number(ui.WInput.value);
  if (!confirm("Create a new blank grid? This will replace the current in-memory state.")) return;
  
  // Reset all state for new file
  _resetAllForNewFile();
  
  state = makeEmptyState(H, W);
  addLayer(state, { name: "cell_type", kind: "counts", init: "zeros", value: 0, seed: 0 });
  selectedLayer = "cell_type";
  _setCurrentFile("new");
  
  // Reset functions to defaults
  fnMeasurements = [];
  layerOps = [];
  saveFunctionsCfg();
  renderFunctionsSpecsTable();
  renderLayerOpsTable();
  
  applyAutoFitZoom();
  syncLayerSelect();
  markDirty();
  saveToLocalStorage();
});

ui.fileInput.addEventListener("change", async () => {
  const f = ui.fileInput.files?.[0];
  if (!f) return;
  const text = await f.text();
  try {
    // Reset all state for new file
    _resetAllForNewFile();
    
    const parsed = JSON.parse(text);
    state = parseState(text);
    ui.HInput.value = String(state.H);
    ui.WInput.value = String(state.W);
    selectedLayer = state.layers[0]?.name || "";
    
    // Apply embedded functions from file
    _tryApplyEmbeddedMeasurementsConfig(parsed);
    _tryApplyEmbeddedLayerOpsConfig(parsed);
    
    // Sync let ops with layers (create missing layers or remove orphaned ops)
    _syncLetOpsWithLayers();
    
    applyAutoFitZoom();
    syncLayerSelect();
    _setCurrentFile(f.name || "loaded");
    markDirty();
    saveToLocalStorage();
  } catch (e) {
    alert(String(e?.message || e));
  } finally {
    ui.fileInput.value = "";
  }
});

ui.HInput.addEventListener("change", () => {
  resizeIfNeeded(Number(ui.HInput.value), Number(ui.WInput.value));
  syncLayerSelect();
  markDirty();
});
ui.WInput.addEventListener("change", () => {
  resizeIfNeeded(Number(ui.HInput.value), Number(ui.WInput.value));
  syncLayerSelect();
  markDirty();
});

// Canvas interactions
let isDown = false;
let rectStart = null;
let lastCell = null;

function eventToCell(ev) {
  const zoom = Number(ui.zoomInput.value);
  const r = ui.canvas.getBoundingClientRect();
  const px = ev.clientX - r.left;
  const py = ev.clientY - r.top;
  const x = clamp(Math.floor(px / zoom), 0, state.W - 1);
  const y = clamp(Math.floor(py / zoom), 0, state.H - 1);
  return { y, x };
}

function drawOverlayRect(y0, x0, y1, x1) {
  const zoom = Number(ui.zoomInput.value);
  const ctx = ui.overlay.getContext("2d");
  ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  const xa = Math.min(x0, x1);
  const xb = Math.max(x0, x1);
  const ya = Math.min(y0, y1);
  const yb = Math.max(y0, y1);
  ctx.strokeStyle = "rgba(255,255,255,.85)";
  ctx.lineWidth = 2;
  ctx.strokeRect(xa * zoom + 1, ya * zoom + 1, (xb - xa + 1) * zoom - 2, (yb - ya + 1) * zoom - 2);
  ctx.fillStyle = "rgba(255,255,255,.10)";
  ctx.fillRect(xa * zoom + 1, ya * zoom + 1, (xb - xa + 1) * zoom - 2, (yb - ya + 1) * zoom - 2);
}

function clearOverlay() {
  const ctx = ui.overlay.getContext("2d");
  ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
}

function applyPaintAtCell(y, x) {
  const meta = state.layers.find((l) => l.name === selectedLayer);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) return;

  const value = ui.toggleEraser.checked ? 0 : meta.kind === "counts" ? clampCounts(ui.paintValue.value) : Math.round(Number(ui.paintValue.value));
  const r = Math.round(Number(ui.brushRadius.value));
  paintCircle(state, selectedLayer, y, x, r, value);
  markDirty();
}

ui.canvas.addEventListener("pointerdown", (ev) => {
  isDown = true;
  ui.canvas.setPointerCapture(ev.pointerId);
  const meta = state.layers.find((l) => l.name === selectedLayer);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) return;

  const { y, x } = eventToCell(ev);
  lastCell = { y, x };

  if (ui.editMode.value === "rectangle") {
    rectStart = { y, x };
    drawOverlayRect(y, x, y, x);
  } else {
    applyPaintAtCell(y, x);
  }
});

ui.canvas.addEventListener("pointermove", (ev) => {
  const { y, x } = eventToCell(ev);
  ui.cursorInfo.textContent = `(y,x): (${y}, ${x})`;

  const mode = ui.inspectMode?.value || "cursor";
  if (mode === "summary") {
    const meta = state.layers.find((l) => l.name === selectedLayer);
    const a = state.data[selectedLayer];
    if (ui.inspectCursorValue) {
      if (!meta || !a) ui.inspectCursorValue.textContent = "";
      else {
        const v = a[y * state.W + x];
        const vv = meta.kind === "categorical" || meta.kind === "counts" ? String(clampCounts(v)) : Number(v).toFixed(6);
        ui.inspectCursorValue.textContent = `${selectedLayer}@(${y},${x}) = ${vv}`;
      }
    }
  } else {
    renderInspectTable(state, y, x);
  }

  if (!isDown) return;

  const meta = state.layers.find((l) => l.name === selectedLayer);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) return;

  if (ui.editMode.value === "rectangle") {
    if (rectStart) drawOverlayRect(rectStart.y, rectStart.x, y, x);
    return;
  }

  if (lastCell && lastCell.y === y && lastCell.x === x) return;
  lastCell = { y, x };
  applyPaintAtCell(y, x);
});

ui.canvas.addEventListener("pointerup", (ev) => {
  isDown = false;
  ui.canvas.releasePointerCapture(ev.pointerId);

  const meta = state.layers.find((l) => l.name === selectedLayer);
  if (!meta || (meta.kind !== "categorical" && meta.kind !== "counts")) {
    rectStart = null;
    clearOverlay();
    return;
  }

  const { y, x } = eventToCell(ev);

  if (ui.editMode.value === "rectangle" && rectStart) {
    const value = ui.toggleEraser.checked ? 0 : meta.kind === "counts" ? clampCounts(ui.paintValue.value) : Math.round(Number(ui.paintValue.value));
    fillRect(state, selectedLayer, rectStart.y, rectStart.x, y, x, value);
    markDirty();
  }

  rectStart = null;
  clearOverlay();
});

ui.canvas.addEventListener("pointerleave", () => {
  ui.cursorInfo.textContent = `(y,x): –`;
});

// Sync let ops with actual layers - create missing layers or remove orphaned ops
function _syncLetOpsWithLayers() {
  const existingLayers = new Set(state.layers.map((l) => l.name));
  const letOpsToRemove = [];
  
  for (let i = 0; i < layerOps.length; i++) {
    const op = layerOps[i];
    if (op.type !== "let" || !op.var) continue;
    
    const varName = String(op.var).trim();
    if (!varName) continue;
    
    if (!existingLayers.has(varName)) {
      // Layer doesn't exist - create it
      try {
        addLayer(state, {
          name: varName,
          kind: "continuous",
          init: "zeros",
          value: 0,
          seed: 0,
          color: "#8B5CF6",
        });
        existingLayers.add(varName);
      } catch (e) {
        // If we can't create it, mark for removal
        console.warn(`Removing orphaned let op for '${varName}':`, e);
        letOpsToRemove.push(i);
      }
    }
  }
  
  // Remove orphaned ops (in reverse order to preserve indices)
  for (let i = letOpsToRemove.length - 1; i >= 0; i--) {
    layerOps.splice(letOpsToRemove[i], 1);
  }
  
  if (letOpsToRemove.length > 0) {
    saveFunctionsCfg();
  }
}

// init
{
  const loaded = tryLoadFromLocalStorage();
  if (loaded) {
    state = loaded;
    selectedLayer = state.layers[0]?.name || "";
  }
  _syncLetOpsWithLayers();
  _setCurrentFile(loaded ? "localStorage" : "demo");
  _inspectInitHistMaskControls();
  syncLayerSelect();
  ui.HInput.value = String(state.H);
  ui.WInput.value = String(state.W);
  applyAutoFitZoom();
  saveToLocalStorage();
  setInterval(() => {
    if (dirtySinceLastSave) saveToLocalStorage();
  }, 1000);
}
requestAnimationFrame(tick);

{
  const wrap = ui.canvasWrap || document.querySelector(".canvasWrap");
  if (wrap && typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => applyAutoFitZoom());
    ro.observe(wrap);
  } else {
    window.addEventListener("resize", () => applyAutoFitZoom());
  }
}

{
  const hist = ui.inspectCanvasHist;
  let raf = 0;
  const schedule = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      inspectSummaryDirty = true;
      _ensureInspectSummaryUpToDate();
    });
  };
  if (hist && typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => schedule());
    ro.observe(hist);
  }
  window.addEventListener("resize", () => schedule());
}
