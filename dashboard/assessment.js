(() => {
  "use strict";

  const root = document.getElementById("assessment");
  if (!root) return;
  const $ = id => document.getElementById(id);
  const MODELS = [
    {id: "SGP4", name: "SGP4", color: "#f2b467", method: "Analytical / semi-analytical propagator", detail: "Traditional baseline. No learned weights or training stage; model form includes perturbations beyond the central-gravity + J2 diagnostic."},
    {id: "PINN", name: "Standalone PINN", color: "#aaabff", method: "Physics-informed neural network", detail: "Trained on central-gravity + J2 equation residuals and exact initial conditions. It receives no ESA orbit labels."},
    {id: "COMBINED", name: "Combined residual model", color: "#6ee7dc", method: "SGP4 + supervised neural correction", detail: "A separate neural correction trained on earlier ESA residuals. It uses additional reference data and does not enforce the PINN physics loss."}
  ];
  const SPLIT_NAMES = {test: "Test · held-out", validation: "Validation", train: "Training"};
  let runIndex = null;
  let payload = null;
  let controller = null;
  let generation = 0;

  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined && text !== null) value.textContent = String(text);
    return value;
  }
  function numeric(value) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
    return null;
  }
  function format(value, digits = 3) {
    const number = numeric(value);
    return number === null ? "N/A" : number.toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
  }
  function scientific(value) {
    const number = numeric(value);
    return number === null ? "N/A" : number.toExponential(2);
  }
  function diagnosticFormat(value, digits) {
    const number = numeric(value);
    return number === null ? "N/A" : number !== 0 && Math.abs(number) < 0.001 ? scientific(number) : format(number, digits);
  }
  function dateTime(value) {
    if (typeof value !== "string") return "time unavailable";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "time unavailable" : date.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
  }
  function safeURL(raw) {
    if (typeof raw !== "string" || !raw) return null;
    try {
      const url = new URL(raw, window.location.origin);
      return url.origin === window.location.origin && url.pathname.startsWith("/assets/model-results/") && !url.username && !url.password ? url.href : null;
    } catch (_) { return null; }
  }
  async function fetchJSON(raw, signal) {
    const url = safeURL(raw);
    if (!url) throw new Error("The benchmark result link is invalid.");
    const response = await fetch(url, {cache: "no-store", credentials: "same-origin", signal});
    if (!response.ok) throw new Error(`Saved benchmark could not be loaded (HTTP ${response.status}).`);
    return response.json();
  }
  function notice(message, state = "loading") {
    const box = $("assessmentNotice");
    box.textContent = message;
    box.dataset.state = state;
    box.hidden = false;
    $("assessmentContent").hidden = true;
  }
  function metricsFor(modelId, split) { return payload?.metrics?.[modelId]?.[split] || {}; }
  function table(title, rows) {
    const wrap = node("div", "assessment-table-scroll");
    const grid = node("table", "assessment-table");
    grid.setAttribute("aria-label", title);
    const head = node("thead");
    const heading = node("tr");
    ["Metric", ...MODELS.map(model => model.name)].forEach((label, index) => {
      const cell = node("th", "", label);
      cell.scope = "col";
      if (index) cell.dataset.model = MODELS[index - 1].id;
      heading.append(cell);
    });
    head.append(heading);
    const body = node("tbody");
    rows.forEach(([label, key, suffix, digits]) => {
      const row = node("tr");
      const headingCell = node("th", "assessment-row-label", label);
      headingCell.scope = "row";
      row.append(headingCell);
      MODELS.forEach(model => {
        const value = payload?.physics_diagnostics?.[model.id]?.[key];
        const cell = node("td", "", `${diagnosticFormat(value, digits)}${numeric(value) === null ? "" : suffix}`);
        cell.dataset.model = model.id;
        row.append(cell);
      });
      body.append(row);
    });
    grid.append(head, body);
    wrap.append(grid);
    return wrap;
  }
  function renderAccuracy(split) {
    const area = $("assessmentAccuracyChart");
    area.replaceChildren();
    const values = MODELS.map(model => ({model, metric: metricsFor(model.id, split)}));
    const max = Math.max(0, ...values.map(row => numeric(row.metric.position_vector_rmse_km) || 0));
    values.forEach(({model, metric}) => {
      const rmse = numeric(metric.position_vector_rmse_km);
      const row = node("div", "assessment-error-row");
      row.setAttribute("role", "listitem");
      const labels = node("div", "assessment-error-labels");
      const name = node("strong", "", model.name); name.style.setProperty("--model-color", model.color);
      labels.append(name, node("strong", "assessment-error-value", rmse === null ? "N/A" : `${format(rmse)} km`));
      const track = node("div", "assessment-error-track");
      track.setAttribute("aria-hidden", "true");
      const fill = node("span", "assessment-error-fill");
      fill.style.setProperty("--model-color", model.color);
      fill.style.width = max > 0 && rmse !== null ? `${Math.max(1, rmse / max * 100)}%` : "0%";
      track.append(fill);
      const velocity = numeric(metric.velocity_vector_rmse_km_s);
      const p95 = numeric(metric.position_p95_km);
      const detail = [
        `${format(metric.future_samples, 0)} future samples`,
        `position P95 ${p95 === null ? "N/A" : `${format(p95)} km`}`,
        `velocity RMSE ${velocity === null ? "N/A" : `${format(velocity * 1000, 2)} m/s`}`
      ].join(" · ");
      row.append(labels, track, node("div", "assessment-error-detail", detail));
      area.append(row);
    });
    const samples = values.map(row => numeric(row.metric.future_samples)).filter(value => value !== null);
    const count = samples.length && samples.every(value => value === samples[0]) ? samples[0] : null;
    $("assessmentAccuracyScope").textContent = `${SPLIT_NAMES[split] || split} interval · ${count === null ? "sample counts unavailable" : `${format(count, 0)} forecast samples per model`} · common timestamps and the same independent orbit reference.`;

    const sgp4 = numeric(metricsFor("SGP4", split).position_vector_rmse_km);
    const pinn = numeric(metricsFor("PINN", split).position_vector_rmse_km);
    const combined = numeric(metricsFor("COMBINED", split).position_vector_rmse_km);
    const correctionTraining = numeric(payload?.training_proofs?.COMBINED?.training_samples);
    const correctionValidation = numeric(payload?.training_proofs?.COMBINED?.validation_samples);
    let finding = "Held-out position errors are unavailable for the selected interval.";
    if (sgp4 !== null && pinn !== null && combined !== null) {
      const pinnRatio = sgp4 > 0 ? pinn / sgp4 : null;
      const combinedDelta = sgp4 > 0 ? (combined / sgp4 - 1) * 100 : null;
      const combinedComparison = combinedDelta === null ? "" : combinedDelta <= 0 ? ` (${format(Math.abs(combinedDelta), 1)}% lower than SGP4)` : ` (${format(combinedDelta, 1)}% higher than SGP4)`;
      finding = `On the selected ${SPLIT_NAMES[split] || split} interval, standalone PINN position RMSE is ${format(pinn)} km versus ${format(sgp4)} km for SGP4${pinnRatio === null ? "" : ` (${format(pinnRatio, 1)}× higher than SGP4)`}. The combined residual model is ${format(combined)} km${combinedComparison}; it used ${format(correctionTraining, 0)} earlier ESA calibration and ${format(correctionValidation, 0)} validation samples. These results describe this one arc and are not an equal-information comparison.`;
    }
    $("assessmentFinding").textContent = finding;
  }
  function renderPhysics() {
    const area = $("assessmentPhysicsTable");
    area.replaceChildren(table("Full-arc approximate physics diagnostics", [
      ["J2 energy relative drift · P95", "j2_energy_relative_drift_p95", "", 2],
      ["Axial angular momentum drift · P95", "axial_angular_momentum_drift_p95_km2_s", " km²/s", 3],
      ["Position / velocity kinematic residual RMS", "kinematic_residual_vector_rms_km_s", " km/s", 6],
      ["Central + J2 acceleration residual RMS", "central_j2_acceleration_residual_vector_rms_km_s2", " km/s²", 2]
    ]));
  }
  function check(label, passed) {
    const row = node("div", "assessment-check");
    const known = typeof passed === "boolean";
    const okay = passed === true;
    row.append(node("span", `assessment-check-mark ${okay ? "is-pass" : known ? "is-fail" : "is-unknown"}`, okay ? "✓" : known ? "!" : "—"));
    row.append(node("span", "", label), node("strong", okay ? "is-pass" : known ? "is-fail" : "is-unknown", okay ? "Verified" : known ? "Not verified" : "Unavailable"));
    return row;
  }
  function renderReliability(split) {
    const area = $("assessmentReliabilityChecks");
    area.replaceChildren();
    const verification = payload?.verification || {};
    const replay = payload?.replay_verification || {};
    const counts = MODELS.map(model => numeric(metricsFor(model.id, split).future_samples));
    const equalCounts = counts.every(value => value !== null && value === counts[0]);
    const checks = [
      ["All saved prediction states finite", verification.all_states_finite],
      ["Shared initial state across methods", verification.same_initial_r_and_v],
      ["Equal sample count in selected split", equalCounts],
      ["Held-out test excluded from fitting", verification.no_test_samples_in_training && verification.correction_frozen_before_test],
      ["PINN benchmark object excluded from training catalog", verification.benchmark_absent_from_pinn_catalog && verification.pinn_no_esa_training_labels],
      ["Saved checkpoints and metrics replayed", replay.passed]
    ];
    const passed = checks.filter(([, status]) => status === true).length;
    $("assessmentCheckCount").textContent = `${passed}/${checks.length} verified`;
    checks.forEach(([label, status]) => area.append(check(label, status)));
    if (numeric(replay.original_checksums_verified) !== null) {
      area.append(node("p", "assessment-replay-detail", `${format(replay.original_checksums_verified, 0)} saved artifact checksums verified; checkpoint replays reproduce the stored predictions.`));
    }
  }
  function renderEfficiency() {
    const area = $("assessmentEfficiencyTable");
    const wrap = node("div", "assessment-table-scroll");
    const grid = node("table", "assessment-table assessment-efficiency-table");
    grid.setAttribute("aria-label", "Model training and prediction cost comparison");
    const points = numeric(payload?.split?.horizon_seconds) !== null && numeric(payload?.split?.step_seconds) > 0
      ? Math.round(payload.split.horizon_seconds / payload.split.step_seconds) + 1 : 181;
    const head = node("thead");
    const heading = node("tr");
    ["Method", "Trainable NN weights + biases", "Training time", `Prediction / ${points} states`, "vs SGP4"].forEach(label => { const th = node("th", "", label); th.scope = "col"; heading.append(th); });
    head.append(heading);
    const body = node("tbody");
    const baseline = numeric(payload?.prediction_timing_seconds?.SGP4);
    MODELS.forEach(model => {
      const proof = payload?.training_proofs?.[model.id] || {};
      const timeSeconds = numeric(payload?.prediction_timing_seconds?.[model.id]);
      const ratio = baseline !== null && baseline > 0 && timeSeconds !== null ? timeSeconds / baseline : null;
      const row = node("tr");
      const method = node("th", "assessment-row-label", model.name); method.scope = "row"; method.dataset.model = model.id;
      row.append(method,
        node("td", "", model.id === "SGP4" ? "0 · none" : numeric(proof.parameter_count) === null ? "N/A" : format(proof.parameter_count, 0)),
        node("td", "", model.id === "SGP4" ? "No training" : numeric(proof.training_seconds) === null ? "N/A" : `${format(proof.training_seconds, 2)} s`),
        node("td", "", timeSeconds === null ? "N/A" : `${format(timeSeconds * 1000, 4)} ms`),
        node("td", "", ratio === null ? "N/A" : `${format(ratio, 1)}×`));
      body.append(row);
    });
    grid.append(head, body); wrap.append(grid); area.replaceChildren(wrap);
    const hw = payload?.hardware || {};
    const cpu = typeof hw.cpu === "string" ? hw.cpu : "CPU model unavailable";
    const repeats = Array.isArray(payload?.files) && payload.files.some(file => file.path === "comparison/propagation_benchmarks.csv") ? "Median of five warmed, order-rotated runs" : "Saved timing summary";
    $("assessmentHardwareNote").textContent = `${repeats} on ${cpu} (${hw.accelerator || "accelerator not recorded"}). Each prediction call covers ${points} samples; training time is listed separately and file I/O is excluded. These are empirical timings, not a Big-O complexity estimate.`;
  }
  function renderMethods() {
    const area = $("assessmentMethodCards"); area.replaceChildren();
    MODELS.forEach(model => {
      const card = node("article", "assessment-method-card");
      card.style.setProperty("--model-color", model.color);
      card.append(node("span", "assessment-method-tag", model.method), node("h4", "", model.name), node("p", "", model.detail));
      area.append(card);
    });
  }
  function parameterCard(modelId) {
    const model = MODELS.find(item => item.id === modelId);
    const proof = payload?.training_proofs?.[modelId] || {};
    const hardware = payload?.hardware || {};
    const runConfig = payload?.split || {};
    const facts = [];
    const fact = (label, value) => facts.push([label, value]);

    if (modelId === "SGP4") {
      fact("Learned parameters", "None. This method has no neural network weights or biases.");
      fact("Inputs", "CelesTrak GP mean orbital elements and their epoch; the benchmark starts all methods from the same position and velocity state.");
      fact("Fixed model constants", "SGP4 uses the WGS-72 gravity parameter set from the SGP4 implementation.");
      fact("Shared benchmark settings", `${format((numeric(runConfig.horizon_seconds) || 10800) / 3600, 1)} h forecast · ${format(runConfig.step_seconds || 60, 0)} s cadence · TEME state frame.`);
    } else if (modelId === "PINN") {
      const width = numeric(hardware.width) || 48;
      const depth = numeric(proof.depth) || 3;
      const count = numeric(proof.parameter_count);
      const catalog = proof.catalog || {};
      const splits = catalog.split_counts || {};
      fact("Architecture", `6 normalized orbital/time features → ${depth} Tanh hidden layers × ${format(width, 0)} units → 6 outputs.`);
      fact(`Why the count is ${format(count, 0)}`, `(6×${format(width, 0)}+${format(width, 0)}) + ${(depth - 1)}×(${format(width, 0)}×${format(width, 0)}+${format(width, 0)}) + (${format(width, 0)}×6+6) = ${format(count, 0)} weights and biases.`);
      fact("Network inputs", "Normalized initial-state features: radius, radial velocity, speed squared, elapsed time τ, normalized z, and normalized vz. The network predicts Taylor-trial corrections to position and velocity.");
      fact("Training objective", "Central-gravity + J2 differential-equation residual with exact initial conditions; no ESA future-track labels.");
      fact("Gravity constants / scales", "μ = 398600.8 km³/s²; Earth radius = 6378.135 km; J2 = 0.001082616; position scale = 7000 km. Velocity and time scales are derived from μ and the position scale.");
      fact("Optimizer settings", `Adam · initial learning rate 0.003 with cosine decay to 0.00001 · gradient norm clipped at 10 · ${format(proof.optimizer_steps_executed, 0)} optimizer steps · batch ${format(hardware.batch_size, 0)} · seed ${format(runConfig.seed, 0)} · float64.`);
      fact("Catalog split", `${format(splits.train, 0)} training · ${format(splits.validation, 0)} validation · ${format(splits.test, 0)} test objects. The independent Sentinel-1D benchmark object is excluded.`);
    } else {
      const config = proof.configuration || {};
      const width = numeric(config.width) || 32;
      const count = numeric(proof.parameter_count);
      const period = numeric(config.period_seconds);
      const scale = Array.isArray(config.scale) ? config.scale : [];
      const positionScale = scale.slice(0, 3).map(value => format(value, 4)).join(" / ");
      const velocityScale = scale.slice(3, 6).map(value => scientific(value)).join(" / ");
      fact("Architecture", `5 time / orbital-phase features → ${format(width, 0)} → ${format(width, 0)} Tanh layers → 6 scaled correction outputs.`);
      fact(`Why the count is ${format(count, 0)}`, `(5×${format(width, 0)}+${format(width, 0)}) + (${format(width, 0)}×${format(width, 0)}+${format(width, 0)}) + (${format(width, 0)}×6+6) = ${format(count, 0)} weights and biases.`);
      fact("Network inputs", `sin φ, cos φ, sin 2φ, cos 2φ, and elapsed time / ${format(config.time_scale_seconds, 0)} s; φ = 2πt / T, with T = ${format(period, 2)} s.`);
      fact("Outputs / targets", "Six SGP4-relative corrections: position and velocity in radial, transverse, and normal axes. Trained on earlier ESA − SGP4 residuals, scaled by training-only RMS values.");
      fact("Training / validation / test", `${format(proof.training_samples, 0)} ESA calibration samples · ${format(proof.validation_samples, 0)} validation samples · ${format(proof.test_samples_received, 0)} test samples received during training. The correction is gated to zero at the common initial time.`);
      fact("Optimizer settings", `AdamW · initial learning rate 0.003 · weight decay 0.0001 · cosine decay to 0.00001 · ${format(proof.optimizer_steps_executed, 0)} steps · gradient norm clipped at 10 · seed ${format(config.seed, 0)}.`);
      fact("Residual scaling R / T / N", `Position (km): ${positionScale}. Velocity (km/s): ${velocityScale}.`);
    }

    const card = node("article", "assessment-param-card");
    card.style.setProperty("--model-color", model.color);
    card.append(node("span", "assessment-method-tag", model.method), node("h4", "", model.name));
    const list = node("dl", "assessment-param-facts");
    facts.forEach(([label, value]) => {
      const group = node("div", "assessment-param-fact");
      group.append(node("dt", "", label), node("dd", "", value));
      list.append(group);
    });
    card.append(list);
    return card;
  }
  function renderParameterDetails() {
    const area = $("assessmentParameterDetails");
    area.replaceChildren(...MODELS.map(model => parameterCard(model.id)));
  }
  function renderDownloads() {
    const area = $("assessmentDownloads"); area.replaceChildren();
    const files = payload?.files || [];
    const wanted = [
      ["Benchmark report ↓", "comparison_report.md"],
      ["Held-out metrics CSV ↓", "comparison/test_metrics.csv"],
      ["Physics diagnostics ↓", "comparison/physics_diagnostics.json"],
      ["Timing runs CSV ↓", "comparison/propagation_benchmarks.csv"]
    ];
    wanted.forEach(([label, path]) => {
      const file = files.find(item => item.path === path);
      const href = file ? safeURL(file.url) : null;
      if (!href) return;
      const link = node("a", "assessment-download", label);
      link.href = href;
      link.download = path.split("/").pop();
      area.append(link);
    });
  }
  function render() {
    if (!payload) return;
    const split = $("modelSplitSelect")?.value || "test";
    $("assessmentSplitLabel").textContent = `${SPLIT_NAMES[split] || split} · synced with Orbital evaluation`;
    $("assessmentRunName").textContent = `${payload.object_name || "Saved orbit"} · NORAD ${payload.norad_id ?? "N/A"}`;
    $("assessmentRunMeta").textContent = `${dateTime(payload.start_utc)} → ${dateTime(payload.end_utc)} · ${payload.reference_label || "Independent orbit reference"}`;
    renderAccuracy(split);
    renderPhysics();
    renderReliability(split);
    renderEfficiency();
    renderParameterDetails();
    renderMethods();
    renderDownloads();
    $("assessmentNotice").hidden = true;
    $("assessmentContent").hidden = false;
  }
  async function loadRun(runId, currentController = null, current = null) {
    if (!currentController) {
      controller?.abort();
      controller = new AbortController();
      currentController = controller;
      current = ++generation;
    }
    const entry = runIndex?.runs.find(item => item.run_id === runId);
    if (!entry) return;
    notice(`Loading assessment for ${entry.object_name || "selected orbit"}…`);
    try {
      const result = await fetchJSON(entry.payload_url, currentController.signal);
      if (current !== generation) return;
      if (!result || result.run_id !== runId || !result.metrics || !MODELS.every(model => result.metrics[model.id])) throw new Error("This run does not include complete SGP4, PINN, and combined metrics.");
      payload = result;
      render();
    } catch (error) {
      if (error.name !== "AbortError" && current === generation) notice(error.message || "Unable to load the benchmark assessment.", "error");
    }
  }
  async function loadIndex() {
    controller?.abort();
    controller = new AbortController();
    const current = ++generation;
    notice("Loading the saved benchmark…");
    try {
      const result = await fetchJSON("/assets/model-results/index.json", controller.signal);
      if (current !== generation) return;
      if (!result || !Array.isArray(result.runs)) throw new Error("The saved benchmark index is incomplete.");
      runIndex = result;
      const requested = $("modelRunSelect")?.value || result.latest_run_id;
      const entry = result.runs.find(item => item.run_id === requested) || result.runs.find(item => item.run_id === result.latest_run_id) || result.runs[0];
      if (!entry) throw new Error("No completed benchmark runs are available.");
      await loadRun(entry.run_id, controller, current);
    } catch (error) {
      if (error.name !== "AbortError" && current === generation) notice(error.message || "Unable to load the saved benchmark.", "error");
    }
  }

  $("assessmentRefresh").addEventListener("click", loadIndex);
  $("modelRunSelect")?.addEventListener("change", event => {
    if (runIndex?.runs.some(item => item.run_id === event.target.value)) loadRun(event.target.value);
  });
  $("modelResultsRefresh")?.addEventListener("click", () => window.setTimeout(loadIndex, 0));
  $("modelSplitSelect")?.addEventListener("change", render);
  loadIndex();
})();
