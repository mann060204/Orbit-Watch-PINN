(() => {
  "use strict";

  const root = document.getElementById("models");
  if (!root) return;
  const $ = id => document.getElementById(id);
  const MODEL_IDS = ["SGP4", "PINN", "COMBINED"];
  const MODELS = {
    SGP4: {name: "SGP4", folder: "sgp4", color: "#f2b467", description: "Analytical orbit propagation from the shared CelesTrak initial input. SGP4 has no neural training step."},
    PINN: {name: "PINN", folder: "pinn", color: "#aaabff", description: "Standalone physics-informed neural network trained on central gravity and J2 equations, with shared initial conditions. It receives no ESA orbit labels."},
    COMBINED: {name: "Combined", folder: "combined", color: "#6ee7dc", description: "SGP4 + a supervised neural correction trained on earlier ESA position and velocity residuals. This correction is a separate network from the standalone PINN and does not enforce its physics loss."}
  };
  const SPLITS = {test: "Test", validation: "Validation", train: "Training"};
  const FIGURE_NAMES = {
    "01_state_predictions": "Position & velocity predictions",
    "02_error_norms": "Position & velocity error over time",
    "03_rtn_errors": "Radial, transverse & normal errors",
    "04_training_history": "Training & validation loss",
    "01_full_window_errors": "All three models · full-window errors",
    "02_held_out_metric_comparison": "Reserved test interval · error comparison",
    "03_held_out_rtn_errors": "Reserved test interval · RTN errors",
    "04_held_out_cumulative_errors": "Reserved test interval · cumulative RMSE & MAE"
  };
  let runIndex = null;
  let data = null;
  let selected = "comparison";
  let split = "test";
  let request = null;
  let generation = 0;

  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined && text !== null) value.textContent = String(text);
    return value;
  }
  function number(value) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
    return null;
  }
  function format(value, decimals = 2) {
    const n = number(value);
    return n === null ? "N/A" : n.toLocaleString("en-US", {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  }
  function scientific(value) {
    const n = number(value);
    if (n === null) return "N/A";
    return n !== 0 && Math.abs(n) < 0.001 ? n.toExponential(4) : format(n, 6);
  }
  function meters(value) {
    const n = number(value);
    return n === null ? null : n * 1000;
  }
  function json(value) { return JSON.stringify(value === undefined ? null : value, null, 2); }
  function utc(value) {
    if (typeof value !== "string") return "Unavailable";
    const timestamp = new Date(value);
    return Number.isNaN(timestamp.getTime()) ? "Unavailable" : timestamp.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
  }
  function utcOffset(seconds) {
    const offset = number(seconds);
    const start = new Date(data.start_utc).getTime();
    return offset === null || !Number.isFinite(start) ? "Unavailable" : utc(new Date(start + offset * 1000).toISOString());
  }
  function safeURL(raw) {
    if (typeof raw !== "string" || !raw) return null;
    try {
      const url = new URL(raw, window.location.origin);
      if (url.origin !== window.location.origin || !url.pathname.startsWith("/assets/model-results/") || url.username || url.password) return null;
      return url.href;
    } catch (_) { return null; }
  }
  function artifactURL(path) {
    if (!data || typeof path !== "string" || path.includes("\\") || path.split("/").some(part => part === "..")) return null;
    const file = (data.files || []).find(item => item.path === path);
    if (file) return safeURL(file.url);
    if (!data.base_url) return null;
    return safeURL(`${data.base_url.replace(/\/$/, "")}/${path.split("/").map(encodeURIComponent).join("/")}`);
  }
  function download(text, path, className = "quiet") {
    const url = artifactURL(path);
    if (!url) return node("span", "secondary", `${text} unavailable`);
    const link = node("a", className, text);
    link.href = url;
    link.download = path.split("/").pop();
    return link;
  }
  function setLink(id, raw, filename) {
    const link = $(id);
    const url = safeURL(raw);
    link.hidden = !url;
    if (url) { link.href = url; link.download = filename || ""; }
    else link.removeAttribute("href");
  }
  function notice(text, state = "loading") {
    const box = $("modelResultsNotice");
    box.replaceChildren(node("span", "", text));
    box.dataset.state = state;
    box.hidden = false;
    const evaluation = $("modelEvaluationNotice");
    evaluation.textContent = state === "loading" ? "Loading the selected experiment’s orbit evaluation…" : `${text} Use Model training above to select or refresh a run.`;
    evaluation.dataset.state = state;
    evaluation.hidden = false;
    if (state === "error") {
      const button = node("button", "quiet", "Retry");
      button.type = "button";
      button.addEventListener("click", () => loadIndex());
      box.append(button);
    }
  }
  async function fetchJSON(url, signal) {
    const target = safeURL(url);
    if (!target) throw new Error("The saved result link is invalid.");
    const response = await fetch(target, {cache: "no-store", signal, credentials: "same-origin"});
    if (!response.ok) throw new Error(`Saved results could not be read (HTTP ${response.status}).`);
    return response.json();
  }
  async function loadIndex() {
    request?.abort();
    const controller = new AbortController();
    request = controller;
    const current = ++generation;
    $("modelResultsRefresh").disabled = true;
    $("modelRunSelect").disabled = true;
    $("modelResultsContent").hidden = true;
    $("modelTrainingContent").hidden = true;
    $("modelRunDownload").hidden = true;
    notice("Loading saved model results…");
    try {
      const index = await fetchJSON("/assets/model-results/index.json", controller.signal);
      if (current !== generation) return;
      if (!index || !Array.isArray(index.runs)) throw new Error("The saved results index is incomplete.");
      runIndex = index;
      const select = $("modelRunSelect");
      select.replaceChildren();
      const runs = index.runs.filter(item => item && typeof item.run_id === "string" && safeURL(item.payload_url));
      if (!runs.length) {
        select.append(node("option", "", "No completed runs"));
        notice("No completed three-model runs are available yet. Run the training workflow, then refresh results.", "empty");
        return;
      }
      for (const item of runs) {
        const option = node("option", "", `${item.object_name || "Orbit"} · ${item.run_id}`);
        option.value = item.run_id;
        select.append(option);
      }
      const requested = data?.run_id || index.latest_run_id;
      select.value = runs.some(item => item.run_id === requested) ? requested : runs[0].run_id;
      select.disabled = false;
      await loadRun(select.value, controller, current);
    } catch (error) {
      if (error.name !== "AbortError" && current === generation) notice(error.message || "Unable to load saved results.", "error");
    } finally {
      if (current === generation) $("modelResultsRefresh").disabled = false;
    }
  }
  async function loadRun(runId, controller = null, current = null) {
    if (!controller) {
      request?.abort();
      controller = new AbortController();
      request = controller;
      current = ++generation;
    }
    const entry = runIndex?.runs.find(item => item.run_id === runId);
    if (!entry) return;
    $("modelResultsContent").hidden = true;
    $("modelTrainingContent").hidden = true;
    $("modelRunDownload").hidden = true;
    notice("Loading this run’s evaluations, training records, and graphs…");
    try {
      const payload = await fetchJSON(entry.payload_url, controller.signal);
      if (current !== generation) return;
      if (!payload || payload.run_id !== runId || !payload.metrics || !MODEL_IDS.every(id => payload.metrics[id])) throw new Error("This run does not contain complete SGP4, PINN, and combined evaluations.");
      data = payload;
      $("modelObjectName").textContent = `${payload.object_name} · NORAD ${payload.norad_id}`;
      $("modelReferenceText").textContent = `Reference: ${payload.reference_label || "Independent orbit reference"}. This is a restituted orbit estimate, not an error-free or final precise orbit.`;
      setLink("modelRunDownload", payload.archive_url, `${payload.run_id}.zip`);
      renderTraining();
      renderView();
      renderFiles();
      $("modelResultsNotice").hidden = true;
      $("modelEvaluationNotice").hidden = true;
      $("modelTrainingContent").hidden = false;
      $("modelResultsContent").hidden = false;
    } catch (error) {
      if (error.name !== "AbortError" && current === generation) notice(error.message || "Unable to read this saved run.", "error");
    }
  }
  function metric(id) { return data.metrics[id]?.[split] || {}; }
  function activeModels() { return selected === "comparison" ? MODEL_IDS : [selected]; }
  function intervalText() {
    const cfg = data.split || {};
    const cadence = number(cfg.step_seconds);
    const from = split === "train" ? cadence : split === "validation" ? (number(cfg.train_end_seconds) ?? 0) + (cadence ?? 0) : (number(cfg.validation_end_seconds) ?? 0) + (cadence ?? 0);
    const to = split === "train" ? cfg.train_end_seconds : split === "validation" ? cfg.validation_end_seconds : cfg.test_end_seconds;
    $("modelIntervalText").textContent = `${SPLITS[split]} samples: ${utcOffset(from)} → ${utcOffset(to)} · ${format(cadence, 0)} s cadence. Full saved arc: ${utc(data.start_utc)} → ${utc(data.end_utc)}.`;
  }
  function addMetricCard(container, label, value, unit, color, note, secondary) {
    const card = node("article", "model-metric-card");
    card.style.setProperty("--model-color", color);
    const title = node("div", "model-metric-card-label");
    title.append(node("i"), node("span", "", label));
    const result = node("div", "model-metric-value", format(value));
    if (value !== null) result.append(node("small", "", unit));
    card.append(title, result, node("div", "model-metric-card-note", note));
    if (secondary) {
      const row = node("div", "model-metric-card-secondary");
      row.append(node("span", "", secondary.label), node("strong", "", secondary.value));
      card.append(row);
    }
    container.append(card);
  }
  function table(title, columns, rows) {
    const box = node("div", "model-table-box");
    box.append(node("h4", "", title));
    const wrap = node("div", "model-table-scroll");
    const grid = node("table");
    const header = node("tr");
    for (const column of columns) {
      const cell = node("th", "", typeof column === "string" ? column : column.label);
      cell.scope = "col";
      if (column.model) cell.dataset.model = column.model;
      header.append(cell);
    }
    const head = node("thead"); head.append(header);
    const body = node("tbody");
    for (const values of rows) {
      const row = node("tr");
      values.forEach((value, index) => {
        const cell = node("td", "", value === undefined || value === null ? "N/A" : value);
        if (columns[index]?.model) cell.dataset.model = columns[index].model;
        row.append(cell);
      });
      body.append(row);
    }
    grid.append(head, body); wrap.append(grid); box.append(wrap);
    return box;
  }
  function renderMetricTables(ids) {
    const area = $("modelMetricTables"); area.replaceChildren();
    const columns = ["Metric", ...ids.map(id => ({label: MODELS[id].name, model: id}))];
    for (const quantity of ["position", "velocity"]) {
      const unit = quantity === "position" ? "m" : "m/s";
      const suffix = quantity === "position" ? "km" : "km_s";
      const scalar = [
        ["Vector RMSE", `${quantity}_vector_rmse_${suffix}`],
        ["Vector MAE", `${quantity}_vector_mae_${suffix}`],
        ["95th percentile", `${quantity}_p95_${suffix}`],
        ["Maximum error", `${quantity}_max_${suffix}`],
        ["Final sample error", `final_${quantity}_error_${suffix}`]
      ];
      const rows = scalar.map(([label, key]) => [label, ...ids.map(id => format(meters(metric(id)[key]), quantity === "velocity" ? 4 : 2))]);
      area.append(table(`${quantity === "position" ? "Position" : "Velocity"} errors (${unit})`, columns, rows));
    }
    for (const quantity of ["position", "velocity"]) {
      const suffix = quantity === "position" ? "km" : "km_s";
      const rows = [];
      for (const measure of ["rmse", "mae"]) {
        ["R · radial", "T · transverse", "N · normal"].forEach((axis, index) => rows.push([`${axis} ${measure.toUpperCase()}`, ...ids.map(id => format(meters(metric(id)[`${quantity}_rtn_${measure}_${suffix}`]?.[index]), quantity === "velocity" ? 4 : 2))]));
      }
      area.append(table(`${quantity === "position" ? "Position" : "Velocity"} in reference RTN (${quantity === "position" ? "m" : "m/s"})`, columns, rows));
    }
  }
  function facts(entries, className) {
    const list = node("dl", className);
    entries.forEach(([label, value]) => {
      const group = node("div");
      group.append(node("dt", "", label), node("dd", "", value)); list.append(group);
    });
    return list;
  }
  function disclosure(title, content) {
    const details = node("details", "model-disclosure");
    details.append(node("summary", "", title), content);
    return details;
  }
  function trainingHistory(id, proof) {
    const history = Array.isArray(data.histories?.[id]) ? data.histories[id] : [];
    if (!history.length) return node("p", "model-no-data", "No saved step-by-step training history is available for this model.");
    const box = node("div", "model-training-history");
    const grid = node("table");
    const head = node("thead"); const heading = node("tr");
    const keys = Object.keys(history[0]);
    for (const key of keys) { const th = node("th", "", key.replaceAll("_", " ")); th.scope = "col"; heading.append(th); }
    head.append(heading);
    const body = node("tbody");
    for (const record of history) {
      const row = node("tr");
      if (number(record.step) !== null && number(record.step) === number(proof.best_step)) row.dataset.selected = "true";
      for (const key of keys) {
        const raw = record[key];
        const value = key === "step" ? format(raw, 0) : number(raw) === null ? raw : scientific(raw);
        const cell = node("td", "", value);
        cell.title = String(raw ?? "N/A"); row.append(cell);
      }
      body.append(row);
    }
    grid.append(head, body); box.append(grid); return box;
  }
  function renderTraining() {
    const area = $("modelTrainingArea"); area.replaceChildren();
    const p = node("div", "model-training-empty");
    const baseline = node("div");
    baseline.append(node("strong", "", "SGP4 · analytical baseline"), node("p", "", "No neural training is needed. All three models start from the same CelesTrak-derived position and velocity."));
    const links = node("div", "model-download-row");
    links.append(download("Initial GP input ↓", "inputs/benchmark/initial_gp.json"), download("Initial TLE ↓", "inputs/derived_initial_tle.txt"), download("Setup proof ↓", "sgp4/training_proof.json"));
    p.append(baseline, links); area.append(p);
    const trained = ["PINN", "COMBINED"];
    const grid = node("div", "model-training-grid"); grid.dataset.count = String(trained.length);
    trained.forEach(id => {
      const proof = data.training_proofs?.[id] || {};
      const card = node("article", "model-training-card"); card.style.setProperty("--model-color", MODELS[id].color);
      card.append(node("div", "training-card-label", id === "PINN" ? "PHYSICS-INFORMED NETWORK" : "SGP4 + NEURAL CORRECTION"));
      card.append(node("h3", "", id === "PINN" ? "Standalone PINN" : "Combined model"));
      card.append(facts([
        ["Optimizer steps", format(proof.optimizer_steps_executed, 0)],
        ["Selected step", format(proof.best_step, 0)],
        ["Parameters", format(proof.parameter_count, 0)],
        ["Training time", `${format(proof.training_seconds)} s`],
        ["Initial validation loss", scientific(proof.initial_validation_physics_loss ?? proof.initial_validation_loss)],
        ["Best validation loss", scientific(proof.best_validation_physics_loss ?? proof.best_validation_loss)]
      ], "model-training-facts"));
      card.append(node("p", "", proof.training_targets || MODELS[id].description));
      if (id === "PINN" && proof.catalog) card.append(node("p", "", `${format(proof.catalog.objects, 0)} catalog objects · benchmark object ${proof.catalog.benchmark_object_excluded === true ? "excluded" : "exclusion not verified"}. Physics loss measures equation residuals; it is not position error against ESA.`));
      if (id === "COMBINED") card.append(node("p", "", `${format(proof.training_samples, 0)} earlier ESA calibration samples · ${format(proof.validation_samples, 0)} validation samples · ${format(proof.test_samples_received, 0)} test samples received during training. This network is calibrated to the saved Sentinel-1D arc.`));
      const links = node("div", "model-download-row");
      links.append(download("Trained weights ↓", `${MODELS[id].folder}/model.pt`), download("Training CSV ↓", `${MODELS[id].folder}/training_history.csv`));
      card.append(links);
      card.append(disclosure("Saved training steps · selected checkpoint highlighted", trainingHistory(id, proof)));
      card.append(disclosure("Complete training proof", node("pre", "model-json", json(proof))));
      grid.append(card);
    });
    area.append(grid);
  }
  function csvRows(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (ch === '"') {
        if (quoted && text[i + 1] === '"') { field += '"'; i++; }
        else quoted = !quoted;
      } else if (ch === "," && !quoted) { row.push(field); field = ""; }
      else if ((ch === "\n" || ch === "\r") && !quoted) {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); if (row.some(value => value !== "")) rows.push(row); row = []; field = "";
      } else field += ch;
    }
    row.push(field); if (row.some(value => value !== "")) rows.push(row);
    return rows;
  }
  function renderPredictions(ids) {
    const area = $("modelPredictionArea"); area.replaceChildren();
    const holder = node("div");
    const description = node("p", "model-metric-note", "Saved position and velocity states in TEME. This table includes the entire arc, including the shared initial sample; changing the evaluation interval only changes the metrics above. Values retain the CSV units: km and km/s.");
    holder.append(description);
    ids.forEach(id => {
      const folder = MODELS[id].folder;
      const urls = node("div", "model-download-row");
      urls.append(download(`${MODELS[id].name} predictions CSV ↓`, `${folder}/predictions.csv`), download("Errors by time CSV ↓", `${folder}/errors_by_time.csv`), download("Model report ↓", `${folder}/report.md`));
      holder.append(urls);
      const body = node("div", "model-prediction-table-body");
      const details = disclosure(`${MODELS[id].name} · view all saved position & velocity states`, body);
      let loaded = false;
      let pending = false;
      const runId = data.run_id;
      details.addEventListener("toggle", async () => {
        if (!details.open || loaded || pending) return;
        pending = true;
        body.replaceChildren(node("p", "model-no-data", "Loading saved trajectory…"));
        try {
          const url = artifactURL(`${folder}/predictions.csv`);
          if (!url) throw new Error("Saved trajectory is unavailable.");
          const response = await fetch(url, {cache: "no-store", credentials: "same-origin"});
          if (!response.ok) throw new Error(`Saved trajectory could not be read (HTTP ${response.status}).`);
          const content = await response.text();
          if (content.length > 5000000) throw new Error("This trajectory is too large to preview. Use the CSV download.");
          if (data.run_id !== runId || !body.isConnected) return;
          const [columns, ...rows] = csvRows(content);
          if (!columns || !rows.length) throw new Error("This saved trajectory has no rows.");
          body.replaceChildren(table(`${format(rows.length, 0)} saved states · exact CSV values`, columns.map(key => key.replaceAll("_", " ")), rows));
          loaded = true;
        } catch (error) {
          if (body.isConnected) body.replaceChildren(node("p", "model-no-data", `${error.message} Close and reopen to retry.`));
        } finally { pending = false; }
      });
      holder.append(details);
    });
    area.append(disclosure("Predicted trajectories & model reports", holder));
  }
  function renderFigures() {
    const gallery = $("modelFigureGallery"); gallery.replaceChildren();
    const folder = selected === "comparison" ? "comparison" : MODELS[selected].folder;
    const figures = (Array.isArray(data.figures) ? data.figures : []).filter(path => typeof path === "string" && path.startsWith(`${folder}/figures/`) && path.endsWith(".png"));
    $("modelFigureCount").textContent = `${figures.length} graphs`;
    $("modelFigureNote").textContent = "Saved figures retain their original full-window or test-interval labels when the metric interval changes. Open any graph at full size; download PNG or SVG.";
    if (!figures.length) gallery.append(node("p", "model-no-data", "No saved figures are available for this view."));
    for (const path of figures) {
      const url = artifactURL(path); if (!url) continue;
      const key = path.split("/").pop().replace(/\.png$/, "");
      const title = FIGURE_NAMES[key] || key.replace(/^\d+_/, "").replaceAll("_", " ");
      const figure = node("figure", "model-figure");
      const link = node("a"); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; link.setAttribute("aria-label", `Open ${title} at full size`);
      const image = node("img"); image.src = url; image.alt = `${MODELS[selected]?.name || "Model comparison"}: ${title}`; image.loading = "lazy"; image.decoding = "async";
      image.addEventListener("error", () => { image.hidden = true; link.append(node("p", "model-no-data", "Graph preview unavailable. Use the download links below.")); }, {once: true});
      link.append(image);
      const caption = node("figcaption");
      const links = node("div", "model-download-row");
      links.append(download("PNG ↓", path, ""), download("SVG ↓", path.replace(/\.png$/, ".svg"), ""));
      caption.append(node("h4", "", title), links); figure.append(link, caption); gallery.append(figure);
    }
  }
  function renderVerification(ids) {
    const area = $("modelVerificationContent"); area.replaceChildren();
    const hardware = data.hardware || {};
    area.append(facts([
      ["Training device", hardware.selected_device || "Unavailable"],
      ["CPU", hardware.cpu || "Unavailable"],
      ["System memory", `${format(hardware.total_memory_gib)} GiB`],
      ["PyTorch", hardware.torch || "Unavailable"],
      ["CPU threads used", format(hardware.torch_threads, 0)],
      ["Saved prediction replay", data.replay_verification?.passed === true ? "Passed" : data.replay_verification?.passed === false ? "Failed" : "Unavailable"]
    ], "model-verification-facts"));
    area.append(table("Saved prediction timing", ["Model", "Time (ms)"], ids.map(id => [MODELS[id].name, format(meters(data.prediction_timing_seconds?.[id]), 4)])));
    area.append(node("p", "model-metric-note", data.timing_note || "Median of warmed prediction measurements. Training and file I/O are excluded; combined timing includes SGP4."));
    const sections = [
      ["Hardware configuration", hardware],
      ["Data split & information available to each model", data.split],
      ["Evaluation checks", data.verification],
      ["Checkpoint replay verification", data.replay_verification],
      ["Physics diagnostics", data.physics_diagnostics]
    ];
    for (const [title, value] of sections) area.append(disclosure(title, node("pre", "model-json", json(value))));
  }
  function renderFiles() {
    if (!data) return;
    const files = (Array.isArray(data.files) ? data.files : []).filter(item => item && typeof item.path === "string" && safeURL(item.url));
    $("modelFileCount").textContent = `${format(files.length, 0)} FILES`;
    const query = $("modelFileSearch").value.toLowerCase().trim();
    const shown = files.filter(item => item.path.toLowerCase().includes(query));
    const area = $("modelFileList"); area.replaceChildren();
    for (const file of shown) {
      const row = node("div", "model-file-row");
      const link = node("a", "", file.path); link.href = safeURL(file.url); link.download = file.path.split("/").pop();
      const size = number(file.bytes);
      const label = size === null ? "" : size >= 1048576 ? `${format(size / 1048576)} MB` : size >= 1024 ? `${format(size / 1024, 1)} KB` : `${size} B`;
      row.append(link, node("span", "", label)); area.append(row);
    }
    if (!shown.length) area.append(node("p", "model-files-empty", query ? "No files match your search." : "No individual artifacts are available."));
  }
  function renderView() {
    if (!data) return;
    intervalText();
    const ids = activeModels();
    const metrics = Object.fromEntries(ids.map(id => [id, metric(id)]));
    $("modelViewTitle").textContent = selected === "comparison" ? `${SPLITS[split]} interval · model comparison` : `${MODELS[selected].name} · ${SPLITS[split].toLowerCase()} evaluation`;
    $("modelViewDescription").textContent = selected === "comparison" ? "Same initial orbit, timestamps, and ESA reference. The combined model uses earlier ESA calibration; standalone PINN uses physics without ESA labels. This is a retrospective benchmark on one satellite." : MODELS[selected].description;
    const counts = ids.map(id => number(metrics[id].future_samples));
    $("modelSampleCount").textContent = counts.every(count => count !== null && count === counts[0]) ? `${format(counts[0], 0)} ${ids.length === 1 ? "samples" : "samples per model"}` : "See sample counts in full metrics";
    const cards = $("modelMetricCards"); cards.replaceChildren(); cards.dataset.kind = selected === "comparison" ? "comparison" : "single";
    if (selected === "comparison") {
      ids.forEach(id => addMetricCard(cards, MODELS[id].name, meters(metric(id).position_vector_rmse_km), "m", MODELS[id].color, `${SPLITS[split]} position RMSE`, {label: "Velocity RMSE", value: `${format(meters(metric(id).velocity_vector_rmse_km_s), 4)} m/s`}));
    } else {
      const m = metric(selected), color = MODELS[selected].color;
      addMetricCard(cards, "Position RMSE", meters(m.position_vector_rmse_km), "m", color, "Root mean square vector error");
      addMetricCard(cards, "Position MAE", meters(m.position_vector_mae_km), "m", color, "Mean vector error");
      addMetricCard(cards, "Velocity RMSE", meters(m.velocity_vector_rmse_km_s), "m/s", color, "Root mean square vector error");
      addMetricCard(cards, "Velocity MAE", meters(m.velocity_vector_mae_km_s), "m/s", color, "Mean vector error");
    }
    const insight = $("modelComparisonInsight"); insight.hidden = true;
    if (selected === "comparison") {
      const baseline = number(metric("SGP4").position_vector_rmse_km), corrected = number(metric("COMBINED").position_vector_rmse_km);
      if (baseline !== null && baseline > 0 && corrected !== null) {
        const change = 100 * (baseline - corrected) / baseline;
        insight.textContent = `Combined position RMSE is ${format(Math.abs(change), 1)}% ${change >= 0 ? "lower" : "higher"} than SGP4 on this ${SPLITS[split].toLowerCase()} interval. The combined network had earlier ESA calibration data; this result does not establish performance across other satellites.`;
        insight.hidden = false;
      }
    }
    renderMetricTables(ids);
    $("modelFullMetrics").textContent = json({interval: split, values_in_saved_units: metrics});
    const metricPath = selected === "comparison" ? "comparison/all_metrics.csv" : `${MODELS[selected].folder}/metrics.csv`;
    setLink("modelMetricsDownload", artifactURL(metricPath), metricPath.split("/").pop());
    renderPredictions(ids); renderFigures(); renderVerification(ids);
    const unavailable = data.unavailable_collision_metrics || {};
    $("modelCollisionDetails").textContent = json(selected === "comparison" ? unavailable : unavailable[selected] || unavailable[MODELS[selected].folder] || unavailable);
  }
  function selectTab(id, updateHash = true, focus = false) {
    if (id !== "comparison" && !MODEL_IDS.includes(id)) return;
    selected = id;
    for (const tab of $("modelResultsTabs").querySelectorAll('[role="tab"]')) {
      const active = tab.dataset.model === id;
      tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
      if (active) { $("modelResultsPanel").setAttribute("aria-labelledby", tab.id); if (focus) tab.focus(); }
    }
    if (updateHash) window.history.replaceState(null, "", id === "comparison" ? "#models" : `#models-${id.toLowerCase()}`);
    renderView();
  }
  function readHash() {
    const match = window.location.hash.match(/^#models(?:-(sgp4|pinn|combined))?$/i);
    if (match) {
      const next = match[1] ? match[1].toUpperCase() : "comparison";
      if (next !== selected) selectTab(next, false);
    }
  }
  $("modelResultsTabs").addEventListener("click", event => {
    const tab = event.target.closest('[role="tab"]'); if (tab) selectTab(tab.dataset.model);
  });
  $("modelResultsTabs").addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const tabs = Array.from($("modelResultsTabs").querySelectorAll('[role="tab"]'));
    const index = tabs.indexOf(event.target); if (index < 0) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[next].dataset.model, true, true);
  });
  $("modelSplitSelect").addEventListener("change", event => { if (SPLITS[event.target.value]) { split = event.target.value; renderView(); } });
  $("modelRunSelect").addEventListener("change", event => loadRun(event.target.value));
  $("modelResultsRefresh").addEventListener("click", () => loadIndex());
  $("modelFileSearch").addEventListener("input", renderFiles);
  window.addEventListener("hashchange", readHash);
  readHash();
  loadIndex();
})();
