"use strict";

// Object notes and conversations are separate from the five-second position loop.
(() => {
  const el = id => document.getElementById(id);
  const conversations = new Map();
  let selectedObject = null;
  let objectInfo = null;
  let infoRequest = null;
  let chatRequest = null;
  let connection = {configured:false, provider:"gemini", model:"gemini-3.1-flash-lite"};
  let statusLoaded = false;
  let serverCompatible = false;
  let settingsSaving = false;
  const gemini = {
    model:"gemini-3.1-flash-lite",
    description:"Your Gemini API key stays encrypted on this computer. Questions and selected object context are sent to Google. Free-tier content may be used to improve Google products. Availability, limits, and charges depend on your API project and model.",
    links:[{title:"Get a Gemini API key ↗", url:"https://aistudio.google.com/apikey"}, {title:"Pricing & data use ↗", url:"https://ai.google.dev/gemini-api/docs/pricing"}],
    accountLinks:[{title:"Gemini API projects ↗", url:"https://aistudio.google.com/projects"}, {title:"Gemini API limits ↗", url:"https://ai.google.dev/gemini-api/docs/rate-limits"}],
    accountNote:"Check the Google AI Studio project linked to your API key and its billing tier."
  };
  const restartNotice = "Restart the dashboard server to load Gemini-only chat, then refresh this page.";
  const geminiStatus = () => connection.providers?.gemini || connection;

  const formatNumber = (value, digits = 3) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits:digits});
  const formatTime = value => {
    if (!value) return "Not available";
    const parsed = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : value + "Z");
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toISOString().replace("T", " ").slice(0,19) + " UTC";
  };
  const readable = value => value == null ? "Not available" : typeof value === "object" ? JSON.stringify(value) : String(value);
  const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  };
  function sourceLink(source, label) {
    if (!source || !source.url) return null;
    let url;
    try { url = new URL(source.url); } catch { return null; }
    if (!["https:", "http:"].includes(url.protocol)) return null;
    const link = node("a", "object-source-link", label || source.title || url.hostname);
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = source.title || url.hostname;
    return link;
  }
  function appendSources(container, sources) {
    const seen = new Set();
    for (const source of sources || []) {
      const link = sourceLink(source);
      if (!link || seen.has(link.href)) continue;
      seen.add(link.href);
      container.append(link);
    }
  }
  function appendProviderLinks(container, sources) {
    for (const source of sources) {
      if (container.childElementCount) container.append(document.createTextNode(" · "));
      const link = sourceLink(source);
      if (link) container.append(link);
    }
  }
  async function request(path, options = {}) {
    const response = await fetch(path, {cache:"no-store", ...options});
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The dashboard returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  const post = (body, signal) => ({method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body), signal});
  function conversation(id = selectedObject) {
    if (!conversations.has(id)) conversations.set(id, {messages:[], error:""});
    return conversations.get(id);
  }
  function setNotice(text) {
    el("objectInfoNotice").textContent = text;
    el("objectInfoNotice").hidden = !text;
  }
  function renderInfo(data) {
    objectInfo = data;
    el("objectStoryTitle").textContent = data.name || `Object ${data.id}`;
    el("objectIdentityMeta").textContent = `NORAD ${data.id} · ${data.international_designator || "International designator unavailable"}`;
    el("objectCategory").textContent = data.category || "Catalog object";
    el("objectCategory").hidden = false;
    el("chatObjectContext").textContent = `${data.name || "Selected object"} · #${data.id}`;
    el("objectHistorySummary").textContent = data.history_summary || "No documented object history is available in the connected sources.";
    el("objectHistoryScope").textContent = data.history_scope || "";
    const sources = new Map((data.sources || []).map(source => [String(source.id), source]));
    const timeline = el("objectTimeline");
    timeline.replaceChildren();
    for (const event of data.history_events || []) {
      const item = node("li", "object-timeline-event");
      item.append(node("span", "timeline-date", readable(event.date)), node("h4", "", event.title || "Documented event"), node("p", "", event.description || ""));
      const links = node("div", "timeline-sources");
      for (const id of event.source_ids || []) {
        const source = sources.get(String(id));
        const link = sourceLink(source, source?.title || `Source ${id}`);
        if (link) links.append(link);
      }
      if (links.childElementCount) item.append(links);
      timeline.append(item);
    }
    timeline.hidden = !timeline.childElementCount;
    const facts = el("objectOrbitFacts");
    facts.replaceChildren();
    const factRows = [...(data.facts || [])];
    if (data.current_state?.altitude_km != null) factRows.push({label:"Geocentric altitude", value:formatNumber(data.current_state.altitude_km,1), unit:"km"});
    if (data.current_state?.speed_km_s != null) factRows.push({label:"Speed", value:formatNumber(data.current_state.speed_km_s,4), unit:"km/s"});
    factRows.push({label:"Element epoch", value:formatTime(data.epoch_utc)}, {label:"Element age", value:formatNumber(data.element_age_hours,1), unit:"hours"});
    for (const fact of factRows) {
      const row = node("div");
      const value = typeof fact.value === "number" ? formatNumber(fact.value,8) : readable(fact.value);
      row.append(node("dt", "", fact.label || "Parameter"), node("dd", "", `${value}${fact.unit ? " " + fact.unit : ""}`));
      facts.append(row);
    }
    const current = data.current_state;
    el("objectStateTime").textContent = current?.sgp4_error ? `SGP4 state unavailable (code ${current.sgp4_error}).` : `State snapshot: ${formatTime(current?.time_utc)} · ${current?.frame || "TEME"}. Refresh info for a new snapshot.`;
    el("objectCatalogNote").textContent = data.catalog_history_note || "These records are the catalog snapshots saved in this project, not the object's full orbital history.";
    const catalog = el("objectCatalogHistory");
    catalog.replaceChildren();
    for (const record of (data.catalog_history || []).slice(0,8)) {
      const row = node("div", "catalog-history-row");
      row.append(node("strong", "", formatTime(record.epoch_utc)), node("span", "", `${formatNumber(record.mean_motion_rev_day,6)} rev/day · ${formatNumber(record.inclination_deg,4)}° inclination`));
      if (record.fetched_at_utc) row.append(node("span", "", `Downloaded ${formatTime(record.fetched_at_utc)}`));
      catalog.append(row);
    }
    if (!catalog.childElementCount) catalog.append(node("p", "object-meta", "No saved catalog epochs for this object."));
    const encounterArea = el("objectEncounterContext");
    encounterArea.replaceChildren();
    const encounters = data.encounters;
    if (!encounters?.run_id) encounterArea.append(node("p", "object-meta", "No completed screening is available. Run a screening to check the selected catalog and time window."));
    else {
      const countSummary = encounters.screened === false ? "This object was not included in the latest screening." : `${encounters.total ?? 0} close-approach events in the latest saved run.`;
      encounterArea.append(node("p", "object-meta", `${countSummary} ${formatTime(encounters.window_start_utc)} → ${formatTime(encounters.window_end_utc)}`));
      if (encounters.threshold_km != null) encounterArea.append(node("p", "object-meta", `Screening threshold: ${formatNumber(encounters.threshold_km)} km.`));
      for (const event of (encounters.events || []).slice(0,3)) {
        const other = event.other_name || `Object #${event.other_id}`;
        const row = node("p", "object-encounter-item");
        row.append(node("strong", "", other), node("span", "", `${formatNumber(event.miss_distance_km,4)} km miss distance · ${formatTime(event.tca_utc)}`));
        encounterArea.append(row);
      }
      encounterArea.append(node("p", "object-meta", encounters.note || "Screening counts are specific to that run. Close-approach distance alone does not determine collision probability."));
    }
    el("objectSources").replaceChildren();
    appendSources(el("objectSources"), data.sources);
    if (!el("objectSources").childElementCount) el("objectSources").append(node("p", "object-meta", "No linked sources available."));
    el("objectInfoContent").hidden = false;
    setNotice("");
  }
  async function loadObjectInfo(id) {
    infoRequest?.abort();
    const controller = new AbortController();
    infoRequest = controller;
    el("refreshObjectInfo").disabled = true;
    setNotice("Loading documented history and orbital context…");
    try {
      const data = await request(`/api/objects/${id}/info`, {signal:controller.signal});
      if (controller !== infoRequest || selectedObject !== id) return;
      if (Number(data.id) !== id) throw new Error("The server returned information for a different object. Please refresh.");
      renderInfo(data);
    } catch (error) {
      if (controller !== infoRequest || error.name === "AbortError") return;
      setNotice(`Object information unavailable: ${error.message}`);
    } finally {
      if (controller === infoRequest) el("refreshObjectInfo").disabled = false;
    }
  }
  function cancelChat(message) {
    if (!chatRequest) return;
    const active = chatRequest;
    chatRequest = null;
    active.controller.abort();
    active.userMessage.status = "failed";
    conversation(active.id).error = message;
  }
  function select(id) {
    id = Number(id);
    if (!Number.isSafeInteger(id) || id <= 0) return;
    if (id !== selectedObject) {
      cancelChat("The request stopped when you switched objects. Send your question again to retry.");
      selectedObject = id;
      objectInfo = null;
      el("objectInfoContent").hidden = true;
      el("objectCategory").hidden = true;
      el("objectStoryTitle").textContent = `Object #${id}`;
      el("objectIdentityMeta").textContent = "Looking up this catalog object…";
      el("chatObjectContext").textContent = `Selected object · #${id}`;
      el("objectChatInput").value = "";
    }
    renderChat();
    loadObjectInfo(id);
  }
  function updateConnection() {
    const configured = serverCompatible && connection.configured;
    el("chatConnectionState").textContent = !statusLoaded ? "Checking Gemini configuration" : !serverCompatible ? connection.message || restartNotice : configured ? `Gemini configured · ${connection.model}` : "Gemini not connected";
    el("chatConnectionState").classList.toggle("is-connected", configured);
    el("chatConnectionState").title = connection.message || "";
    el("chatSettingsButton").textContent = configured ? "Gemini settings" : "Connect Gemini";
    el("sendObjectChat").disabled = !configured || !selectedObject || Boolean(chatRequest) || settingsSaving;
    el("objectChatInput").disabled = !serverCompatible || !selectedObject || Boolean(chatRequest) || settingsSaving;
    el("sendObjectChat").innerText = chatRequest ? "Waiting…" : "Send ↑";
    el("chatPrivacyNote").textContent = "Questions, chat history, and selected object data are sent to Google through Gemini. Responses may be incomplete; check the linked sources.";
    updateSettings();
  }
  function updateSettings(resetModel = false) {
    const profile = geminiStatus();
    if (resetModel) el("chatModelName").value = profile.model || gemini.model;
    el("chatSettingsTitle").textContent = "Connect to Gemini";
    el("chatSettingsDescription").textContent = gemini.description;
    el("chatApiKeyLabel").textContent = "Gemini API key";
    el("chatApiKey").required = !profile.configured;
    el("chatApiKey").placeholder = profile.configured ? "Leave blank to keep the Gemini key" : "Paste your Gemini API key";
    el("chatKeyHint").textContent = profile.key_source === "environment"
      ? "Using the Gemini key supplied to the server environment. Saving a key creates an encrypted local setting."
      : profile.configured ? "Your Gemini key is hidden. Enter a new key only to replace it."
      : "The key is never included in the conversation.";
    el("chatModelHint").textContent = "Use a model available to your Gemini API account.";
    el("chatProviderLinks").replaceChildren();
    appendProviderLinks(el("chatProviderLinks"), gemini.links);
    el("chatProviderNotice").hidden = serverCompatible;
    el("chatProviderNotice").textContent = serverCompatible ? "" : statusLoaded ? connection.message || restartNotice : "Checking that the dashboard server supports Gemini-only chat…";
    el("disconnectChat").hidden = profile.key_source !== "saved";
    el("disconnectChat").textContent = "Remove saved Gemini key";
    el("disconnectChat").disabled = settingsSaving || !serverCompatible;
    el("saveChatSettings").disabled = settingsSaving || !serverCompatible;
    for (const id of ["chatApiKey", "chatModelName"]) el(id).disabled = settingsSaving || !serverCompatible;
  }
  function applyConnection(updated) {
    serverCompatible = updated.provider === "gemini" && Array.isArray(updated.supported_providers)
      && updated.supported_providers.length === 1 && updated.supported_providers[0] === "gemini";
    if (!serverCompatible) cancelChat(restartNotice);
    connection = serverCompatible ? updated : {configured:false, provider:"gemini", model:gemini.model, message:restartNotice};
    if (!el("chatSettingsDialog").open) {
      updateSettings(true);
    }
    statusLoaded = true;
    renderChat();
  }
  async function loadConnection() {
    try {
      applyConnection(await request("/api/chat/status"));
    } catch (error) {
      serverCompatible = false;
      connection = {...connection, configured:false, message:error.message};
    }
    statusLoaded = true;
    renderChat();
  }
  function renderChat() {
    const log = el("objectChatMessages");
    const thread = conversation();
    log.replaceChildren();
    if (!thread.messages.length) {
      const welcome = node("div", "chat-welcome");
      welcome.append(node("span", "chat-welcome-icon", "✧"), node("h4", "", "Ask about this object"), node("p", "", connection.configured ? "Ask about the selected object, explore its documented history, or get an explanation of orbital mechanics." : "Connect your Gemini API account to ask questions. The object history and source links are available alongside the chat."));
      if (!connection.configured && statusLoaded) {
        const button = node("button", "chat-connect-prompt", "Connect Gemini →");
        button.type = "button";
        button.addEventListener("click", openSettings);
        welcome.append(button);
      }
      log.append(welcome);
    }
    for (const message of thread.messages) {
      const entry = node("article", `chat-message chat-message-${message.role}`);
      entry.append(node("span", "chat-message-author", message.role === "user" ? "You" : "Orbital Watch · Gemini"), node("div", "chat-message-text", message.content));
      if (message.sources?.length) {
        const sources = node("div", "chat-message-sources");
        sources.append(node("span", "", "Sources supplied with this answer"));
        appendSources(sources, message.sources);
        entry.append(sources);
      }
      if (message.status === "failed") entry.append(node("span", "chat-message-status", "No answer received"));
      if (message.incomplete) entry.append(node("span", "chat-message-status", "The model reached its answer limit. Ask a follow-up to continue."));
      log.append(entry);
    }
    if (chatRequest?.id === selectedObject) log.append(node("p", "chat-waiting", "Gemini is responding…"));
    el("chatRequestError").textContent = thread.error;
    el("chatRequestError").hidden = !thread.error;
    const accountLinks = el("chatAccountLinks");
    accountLinks.hidden = !/Gemini.*(?:credit|quota|billing|spend limit|usage limit|rate.limit|exhausted)/i.test(thread.error);
    accountLinks.replaceChildren();
    if (!accountLinks.hidden) {
      appendProviderLinks(accountLinks, gemini.accountLinks);
      accountLinks.append(node("br"), node("span", "", gemini.accountNote));
    }
    el("chatSuggestions").hidden = thread.messages.length > 0;
    el("clearObjectChat").disabled = !thread.messages.length;
    updateConnection();
    log.scrollTop = log.scrollHeight;
  }
  async function sendMessage(event) {
    event.preventDefault();
    const message = el("objectChatInput").value.trim();
    if (!message || !selectedObject || chatRequest || settingsSaving || !serverCompatible) return;
    if (!connection.configured) { openSettings(); return; }
    const id = selectedObject;
    const thread = conversation(id);
    const history = [];
    let remainingHistory = 20000;
    for (const item of thread.messages.filter(item => item.status !== "failed").slice(-8).reverse()) {
      if (remainingHistory <= 0) break;
      const content = item.content.slice(0, Math.min(8000, remainingHistory));
      history.unshift({role:item.role, content});
      remainingHistory -= content.length;
    }
    const userMessage = {role:"user", content:message, status:"pending"};
    thread.messages.push(userMessage);
    thread.messages = thread.messages.slice(-8);
    thread.error = "";
    const active = {id, userMessage, controller:new AbortController()};
    chatRequest = active;
    el("objectChatInput").value = "";
    renderChat();
    try {
      const payload = {object_id:id, message, history, provider:"gemini"};
      const data = await request("/api/chat", post(payload, active.controller.signal));
      if (chatRequest !== active || selectedObject !== id) return;
      if (Number(data.object_id) !== id) throw new Error("The answer belongs to a different object. Please try again.");
      if (data.provider !== "gemini") throw new Error("The dashboard returned an unexpected chat configuration. Refresh the page and check Gemini settings.");
      if (typeof data.answer !== "string" || !data.answer.trim()) throw new Error("The model returned no answer. Please try again.");
      userMessage.status = "complete";
      thread.messages.push({role:"assistant", content:data.answer, sources:data.sources || [], incomplete:Boolean(data.incomplete), status:"complete"});
      thread.messages = thread.messages.slice(-8);
    } catch (error) {
      if (chatRequest !== active || error.name === "AbortError") return;
      userMessage.status = "failed";
      thread.error = error.message;
      el("objectChatInput").value = message;
    } finally {
      if (chatRequest === active) {
        chatRequest = null;
        renderChat();
        el("objectChatInput").focus({preventScroll:true});
      }
    }
  }
  function openSettings() {
    if (settingsSaving) return;
    el("chatSettingsMessage").hidden = true;
    el("chatSettingsMessage").textContent = "";
    el("chatApiKey").value = "";
    updateSettings(true);
    updateConnection();
    el("chatSettingsDialog").showModal();
    (serverCompatible ? connection.configured ? el("chatModelName") : el("chatApiKey") : el("closeChatSettings")).focus();
  }
  async function saveSettings(event) {
    event.preventDefault();
    if (settingsSaving) return;
    if (!serverCompatible) { updateSettings(); return; }
    const apiKey = el("chatApiKey").value.trim();
    const model = el("chatModelName").value.trim();
    if (!model || (!apiKey && !geminiStatus().configured)) return;
    cancelChat("The request stopped while the connection settings changed. Send your question again to retry.");
    settingsSaving = true;
    renderChat();
    el("chatSettingsMessage").textContent = "Saving the encrypted local connection…";
    el("chatSettingsMessage").hidden = false;
    try {
      const config = {provider:"gemini", model};
      if (apiKey) config.api_key = apiKey;
      applyConnection(await request("/api/chat/config", post(config)));
      el("chatApiKey").value = "";
      el("chatSettingsDialog").close();
    } catch (error) {
      el("chatSettingsMessage").textContent = error.message;
    } finally {
      el("chatApiKey").value = "";
      settingsSaving = false;
      renderChat();
    }
  }
  async function removeKey() {
    if (settingsSaving) return;
    if (!serverCompatible) { updateSettings(); return; }
    cancelChat("The request stopped while the connection settings changed. Send your question again to retry.");
    settingsSaving = true;
    renderChat();
    el("chatSettingsMessage").hidden = false;
    el("chatSettingsMessage").textContent = "Removing the saved connection…";
    try {
      applyConnection(await request("/api/chat/config", post({clear:true, provider:"gemini"})));
      el("chatSettingsMessage").textContent = geminiStatus().configured
        ? "Saved Gemini key removed. The server still has a Gemini key in its environment."
        : "Saved Gemini key removed. Connect again whenever you need Gemini.";
    } catch (error) { el("chatSettingsMessage").textContent = error.message; }
    finally {
      el("chatApiKey").value = "";
      settingsSaving = false;
      renderChat();
    }
  }

  window.addEventListener("object-selected", event => select(event.detail?.id));
  el("refreshObjectInfo").addEventListener("click", () => { if (selectedObject) loadObjectInfo(selectedObject); });
  el("objectChatForm").addEventListener("submit", sendMessage);
  el("objectChatInput").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (el("objectChatInput").value.trim() && !chatRequest) el("objectChatForm").requestSubmit();
    }
  });
  el("chatSuggestions").addEventListener("click", event => {
    const button = event.target.closest("[data-question]");
    if (!button) return;
    el("objectChatInput").value = button.dataset.question;
    el("objectChatInput").focus({preventScroll:true});
  });
  el("clearObjectChat").addEventListener("click", () => {
    cancelChat("");
    conversations.set(selectedObject, {messages:[], error:""});
    renderChat();
  });
  el("chatSettingsButton").addEventListener("click", openSettings);
  el("closeChatSettings").addEventListener("click", () => el("chatSettingsDialog").close());
  el("chatSettingsDialog").addEventListener("close", () => { el("chatApiKey").value = ""; });
  el("chatSettingsForm").addEventListener("submit", saveSettings);
  el("disconnectChat").addEventListener("click", removeKey);
  renderChat();
  loadConnection();
  if (window.orbitalSelectedObjectId) select(window.orbitalSelectedObjectId);
})();
