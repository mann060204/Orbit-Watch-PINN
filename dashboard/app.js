"use strict";

const $ = id => document.getElementById(id);
const number = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const escapeHTML = value => String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const dateOf = value => new Date(/[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : value + "Z");
const utc = value => value ? dateOf(value).toISOString().replace("T", " ").slice(0, 19) + " UTC" : "—";
const percent = value => value == null ? "—" : number(value * 100, 2) + "%";
const distance = value => value == null ? "—" : value < 1 ? number(value * 1000, 2) + " m" : number(value, 3) + " km";
let state = null, objects = new Map(), events = [], visibleEvents = 60, activeEvent = null;
let catalogVersion = null, runVersion = null, selectedId = 25544, orbitPath = [];
let liveStates = [], liveTime = null, receivedAt = 0, paused = false, liveError = null;

async function api(path, body) {
  const options = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  const response = await fetch(path, {...options, signal:AbortSignal.timeout(60000)});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
function showError(message) {
  $("errorBanner").textContent = message || "";
  $("errorBanner").hidden = !message;
}
function bindDownload(id, filename) {
  const element = $(id);
  element.classList.toggle("disabled", !runVersion);
  element.href = runVersion ? `/api/download/${runVersion}/${filename}` : "#";
}
async function loadCatalog() {
  const payload = await api("/api/catalog");
  objects = new Map(payload.objects.map(object => [object.id, object]));
  $("objectOptions").innerHTML = payload.objects.map(object => `<option value="${object.id} · ${escapeHTML(object.name)}"></option>`).join("");
  if (!objects.has(selectedId)) selectedId = payload.objects[0]?.id;
  if (selectedId) await selectObject(selectedId);
}
async function selectObject(id) {
  const object = objects.get(Number(id));
  if (!object) return;
  selectedId = object.id;
  window.orbitalSelectedObjectId = object.id;
  window.dispatchEvent(new CustomEvent("object-selected", {detail:{id:object.id}}));
  $("objectSearch").value = `${object.id} · ${object.name}`;
  orbitPath = [];
  try {
    const result = await api(`/api/orbit/${object.id}`);
    if (selectedId === object.id) orbitPath = result.positions;
  } catch (error) { showError(error.message); }
  updateObjectDetails();
}
function updateObjectDetails() {
  const object = objects.get(selectedId);
  if (!object) return;
  const row = liveStates.find(item => item[0] === selectedId);
  const age = (Date.now() - dateOf(object.epoch).getTime()) / 3600000;
  let detail = "State unavailable";
  if (row && row[7] === 0 && row[1] !== null) {
    const altitude = Math.hypot(...row.slice(1,4)) - 6378.137;
    const speed = Math.hypot(...row.slice(4,7));
    detail = `${number(altitude, 1)} km geocentric altitude · ${number(speed, 3)} km/s`;
  }
  $("objectDetails").innerHTML = `<b>${escapeHTML(object.name)} <span class="secondary">#${object.id}</span></b>${detail}<br>Epoch ${escapeHTML(utc(object.epoch))}<br><span class="${Math.abs(age)>72?"amber":""}">${number(age, 1)} h element age</span>`;
}
function updateSummary() {
  const summary = state.summary;
  const busy = state.job.running;
  $("runButton").disabled = busy;
  $("refreshButton").disabled = busy;
  $("jobProgress").value = state.job.progress;
  $("jobMessage").textContent = state.job.message;
  $("autoRefresh").checked = state.auto_refresh;
  $("catalogCount").textContent = number(state.data.unique_objects);
  $("catalogSub").textContent = `${state.data.sources.length} CelesTrak groups · latest published elements`;
  const groupNames = {stations:"Space stations", "cosmos-2251-debris":"COSMOS 2251 group", "iridium-33-debris":"IRIDIUM 33 group", "fengyun-1c-debris":"FENGYUN 1C group", active:"Active satellites"};
  $("sourceGroups").innerHTML = state.data.sources.map(source => `<div class="source-row"><span>${escapeHTML(groupNames[source.group] || source.group)}</span><strong>${number(source.record_count)}</strong></div>`).join("");
  const fetchTimes = state.data.sources.map(source => dateOf(source.fetched_at_utc).getTime());
  $("fetchTime").textContent = utc(new Date(Math.max(...fetchTimes)).toISOString());
  showError(state.job.error || liveError);
  if (!summary) return;
  $("screenedCount").textContent = number(summary.screening.screened_objects);
  $("screenedSub").textContent = `${number(summary.input_objects-summary.screening.screened_objects)} objects excluded · ${number(summary.screening.shared_orbit_pairs_excluded||0)} shared-orbit pairs separated`;
  $("eventCount").textContent = number(summary.screening.conjunction_events);
  $("eventSub").textContent = `Within ${summary.config.threshold_km} km · ${summary.config.horizon_hours} hour window`;
  $("closest").textContent = distance(summary.closest_approach_km);
  $("windowText").textContent = `${utc(summary.window_start_utc)} → ${utc(summary.window_end_utc)} · ${summary.config.step_seconds}s intervals, refined with SGP4`;
  const classification = summary.classification;
  const labeledPairs = classification.evaluated_pairs || 0;
  for (const [id, key, render] of [["accuracy", "accuracy", percent], ["precision", "precision", percent],
      ["recall", "recall", percent], ["f1", "f1_score", value => number(value, 3)]]) {
    const value = classification[key];
    $(id).textContent = value == null ? "N/A" : render(value);
    $(id).title = value == null ? (labeledPairs
      ? "These reference pairs do not provide the cases needed to calculate this metric."
      : "Independent reference outcomes are needed to calculate this metric.")
      : "Evaluated on the supplied labeled object pairs.";
  }
  $("labelStatus").textContent = labeledPairs ? "SUPPLIED REFERENCE LABELS" : "NOT EVALUATED";
  $("classificationScope").textContent = `${number(labeledPairs)} supplied reference pairs scored for this screening run.`;
  $("evaluationReason").textContent = labeledPairs ? classification.reason
    : `Accuracy, precision, recall, and F1 need independent yes/no outcomes showing which object pairs came within ${number(summary.config.threshold_km, 2)} km during this run's forecast window. No such labels have been supplied.`;
  $("propagationRate").textContent = percent(summary.propagation.success_rate);
  $("propagationRate").title = `${number(summary.propagation.valid_states)} of ${number(summary.propagation.attempted_states)} state calculations completed successfully.`;
  $("referenceStatus").textContent = summary.validation.passed ? "Passed" : "Failed";
  $("referenceStatus").title = `Position vector difference: ${summary.validation.position_vector_error_km} km. Implementation agreement, not observed accuracy.`;
  $("resultFolder").textContent = `results/${summary.run_id}/`;
  $("runtimeText").textContent = `${number(summary.propagation.attempted_states)} state samples · ${number(summary.timing_seconds.total, 1)} s run time · full TEME trajectories saved`;
  [["csvDownload","conjunctions.csv"],["zipDownload","all.zip"],["reportDownload","report.md"],["metricsDownload","metrics.csv"]].forEach(([id,file])=>bindDownload(id,file));
}
async function pollState() {
  try {
    state = await api("/api/state");
    if (catalogVersion !== state.data.generated_at_utc) {
      await loadCatalog();
      catalogVersion = state.data.generated_at_utc;
    }
    if (state.summary && runVersion !== state.summary.run_id) {
      const result = await api("/api/conjunctions");
      events = result.events;
      runVersion = result.run_id;
      visibleEvents = 60;
      activeEvent = null;
      renderEvents();
      if (events.length) selectEvent(events[0].event_id);
      else clearEventDetail();
    }
    updateSummary();
  } catch (error) {
    $("connection").textContent = "Server unavailable";
    showError(`Dashboard connection: ${error.message}. Check that run_dashboard.ps1 is running.`);
  } finally { setTimeout(pollState, 3000); }
}
async function pollLive() {
  try {
    if (!paused) {
      const result = await api("/api/live");
      liveStates = result.states;
      liveTime = result.time_utc;
      receivedAt = performance.now();
      $("liveTime").textContent = `${utc(liveTime)} · model positions, refreshed every 5 s`;
      $("liveValid").textContent = `${number(liveStates.filter(row=>row[7]===0&&row[1]!==null).length)} valid states`;
      $("connection").textContent = "Live propagation";
      updateObjectDetails();
      liveError = null;
    }
  } catch (error) {
    liveError = `Live positions unavailable: ${error.message}`;
    $("connection").textContent = "Live data interrupted";
    showError(liveError);
  } finally { setTimeout(pollLive, 5000); }
}
function renderEvents() {
  const query = $("eventSearch").value.trim().toLowerCase();
  const filtered = events.filter(event => `${event.object1_name} ${event.object2_name} ${event.object1_id} ${event.object2_id}`.toLowerCase().includes(query));
  $("tableCount").textContent = number(filtered.length);
  $("eventRows").innerHTML = filtered.slice(0,visibleEvents).map(event=>`<tr class="${activeEvent===event.event_id?"selected":""}"><td><button class="pair-button" data-event="${event.event_id}">${escapeHTML(event.object1_name)}<span>↔ ${escapeHTML(event.object2_name)}</span><span>#${event.object1_id} / #${event.object2_id}</span></button></td><td class="distance-cell">${distance(event.miss_distance_km)}</td><td>${escapeHTML(utc(event.tca_utc).slice(5,19))}</td><td>${number(event.relative_speed_km_s,3)} km/s</td><td class="${Math.max(event.object1_epoch_age_hours_at_tca,event.object2_epoch_age_hours_at_tca)>72?"amber":""}">${number(Math.max(event.object1_epoch_age_hours_at_tca,event.object2_epoch_age_hours_at_tca),1)} h max</td></tr>`).join("") || `<tr><td colspan="5" class="empty-cell">${runVersion ? "No matching close approaches inside this screening threshold." : "Run a screening to see close approaches."}</td></tr>`;
  $("showMore").hidden = filtered.length <= visibleEvents;
}
function clearEventDetail() {
  $("detailTitle").textContent = "No close approaches reported";
  $("detailFacts").textContent = "No pairs were flagged inside the chosen threshold. This does not certify collision-free space.";
  $("distanceChart").innerHTML = "";
}
function selectEvent(id) {
  const event = events.find(item=>item.event_id===id);
  if (!event) return;
  activeEvent = id;
  $("detailTitle").textContent = `${event.object1_name} ↔ ${event.object2_name}`;
  $("detailFacts").innerHTML = `<strong class="distance-big">${distance(event.miss_distance_km)}</strong><strong>${escapeHTML(utc(event.tca_utc))}</strong><br>${number(event.relative_speed_km_s,4)} km/s relative speed<br>${escapeHTML(event.event_id)} · ${event.at_window_boundary?"Window boundary minimum":"Refined closest approach"}<br>Collision probability: unavailable`;
  $("detailNote").textContent = event.refinement_failed ? "Some intervals could not be refined. Review refinement_errors.json before using this result." : "Small separations may include co-orbiting or docked objects. Element errors and object sizes are not represented in this curve.";
  drawDistanceChart(event.distance_curve);
  renderEvents();
}
function drawDistanceChart(curve) {
  const values = curve.filter(point=>point[1]!==null);
  if (values.length<2) { $("distanceChart").innerHTML=""; return; }
  const [left,right,top,bottom]=[42,317,15,145];
  const minX=Math.min(...values.map(p=>p[0])),maxX=Math.max(...values.map(p=>p[0]));
  const maxY=Math.max(...values.map(p=>p[1]),0.001)*1.08;
  const x=value=>left+(value-minX)/(maxX-minX||1)*(right-left);
  const y=value=>bottom-value/maxY*(bottom-top);
  let svg="";
  for(let i=0;i<=3;i++) { const yy=top+i*(bottom-top)/3; svg+=`<line x1="${left}" x2="${right}" y1="${yy}" y2="${yy}" stroke="#273544"/><text class="chart-label" x="${left-6}" y="${yy+4}" text-anchor="end">${number(maxY*(1-i/3),2)}</text>`; }
  const path=values.map((p,i)=>`${i?"L":"M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(" ");
  svg+=`<path d="${path}" stroke="#6ee7dc" stroke-width="2" fill="none"/><line x1="${x(0)}" x2="${x(0)}" y1="${top}" y2="${bottom}" stroke="#f2b467" stroke-dasharray="3 4"/><text class="chart-label" x="${left}" y="164">${number(minX)}s</text><text class="chart-label" x="${x(0)}" y="164" text-anchor="middle">TCA</text><text class="chart-label" x="${right}" y="164" text-anchor="end">+${number(maxX)}s</text><text class="chart-label" x="${left}" y="10">Separation (km)</text>`;
  $("distanceChart").innerHTML=svg;
}

$("analysisForm").addEventListener("submit",async event=>{
  event.preventDefault();
  const config={horizon_hours:Number($("horizon").value),step_seconds:Number($("step").value),threshold_km:Number($("threshold").value),max_epoch_age_hours:72,include_stale:$("includeStale").checked,acceleration_pad_km_s2:0.05};
  try { $("runButton").disabled=true; await api("/api/analyze",config); $("jobMessage").textContent="Starting screening…"; showError(null); }
  catch(error){showError(error.message);$("runButton").disabled=false;}
});
$("refreshButton").addEventListener("click",async()=>{try{$("refreshButton").disabled=true;await api("/api/refresh",{});}catch(error){showError(error.message);$("refreshButton").disabled=false;}});
$("autoRefresh").addEventListener("change",async()=>{try{await api("/api/auto-refresh",{enabled:$("autoRefresh").checked});}catch(error){showError(error.message);}});
$("eventSearch").addEventListener("input",()=>{visibleEvents=60;renderEvents();});
$("showMore").addEventListener("click",()=>{visibleEvents+=100;renderEvents();});
$("eventRows").addEventListener("click",event=>{const button=event.target.closest("[data-event]");if(button)selectEvent(button.dataset.event);});
$("objectSearch").addEventListener("change",()=>{const value=$("objectSearch").value.trim();const object=objects.get(Number(value.split(" · ")[0]))||[...objects.values()].find(item=>item.name.toLowerCase()===value.toLowerCase());if(object)selectObject(object.id);});
$("pauseButton").addEventListener("click",()=>{paused=!paused;$("pauseButton").textContent=paused?"Resume live view":"Pause view";$("connection").textContent=paused?"View paused":"Resuming live view";});
$("zipDownload").addEventListener("click",()=>{$("runtimeText").textContent="Preparing the complete archive; trajectory files can make this a large download.";});

// Orthographic TEME view. Canvas draws data and coordinate geometry, not a map.
const canvas=$("orbitCanvas"),ctx=canvas.getContext("2d");
let yaw=-0.5,pitch=0.45,zoom=1,drag=null,width=0,height=0,pickPoints=[];
const earth=6378.137;
function rotate([x,y,z]){const cx=Math.cos(yaw),sx=Math.sin(yaw),cy=Math.cos(pitch),sy=Math.sin(pitch);const xx=cx*x+sx*y,yy=-sx*x+cx*y;return[xx,cy*z-sy*yy,sy*z+cy*yy];}
function project(p){const [x,y,z]=rotate(p),scale=Math.min(width,height)*.285/earth*zoom;return{x:width/2+x*scale,y:height/2-y*scale,z,rx:x,ry:y,scale};}
function hiddenByEarth(p){return p.rx*p.rx+p.ry*p.ry<earth*earth && p.z<Math.sqrt(Math.max(0,earth*earth-p.rx*p.rx-p.ry*p.ry));}
function drawCurve(points,color,lineWidth=1,earthGrid=false){ctx.beginPath();let pen=false;for(const point of points){if(!point){pen=false;continue;}const p=project(point);if((earthGrid&&p.z<0)||(!earthGrid&&hiddenByEarth(p))){pen=false;continue;}if(pen)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);pen=true;}ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.stroke();}
function drawGlobe(){
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  if(rect.width!==width||rect.height!==height||canvas.width!==Math.round(rect.width*dpr)){width=rect.width;height=rect.height;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);}
  ctx.clearRect(0,0,width,height);if(!width||!height)return requestAnimationFrame(drawGlobe);
  const r=Math.min(width,height)*.285*zoom,cx=width/2,cy=height/2;
  ctx.strokeStyle="#213548";ctx.lineWidth=1;
  for(const factor of [1.38,1.75]){ctx.beginPath();ctx.ellipse(cx,cy,r*factor,r*factor,0,0,Math.PI*2);ctx.setLineDash([2,7]);ctx.stroke();ctx.setLineDash([]);}
  const glow=ctx.createRadialGradient(cx,cy,r*.95,cx,cy,r*1.1);glow.addColorStop(0,"#237a8533");glow.addColorStop(1,"#237a8500");ctx.fillStyle=glow;ctx.beginPath();ctx.arc(cx,cy,r*1.1,0,Math.PI*2);ctx.fill();
  const sphere=ctx.createRadialGradient(cx-r*.36,cy-r*.38,r*.05,cx,cy,r);sphere.addColorStop(0,"#214355");sphere.addColorStop(.6,"#142d3c");sphere.addColorStop(1,"#0c1b29");ctx.fillStyle=sphere;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.fill();ctx.strokeStyle="#3c7084";ctx.stroke();
  for(let lat=-60;lat<=60;lat+=30){const a=lat*Math.PI/180;const points=[];for(let i=0;i<=120;i++){const t=i/120*Math.PI*2;points.push([earth*Math.cos(a)*Math.cos(t),earth*Math.cos(a)*Math.sin(t),earth*Math.sin(a)]);}drawCurve(points,"#3c697d77",.7,true);}
  for(let lon=0;lon<360;lon+=30){const a=lon*Math.PI/180;const points=[];for(let i=0;i<=60;i++){const t=-Math.PI/2+i/60*Math.PI;points.push([earth*Math.cos(t)*Math.cos(a),earth*Math.cos(t)*Math.sin(a),earth*Math.sin(t)]);}drawCurve(points,"#3c697d77",.7,true);}
  if(orbitPath.length)drawCurve(orbitPath,"#6ee7dcaa",1.3);
  pickPoints=[];
  // Keep dots at the most recent direct SGP4 sample; do not imply interpolated ground truth.
  const rows=liveStates.filter(row=>row[7]===0&&row[1]!==null).map(row=>({row,p:project(row.slice(1,4))})).sort((a,b)=>a.p.z-b.p.z);
  for(const {row,p} of rows){if(hiddenByEarth(p)||p.x<0||p.x>width||p.y<0||p.y>height)continue;const object=objects.get(row[0]);const stale=object&&Math.abs((Date.now()-dateOf(object.epoch))/3600000)>72;const selected=row[0]===selectedId;const color=selected?"#fff":stale?"#657589":object?.groups.includes("stations")?"#6ee7dc":"#f2b467";ctx.fillStyle=color;ctx.globalAlpha=selected?1:stale?.45:.85;ctx.beginPath();ctx.arc(p.x,p.y,selected?3.5:1.5,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;pickPoints.push({id:row[0],x:p.x,y:p.y});if(selected){ctx.strokeStyle="#6ee7dc";ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke();ctx.font="12px 'Segoe UI',sans-serif";ctx.fillStyle="#d9fff7";ctx.fillText(object?.name||String(row[0]),p.x+11,p.y-9);}}
  if(!liveStates.length){ctx.fillStyle="#b5c7d8";ctx.font="14px 'Segoe UI',sans-serif";ctx.textAlign="center";ctx.fillText("Loading SGP4 positions…",cx,cy+r+36);ctx.textAlign="left";}
  requestAnimationFrame(drawGlobe);
}
canvas.addEventListener("pointerdown",event=>{drag={x:event.clientX,y:event.clientY,moved:false};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.moved=drag.moved||Math.abs(dx)+Math.abs(dy)>2;yaw+=dx*.006;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.006));drag.x=event.clientX;drag.y=event.clientY;});
canvas.addEventListener("pointerup",event=>{if(drag&&!drag.moved){const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;const closest=pickPoints.map(p=>({...p,d:Math.hypot(p.x-x,p.y-y)})).sort((a,b)=>a.d-b.d)[0];if(closest&&closest.d<13)selectObject(closest.id);}drag=null;});
canvas.addEventListener("pointercancel",()=>{drag=null;});
canvas.addEventListener("wheel",event=>{event.preventDefault();zoom=Math.max(.45,Math.min(2.4,zoom*Math.exp(-event.deltaY*.001)));},{passive:false});
canvas.addEventListener("keydown",event=>{const actions={ArrowLeft:()=>yaw-=.1,ArrowRight:()=>yaw+=.1,ArrowUp:()=>pitch-=.1,ArrowDown:()=>pitch+=.1,"+":()=>zoom=Math.min(2.4,zoom+.1),"-":()=>zoom=Math.max(.45,zoom-.1)};if(actions[event.key]){event.preventDefault();actions[event.key]();}});
$("resetView").addEventListener("click",()=>{yaw=-.5;pitch=.45;zoom=1;});
setInterval(()=>{$("clock").textContent=utc(new Date().toISOString());},1000);
pollState();pollLive();requestAnimationFrame(drawGlobe);
