// In-memory DOM behavior checks using the actual published scientific results.
// No Windows/browser automation or generated orbit data is used here.
// Setup: npm install --prefix .local/ui-check --no-save --package-lock=false --ignore-scripts happy-dom
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {Window} from '../.local/ui-check/node_modules/happy-dom/lib/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = await readFile(path.join(root, 'dashboard/index.html'), 'utf8');
const script = await readFile(path.join(root, 'dashboard/model-results.js'), 'utf8');
const index = JSON.parse(await readFile(path.join(root, 'dashboard/model-results/index.json'), 'utf8'));
const payloadFor = async id => JSON.parse(await readFile(path.join(root, 'dashboard/model-results', id, 'payload.json'), 'utf8'));
const latest = await payloadFor(index.latest_run_id);

async function setup(mode = 'normal') {
  const window = new Window({url:'http://127.0.0.1:8787/#models', settings:{
    enableJavaScriptEvaluation:true, disableJavaScriptFileLoading:true,
    disableCSSFileLoading:true, disableIframePageLoading:true,
  }});
  window.document.write(html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ''));
  let fail = mode === 'error';
  window.fetch = async raw => {
    const url = new URL(raw);
    assert.equal(url.origin, 'http://127.0.0.1:8787');
    assert.ok(url.pathname.startsWith('/assets/model-results/'));
    if (fail) return new Response('offline test', {status:503});
    if (mode === 'empty' && url.pathname.endsWith('/index.json')) {
      return Response.json({schema_version:1,latest_run_id:null,runs:[]});
    }
    const relative = decodeURIComponent(url.pathname.slice('/assets/'.length));
    const target = path.resolve(root, 'dashboard', relative);
    assert.ok(target.startsWith(path.join(root, 'dashboard/model-results')+path.sep));
    return new Response(await readFile(target));
  };
  const errors = [];
  window.addEventListener('error', e => errors.push(String(e.error || e.message)));
  window.eval(script);
  const get = id => window.document.getElementById(id);
  const wait = async predicate => {
    for (let n=0; n<100; n++) {
      if (predicate()) return;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    throw new Error(`DOM state did not settle: ${get('modelResultsNotice').textContent}; errors=${errors}`);
  };
  return {window,get,wait,errors,recover:()=>{fail=false;}};
}

const env = await setup();
try {
  const {window,get,wait,errors} = env;
  await wait(()=>!get('modelResultsContent').hidden);
  assert.equal(get('modelRunSelect').value, index.latest_run_id);
  assert.equal(get('modelRunSelect').options.length, index.runs.length);
  assert.ok(get('orbitCanvas').compareDocumentPosition(get('training')) & window.Node.DOCUMENT_POSITION_FOLLOWING);
  assert.ok(get('training').compareDocumentPosition(get('models')) & window.Node.DOCUMENT_POSITION_FOLLOWING);
  assert.equal(get('modelTrainingContent').hidden, false);
  assert.equal(get('modelEvaluationNotice').hidden, true);
  assert.equal(get('modelTrainingArea').querySelectorAll('.model-training-card').length, 2);
  const trainingCard = get('modelTrainingArea').querySelector('.model-training-card');
  const trainingDetails = trainingCard.querySelector('details');
  trainingDetails.open = true;
  assert.equal(get('modelMetricCards').children.length, 3);
  assert.match(get('modelSampleCount').textContent, /60 samples per model/);
  const full = () => JSON.parse(get('modelFullMetrics').textContent);
  assert.deepEqual(full().values_in_saved_units, Object.fromEntries(['SGP4','PINN','COMBINED'].map(n=>[n,latest.metrics[n].test])));
  for (const [id, tab, figures] of [
    ['SGP4','modelTabSGP4',3],['PINN','modelTabPINN',4],['COMBINED','modelTabCombined',4],
  ]) {
    get(tab).click();
    assert.equal(get(tab).getAttribute('aria-selected'), 'true');
    assert.equal(get('modelMetricCards').children.length, 4);
    assert.deepEqual(full().values_in_saved_units[id], latest.metrics[id].test);
    assert.equal(get('modelFigureGallery').querySelectorAll('img').length, figures);
    assert.equal(get('modelTrainingArea').querySelectorAll('.model-training-card').length, 2);
    assert.equal(get('modelTrainingArea').querySelector('.model-training-card'), trainingCard);
    assert.equal(trainingDetails.open, true);
    assert.match(get('modelMetricsDownload').href, new RegExp(`/${id.toLowerCase()}/metrics.csv$`));
    assert.equal(get('modelTrainingArea').querySelectorAll('tr[data-selected="true"]').length, 2);
  }
  get('modelSplitSelect').value = 'validation';
  get('modelSplitSelect').dispatchEvent(new window.Event('change'));
  assert.equal(full().interval, 'validation');
  assert.deepEqual(full().values_in_saved_units.COMBINED, latest.metrics.COMBINED.validation);
  assert.match(get('modelSampleCount').textContent, /30 samples/);
  get('modelSplitSelect').value = 'train';
  get('modelSplitSelect').dispatchEvent(new window.Event('change'));
  assert.deepEqual(full().values_in_saved_units.COMBINED, latest.metrics.COMBINED.train);
  assert.match(get('modelSampleCount').textContent, /90 samples/);
  get('modelTabCombined').dispatchEvent(new window.KeyboardEvent('keydown',{key:'Home',bubbles:true}));
  assert.equal(get('modelTabComparison').getAttribute('aria-selected'), 'true');
  assert.equal(get('modelFigureGallery').querySelectorAll('img').length, 4);
  const inner = get('modelPredictionArea').querySelector('.model-prediction-table-body').parentElement;
  inner.open = true;
  inner.dispatchEvent(new window.Event('toggle'));
  try { await wait(()=>inner.querySelectorAll('tbody tr').length === 181); }
  catch (error) { throw new Error(`${error.message}; connected=${inner.isConnected}; trajectory=${inner.textContent}`); }
  assert.match(inner.querySelector('tbody tr').textContent, /2026-09-13/);
  if (index.runs.length > 1) {
    const other = index.runs.find(r=>r.run_id!==index.latest_run_id);
    get('modelRunSelect').value = other.run_id;
    get('modelRunSelect').dispatchEvent(new window.Event('change'));
    await wait(()=>!get('modelResultsContent').hidden && get('modelRunDownload').href.includes(other.run_id));
    const old = await payloadFor(other.run_id);
    assert.deepEqual(full().values_in_saved_units.PINN, old.metrics.PINN.train);
    assert.match(get('modelTrainingArea').querySelector('a[href$="/pinn/model.pt"]').href, new RegExp(other.run_id));
  }
  get('modelFileSearch').value = 'model.pt';
  get('modelFileSearch').dispatchEvent(new window.Event('input'));
  assert.equal(get('modelFileList').querySelectorAll('a').length, latest.files.filter(f=>f.path.includes('model.pt')).length);
  assert.ok([...get('modelFileList').querySelectorAll('a')].some(link=>link.textContent==='pinn/model.pt'));
  assert.ok([...get('modelFileList').querySelectorAll('a')].some(link=>link.textContent==='combined/model.pt'));
  for (const link of [...get('models').querySelectorAll('a[href]'), ...get('modelTrainingArea').querySelectorAll('a[href]')]) {
    assert.ok(link.getAttribute('href') === '#training' || link.href.startsWith('http://127.0.0.1:8787/assets/model-results/'));
  }
  assert.deepEqual(errors, []);
} finally { await env.window.happyDOM.close(); }

const empty = await setup('empty');
try {
  await empty.wait(()=>empty.get('modelResultsNotice').dataset.state==='empty');
  assert.equal(empty.get('modelResultsContent').hidden, true);
  assert.equal(empty.get('modelTrainingContent').hidden, true);
  assert.equal(empty.get('modelEvaluationNotice').dataset.state, 'empty');
  assert.equal(empty.get('modelRunDownload').hidden, true);
} finally { await empty.window.happyDOM.close(); }

const recovery = await setup('error');
try {
  await recovery.wait(()=>recovery.get('modelResultsNotice').dataset.state==='error');
  assert.equal(recovery.get('modelResultsContent').hidden, true);
  assert.equal(recovery.get('modelTrainingContent').hidden, true);
  assert.equal(recovery.get('modelEvaluationNotice').dataset.state, 'error');
  recovery.recover();
  recovery.get('modelResultsNotice').querySelector('button').click();
  await recovery.wait(()=>!recovery.get('modelResultsContent').hidden);
} finally { await recovery.window.happyDOM.close(); }

console.log('PASS: all model tabs, exact split values, two-run selection, 181-state preview, weights/download links, keyboard navigation, empty state and error recovery.');
