// In-memory checks of the actual Gemini-only UI. No external requests or real keys.
// Setup: npm install --prefix .local/ui-check --no-save --package-lock=false --ignore-scripts happy-dom
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {Window} from '../.local/ui-check/node_modules/happy-dom/lib/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = await readFile(path.join(root, 'dashboard/index.html'), 'utf8');
const script = await readFile(path.join(root, 'dashboard/assistant.js'), 'utf8');

async function setup({legacy=null}={}) {
  const window = new Window({url:'http://127.0.0.1:8787/', settings:{
    enableJavaScriptEvaluation:true, disableJavaScriptFileLoading:true,
    disableCSSFileLoading:true, disableIframePageLoading:true,
  }});
  window.document.write(html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ''));
  const get = id => window.document.getElementById(id);
  const profile = {provider:'gemini',configured:false,model:'gemini-3.1-flash-lite',key_source:null};
  const status = () => {
    if (legacy === 'openai-only') return {provider:'openai',configured:true,model:'old-openai-model',key_source:'saved'};
    if (legacy) {
      const profiles = {
        openai:{provider:'openai',configured:true,model:'old-openai-model',key_source:'saved'},
        gemini:{...profile,configured:true,key_source:'saved'},
      };
      return {...profiles[legacy],providers:profiles,supported_providers:['openai','gemini']};
    }
    return {...profile,providers:{gemini:{...profile}},supported_providers:['gemini']};
  };
  const calls = [];
  const controls = {chatMode:'answer', configError:false, pendingSignal:null};
  window.fetch = async (raw, options={}) => {
    const url = new URL(raw, window.location.href);
    assert.equal(url.origin, 'http://127.0.0.1:8787');
    const body = options.body ? JSON.parse(options.body) : undefined;
    calls.push({path:url.pathname,body});
    if (url.pathname === '/api/chat/status') return Response.json(status());
    if (url.pathname === '/api/objects/35809/info') return Response.json({id:35809,name:'IRIDIUM 33 DEB',history_summary:'Documented history.'});
    assert.equal(legacy,null,'Old servers must never receive configuration or chat requests');
    if (url.pathname === '/api/chat/config') {
      assert.equal(body.provider,'gemini');
      if (controls.pendingSignal) assert.equal(controls.pendingSignal.aborted,true,'Chat must be cancelled before configuring or removing a key');
      if (controls.configError) return Response.json({error:'Connection could not be saved.'},{status:400});
      if (body.clear) { profile.configured=false; profile.key_source=null; }
      else {
        if (!profile.configured) assert.ok(body.api_key,'Unconfigured Gemini needs a key');
        profile.configured=true; profile.key_source='saved'; profile.model=body.model;
      }
      return Response.json(status());
    }
    if (url.pathname === '/api/chat') {
      assert.equal('api_key' in body,false);
      assert.equal(body.provider,'gemini','Every question must be bound to Gemini');
      if (controls.chatMode === 'pending') {
        controls.pendingSignal=options.signal;
        return new Promise((resolve,reject) => {
          options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true});
        });
      }
      if (controls.chatMode === 'quota') return Response.json({error:'Gemini reports a quota limit for this API project.'},{status:429});
      return Response.json({object_id:35809,provider:controls.chatMode === 'wrong-provider'?'openai':'gemini',answer:'Answer from Gemini',sources:[
        {title:'CelesTrak reference',url:'https://celestrak.org/'},
        {title:'Unsafe source',url:'javascript:alert(1)'},
      ]});
    }
    throw new Error(`Unexpected request: ${url.pathname}`);
  };
  const errors=[];
  window.addEventListener('error',event=>errors.push(String(event.error || event.message)));
  window.eval(script);
  const wait = async predicate => {
    for (let n=0;n<100;n++) {
      if (predicate()) return;
      await new Promise(resolve=>setTimeout(resolve,10));
    }
    throw new Error(`DOM did not settle. Errors: ${errors}; settings: ${get('chatSettingsMessage').textContent}`);
  };
  await wait(()=>!get('chatConnectionState').textContent.includes('Checking Gemini'));
  window.dispatchEvent(new window.CustomEvent('object-selected',{detail:{id:35809}}));
  await wait(()=>get('objectStoryTitle').textContent === 'IRIDIUM 33 DEB');
  const submit = id => get(id).dispatchEvent(new window.Event('submit',{cancelable:true,bubbles:true}));
  const send = text => { get('objectChatInput').value=text; submit('objectChatForm'); };
  return {window,get,wait,submit,send,calls,controls,errors};
}

const env=await setup();
try {
  const {window,get,wait,submit,send,calls,controls,errors}=env;
  assert.equal(get('chatProvider'),null,'Provider selection must be removed');
  assert.match(get('chatConnectionState').textContent,/Gemini not connected/);
  assert.equal(get('sendObjectChat').disabled,true);
  send('Explain this object');
  assert.equal(get('chatSettingsDialog').open,true);
  assert.equal(calls.some(call=>call.path==='/api/chat'),false);
  assert.equal(get('chatApiKey').required,true);
  assert.equal(get('chatModelName').value,'gemini-3.1-flash-lite');
  assert.equal(get('disconnectChat').hidden,true);
  assert.match(get('chatSettingsDescription').textContent,/improve Google products/);
  assert.ok(get('chatProviderLinks').querySelector('a[href="https://aistudio.google.com/apikey"]'));
  const before=calls.length;
  submit('chatSettingsForm');
  assert.equal(calls.length,before,'Unconfigured Gemini must not submit an empty key');
  get('chatApiKey').value='fake-gemini-test-key';
  submit('chatSettingsForm');
  await wait(()=>!get('chatSettingsDialog').open && get('chatConnectionState').textContent.includes('Gemini configured'));
  assert.equal(get('chatApiKey').value,'');
  assert.deepEqual(calls.filter(call=>call.path==='/api/chat/config').at(-1).body,{provider:'gemini',model:'gemini-3.1-flash-lite',api_key:'fake-gemini-test-key'});
  assert.equal(calls.filter(call=>call.path==='/api/chat/status').length,1,'Successful configuration uses returned status');
  send('Gemini question');
  await wait(()=>get('objectChatMessages').textContent.includes('Answer from Gemini'));
  assert.match(get('objectChatMessages').textContent,/Orbital Watch · Gemini/);
  assert.ok(get('objectChatMessages').querySelector('a[href="https://celestrak.org/"]'));
  assert.equal(get('objectChatMessages').querySelector('a[href^="javascript:"]'),null);
  assert.deepEqual(calls.filter(call=>call.path==='/api/chat').at(-1).body.history,[]);
  assert.match(get('chatPrivacyNote').textContent,/sent to Google through Gemini/);

  controls.chatMode='quota';
  send('Quota question');
  await wait(()=>!get('chatRequestError').hidden);
  assert.equal(get('objectChatInput').value,'Quota question','A failed question remains available to retry');
  assert.equal(get('chatAccountLinks').hidden,false);
  assert.ok(get('chatAccountLinks').querySelector('a[href="https://aistudio.google.com/projects"]'));
  assert.equal(get('chatAccountLinks').textContent.includes('OpenAI'),false);

  controls.chatMode='pending';
  send('Pending question');
  await wait(()=>Boolean(controls.pendingSignal));
  assert.equal(calls.filter(call=>call.path==='/api/chat').at(-1).body.history.some(item=>item.content==='Quota question'),false,'Failed questions must not be sent as completed history');
  get('chatSettingsButton').click();
  assert.equal(get('chatApiKey').required,false,'Only the saved Gemini key may be reused');
  submit('chatSettingsForm');
  await wait(()=>!get('chatSettingsDialog').open);
  assert.equal(controls.pendingSignal.aborted,true);
  assert.deepEqual(calls.filter(call=>call.path==='/api/chat/config').at(-1).body,{provider:'gemini',model:'gemini-3.1-flash-lite'});

  controls.pendingSignal=null;
  send('Pending removal question');
  await wait(()=>Boolean(controls.pendingSignal));
  get('chatSettingsButton').click();
  assert.equal(get('disconnectChat').hidden,false);
  get('disconnectChat').click();
  await wait(()=>get('chatSettingsMessage').textContent.includes('Saved Gemini key removed'));
  assert.equal(controls.pendingSignal.aborted,true);
  assert.deepEqual(calls.filter(call=>call.path==='/api/chat/config').at(-1).body,{clear:true,provider:'gemini'});
  assert.match(get('chatConnectionState').textContent,/Gemini not connected/);
  assert.equal(get('chatApiKey').required,true);
  assert.equal(get('sendObjectChat').disabled,true);

  controls.configError=true;
  get('chatApiKey').value='fake-replacement-test-key';
  submit('chatSettingsForm');
  await wait(()=>get('chatSettingsMessage').textContent.includes('could not be saved'));
  assert.equal(get('chatApiKey').value,'');
  assert.match(get('chatConnectionState').textContent,/Gemini not connected/);
  controls.configError=false;
  get('chatApiKey').value='fake-replacement-test-key';
  submit('chatSettingsForm');
  await wait(()=>!get('chatSettingsDialog').open);
  assert.match(get('chatConnectionState').textContent,/Gemini configured/);
  assert.equal(get('chatApiKey').value,'');

  controls.chatMode='wrong-provider';
  send('Reject a mismatched answer');
  await wait(()=>get('chatRequestError').textContent.includes('unexpected chat configuration'));
  assert.equal(get('objectChatInput').value,'Reject a mismatched answer');
  assert.equal(get('chatAccountLinks').hidden,true);
  get('chatSettingsButton').click();
  get('chatApiKey').value='clear-on-close';
  get('closeChatSettings').click();
  await wait(()=>get('chatApiKey').value==='');
  assert.equal(window.localStorage.length,0);
  assert.equal(window.sessionStorage.length,0);
  assert.deepEqual(errors,[]);
} finally { await env.window.happyDOM.close(); }

for (const kind of ['openai-only','openai','gemini']) {
  const env=await setup({legacy:kind});
  try {
    const {window,get,submit,send,calls,errors}=env;
    assert.match(get('chatConnectionState').textContent,/Restart the dashboard server/);
    assert.equal(get('sendObjectChat').disabled,true);
    assert.equal(get('objectChatInput').disabled,true);
    get('chatSettingsButton').click();
    assert.match(get('chatProviderNotice').textContent,/Restart the dashboard server to load Gemini-only chat/);
    assert.equal(get('chatProviderNotice').hidden,false);
    for (const id of ['saveChatSettings','disconnectChat','chatApiKey','chatModelName']) assert.equal(get(id).disabled,true);
    assert.equal(get('chatApiKey').required,true,'Legacy configured status must not count as a Gemini connection');
    assert.equal(get('chatModelName').value,'gemini-3.1-flash-lite');
    get('chatApiKey').value='never-send-to-old-server';
    submit('chatSettingsForm');
    get('disconnectChat').click();
    send('Must not send to the old server');
    assert.equal(calls.some(call=>call.path==='/api/chat/config' || call.path==='/api/chat'),false);
    assert.doesNotMatch(window.document.body.textContent,/OpenAI|old-openai-model/);
    assert.equal(window.localStorage.length,0);
    assert.equal(window.sessionStorage.length,0);
    get('closeChatSettings').click();
    await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(get('chatApiKey').value,'');
    assert.deepEqual(errors,[]);
  } finally { await env.window.happyDOM.close(); }
}

console.log('Gemini-only chat dashboard DOM checks passed: setup, key reuse/removal, answers and sources, quota recovery, cancellation, and all old-server guards.');
