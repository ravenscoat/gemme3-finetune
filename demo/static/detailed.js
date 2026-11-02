const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let meta, evaluation, current, jsonView = false, busy = false, savedIndex = 0, datasetOffset = 0, datasetTotal = 0, datasetRows = [], searchSequence = 0;
let selectedDataset='bitext', datasetCatalog={}, datasetRevision=0;
let developmentSignature='', developmentIndex=0;
const pageNames = {playground:'Playground', dataset:'Dataset explorer', evaluation:'Evaluation'};
$('#page-evaluation').insertAdjacentHTML('beforeend','<div id="other-evaluation" hidden></div>');

async function api(url, options) {
  if (!options?.method || options.method==='GET') url+=(url.includes('?')?'&':'?')+'dataset='+encodeURIComponent(selectedDataset);
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'The request could not be completed.');
  return body;
}
function notify(message) {
  $('#toast').textContent = message; $('#toast').classList.add('visible');
  setTimeout(() => $('#toast').classList.remove('visible'), 4000);
}
function page(name) {
  if (!pageNames[name]) name = 'playground';
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-'+name));
  document.querySelectorAll('.nav-item').forEach(p => p.classList.toggle('active', p.dataset.page === name));
  $('#page-crumb').textContent = pageNames[name];
  history.replaceState(null, '', '#'+name);
  window.scrollTo({top:0, behavior:'smooth'});
  if (name === 'dataset') loadDataset();
  if (name === 'evaluation' && meta) refreshMetrics();
}
document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', () => page(button.dataset.page)));
$('.brand').addEventListener('click', e => { e.preventDefault(); page('playground'); });

function recordHTML(record) {
  if (meta?.kind==='qa') return `<div class="record-tags"><span class="tag">${esc(record.domain.replaceAll('_',' '))}</span></div><div class="conversation-label">INSURANCE QUESTION</div><div class="customer-bubble">${esc(record.question)}</div><div class="conversation-label">SOURCE-STYLE ANSWER</div><div class="response-block"><span class="response-mark">✳</span><p class="response-text">${esc(record.answer)}</p></div>`;
  if (meta?.kind==='workflow') return `<div class="record-tags">${[record.category,record.subcategory,record.difficulty].map(v=>`<span class="tag">${esc(v.replaceAll('_',' '))}</span>`).join('')}</div><div class="conversation-label">CUSTOMER MESSAGE</div><div class="customer-bubble">${esc(record.input.customer_message)}</div><div class="conversation-label">SCENARIO CONTEXT</div>${fieldsHTML(record.input.context)}<div class="conversation-label">EXPECTED WORKFLOW</div>${fieldsHTML(record.expected_output)}${record.input.conversation_history.length?'<details class="workflow-history"><summary>Conversation history</summary><pre class="json-output">'+esc(JSON.stringify(record.input.conversation_history,null,2))+'</pre></details>':''}`;
  if (meta?.kind==='clinical') return `<div class="record-tags"><span class="tag">${esc(record.admission_type.replaceAll('_',' '))}</span><span class="tag category">STRUCTURED ADMISSION</span></div>${fieldsHTML(record)}<p class="footnote">Numeric values and missingness follow the demo format. This view does not establish clinical correctness.</p>`;
  return `<div class="record-tags"><span class="tag">${esc(record.intent.replaceAll('_',' '))}</span><span class="tag category">${esc(record.category.replaceAll('_',' '))}</span><span class="tag" title="Language-variation tags">${esc(record.tags)}</span></div><div class="conversation-label">CUSTOMER REQUEST</div><div class="customer-bubble">${esc(record.instruction)}</div><div class="conversation-label">SUPPORT REPLY</div><div class="response-block"><span class="response-mark">✳</span><p class="response-text">${esc(record.response)}</p></div>`;
}
function renderOutput() {
  if (!current) return;
  $('#download').disabled=busy||!!current.pending||(!current.record&&!current.text);
  $('#json-toggle').disabled=!current.record;
  const source = $('#output-source');
  source.classList.toggle('live', !!current.live);
  source.textContent = current.empty ? 'No structurally valid saved outputs' : current.pending ? 'Assessment examples pending' : current.live ? 'Live model output' : `Saved model output · #${current.sample_id}`;
  if (current.empty) {
    $('#output-body').innerHTML='<div class="empty-state"><h3>No saved output passed the structure check.</h3><p>The assessment is complete. Its failed attempts remain counted in Evaluation.</p></div>';
  } else if (current.pending) {
    $('#output-body').innerHTML='<div class="empty-state"><span>◴</span><h3>Your assessment is in progress.</h3><p>New saved model outputs will appear here when they are ready. You can browse this dataset now.</p></div>';
  } else if (jsonView && current.record) {
    $('#output-body').innerHTML = `<pre class="json-output">${esc(JSON.stringify(current.record, null, 2))}</pre>`;
  } else if (current.record) {
    $('#output-body').innerHTML = recordHTML(current.record);
  } else {
    $('#output-body').innerHTML = `<pre class="stream-output">${esc(current.text || 'Preparing your record…')}</pre>${busy ? '<span class="stream-cursor"></span>' : ''}`;
    if (current.validation_error) $('#output-body').innerHTML += `<div class="result-note">Output kept as generated. Schema check: ${esc(current.validation_error)}</div>`;
  }
  const details = current.pending ? 'No generated records are being substituted' : current.live ? (current.seconds ? `${current.seconds.toFixed(1)} seconds · ${current.new_tokens} tokens` : 'Live generation') : 'Recorded assessment output';
  const state = current.error ? 'Generation failed' : current.record ? '✓ Structure checked' : current.status === 'complete' ? 'Schema check failed' : 'Waiting for output';
  $('#output-footer').innerHTML = `<span>${esc(details)}</span><span>${state}</span>`;
  if (current.hit_token_limit) $('#output-body').innerHTML += '<div class="result-note">The generation reached its token limit. Inspect the raw output for completeness.</div>';
  if (current.record && current.requested_intent && groupValue(current.record) !== current.requested_intent) $('#output-body').innerHTML += '<div class="result-note">The model generated a different focus from the one requested. Focus is a prompt instruction, not a guarantee.</div>';
  if(current.source_warnings?.length) $('#output-body').innerHTML+='<div class="result-note">'+current.source_warnings.map(esc).join('<br>')+'</div>';
}
function renderReferences(references) {
  if (!references?.length) {
    $('#references').innerHTML = '<div class="card reference-placeholder">Training references appear when a complete record is available.</div>';
    return;
  }
  $('#references').innerHTML = references.map((r,i) => `<article class="card reference-card" tabindex="0" role="button" aria-label="Open training reference ${i+1}" data-reference="${i}"><div class="reference-top"><span>REFERENCE ${String(i+1).padStart(2,'0')}</span><strong>${Math.round(r.similarity*100)}% ${esc(r.similarity_method||'lexical similarity')}</strong></div><span class="tag">${esc(groupValue(r.record).replaceAll('_',' '))}</span><p>${esc(recordPreview(r.record))}</p><span class="reference-link">Source ${esc(r.source_line)} · View full record ↗</span></article>`).join('');
  document.querySelectorAll('[data-reference]').forEach(card => {
    const open = () => openRecord(references[Number(card.dataset.reference)].record, references[Number(card.dataset.reference)].source_line);
    card.addEventListener('click',open);
    card.addEventListener('keydown',e => { if (e.key==='Enter' || e.key===' ') { e.preventDefault(); open(); } });
  });
}
function renderCopyNote() {
  if (!current?.record || jsonView) return;
  const notes = [];
  if (current.request_seen) notes.push('The normalized customer request appears in training.');
  if (current.response_seen) notes.push('The normalized reply appears in training.');
  if (current.record_seen) notes.push('The full normalized record appears in training.');
  if (notes.length) $('#output-body').innerHTML += `<div class="result-note">↳ ${notes.join(' ')} Similarity and novelty are separate checks.</div>`;
}
async function loadSaved(index) {
  if (busy) return;
  $('#next-saved').disabled = true;
  try {
    const revision=datasetRevision;
    const saved = await api('/api/saved?index='+index);
    if (busy || revision!==datasetRevision) return;
    current = saved;
    savedIndex = current.index;
    renderOutput(); renderCopyNote(); renderReferences(current.references);
  } catch (error) { notify(error.message); }
  finally { $('#next-saved').disabled = busy || !!current?.pending || !!current?.empty; }
}
$('#next-saved').addEventListener('click', () => loadSaved(savedIndex+1));
$('#json-toggle').addEventListener('click', () => { jsonView=!jsonView; $('#json-toggle').classList.toggle('selected',jsonView); renderOutput(); renderCopyNote(); });
$('#download').addEventListener('click', () => {
  if (!current) return;
  const body = current.record || {raw_output:current.text, schema_error:current.validation_error};
  const url = URL.createObjectURL(new Blob([JSON.stringify(body,null,2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download=`gemma-${current.live?'live':'saved-'+current.sample_id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});
function promptCount() { $('#char-count').textContent = `${$('#prompt').value.length} / 1,000`; }
$('#prompt').addEventListener('input',promptCount); promptCount();
document.querySelectorAll('[data-suggestion]').forEach(b => b.addEventListener('click', () => { $('#prompt').value=b.dataset.suggestion; promptCount(); $('#prompt').focus(); }));
$('#temperature').addEventListener('input', () => $('#temperature-value').textContent=Number($('#temperature').value).toFixed(1));

function setBusy(value) {
  busy=value;
  $('#generate').disabled=value; $('#load-model').disabled=value; $('#next-saved').disabled=value||!!current?.pending; $('#download').disabled=value||!!current?.pending||!current;
  $('#dataset-selector').disabled=value;
  $('#generate').innerHTML=value ? '<span>Creating your example…</span><span class="stream-cursor"></span>' : '<span>Generate a record</span><span>↗</span>';
}
async function refreshStatus() {
  try {
    const status = await api('/api/status');
    const state = status.engine.state;
    $('#engine-dot').className='status-dot '+(state==='ready'?'ready': ['loading','generating','training','evaluating'].includes(state)?'busy':state==='error'?'error':'');
    $('#engine-text').textContent=({idle:'Load model',loading:'Model loading',ready:'Model ready',generating:'Model generating',training:'GPU training',evaluating:'GPU evaluating',error:'Retry model load'})[state] || state;
    const reserved=['training','evaluating'].includes(state);
    $('#gpu-notice').hidden=!reserved;
    $('#generate').disabled=busy||reserved; $('#load-model').disabled=busy||reserved;
    if(current?.pending && !busy) loadSaved(0);
  } catch { $('#engine-text').textContent='Server unavailable'; $('#engine-dot').className='status-dot error'; }
}
async function runRequest(request) {
  if (busy) return;
  const loadingOnly=request.action==='load';
  const began=performance.now();
  setBusy(true); $('#job-message').classList.remove('error');
  $('#job-message').textContent=loadingOnly?'Loading the selected adapter into GPU memory…':'Connecting to your fine-tuned model…';
  if (!loadingOnly) {
    current={live:true, text:'',record:null}; jsonView=false; $('#json-toggle').classList.remove('selected'); renderOutput(); renderReferences([]);
  }
  try {
    const {id}=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...request,dataset:selectedDataset})});
    while (true) {
      const job=await api('/api/jobs/'+id);
      const seconds=Math.floor((performance.now()-began)/1000);
      if (job.status==='error') throw new Error(job.error);
      if (job.status==='complete') {
        if (!loadingOnly) { current=job; renderOutput(); renderCopyNote(); renderReferences(job.references); }
        $('#job-message').textContent=loadingOnly?'Model loaded. Your playground is ready.':`Generated live by the full-pass model in ${job.seconds.toFixed(1)} seconds.`;
        break;
      }
      $('#job-message').textContent=job.status==='loading' ? `Loading the model on your GPU… ${seconds}s. The first load can take a few minutes.` : `Generating a new record… ${seconds}s`;
      if (!loadingOnly && job.text) { current={...job,live:true}; renderOutput(); }
      await refreshStatus();
      await new Promise(resolve=>setTimeout(resolve,900));
    }
  } catch (error) {
    if (!loadingOnly) { current={...current,error:error.message}; renderOutput(); }
    $('#job-message').textContent=error.message;
    $('#job-message').classList.add('error');
    notify(error.message);
  } finally { setBusy(false); await refreshStatus(); }
}
$('#load-model').addEventListener('click', () => runRequest({action:'load'}));
$('#generate').addEventListener('click', () => {
  const prompt=$('#prompt').value.trim();
  if (!prompt) { $('#prompt').focus(); return notify('Enter a generation prompt first.'); }
  runRequest({prompt, intent:$('#intent').value, temperature:Number($('#temperature').value), seed:Number($('#seed').value)});
});

function openRecord(record,sourceLine) {
  $('#dialog-content').innerHTML=`<p class="footnote">${esc(meta.label)} training split · Source ${esc(sourceLine)}</p>${recordHTML(record)}`;
  $('#record-dialog').showModal();
}
$('#close-dialog').addEventListener('click',()=>$('#record-dialog').close());
$('#record-dialog').addEventListener('click',e=> { if(e.target===$('#record-dialog')) { const r=e.target.getBoundingClientRect(); if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close(); } });
async function loadDataset() {
  const sequence=++searchSequence;
  const params=new URLSearchParams({q:$('#dataset-search').value,intent:$('#dataset-intent').value,offset:datasetOffset,limit:8});
  try {
    const data=await api('/api/dataset?'+params);
    if(sequence!==searchSequence)return;
    datasetRows=data.records; datasetTotal=data.total;
    $('#dataset-count').textContent=data.total.toLocaleString()+' examples';
    $('#dataset-rows').innerHTML=data.records.length ? data.records.map((r,i)=>`<tr tabindex="0" data-row="${i}"><td>${esc(r.source_line)}</td><td>${esc(recordPreview(r.record))}</td><td><span class="tag">${esc(groupValue(r.record).replaceAll('_',' '))}</span></td><td>↗</td></tr>`).join('') : '<tr><td colspan="4">No examples match this search. Try another phrase or group.</td></tr>';
    $('#dataset-range').textContent=data.total ? `${datasetOffset+1}–${Math.min(datasetOffset+8,data.total)} of ${data.total.toLocaleString()} examples` : '0 examples';
    $('#prev-page').disabled=datasetOffset===0; $('#next-page').disabled=datasetOffset+8>=datasetTotal;
    document.querySelectorAll('[data-row]').forEach(row=>{
      const open=()=> { const data=datasetRows[Number(row.dataset.row)]; openRecord(data.record,data.source_line); };
      row.addEventListener('click',open); row.addEventListener('keydown',e=>{if(e.key==='Enter')open();});
    });
  } catch(error) { notify(error.message); }
}
let searchTimeout;
$('#dataset-search').addEventListener('input',()=>{clearTimeout(searchTimeout);searchTimeout=setTimeout(()=>{datasetOffset=0;loadDataset();},250);});
$('#dataset-intent').addEventListener('change',()=>{datasetOffset=0;loadDataset();});
$('#prev-page').addEventListener('click',()=>{datasetOffset=Math.max(0,datasetOffset-8);loadDataset();});
$('#next-page').addEventListener('click',()=>{if(datasetOffset+8<datasetTotal){datasetOffset+=8;loadDataset();}});

function renderMetrics(data) {
  renderDevelopmentMetrics(data.development);
  if(!$('#original-evaluation-heading')){
    const heading=document.createElement('div');heading.id='original-evaluation-heading';heading.className='section-heading';
    heading.innerHTML='<div><h2>Original frozen assessment</h2><p>Earlier cohorts below; these scores were not recomputed for the newer generation policies.</p></div>';
    $('#page-evaluation').insertBefore(heading,$('#evaluation-stats'));
  }
  renderAdvancedMetrics(data.advanced);
  $('#comparison-table').closest('.comparison-card').hidden=selectedDataset!=='bitext'; $('.evaluation-grid').hidden=selectedDataset!=='bitext';
  $('#other-evaluation').hidden=selectedDataset==='bitext';
  if(selectedDataset!=='bitext') return renderOtherMetrics(data);
  const full=data.stages[2];
  $('#evaluation-stats').innerHTML=`<div class="card eval-stat"><div class="eyebrow">REPLY DISTRIBUTION</div><strong>${full.reply_mauve.toFixed(3)}</strong><p>MAUVE similarity · higher is closer</p></div><div class="card eval-stat"><div class="eyebrow">STRUCTURE</div><strong>${full.valid}/${full.attempted}</strong><p>Valid records after outer-fence removal</p></div><div class="card eval-stat orange"><div class="eyebrow">SOURCE REPRODUCTION</div><strong>${(full.reply_copy*100).toFixed(0)}%</strong><p>Replies matching normalized training text</p></div>`;
  const rows=[['Valid records','valid'],['Intents covered, of 39','intents'],['Intent distance ↓','intent_tv'],['Category distance ↓','category_tv'],['Request MAUVE ↑','request_mauve'],['Reply MAUVE ↑','reply_mauve'],['Training request matches','request_copy'],['Training reply matches','reply_copy'],['Exact record matches','record_copy']];
  $('#comparison-table').innerHTML='<table class="comparison-table"><thead><tr><th>MEASURE</th>'+data.stages.map(s=>`<th>${esc(s.name)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(([title,key])=>'<tr><td>'+title+'</td>'+data.stages.map(s=>'<td>'+(key==='valid'?`${s.valid}/${s.attempted}`:key==='intents'?s[key]:key.includes('copy')?(s[key]*100).toFixed(1)+'%':s[key].toFixed(3))+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  const counts=Object.keys(meta.intents).sort().map(intent=>[intent,data.intent_counts[intent]||0]);
  const max=Math.max(...counts.map(x=>x[1]));
  $('#intent-bars').innerHTML=counts.map(([intent,count])=>`<div class="intent-row ${count?'':'missing'}"><span title="${esc(intent)}">${esc(intent.replaceAll('_',' '))}</span><div class="bar-track"><i style="width:${count/max*100}%"></i></div><span>${count}</span></div>`).join('');
  $('#test-results').innerHTML=`<div class="test-stat"><span>Held-out records</span><strong>${data.test.test_records.toLocaleString()}</strong></div><div class="test-stat"><span>Mean record loss</span><strong>${data.test.mean_record_loss.toFixed(4)}</strong></div><div class="test-stat"><span>Token-weighted perplexity</span><strong>${data.test.token_weighted_perplexity.toFixed(3)}</strong></div>`;
}

function renderDevelopmentMetrics(data) {
  let section=$('#development-evaluation');
  if(!section){section=document.createElement('section');section.id='development-evaluation';section.className='development-evaluation';$('#page-evaluation').insertBefore(section,$('#evaluation-stats'));}
  section.hidden=!data;if(!data)return;
  const signature=selectedDataset+data.records_sha256+data.metrics_sha256+(data.review?.judgments_sha256||'')+JSON.stringify(data.notes)+JSON.stringify(data.final_trial);
  if(signature===developmentSignature)return;
  developmentSignature=signature;developmentIndex=0;
  const a=data.chosen,b=data.baseline,num=v=>v==null?'—':Number(v).toFixed(3),pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
  const rows=[];
  const add=(title,old,value,format=num)=>rows.push([title,format(old),format(value)]);
  const featureNames={intent:'Intent proportion distance',category:'Category proportion distance',subcategory:'Subcategory proportion distance',difficulty:'Difficulty proportion distance','expected_output.routing_decision':'Routing proportion distance',actions:'Action-count proportion distance','input.context.state':'State proportion distance','input.context.channel':'Channel proportion distance'};
  for(const [field,title] of Object.entries(featureNames))if(a.feature_fidelity?.[field])add(title,b?.feature_fidelity?.[field]?.TV,a.feature_fidelity[field].TV);
  if(a.fidelity_vs_training.categorical.domain)add('Topic proportion distance',null,a.fidelity_vs_training.categorical.domain.total_variation);
  for(const [field,value] of Object.entries(a.fidelity_vs_training.categorical)){
    if(field==='domain'||a.feature_fidelity?.[field])continue;
    add(field.replaceAll('_',' ')+' proportion distance',b?.fidelity_vs_training?.categorical?.[field]?.total_variation,value.total_variation);
  }
  const names={request_characters:'Request length distance · characters',reply_characters:meta.kind==='workflow'?'Workflow length distance · characters':'Answer length distance · characters',required_action_count:'Action-count distance',age:'Age distance · years',length_of_stay_days:'Stay-length distance · days',creatinine_mg_dl:'Creatinine distance · mg/dL',glucose_mg_dl:'Glucose distance · mg/dL',hemoglobin_g_dl:'Hemoglobin distance · g/dL'};
  for(const [field,value] of Object.entries(a.fidelity_vs_training.numeric))add(names[field]||field,b?.fidelity_vs_training?.numeric?.[field]?.wasserstein,value.wasserstein);
  add('Training request reuse',b?.training_request_match_rate,a.training_request_match_rate,pct);
  add(meta.kind==='workflow'?'Training workflow-label reuse':'Training reply reuse',b?.training_reply_match_rate,a.training_reply_match_rate,pct);
  add('Whole-record training matches',b?.exact_training_record_match_rate,a.exact_training_record_match_rate,pct);
  if(a.prefix_version_flags!=null)add('Diagnosis/version prefix flags',b?.prefix_version_flags,a.prefix_version_flags,v=>v==null?'—':String(v));
  if(a.fidelity_vs_training.spearman_correlations)add('Numeric correlation gap',b?.fidelity_vs_training?.spearman_correlations?.mean_absolute_difference,a.fidelity_vs_training.spearman_correlations.mean_absolute_difference);
  const triple=v=>v?['min','median','max'].map(k=>v[k]==null?'—':Number(v[k]).toFixed(2)).join(' / '):'—';
  const ranges='<details class="development-ranges"><summary>Inspect ranges, medians and missing values</summary><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>FIELD</th><th>SOURCE MIN / MEDIAN / MAX</th><th>SELECTED MIN / MEDIAN / MAX</th><th>MISSING SOURCE / SELECTED</th><th>OUTSIDE SOURCE RANGE</th></tr></thead><tbody>'+Object.entries(a.fidelity_vs_training.numeric).map(([f,v])=>`<tr><td>${esc(names[f]||f)}</td><td>${esc(triple(v.reference))}</td><td>${esc(triple(v.generated))}</td><td>${pct(v.reference_missing_rate)} / ${pct(v.generated_missing_rate)}</td><td>${pct(v.generated_outside_reference_range_rate)}</td></tr>`).join('')+'</tbody></table></div><p class="inside-note">Schema-valid rows only. Source ranges are observed training values, not clinical safety limits. Staying inside them does not guarantee matching distributions.</p></details>';
  let html=`<div class="development-heading"><div class="eyebrow">LATEST DEVELOPMENT EVIDENCE</div><h2>Closer to the source.<br><em>With the tradeoffs visible.</em></h2><p>${esc(data.policy)}</p><span class="subtle-pill">Frozen cohort · ${a.attempted} attempts</span></div>
  <div class="development-notice">Saved development results. Live generation still uses the original saved defaults. Supplied category or diagnosis conditions are external controls; matching them is not proof that Gemma learned their frequencies.</div>
  <div class="evaluation-stats"><div class="card eval-stat"><div class="eyebrow">RECORD STRUCTURE</div><strong>${a.structure_valid}/${a.attempted}</strong><p>All attempts remain in the evaluation</p></div><div class="card eval-stat"><div class="eyebrow">SOURCE LABEL CHECKS</div><strong>${a.source_constraint_valid}/${a.attempted}</strong><p>Selected labels and basic constraints; not every action or fact</p></div><div class="card eval-stat orange"><div class="eyebrow">WHOLE-RECORD REUSE</div><strong>${pct(a.exact_training_record_match_rate)}</strong><p>Normalized matches among valid records</p></div></div>
  <div class="card comparison-card"><div class="card-title"><h3>Does the distribution move closer?</h3><span class="tiny-label">LOWER DISTANCE IS CLOSER</span></div><p class="inside-note">Baseline: ${esc(data.baseline_label||'No new comparison; original assessment retained')}. Both columns compare their cohort with the complete training split. These are descriptive distances, not accuracy percentages.</p><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>MEASURE</th><th>BASELINE</th><th>SELECTED COHORT</th></tr></thead><tbody>${rows.map(r=>'<tr>'+r.map(v=>`<td>${esc(v)}</td>`).join('')+'</tr>').join('')}</tbody></table></div>${ranges}<div class="development-notes">${data.notes.map(n=>`<p>↳ ${esc(n)}</p>`).join('')}</div></div>`;
  if(data.review){
    const r=data.review,d=data.workflow_diagnostics;
    html+=`<div class="card info-card development-review"><div class="eyebrow">ALL ${r.n} SCENARIOS · ONE AI REVIEWER</div><h3>Do the actions fit the situation?</h3><div class="review-counts"><div><strong>${r.counts.Pass||0}</strong><span>Pass</span></div><div><strong>${r.counts.Issue||0}</strong><span>Issue</span></div><div><strong>${r.counts.Unsure||0}</strong><span>Unsure</span></div></div><p>${esc(r.limits)}</p><p>${d.count_outside_group_support} records have action counts outside their observed training subgroup; ${d.records_with_unseen_actions} contain a new action label. These flags do not automatically mean the actions are wrong.</p><details><summary>Read all 64 observations</summary><div class="dataset-table-wrap"><table class="dataset-table"><thead><tr><th>SAMPLE</th><th>VERDICT</th><th>OBSERVATION</th></tr></thead><tbody>${r.judgments.map(j=>`<tr><td>${j.sample_id}</td><td>${esc(j.verdict)}</td><td>${esc(j.reason)}</td></tr>`).join('')}</tbody></table></div></details></div>`;
  }
  html+=`<div class="card development-example"><div class="card-title"><h3>Inspect this cohort</h3><span id="development-position" class="tiny-label"></span><button id="development-prev" class="secondary-button">← Previous</button><button id="development-next" class="secondary-button">Next →</button>${data.flagged_indices?.length?'<button id="development-flagged" class="secondary-button">Show a flagged example ↗</button>':''}</div><p class="inside-note">Every schema-valid record is available here, including flagged records. Compare its style with the training split in Dataset explorer.</p><div id="development-record">Loading saved example…</div></div><div class="development-links">${['bitext','reliability'].includes(selectedDataset)?'<a href="/evidence/distributions" target="_blank" rel="noreferrer">Explore distribution charts ↗</a>':''}</div>`;
  if(data.final_trial)html+='<div class="card info-card advanced-finding"><div class="eyebrow">FINAL BOUNDED ATTEMPT · REJECTED</div><h3>A tested change we did not keep.</h3><p>A consistency instruction was tested on four flagged scenarios and eight passing controls, with fixed inputs. The baseline produced 12/12 valid workflows; the candidate produced 11/12. It still repeated an already-completed payment and worsened action-count similarity. No flagged case became a Pass, and one passing control became uncertain.</p><p>The existing model and generation policy remain unchanged. This selected-case AI review does not estimate population accuracy. Experiments are frozen for submission.</p></div>';
  section.innerHTML=html;
  $('#development-prev').addEventListener('click',()=>loadDevelopmentExample(developmentIndex-1));
  $('#development-next').addEventListener('click',()=>loadDevelopmentExample(developmentIndex+1));
  if($('#development-flagged'))$('#development-flagged').addEventListener('click',()=>loadDevelopmentExample(data.flagged_indices.find(i=>i>developmentIndex)??data.flagged_indices[0]));
  loadDevelopmentExample(0);
}

async function loadDevelopmentExample(index) {
  const revision=datasetRevision,signature=developmentSignature;
  try {
    const result=await api('/api/development/saved?index='+index);
    if(revision!==datasetRevision||signature!==developmentSignature)return;
    developmentIndex=result.index;
    $('#development-position').textContent=`${result.index+1} / ${result.total} · sample ${result.sample_id}`;
    $('#development-record').innerHTML=recordHTML(result.record)+(result.source_warnings.length?'<p class="result-note">Source-value flags: '+esc(result.source_warnings.join('; '))+'</p>':'')+
      (result.whole_record_match?'<p class="result-note">This whole record matches normalized training content. Similarity does not establish novelty.</p>':'')+
      (result.review?'<p class="result-note">AI review · '+esc(result.review.verdict)+': '+esc(result.review.reason)+'</p>':'')+
      '<details><summary>Full JSON record</summary><pre class="json-output">'+esc(JSON.stringify(result.record,null,2))+'</pre></details>';
  }catch(error){if(revision===datasetRevision)$('#development-record').textContent=error.message;}
}

function renderAdvancedMetrics(data) {
  let section=$('#advanced-evaluation');
  if(!section){section=document.createElement('section');section.id='advanced-evaluation';section.className='advanced-evaluation';$('#page-evaluation').append(section);}
  section.hidden=!data||!Object.keys(data).length;if(section.hidden)return;
  const number=v=>Number(v).toFixed(3);
  let html='<div class="section-heading"><div><h2>Original assessment: deeper checks</h2><p>These earlier cohorts are separate from the latest development comparison above.</p></div></div>';
  if(data.review){
    const r=data.review;
    html+=`<div class="card info-card advanced-review"><div class="eyebrow">RELEVANCE &amp; COHERENCE · ${r.n} STRATIFIED EXAMPLES</div><h3>Do the fields tell the same story?</h3><div class="review-counts"><div><strong>${r.counts.Pass}</strong><span>Pass</span></div><div><strong>${r.counts.Issue}</strong><span>Issue</span></div><div><strong>${r.counts.Unsure}</strong><span>Unsure</span></div></div><p>One AI reviewer checked request/answer alignment, labels and internal consistency. Pass means relevant and coherent within this rubric; insurance facts are not certified. This stratified sample is not an accuracy estimate, and the judge has no human calibration.</p><details><summary>Read every judgment</summary><div class="dataset-table-wrap"><table class="dataset-table"><thead><tr><th>REVIEW / SAMPLE</th><th>VERDICT</th><th>REASON</th></tr></thead><tbody>${r.judgments.map(j=>`<tr><td>${esc(j.review_id)} / ${esc(j.sample_id)}</td><td>${esc(j.verdict)}</td><td>${esc(j.reason)}</td></tr>`).join('')}</tbody></table></div></details></div>`;
  }
  if(data.prdc){
    const table=k=>'<div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>FIELD / COLLECTION</th><th>Precision ↑</th><th>Recall ↑</th><th>Density</th><th>Coverage ↑</th></tr></thead><tbody>'+Object.entries(data.prdc.fields).flatMap(([field,v])=>['base','tuned','validation_control'].map(label=>`<tr><td>${esc(field)} · ${label==='base'?'Original':label==='tuned'?'Fine-tuned':'Source validation control'} (n=${v.n})</td>${['precision','recall','density','coverage'].map(metric=>`<td>${number(v.scores[label][k][metric])}</td>`).join('')}</tr>`)).join('')+'</tbody></table></div>';
    html+='<div class="card comparison-card"><div class="card-title"><h3>Resemblance and variety in text embeddings</h3><span class="tiny-label">EXPERIMENTAL PRDC</span></div>'+table('5')+'<p class="inside-note">Precision checks whether generated examples sit near source examples; recall and coverage check how much source variety is represented. Density describes concentration and can exceed 1. Default neighborhood: 5. Small equal-size samples, structurally valid outputs only. High scores do not establish factual correctness or privacy.</p><details><summary>Check neighborhood sensitivity: 10</summary>'+table('10')+'</details></div>';
  }
  if(data.sdmetrics){
    const collections=data.sdmetrics.collections,tuned=collections.tuned;
    html+='<div class="card comparison-card"><div class="card-title"><h3>Clinical table resemblance</h3><span class="tiny-label">SDMETRICS · DEMO ONLY</span></div><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>COLLECTION</th><th>Records</th><th>Statistical quality ↑</th></tr></thead><tbody>'+Object.entries(collections).map(([label,v])=>`<tr><td>${label==='base'?'Original':label==='tuned'?'Fine-tuned':'Source validation control'}</td><td>${v.n}</td><td>${number(v.overall_quality)}</td></tr>`).join('')+'</tbody></table></div><p class="inside-note">Combines column-shape and column-pair similarity. Not clinical accuracy. The control has only 21 admissions, so it is not sample-size matched. Source: 230 training admissions from 80 patients.</p><details open><summary>Missing labs and observed-range coverage</summary><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>FIELD</th><th>Source missing</th><th>Generated missing</th><th>Range coverage</th></tr></thead><tbody>'+['creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl'].map(field=>{const v=tuned.per_column[field];return `<tr><td>${esc(field.replaceAll('_',' '))}</td><td>${(100*v.training_missing_rate).toFixed(1)}%</td><td>${(100*v.generated_missing_rate).toFixed(1)}%</td><td>${number(v.range_coverage)}</td></tr>`;}).join('')+'</tbody></table></div></details></div>';
  }
  if(data.icd){const r=data.icd.tuned;html+=`<div class="card info-card advanced-finding"><div class="eyebrow">EXPLORATORY FAILURE CHECK</div><h3>${r.flagged} / ${r.n} diagnosis/version pairs flagged.</h3><p>Some generated diagnoses start with a character incompatible with their stated ICD version. The same screen flagged 0 / ${data.icd.training_reference.n} training records. This is a narrow prefix check, not full code validation. It shows why correct JSON and similar averages cannot certify a clinical record.</p></div>`;}
  if(data.factual_spot_checks?.length){html+='<div class="card info-card advanced-finding"><div class="eyebrow">TARGETED FACTUAL SPOT-CHECKS</div><h3>Two selected claims conflict with official sources.</h3><p>These examples were checked because the review raised concerns; they are not a factual error-rate estimate.</p>'+data.factual_spot_checks.map(item=>`<p><strong>${esc(item.review_id)} · sample ${esc(item.sample_id)}</strong><br>${esc(item.claim_summary)}<br>${esc(item.evidence_summary)} <a href="${esc(item.source)}" target="_blank" rel="noreferrer">Official evidence ↗</a></p>`).join('')+'</div>';}
  section.innerHTML=html;
}

async function init() {
  try {
    datasetCatalog=await api('/api/catalog');
    $('#dataset-selector').innerHTML=Object.entries(datasetCatalog).map(([key,value])=>`<option value="${key}">${esc(value.label)}</option>`).join('');
    $('#dataset-selector').addEventListener('change',()=>switchDataset($('#dataset-selector').value));
    await switchDataset('bitext'); page(location.hash.slice(1)||'playground');
    refreshStatus();setInterval(refreshStatus,5000);setInterval(()=>{if($('#page-evaluation').classList.contains('active'))refreshMetrics();},10000);
  } catch(error) { notify('Could not load the local workspace: '+error.message); }
}
init();

function fieldsHTML(record) {
  return '<dl class="record-fields">'+Object.entries(record).map(([key,value])=>`<div><dt>${esc(key.replaceAll('_',' '))}</dt><dd>${esc(value===null?'Not recorded':Array.isArray(value)?value.join(', ')||'None':typeof value==='object'?JSON.stringify(value):String(value))}</dd></div>`).join('')+'</dl>';
}
function groupValue(record) {return String(record[meta?.group_field]??'Record');}
function recordPreview(record) {
  if(meta?.kind==='qa')return record.question;
  if(meta?.kind==='workflow')return record.input.customer_message;
  if(meta?.kind==='clinical')return `${record.gender} · Age ${record.age_topcoded?'90+':record.age} · ${record.admission_type} · Stay ${record.length_of_stay_days} days · ${record.insurance}`;
  return record.instruction;
}
async function refreshMetrics() {
  const revision=datasetRevision;
  try {const result=await api('/api/metrics');if(revision===datasetRevision){evaluation=result;renderMetrics(result);}}
  catch(error){notify(error.message);}
}
async function switchDataset(key) {
  const revision=++datasetRevision;selectedDataset=key;current=null;datasetOffset=0;savedIndex=0;jsonView=false;
  $('#dataset-selector').value=key;$('#json-toggle').classList.remove('selected');$('#prompt').value='Generate synthetic data.';promptCount();
  $('#generate').disabled=true;$('#load-model').disabled=true;
  $('#dataset-search').value='';$('#output-body').innerHTML='<div class="empty-state">Loading this workspace…</div>';$('#job-message').textContent='';
  try {
    const result=await api('/api/meta');if(revision!==datasetRevision)return;meta=result;
    $('#source-scope').textContent=meta.scope;
    $('.stat-strip>div:nth-child(1) strong').textContent=meta.training_rows.toLocaleString();
    $('.stat-strip>div:nth-child(2) strong').textContent=meta.groups.length+' '+(meta.group_field==='intent'?'intents':'groups');
    $('.stat-strip>div:nth-child(2) small').textContent=meta.kind==='clinical'?'100 patients · disjoint splits':meta.group_field.replaceAll('_',' ')+' distribution';
    $('.stat-strip>div:nth-child(3) strong').textContent=meta.epochs===1?'One full pass':'Three training passes';
    $('.stat-strip>div:nth-child(3) small').textContent='Selected '+meta.checkpoint.replace('checkpoint-','checkpoint · ');
    $('.diagram-node small').textContent=meta.label;
    $('.subtle-pill').textContent='✦ '+meta.label+' adapter';
    $('.sidebar-bottom>div>span').textContent=meta.kind==='clinical'?'Clinical record generation':meta.kind==='workflow'?'Workflow scenario generation':'Insurance generation';
    $('.diagram-footer').textContent='Fine-tuned for '+meta.label;
    $('.section-heading>div>p').textContent=meta.kind==='clinical'?'Create a new structured hospital-admission record.':meta.kind==='workflow'?'Create a scenario with context and expected workflow actions.':'Your prompt in. A new question-and-answer record out.';
    $('.composer-foot').textContent=meta.kind==='clinical'?'Structured admissions from the 100-patient demo.':meta.kind==='workflow'?'One scenario. Its context. Expected workflow labels.':'One request. One reply. One complete example.';
    const suggestions=meta.kind==='clinical'?['Generate a fictional hospital admission record.','Generate a record with missing laboratory values.','Generate an emergency admission example.']:meta.kind==='workflow'?['Generate a new claim intake scenario.','Generate a scenario requiring human handoff.','Generate a payment-processing scenario.']:meta.kind==='qa'?['Generate a new question and answer about life insurance.','Generate a new question and answer about health insurance.','Generate a new question and answer about auto insurance.']:['Generate a new example about checking insurance coverage.','Generate a new example about reporting an insurance claim.','Generate a new example about renewing an insurance policy.'];
    const names=meta.kind==='clinical'?['New admission ↗','Missing labs ↗','Emergency ↗']:meta.kind==='workflow'?['Claim intake ↗','Human handoff ↗','Payments ↗']:meta.kind==='qa'?['Life insurance ↗','Health insurance ↗','Auto insurance ↗']:['Check coverage ↗','File a claim ↗','Renew a policy ↗'];
    document.querySelectorAll('[data-suggestion]').forEach((b,i)=>{b.dataset.suggestion=suggestions[i];b.textContent=names[i];});
    $('label[for="intent"]').innerHTML='Focus on '+esc(meta.group_field.replaceAll('_',' '))+' <span>Optional</span>';
    const options=meta.groups.map(v=>`<option value="${esc(v)}">${esc(v.replaceAll('_',' '))}</option>`).join('');
    $('#intent').innerHTML='<option value="">Let the model choose</option>'+options;
    $('#dataset-intent').innerHTML='<option value="">All '+meta.groups.length+' groups</option>'+options;
    $('#dataset-intent').setAttribute('aria-label','Filter by '+meta.group_field.replaceAll('_',' '));
    $('#page-dataset .page-intro p').textContent=`Browse ${meta.training_rows.toLocaleString()} source records from the ${meta.label} training split.`;
    $('#page-evaluation .page-intro p').textContent='Latest development evidence, followed by the original frozen assessments. Compare distribution matching, copying and visible failure cases.';
    document.querySelectorAll('.dataset-table th')[0].textContent='SOURCE RECORD';document.querySelectorAll('.dataset-table th')[1].textContent=meta.kind==='clinical'?'ADMISSION PREVIEW':'CUSTOMER REQUEST';document.querySelectorAll('.dataset-table th')[2].textContent=meta.group_field.replaceAll('_',' ').toUpperCase();
    $('.dataset-bottom .info-card h3').textContent=meta.fields.length+' fields. One coherent record.';
    $('.dataset-bottom .info-card p').textContent='The generator learns this record structure and its source distributions.';
    $('.field-list').innerHTML=meta.fields.map(f=>'<code>'+esc(f)+'</code>').join('');
    document.querySelectorAll('.dataset-bottom .info-card p')[1].textContent=meta.scope;
    $('.dataset-bottom .info-card a').href=({bitext:'https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset',insuranceqa:'https://github.com/shuzi/insuranceQA/tree/master/V2',reliability:'https://huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark',mimic_demo:'https://physionet.org/content/mimic-iv-demo/2.2/'})[key];
    $('.reference-section>.footnote').textContent=meta.kind==='clinical'?'References use equally weighted field similarities, with numeric differences scaled by training ranges. Similarity is not clinical correctness or causal attribution.':'References use shared wording to find nearby training requests. Similarity is not correctness or causal attribution.';
    $('.advanced p').textContent='The default prompt matches this dataset’s assessment. Live generation uses one record at a time; custom prompts and focus are exploratory.';
    await refreshMetrics();if(revision!==datasetRevision)return;
    await loadSaved(key==='bitext'?12:0);if(revision!==datasetRevision)return;
    await refreshStatus();
    if($('#page-dataset').classList.contains('active'))loadDataset();
  } catch(error){notify('Could not load dataset: '+error.message);}
}
function renderOtherMetrics(result) {
  const section=$('#other-evaluation');
  if(!result.labels) {
    $('#evaluation-stats').innerHTML='<div class="card info-card"><h3>Assessment in progress</h3><p>'+esc(result.progress?.phase||'Queued')+' · '+(result.progress?.completed||0)+' / '+(result.progress?.total||128)+'</p></div>';
    section.innerHTML='<div class="card info-card"><h3>Measured results will appear here.</h3><p>The GPU is generating frozen original-model and fine-tuned-model outputs. No scores are being estimated or filled in. This page refreshes while open.</p></div>';return;
  }
  const base=result.labels.base,tuned=result.labels.tuned;
  const percent=v=>v===null||v===undefined?'—':(100*v).toFixed(1)+'%';
  const number=v=>v===null||v===undefined?'—':Number(v).toFixed(3);
  const group=meta.group_field;
  $('#evaluation-stats').innerHTML=`<div class="card eval-stat"><div class="eyebrow">STRUCTURE</div><strong>${tuned.structure_valid}/${tuned.attempted}</strong><p>Structurally valid fine-tuned outputs</p></div><div class="card eval-stat"><div class="eyebrow">TOPIC DISTRIBUTION</div><strong>${number(tuned.fidelity_vs_training.categorical[group]?.total_variation)}</strong><p>Total variation · lower is closer</p></div><div class="card eval-stat orange"><div class="eyebrow">SOURCE REPRODUCTION</div><strong>${percent(tuned.exact_training_record_match_rate)}</strong><p>Full-record matches among valid outputs</p></div>`;
  const rows=[['Structurally valid records',`${base.structure_valid}/${base.attempted}`,`${tuned.structure_valid}/${tuned.attempted}`],['Source constraints satisfied',`${base.source_constraint_valid}/${base.attempted}`,`${tuned.source_constraint_valid}/${tuned.attempted}`],['Unique record rate',percent(base.unique_record_rate),percent(tuned.unique_record_rate)],['Full-record copying',percent(base.exact_training_record_match_rate),percent(tuned.exact_training_record_match_rate)],['Request copying',percent(base.training_request_match_rate),percent(tuned.training_request_match_rate)],[meta.kind==='workflow'?'Workflow-label copying':'Reply copying',percent(base.training_reply_match_rate),percent(tuned.training_reply_match_rate)]];
  for(const [name,values] of Object.entries(tuned.fidelity_vs_training.categorical))rows.push([name.replaceAll('_',' ')+' distance ↓',number(base.fidelity_vs_training.categorical[name]?.total_variation),number(values.total_variation)]);
  let html='<div class="card comparison-card"><div class="card-title"><h3>Original vs. fine-tuned</h3><span class="tiny-label">128 ATTEMPTS EACH</span></div><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>MEASURE</th><th>Original</th><th>Fine-tuned</th></tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>`<td>${esc(v)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div><p class="inside-note">Distribution and copying measures use structurally valid records. Failed attempts stay in the denominator for validity. Structure and vocabulary constraints do not establish semantic correctness.</p></div>';
  const numeric=Object.entries(tuned.fidelity_vs_training.numeric);
  if(numeric.length)html+='<div class="card comparison-card"><div class="card-title"><h3>Ranges, lengths and missingness</h3></div><div class="dataset-table-wrap"><table class="comparison-table"><thead><tr><th>FIELD</th><th>Generated range</th><th>Training range</th><th>Wasserstein ↓</th><th>Missing: generated / source</th></tr></thead><tbody>'+numeric.map(([name,v])=>`<tr><td>${esc(name.replaceAll('_',' '))}</td><td>${v.generated?esc(v.generated.min+'–'+v.generated.max):'—'}</td><td>${v.reference?esc(v.reference.min+'–'+v.reference.max):'—'}</td><td>${number(v.wasserstein)}</td><td>${percent(v.generated_missing_rate)} / ${percent(v.reference_missing_rate)}</td></tr>`).join('')+'</tbody></table></div><p class="inside-note">Distances retain each field’s units. Text lengths count characters. Source ranges are observations, not clinical safety limits.</p></div>';
  html+='<div class="card test-card"><div class="eyebrow">FROZEN HELD-OUT CHECK</div><h2>'+ (result.test?'Predictions on reserved records.':'Held-out scoring is running.')+'</h2>'+(result.test?`<div class="test-stat"><span>Records scored</span><strong>${result.test.records}</strong></div><div class="test-stat"><span>Mean completion loss</span><strong>${number(result.test.mean_record_loss)}</strong></div><div class="test-stat"><span>Token-weighted perplexity</span><strong>${number(result.test.perplexity)}</strong></div>`:'<p>Generation settings and outputs are frozen before test scoring. Results appear when scoring finishes.</p>')+'<p>'+esc(meta.scope)+'</p></div>';
  html+='<p class="footnote">Exploratory 128-attempt comparison. No universal pass threshold or accuracy percentage is claimed. Semantic quality review remains separate.</p>';
  section.innerHTML=html;
}
