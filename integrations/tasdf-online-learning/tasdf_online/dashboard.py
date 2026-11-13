"""Standalone and toggleable debug/monitoring dashboard for TASDF batch retraining.

Can be mounted into the ML backend (`DASHBOARD_ENABLED=1`) and/or run as an independent
process (`python -m tasdf_online.dashboard`) pointing at `BACKEND_URL`.
"""
from __future__ import annotations

import os
from typing import Callable

import requests
from flask import Blueprint, Flask, Response, jsonify, request


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TASDF Training Monitor</title>
<style>
body{margin:0;padding:16px;background:#0f172a;color:#e2e8f0;font:13px/1.45 ui-monospace,SFMono-Regular,monospace}
h1{font-size:16px;margin:0 0 8px;color:#38bdf8} h2{font-size:13px;margin:0 0 8px;color:#94a3b8;text-transform:uppercase}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#1e293b;padding:10px;border-radius:6px;margin-bottom:12px}
input,button{background:#0f172a;color:#e2e8f0;border:1px solid #334155;padding:5px 9px;border-radius:4px;font:inherit}
button{cursor:pointer;background:#0284c7;border-color:#0369a1} button.warn{background:#b91c1c;border-color:#991b1b}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:12px}
.card{background:#1e293b;border:1px solid #334155;border-radius:6px;padding:12px}
table{width:100%;border-collapse:collapse} th,td{padding:4px 6px;border-bottom:1px solid #334155;text-align:left}
.pill{padding:2px 6px;border-radius:4px;background:#334155} .ok{color:#4ade80} .bad{color:#f87171}
</style></head><body>
<div class="bar">
  <h1>TASDF Training Monitor</h1>
  <span id="status" class="pill">connecting...</span>
  <input id="tok" type="password" placeholder="X-OL-Token" style="width:140px">
  <button onclick="saveTok()">Connect</button>
  <button onclick="act('/train',{force:true})">Train now</button>
  <button onclick="act('/train',{force:true,dry_run:true})">Dry run (1 epoch)</button>
  <button onclick="act('/refresh_ls_predictions',{})">Refresh LS</button>
  <button class="warn" onclick="act('/rollback',{})">Rollback to previous</button>
  <button onclick="toggleCapture()">Toggle Capture</button>
</div>
<div class="card" style="margin-bottom:12px">
  <h2>1. Latest long-term job</h2>
  <div id="job_meta" style="color:#94a3b8">No training job yet.</div>
  <div id="job_loss" style="margin-top:6px;color:#94a3b8"></div>
</div>
<div class="grid">
  <div class="card"><h2>2. Bundles (production / previous / candidates)</h2><table id="vers"></table></div>
  <div class="card"><h2>3. Held-out val Dice per job (is it really improving?)</h2><table id="trend"></table></div>
</div>
<script>
const tokEl=document.getElementById('tok'); tokEl.value=sessionStorage.getItem('ol_tok')||'';
let capOn=true;
function hdr(){return {'X-OL-Token':tokEl.value,'Content-Type':'application/json'};}
function saveTok(){sessionStorage.setItem('ol_tok',tokEl.value);poll();}
async function act(p,b){await fetch(p,{method:'POST',headers:hdr(),body:JSON.stringify(b)});poll();}
async function toggleCapture(){await act('/dashboard/toggle',{enabled:!capOn});}
function n(v,d=4){return (v===null||v===undefined||Number.isNaN(v))?'-':Number(v).toFixed(d);}
async function poll(){
  const r=await fetch('/telemetry/live',{headers:hdr()});
  const st=document.getElementById('status');
  if(!r.ok){st.textContent='HTTP '+r.status+' (set X-OL-Token)';st.className='pill bad';return;}
  const d=await r.json(); capOn=d.dashboard_enabled!==false;
  const p=d.pending||{}, sc=d.schedule||{}, dr=d.last_dry_run;
  st.textContent=`production=${d.production} | previous=${d.previous} | ${d.device} | ${d.status} | training=${d.training} | `+
    `next run ${sc.next_run||'off'} (${sc.cron||'-'}) | new labels ${p.new??'?'}/${p.threshold??'?'} (train ${p.train??'?'}, val ${p.val??'?'})`+
    (dr?` | dry run: ${dr.ms_per_image} ms/img, ${n(dr.peak_vram_mib,0)} MiB`:'')+(d.last_skip?` | postponed: ${d.last_skip.reason}`:'');
  st.className='pill ok';
  if(d.last_error) st.textContent+=` | last_error: ${d.last_error}`;
  const job=(d.jobs||[]).slice(-1)[0];
  if(job){
    const m=(job.metrics||{})[job.stage]||{};
    document.getElementById('job_meta').textContent=
      `Job ${job.job_id} (${job.kind}) -> ${job.bundle} (${job.seconds}s) | ${job.accepted?'PROMOTED':'REJECTED'} (${(job.gate_reasons||[]).join(', ')||'ok'}) | `+
      `train ${job.train_images} img, val ${job.val_images} img | val ${n(m.production_val)} -> ${n(m.val)} | train ${n(m.train)}`+(job.note?` | ${job.note}`:'');
    document.getElementById('job_loss').textContent='epoch: loss / score  '+(job.history||[]).map(h=>`${h.epoch}: ${h.loss}/${h.score??'-'}`).join('  ');
  }
  let vr='<tr><th>Bundle</th><th>Status</th><th>Stage</th><th>D10 mDice</th><th>BC</th><th>val</th><th>train</th></tr>';
  for(const [k,info] of Object.entries(d.bundles||{})){
    const ptr=k===d.production?' <span class="pill ok">production</span>':(k===d.previous?' <span class="pill">previous</span>':'');
    for(const [stg,m] of Object.entries(info.metrics||{})){
      const d10=m.d10||{}, dc=d10.dice||{};
      const gr=(info.gate_reasons||[]).join(', ');
      vr+=`<tr><td>${k}${ptr}</td><td>${info.status}${gr?`<br><small style="color:#f87171">${gr}</small>`:''}</td><td>${stg}</td>`+
          `<td>${n(d10.pooled_mDice)}</td><td>${n(dc[4]??dc['4'],3)}</td><td>${n(m.val)}</td><td>${n(m.train)}</td></tr>`;
    }
  }
  document.getElementById('vers').innerHTML=vr;
  let tr='<tr><th>When</th><th>Bundle</th><th>val images</th><th>production val</th><th>candidate val</th><th>Promoted</th></tr>';
  for(const t of (d.val_trend||[]).slice(-15).reverse()){
    tr+=`<tr><td>${new Date(t.at*1000).toLocaleString()}</td><td>${t.bundle}</td><td>${t.val_images}</td><td>${n(t.production_val)}</td><td>${n(t.val)}</td><td>${t.accepted?'yes':'no'}</td></tr>`;
  }
  document.getElementById('trend').innerHTML=tr;
}
setInterval(poll,2000); poll();
</script></body></html>
"""


def create_dashboard_blueprint(svc, ops_allowed: Callable[[], bool]) -> Blueprint:
    """Create a toggleable debug Blueprint mounted on the ML backend."""
    bp = Blueprint("tasdf_dashboard", __name__)

    @bp.get("/dashboard")
    def dashboard_view():
        if not svc.dashboard_enabled:
            return jsonify(error="dashboard is disabled (POST /dashboard/toggle or set DASHBOARD_ENABLED=0)"), 404
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @bp.post("/dashboard/toggle")
    def dashboard_toggle():
        if not ops_allowed():
            return jsonify(error="unauthorized"), 401
        payload = request.get_json(silent=True) or {}
        svc.dashboard_enabled = bool(payload.get("enabled", not svc.dashboard_enabled))
        return jsonify(dashboard_enabled=svc.dashboard_enabled)

    return bp


def create_standalone_app(backend_url: str) -> Flask:
    """Run the debug dashboard as a completely separate process proxying to `backend_url`."""
    app = Flask(__name__)
    target = backend_url.rstrip("/")

    @app.get("/dashboard")
    def index():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.route("/<path:subpath>", methods=["GET", "POST"])
    def proxy(subpath: str):
        headers = {k: v for k, v in request.headers if k.lower() in {"x-ol-token", "content-type"}}
        resp = requests.request(request.method, f"{target}/{subpath}", headers=headers, data=request.get_data(), timeout=30)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers.get("Content-Type", "application/json"))

    return app


if __name__ == "__main__":
    url = os.getenv("BACKEND_URL", "http://127.0.0.1:9090")
    port = int(os.getenv("DASHBOARD_PORT", "9091"))
    create_standalone_app(url).run(host="0.0.0.0", port=port, threaded=True)
