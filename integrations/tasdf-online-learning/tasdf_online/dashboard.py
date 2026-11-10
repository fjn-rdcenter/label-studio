"""Standalone and toggleable debug/monitoring dashboard for TASDF online learning.

Can be mounted into the ML backend (`DASHBOARD_ENABLED=1`) and/or run as an independent
process (`python -m tasdf_online.dashboard`) pointing at `BACKEND_URL`.
"""
from __future__ import annotations

import base64
import os
from typing import Callable

import cv2
import numpy as np
import requests
import torch
from flask import Blueprint, Flask, Response, jsonify, request

# BGR palette matching label-config.xml: BG=dark, ZP=blue, TE=green, ICM=red, BC=amber
PALETTE = np.array([[20, 22, 28], [246, 130, 59], [94, 197, 34], [68, 68, 239], [11, 158, 245]], dtype=np.uint8)


def _png_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("failed to encode PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _colorize_mask(gray_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    color = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
    fg = (mask > 0)[..., None]
    return np.where(fg, (0.45 * base + 0.55 * color).astype(np.uint8), base)


def _correction_bgr(gray_u8: np.ndarray, ref_mask: np.ndarray, user_mask: np.ndarray) -> np.ndarray:
    out = (cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR) * 0.45).astype(np.uint8)
    added = (ref_mask == 0) & (user_mask > 0)
    removed = (ref_mask > 0) & (user_mask == 0)
    reassigned = (ref_mask > 0) & (user_mask > 0) & (ref_mask != user_mask)
    out[added] = [94, 210, 40]       # green: added foreground
    out[removed] = [60, 60, 235]     # red: removed foreground
    out[reassigned] = [20, 210, 250] # yellow: class reassigned
    return out


def render_visuals(
    img: np.ndarray,
    ref_mask: np.ndarray,
    before_probs: torch.Tensor,
    after_probs: torch.Tensor,
    topo_after: torch.Tensor,
    user_mask: np.ndarray,
) -> dict[str, str | float]:
    """Render 5 debug PNG data-URIs (256x256) and ITGM skeleton stats for the latest episode."""
    g8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    pb = before_probs.argmax(0).numpy().astype(np.uint8)
    pa = after_probs.argmax(0).numpy().astype(np.uint8)
    diff = (0.5 * (after_probs - before_probs).abs().sum(0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    topo_np = topo_after.numpy()
    skel = topo_np.max(0) > 0.30
    heat[skel] = [255, 255, 0]  # cyan skeleton ridge overlay
    return {
        "input_overlay": _png_uri(_colorize_mask(g8, user_mask)),
        "correction_map": _png_uri(_correction_bgr(g8, ref_mask, user_mask)),
        "pred_before": _png_uri(_colorize_mask(g8, pb)),
        "pred_after": _png_uri(_colorize_mask(g8, pa)),
        "prob_diff_heatmap": _png_uri(heat),
        "topo_mean": float(topo_np.mean()),
        "topo_p99": float(np.percentile(topo_np, 99)),
    }


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TASDF Online Learning Debug Monitor</title>
<style>
body{margin:0;padding:16px;background:#0f172a;color:#e2e8f0;font:13px/1.45 ui-monospace,SFMono-Regular,monospace}
h1{font-size:16px;margin:0 0 8px;color:#38bdf8} h2{font-size:13px;margin:0 0 8px;color:#94a3b8;text-transform:uppercase}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#1e293b;padding:10px;border-radius:6px;margin-bottom:12px}
input,button{background:#0f172a;color:#e2e8f0;border:1px solid #334155;padding:5px 9px;border-radius:4px;font:inherit}
button{cursor:pointer;background:#0284c7;border-color:#0369a1} button.warn{background:#b91c1c;border-color:#991b1b}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:12px}
.card{background:#1e293b;border:1px solid #334155;border-radius:6px;padding:12px}
.imgs{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;text-align:center}
.imgs img{width:100%;image-rendering:pixelated;border:1px solid #475569;border-radius:4px;background:#000}
table{width:100%;border-collapse:collapse} th,td{padding:4px 6px;border-bottom:1px solid #334155;text-align:left}
.pill{padding:2px 6px;border-radius:4px;background:#334155} .ok{color:#4ade80} .bad{color:#f87171}
</style></head><body>
<div class="bar">
  <h1>TASDF Online Learning — Standalone Debug Monitor</h1>
  <span id="status" class="pill">connecting...</span>
  <input id="tok" type="password" placeholder="X-OL-Token" style="width:140px">
  <button onclick="saveTok()">Connect</button>
  <button onclick="act('/refresh_ls_predictions',{})">Refresh LS</button>
  <button class="warn" onclick="act('/rollback',{version:'v0'})">Rollback v0</button>
  <button onclick="toggleCapture()">Toggle Capture</button>
</div>
<div class="card" style="margin-bottom:12px">
  <h2>1. Spatial Absorption Probe (Latest Episode)</h2>
  <div id="probe_meta" style="margin-bottom:8px;color:#94a3b8">No episode recorded yet.</div>
  <div class="imgs">
    <div><img id="im_in"><br>1. Input + User Mask</div>
    <div><img id="im_corr"><br>2. Correction M_Δ</div>
    <div><img id="im_bef"><br>3. Pred Before</div>
    <div><img id="im_aft"><br>4. Pred After</div>
    <div><img id="im_diff"><br>5. ΔP Heatmap + ITGM</div>
  </div>
</div>
<div class="grid">
  <div class="card"><h2>2. 6-Group Layer Weight Shifts</h2><table id="shifts"></table></div>
  <div class="card"><h2>3. Version Lineage & Holdout mDice (v0 → vt)</h2><table id="vers"></table></div>
</div>
<script>
const tokEl=document.getElementById('tok'); tokEl.value=sessionStorage.getItem('ol_tok')||'';
let capOn=true;
function hdr(){return {'X-OL-Token':tokEl.value,'Content-Type':'application/json'};}
function saveTok(){sessionStorage.setItem('ol_tok',tokEl.value);poll();}
async function act(p,b){await fetch(p,{method:'POST',headers:hdr(),body:JSON.stringify(b)});poll();}
async function toggleCapture(){await act('/dashboard/toggle',{enabled:!capOn});}
async function poll(){
  const r=await fetch('/telemetry/live',{headers:hdr()});
  const st=document.getElementById('status');
  if(!r.ok){st.textContent='HTTP '+r.status+' (set X-OL-Token)';st.className='pill bad';return;}
  const d=await r.json(); capOn=d.dashboard_enabled!==false;
  st.textContent=`active=${d.active} | ${d.device} | status=${d.status} | queue=${d.queue} | capture=${capOn?'ON':'OFF'}`;
  st.className='pill ok';
  const ep=(d.episodes||[]).slice(-1)[0];
  if(ep){
    const lp=ep.local_probe||{};
    document.getElementById('probe_meta').textContent=
      `Ep ${ep.version} (task #${ep.task_id}, ${ep.seconds}s) | Gate: ${ep.accepted?'ACCEPTED':'REJECTED'} (${(ep.gate_reasons||[]).join(', ')||'ok'}) | `+
      `Local Dice: ${(lp.dice_before||0).toFixed(4)} → ${(lp.dice_after||0).toFixed(4)} | AR_Δ: ${(lp.absorption_rate_pct||0).toFixed(1)}% (${lp.absorbed_pixels||0}/${lp.corrected_pixels||0} px)`;
    const sh=ep.layer_shifts||{};
    let rows='<tr><th>Group</th><th>||ΔW||₂</th><th>Rel %</th><th>Drift v0</th></tr>';
    for(const [g,v] of Object.entries(sh)){
      if(typeof v==='object') rows+=`<tr><td>${g}</td><td>${v.l2_shift.toFixed(6)}</td><td>${v.rel_shift_pct.toFixed(4)}%</td><td>${v.drift_from_v0.toFixed(6)}</td></tr>`;
    }
    rows+=`<tr><td colspan="2">PixelHead Cosine Drift vs v0</td><td colspan="2">${(sh.pixel_head_cosine_drift||0).toFixed(6)}</td></tr>`;
    document.getElementById('shifts').innerHTML=rows;
  }
  const v=d.last_visuals||{};
  if(v.input_overlay){
    document.getElementById('im_in').src=v.input_overlay; document.getElementById('im_corr').src=v.correction_map;
    document.getElementById('im_bef').src=v.pred_before; document.getElementById('im_aft').src=v.pred_after;
    document.getElementById('im_diff').src=v.prob_diff_heatmap;
  }
  let vr='<tr><th>Ver</th><th>Status</th><th>mDice</th><th>ZP</th><th>TE</th><th>ICM</th><th>BC</th></tr>';
  for(const [k,info] of Object.entries(d.version_history||{})){
    const m=info.metrics||{}, dc=m.dice||{};
    vr+=`<tr><td>${k}</td><td>${info.status}</td><td>${(m.pooled_mDice||0).toFixed(4)}</td>`+
        `<td>${(dc[1]||0).toFixed(3)}</td><td>${(dc[2]||0).toFixed(3)}</td><td>${(dc[3]||0).toFixed(3)}</td><td>${(dc[4]||0).toFixed(3)}</td></tr>`;
  }
  document.getElementById('vers').innerHTML=vr;
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
        if not svc.dashboard_enabled:
            svc.telemetry.pop("last_visuals", None)
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
