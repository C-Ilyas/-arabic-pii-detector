"""
app.py
FastAPI app for interactively testing the Arabic PII detector.

"""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TORCH_MODEL = ROOT / "models" / "arabic-pii-detector"
ONNX_MODEL = ROOT / "models" / "arabic-pii-detector-onnx"


def _resolve_backend() -> tuple[str, str]:
    backend = os.getenv("PII_BACKEND", "auto").lower()
    model = os.getenv("PII_MODEL")
    if backend == "torch":
        return "torch", model or str(TORCH_MODEL)
    if backend == "onnx":
        return "onnx", model or str(ONNX_MODEL)
    # auto
    if TORCH_MODEL.exists():
        return "torch", model or str(TORCH_MODEL)
    if ONNX_MODEL.exists():
        return "onnx", model or str(ONNX_MODEL)
    raise FileNotFoundError(
        "No model found. Expected models/arabic-pii-detector (torch) or "
        "models/arabic-pii-detector-onnx. Set PII_MODEL / PII_BACKEND to override."
    )


BACKEND, MODEL_PATH = _resolve_backend()


def _load_detector():
    if BACKEND == "torch":
        from scripts.infer import PIIDetector
        return PIIDetector(MODEL_PATH)
    from scripts.benchmark_onnx import ONNXPIIDetector
    return ONNXPIIDetector(MODEL_PATH)


EXAMPLES: List[str] = [
    "بريدي الإلكتروني هو ahmed.salem@gmail.com",
    "اسمي محمد عبد الرحمن الأحمد ورقم تليفوني 01012345678",
    "رقم الآيبان الخاص بي DZ5800400174001001050032",
    "راني سفيان بوزيد ورقمي 0698123456 من الجزائر العاصمة",
    "صاحبي كريم حدّاد يسكن في حي بلوزداد، وهران، إيميلو karim.h@univ-alger.dz",
    "Hi, my name is Sarah Mansour, راسليني على sarah.mansour@hotmail.fr",
    "العميل خالد فاروق، الهاتف +213661234567، الإيميل khaled@yahoo.fr، يسكن في باب الواد، الجزائر",
    "رقم حسابك المختصر 4823 ورقم حسابك البنكي 100200300400500",
    "رقم هاتفي ٠٦٩٨١٢٣٤٥٦ وحسابي رقم ٤٨٢٣",
    "رقم التعريف الجبائي NIF 099816000123456، وكود SWIFT هو BNPAFRPP",
]


app = FastAPI(title="Arabic PII Detector", version="1.0.0")
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = _load_detector()
    return _detector


@app.on_event("startup")
def _warmup():
    # Load model eagerly so the first request isn't slow.
    get_detector().predict("مرحبا")


class DetectRequest(BaseModel):
    text: str
    max_length: int = 256


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class DetectResponse(BaseModel):
    redacted_text: str
    entities: List[Entity]
    inference_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "model": MODEL_PATH}


@app.get("/examples")
def examples():
    return {"examples": EXAMPLES}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    text = req.text or ""
    if not text.strip():
        return {"redacted_text": "", "entities": [], "inference_ms": 0.0}
    t0 = time.perf_counter()
    result = get_detector().predict(text, max_length=req.max_length)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Arabic PII Detector</title>
<style>
  :root { --bg:#0f1419; --card:#1a2029; --accent:#4f9dde; --txt:#e6e6e6; --muted:#8a94a3; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "Segoe UI", Tahoma, sans-serif;
         background:var(--bg); color:var(--txt); }
  .wrap { max-width: 920px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: .85rem; margin-bottom: 18px; }
  textarea { width:100%; min-height:120px; background:var(--card); color:var(--txt);
             border:1px solid #2a3340; border-radius:10px; padding:12px; font-size:1.05rem;
             resize:vertical; }
  .row { display:flex; gap:10px; margin:12px 0; flex-wrap:wrap; align-items:center; }
  button { background:var(--accent); color:#fff; border:0; padding:10px 18px;
           border-radius:8px; font-size:.95rem; cursor:pointer; }
  button.ghost { background:transparent; border:1px solid #2a3340; color:var(--muted); }
  select { background:var(--card); color:var(--txt); border:1px solid #2a3340;
           border-radius:8px; padding:9px; flex:1; min-width:200px; }
  .card { background:var(--card); border:1px solid #2a3340; border-radius:10px;
          padding:16px; margin-top:16px; }
  .label-h { color:var(--muted); font-size:.8rem; text-transform:uppercase;
             letter-spacing:.5px; margin-bottom:8px; }
  .redacted { font-size:1.1rem; line-height:1.9; }
  .chip { display:inline-block; padding:1px 6px; border-radius:6px; font-weight:600;
          font-size:.85rem; }
  table { width:100%; border-collapse:collapse; margin-top:6px; font-size:.92rem; }
  th, td { text-align:right; padding:7px 10px; border-bottom:1px solid #2a3340; }
  th { color:var(--muted); font-weight:600; }
  .ent-text { font-weight:600; }
  .muted { color:var(--muted); }
  .empty { color:var(--muted); font-style:italic; }
  code { background:#11161d; padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔒 Arabic PII Detector</h1>
  <div class="sub">Backend: <code id="backend">…</code> · يكشف الأسماء، الإيميل، الهاتف، العنوان، أرقام الحسابات، والآيبان</div>

  <textarea id="text" placeholder="اكتب نصاً عربياً أو إنجليزياً للاختبار..."></textarea>

  <div class="row">
    <button onclick="run()">كشف PII</button>
    <select id="examples" onchange="pick()"><option value="">— أمثلة جاهزة —</option></select>
    <button class="ghost" onclick="clearAll()">مسح</button>
  </div>

  <div id="out"></div>
</div>

<script>
const COLORS = {
  PERSON:"#e06c75", EMAIL:"#56b6c2", PHONE_NUMBER:"#98c379", ADDRESS:"#e5c07b",
  ACCOUNT_NUMBER:"#c678dd", BANK_ACCOUNT_NUMBER:"#d19a66", IBAN:"#61afef"
};
function chip(label){ const c = COLORS[label]||"#888";
  return `<span class="chip" style="background:${c}33;color:${c};border:1px solid ${c}">${label}</span>`; }

async function loadMeta(){
  const h = await (await fetch('/health')).json();
  document.getElementById('backend').textContent = h.backend + " · " + h.model;
  const ex = (await (await fetch('/examples')).json()).examples;
  const sel = document.getElementById('examples');
  ex.forEach((t,i)=>{ const o=document.createElement('option'); o.value=t;
    o.textContent=(i+1)+'. '+(t.length>55?t.slice(0,55)+'…':t); sel.appendChild(o); });
}
function pick(){ const v=document.getElementById('examples').value;
  if(v){ document.getElementById('text').value=v; run(); } }
function clearAll(){ document.getElementById('text').value=''; document.getElementById('out').innerHTML=''; }

function renderRedacted(text){
  // colorize the [TAG] placeholders in the redacted string
  return text.replace(/\\[([A-Z_]+)\\]/g, (m,l)=>chip(l));
}

async function run(){
  const text = document.getElementById('text').value;
  const out = document.getElementById('out');
  if(!text.trim()){ out.innerHTML=''; return; }
  out.innerHTML = '<div class="card muted">… جاري الكشف</div>';
  let r;
  try { r = await (await fetch('/detect',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text})})).json(); }
  catch(e){ out.innerHTML = '<div class="card empty">خطأ في الاتصال</div>'; return; }

  let rows = r.entities.length
    ? r.entities.map(e=>`<tr><td class="ent-text">${e.text}</td><td>${chip(e.label)}</td>
        <td class="muted">${e.start}–${e.end}</td><td class="muted">${e.confidence}</td></tr>`).join('')
    : '<tr><td colspan="4" class="empty">لم يتم العثور على أي PII</td></tr>';

  out.innerHTML = `
    <div class="card">
      <div class="label-h">النص بعد الإخفاء</div>
      <div class="redacted">${renderRedacted(r.redacted_text)}</div>
    </div>
    <div class="card">
      <div class="label-h">الكيانات المكتشفة (${r.entities.length}) · ⏱ ${r.inference_ms} ms</div>
      <table><thead><tr><th>النص</th><th>النوع</th><th>الموضع</th><th>الثقة</th></tr></thead>
      <tbody>${rows}</tbody></table>
    </div>`;
}
document.getElementById('text').addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter') run(); });
loadMeta();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
