"""草莓采摘检测 — 两模型版本（检测 + 分类），网页查看结果"""

import base64
import csv
import json
import math
import mimetypes
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import cv2
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────────────────────────────────────
def _load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_cfg = _load_config()
DET_MODEL_PATH       = _cfg.get("det_model", "runs/detect/strawberry_stem_detection/weights/best.pt")
CLS_MODEL_PATH       = _cfg.get("cls_model", "runs/classify/strawberry_ripeness/weights/best.pt")
SOURCE_DIR           = _cfg.get("source", "dataset/test/images")
SAVE_DIR             = _cfg.get("save_dir", "harvest_results")
DET_CONF             = _cfg.get("det_conf", 0.1)
CLS_UNRIPE_THRESHOLD = _cfg.get("cls_unripe_threshold", 0.6)
PORT                 = _cfg.get("port", 8000)

COLORS = {
    "ripe":     (0, 255, 0),
    "overripe": (0, 165, 255),
    "unripe":   (128, 128, 128),
    "stem":     (255, 255, 0),
    "pick":     (0, 255, 255),
    "path":     (255, 0, 255),
}

# ── 模型 & 结果（全局）────────────────────────────────────────────────────────
det_model = None
cls_model = None
results_summary: dict = {}
processing_done = False


# ── 检测逻辑 ──────────────────────────────────────────────────────────────────

def classify_strawberry(img, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "unripe", 0.0
    res = cls_model.predict(source=crop, verbose=False)
    probs = res[0].probs
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)
    cls_name = res[0].names[cls_id]
    if cls_name == "unripe" and conf < CLS_UNRIPE_THRESHOLD:
        return "ripe", conf
    return cls_name, conf


def find_nearest_stem(strawberry_box, stems):
    sb = strawberry_box
    s_cx, s_cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
    best_stem, min_dist = None, float("inf")
    for stem in stems:
        b = stem["box"]
        dist = ((s_cx - (b[0]+b[2])/2)**2 + (s_cy - (b[1]+b[3])/2)**2) ** 0.5
        if dist < min_dist:
            min_dist, best_stem = dist, stem
    return best_stem


def get_pick_point(stem_box, offset_ratio=0.3):
    b = stem_box
    return int((b[0]+b[2])/2), int(b[3] - (b[3]-b[1]) * offset_ratio)


def solve_tsp_nn(points):
    """最近邻启发式 TSP，返回 (排序后的点列表, 总距离)。"""
    if len(points) <= 1:
        return points, 0.0
    remaining = list(range(len(points)))
    order = [remaining.pop(0)]
    total_dist = 0.0
    while remaining:
        last = points[order[-1]]["pixel"]
        best_i, best_d = 0, float("inf")
        for i, idx in enumerate(remaining):
            p = points[idx]["pixel"]
            d = math.hypot(last[0] - p[0], last[1] - p[1])
            if d < best_d:
                best_i, best_d = i, d
        total_dist += best_d
        order.append(remaining.pop(best_i))
    return [points[i] for i in order], round(total_dist, 1)


def draw_pick_path(img, ordered_points):
    """在图像上绘制采摘路径（箭头 + 序号）。"""
    color = COLORS["path"]
    for i, pt in enumerate(ordered_points):
        px, py = pt["pixel"]
        cv2.circle(img, (px, py), 14, color, 2)
        cv2.putText(img, str(i + 1), (px - 6, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i > 0:
            prev = ordered_points[i - 1]["pixel"]
            cv2.arrowedLine(img, (prev[0], prev[1]), (px, py), color, 2, tipLength=0.15)


def process_image(image_path: Path) -> dict:
    img = cv2.imread(str(image_path))
    det_res = det_model.predict(source=str(image_path), conf=DET_CONF, verbose=False)

    stems, strawberries = [], []
    for result in det_res:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            entry = {"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(box.conf[0])}
            (stems if int(box.cls[0]) == 0 else strawberries).append(entry)

    pick_points, overripe_list, unripe_list = [], [], []

    for s in strawberries:
        b = s["box"]
        ripeness, conf = classify_strawberry(img, b)
        color = COLORS.get(ripeness, (128, 128, 128))

        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(img, f"{ripeness} {conf:.2f}", (int(b[0]), int(b[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        nearest_stem = find_nearest_stem(b, stems)

        if ripeness == "ripe" and nearest_stem is not None:
            px, py = get_pick_point(nearest_stem["box"])
            pick_points.append({"pixel": [px, py], "confidence": round(conf, 3)})
            cv2.drawMarker(img, (px, py), COLORS["pick"], cv2.MARKER_CROSS, 20, 3)
            cv2.putText(img, "PICK", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["pick"], 2)
            sb = nearest_stem["box"]
            cv2.rectangle(img, (int(sb[0]), int(sb[1])), (int(sb[2]), int(sb[3])), COLORS["stem"], 2)
        elif ripeness == "overripe":
            overripe_list.append({"bbox": [round(float(x), 1) for x in b], "confidence": round(conf, 3)})
            cv2.putText(img, "OVERRIPE", (int(b[0]), int(b[3])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["overripe"], 2)
        else:
            unripe_list.append({"confidence": round(conf, 3)})

    # TSP 路径规划
    path_distance = 0.0
    if len(pick_points) >= 2:
        pick_points, path_distance = solve_tsp_nn(pick_points)
        draw_pick_path(img, pick_points)

    Path(SAVE_DIR).mkdir(exist_ok=True)
    out_path = Path(SAVE_DIR) / image_path.name
    cv2.imwrite(str(out_path), img)

    return {
        "image": image_path.name,
        "ripe": len(pick_points),
        "overripe": len(overripe_list),
        "unripe": len(unripe_list),
        "total": len(strawberries),
        "pick_points": pick_points,
        "overripe_list": overripe_list,
        "path_distance": path_distance,
    }


def _harvest_advice(totals: dict) -> dict:
    total = totals["total"]
    if total == 0:
        return {"zh": "未检测到草莓", "en": "No strawberries detected"}
    ripe_ratio = totals["ripe"] / total
    overripe_ratio = totals["overripe"] / total
    unripe_ratio = totals["unripe"] / total
    if ripe_ratio >= 0.7:
        return {
            "zh": f"建议立即采摘 — 成熟率 {ripe_ratio*100:.1f}%，已达采摘标准",
            "en": f"Recommend immediate harvest — ripe ratio {ripe_ratio*100:.1f}%, ready for picking",
        }
    if overripe_ratio >= 0.3:
        return {
            "zh": f"注意过熟 — 过熟率 {overripe_ratio*100:.1f}%，需尽快处理",
            "en": f"Warning: overripe — overripe ratio {overripe_ratio*100:.1f}%, handle ASAP",
        }
    if unripe_ratio >= 0.6:
        return {
            "zh": f"建议等待 — 未成熟率 {unripe_ratio*100:.1f}%，草莓尚未就绪",
            "en": f"Recommend waiting — unripe ratio {unripe_ratio*100:.1f}%, not ready yet",
        }
    return {
        "zh": f"部分可采摘 — 成熟率 {ripe_ratio*100:.1f}%，可选择性采摘成熟果实",
        "en": f"Partially harvestable — ripe ratio {ripe_ratio*100:.1f}%, selectively pick ripe ones",
    }


def run_detection():
    global results_summary, processing_done
    img_dir = Path(SOURCE_DIR)
    images = [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in sorted(img_dir.glob(ext))]
    print(f"处理 {len(images)} 张图像...")

    per_image = []
    totals = {"ripe": 0, "overripe": 0, "unripe": 0, "total": 0}
    all_overripe = []

    for img_path in images:
        r = process_image(img_path)
        per_image.append(r)
        for k in totals:
            totals[k] += r[k]
        all_overripe.extend([{"image": r["image"], **o} for o in r.get("overripe_list", [])])
        print(f"  {r['image']}: ripe={r['ripe']} overripe={r['overripe']} unripe={r['unripe']}")

    ripe_ratio = round(totals["ripe"] / totals["total"] * 100, 1) if totals["total"] else 0
    advice = _harvest_advice(totals)

    # 保存完整 JSON 报告
    report = {
        "totals": totals,
        "ripe_ratio_pct": ripe_ratio,
        "advice": advice,
        "images": per_image,
        "overripe_details": all_overripe,
    }
    Path(SAVE_DIR).mkdir(exist_ok=True)
    with open(Path(SAVE_DIR) / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    results_summary = report
    processing_done = True
    print(f"\n===== 汇总 =====")
    print(f"Total: {totals['total']}  Ripe: {totals['ripe']}  Unripe: {totals['unripe']}  Overripe: {totals['overripe']}")
    print(f"Ripe ratio: {ripe_ratio}%")
    print(f"建议: {advice}")
    print(f"报告: {SAVE_DIR}/report.json")

    # 生成静态 HTML 文件
    generate_static_files(report)


# ── 静态 HTML 生成 ────────────────────────────────────────────────────────────

def _img_to_base64(filepath: str) -> str:
    """将图片文件转为 base64 data URI。"""
    p = Path(filepath)
    if not p.exists():
        return ""
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = p.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _read_chartjs() -> str:
    """读取本地 Chart.js 文件内容，用于内嵌到静态 HTML。"""
    p = Path(__file__).parent / "chart.umd.min.js"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _embed_chartjs(html: str) -> str:
    """将 CDN 引用的 Chart.js 替换为内嵌脚本。"""
    chartjs_code = _read_chartjs()
    if chartjs_code:
        html = html.replace(
            '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>',
            f'<script>{chartjs_code}</script>'
        )
    return html


def generate_static_files(report: dict):
    """生成可直接在浏览器打开的静态 HTML 文件。"""
    Path(SAVE_DIR).mkdir(exist_ok=True)
    _generate_static_dashboard(report)
    _generate_static_evaluation()
    print(f"静态页面: {SAVE_DIR}/index.html")
    print(f"静态页面: {SAVE_DIR}/evaluation.html")


def _generate_static_dashboard(report: dict):
    """生成静态 Dashboard HTML，数据和图片全部内嵌，单文件可移植。"""
    # 将检测结果图片转为 base64 内嵌到报告数据中
    report_with_b64 = json.loads(json.dumps(report))  # deep copy
    img_b64_map = {}
    for img_info in report_with_b64.get("images", []):
        name = img_info["image"]
        if name not in img_b64_map:
            # 优先从结果目录读取（带标注的图），回退到源目录
            p = Path(SAVE_DIR) / name
            if not p.exists():
                p = Path(SOURCE_DIR) / name
            img_b64_map[name] = _img_to_base64(str(p))
    data_json = json.dumps(report_with_b64, ensure_ascii=False)
    b64_map_json = json.dumps(img_b64_map, ensure_ascii=False)

    html = DASHBOARD_HTML
    # 内嵌 Chart.js
    html = _embed_chartjs(html)
    # 替换导航链接为静态文件路径
    html = html.replace('href="/"', 'href="./index.html"')
    html = html.replace('href="/evaluation"', 'href="./evaluation.html"')
    # 图片通过 base64 映射表查找
    html = html.replace("'/image/' + img.image", "_IMG_B64[img.image]")
    # 替换 JS 的 fetch 逻辑为直接使用内嵌数据
    old_load = """async function load() {
  const r = await fetch('/results');
  const d = await r.json();
  if (!d.done) { setTimeout(load, 2000); return; }"""
    new_load = f"""const _EMBEDDED_DATA = {data_json};
const _IMG_B64 = {b64_map_json};
async function load() {{
  const d = _EMBEDDED_DATA;
  d.done = true;"""
    html = html.replace(old_load, new_load)

    out = Path(SAVE_DIR) / "index.html"
    out.write_text(html, encoding="utf-8")


def _generate_static_evaluation():
    """生成静态 Evaluation HTML，训练指标内嵌，图片转 base64。"""
    det_raw = _parse_csv(f"{DET_RUNS_DIR}/results.csv")
    cls_raw = _parse_csv(f"{CLS_RUNS_DIR}/results.csv")

    det = {}
    if det_raw:
        det = {
            "epochs": [int(e) for e in det_raw.get("epoch", [])],
            "train_box_loss": det_raw.get("train/box_loss", []),
            "train_cls_loss": det_raw.get("train/cls_loss", []),
            "train_dfl_loss": det_raw.get("train/dfl_loss", []),
            "precision": det_raw.get("metrics/precision(B)", []),
            "recall": det_raw.get("metrics/recall(B)", []),
            "mAP50": det_raw.get("metrics/mAP50(B)", []),
            "mAP50_95": det_raw.get("metrics/mAP50-95(B)", []),
            "val_box_loss": det_raw.get("val/box_loss", []),
            "val_cls_loss": det_raw.get("val/cls_loss", []),
            "val_dfl_loss": det_raw.get("val/dfl_loss", []),
        }
    cls = {}
    if cls_raw:
        cls = {
            "epochs": [int(e) for e in cls_raw.get("epoch", [])],
            "train_loss": cls_raw.get("train/loss", []),
            "accuracy_top1": cls_raw.get("metrics/accuracy_top1", []),
            "accuracy_top5": cls_raw.get("metrics/accuracy_top5", []),
            "val_loss": cls_raw.get("val/loss", []),
        }

    # 收集训练图片并转 base64
    def _list_images_b64(directory: str) -> list:
        d = Path(directory)
        if not d.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        result = []
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in exts:
                result.append({"name": f.name, "data": _img_to_base64(str(f))})
        return result

    det_imgs = _list_images_b64(DET_RUNS_DIR)
    cls_imgs = _list_images_b64(CLS_RUNS_DIR)

    metrics_json = json.dumps({"detection": det, "classification": cls}, ensure_ascii=False)
    det_imgs_json = json.dumps(det_imgs, ensure_ascii=False)
    cls_imgs_json = json.dumps(cls_imgs, ensure_ascii=False)

    html = EVALUATION_HTML
    # 内嵌 Chart.js
    html = _embed_chartjs(html)
    # 替换导航链接
    html = html.replace('href="/"', 'href="./index.html"')
    html = html.replace('href="/evaluation"', 'href="./evaluation.html"')

    # 替换 JS 的 fetch 逻辑为直接使用内嵌数据
    old_init_start = """async function init() {
  const [mRes, iRes] = await Promise.all([
    fetch('/api/training/metrics').then(r=>r.json()),
    fetch('/api/training/available-images').then(r=>r.json())
  ]);"""
    new_init_start = f"""const _METRICS = {metrics_json};
const _DET_IMGS = {det_imgs_json};
const _CLS_IMGS = {cls_imgs_json};
async function init() {{
  const mRes = _METRICS;
  const iRes = {{detection: _DET_IMGS.map(i=>i.name), classification: _CLS_IMGS.map(i=>i.name)}};"""
    html = html.replace(old_init_start, new_init_start)

    # 替换图片加载：使用 base64 数据而非 API 路径
    old_add_image = """function addImage(container, modelType, filename) {
  const src = '/api/training/image/'+modelType+'/'+filename;
  const d = document.createElement('div');
  d.className = 'img-card';
  d.innerHTML = '<img src="'+src+'" onclick="openLightbox(\\x27'+src+'\\x27)"><div class="info">'+filename+'</div>';
  container.appendChild(d);
}"""
    new_add_image = """function addImage(container, modelType, filename) {
  const imgs = modelType === 'detect' ? _DET_IMGS : _CLS_IMGS;
  const found = imgs.find(i => i.name === filename);
  const src = found ? found.data : '';
  const d = document.createElement('div');
  d.className = 'img-card';
  d.innerHTML = '<img src="'+src+'" onclick="openLightbox(\\x27'+src+'\\x27)"><div class="info">'+filename+'</div>';
  container.appendChild(d);
}"""
    html = html.replace(old_add_image, new_add_image)

    out = Path(SAVE_DIR) / "evaluation.html"
    out.write_text(html, encoding="utf-8")


# ── 网页 HTML ─────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>草莓检测结果</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',sans-serif; background:#1a1a2e; color:#eee; padding:24px; }
h1 { text-align:center; padding:14px; background:#e94560; border-radius:8px; margin-bottom:20px; font-size:1.4em; letter-spacing:1px; }
#status { text-align:center; color:#aaa; margin-bottom:18px; font-size:0.95em; }
.nav-bar { text-align:center; margin-bottom:18px; }
.nav-bar a { color:#e94560; text-decoration:none; margin:0 12px; font-size:0.95em; border-bottom:2px solid transparent; padding-bottom:2px; }
.nav-bar a:hover, .nav-bar a.active { border-bottom-color:#e94560; }
#lang-toggle { position:fixed; top:16px; right:24px; z-index:50; background:#0f3460; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85em; color:#eee; border:1px solid #e94560; }
#lang-toggle:hover { background:#16213e; }
.summary { display:flex; gap:14px; justify-content:center; flex-wrap:wrap; margin-bottom:20px; }
.card { background:#0f3460; border-radius:10px; padding:16px 26px; text-align:center; min-width:110px; }
.card .num { font-size:2em; font-weight:bold; }
.card .lbl { font-size:0.8em; color:#aaa; margin-top:4px; }
.ripe .num { color:#00ff00; }
.unripe .num { color:#aaa; }
.overripe .num { color:#ffa500; }
.total .num { color:#fff; }
.ratio .num { color:#00ffff; }
#advice-box { max-width:700px; margin:0 auto 24px; padding:14px 20px; border-radius:8px;
  background:#0f3460; border-left:4px solid #e94560; font-size:1.05em; line-height:1.5; }
#advice-box .adv-label { font-size:0.78em; color:#aaa; margin-bottom:4px; }
.middle { display:flex; gap:24px; justify-content:center; flex-wrap:wrap; margin-bottom:28px; align-items:flex-start; }
.pie-wrap { background:#0f3460; border-radius:10px; padding:20px; width:280px; }
.pie-wrap h3 { text-align:center; margin-bottom:12px; font-size:0.95em; color:#ccc; }
h2 { text-align:center; margin-bottom:16px; font-size:1.1em; color:#ccc; }
.grid { display:flex; flex-wrap:wrap; gap:16px; justify-content:center; }
.img-card { background:#16213e; border-radius:8px; overflow:hidden; width:300px; }
.img-card img { width:100%; display:block; cursor:pointer; }
.img-card .info { padding:10px; font-size:0.82em; display:flex; gap:8px; flex-wrap:wrap; }
.tag { padding:2px 8px; border-radius:4px; }
.tag-ripe     { color:#00ff00; border:1px solid #00ff00; }
.tag-overripe { color:#ffa500; border:1px solid #ffa500; }
.tag-unripe   { color:#aaa;    border:1px solid #555; }
.tag-path     { color:#f0f;    border:1px solid #f0f; }
.img-name { padding:0 10px 8px; font-size:0.75em; color:#666; word-break:break-all; }
#lightbox { display:none; position:fixed; inset:0; background:#000a; z-index:99;
  align-items:center; justify-content:center; }
#lightbox.show { display:flex; }
#lightbox img { max-width:92vw; max-height:92vh; border-radius:8px; }
#lightbox-close { position:absolute; top:16px; right:24px; font-size:2em;
  cursor:pointer; color:#fff; line-height:1; }
</style>
</head>
<body>
<div id="lang-toggle" onclick="toggleLang()">EN</div>
<h1 data-i18n="title">草莓检测结果</h1>
<div class="nav-bar">
  <a href="/" class="active" data-i18n="nav_dashboard">检测面板</a>
  <a href="/evaluation" data-i18n="nav_evaluation">模型评估</a>
</div>
<div id="status" data-i18n="status_loading">检测中，请稍候...</div>

<div class="summary">
  <div class="card total">   <div class="num" id="t-total">-</div>   <div class="lbl" data-i18n="lbl_total">总计</div></div>
  <div class="card ripe">    <div class="num" id="t-ripe">-</div>    <div class="lbl" data-i18n="lbl_ripe">成熟</div></div>
  <div class="card unripe">  <div class="num" id="t-unripe">-</div>  <div class="lbl" data-i18n="lbl_unripe">未成熟</div></div>
  <div class="card overripe"><div class="num" id="t-overripe">-</div><div class="lbl" data-i18n="lbl_overripe">过熟</div></div>
  <div class="card ratio">   <div class="num" id="t-ratio">-</div>   <div class="lbl" data-i18n="lbl_ratio">成熟率</div></div>
</div>

<div id="advice-box" style="display:none">
  <div class="adv-label" data-i18n="advice_label">采摘建议</div>
  <div id="advice-text"></div>
</div>

<div class="middle">
  <div class="pie-wrap">
    <h3 data-i18n="pie_title">成熟度分布</h3>
    <canvas id="pie" width="240" height="240"></canvas>
  </div>
</div>

<h2 data-i18n="images_title">检测图像</h2>
<div class="grid" id="grid"></div>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">&#x2715;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const LANG = {
  zh: {
    title:"草莓检测结果", nav_dashboard:"检测面板", nav_evaluation:"模型评估",
    status_loading:"检测中，请稍候...", status_done:"检测完成",
    lbl_total:"总计", lbl_ripe:"成熟", lbl_unripe:"未成熟", lbl_overripe:"过熟", lbl_ratio:"成熟率",
    advice_label:"采摘建议", pie_title:"成熟度分布",
    images_title:"检测图像", path_label:"路径",
  },
  en: {
    title:"Strawberry Detection", nav_dashboard:"Dashboard", nav_evaluation:"Model Evaluation",
    status_loading:"Processing, please wait...", status_done:"Detection complete",
    lbl_total:"Total", lbl_ripe:"Ripe", lbl_unripe:"Unripe", lbl_overripe:"Overripe", lbl_ratio:"Ripe Ratio",
    advice_label:"Harvest Advice", pie_title:"Ripeness Distribution",
    images_title:"Detection Images", path_label:"path",
  }
};
let currentLang = localStorage.getItem('lang') || 'zh';
function applyLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (LANG[lang][k]) el.textContent = LANG[lang][k];
  });
  document.title = LANG[lang].title;
  document.getElementById('lang-toggle').textContent = lang === 'zh' ? 'EN' : '中文';
  if (window._adviceData) {
    document.getElementById('advice-text').textContent =
      typeof window._adviceData === 'object' ? (window._adviceData[lang] || '') : window._adviceData;
  }
}
function toggleLang() { applyLang(currentLang === 'zh' ? 'en' : 'zh'); }

let pieChart = null;
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('show'); }
document.getElementById('lightbox').addEventListener('click', function(e){ if(e.target===this) closeLightbox(); });

function renderPie(ripe, unripe, overripe) {
  const ctx = document.getElementById('pie').getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(ctx, {
    type:'pie',
    data:{ labels:['Ripe','Unripe','Overripe'],
      datasets:[{ data:[ripe,unripe,overripe], backgroundColor:['#00cc44','#666','#ff8800'],
        borderColor:['#1a1a2e','#1a1a2e','#1a1a2e'], borderWidth:2 }]},
    options:{ plugins:{ legend:{ labels:{ color:'#eee', font:{size:13} }}}}
  });
}

async function load() {
  const r = await fetch('/results');
  const d = await r.json();
  if (!d.done) { setTimeout(load, 2000); return; }

  document.querySelector('[data-i18n="status_loading"]').setAttribute('data-i18n','status_done');
  document.getElementById('status').textContent = LANG[currentLang].status_done;
  const t = d.totals;
  document.getElementById('t-total').textContent   = t.total;
  document.getElementById('t-ripe').textContent    = t.ripe;
  document.getElementById('t-unripe').textContent  = t.unripe;
  document.getElementById('t-overripe').textContent= t.overripe;
  document.getElementById('t-ratio').textContent   = d.ripe_ratio_pct + '%';

  window._adviceData = d.advice;
  document.getElementById('advice-box').style.display = '';
  document.getElementById('advice-text').textContent =
    typeof d.advice === 'object' ? (d.advice[currentLang] || '') : d.advice;

  renderPie(t.ripe, t.unripe, t.overripe);

  const grid = document.getElementById('grid');
  const pathLbl = LANG[currentLang].path_label;
  for (const img of d.images) {
    const src = '/image/' + img.image;
    const div = document.createElement('div');
    div.className = 'img-card';
    const pathTag = img.path_distance > 0
      ? '<span class="tag tag-path">' + pathLbl + ' ' + Math.round(img.path_distance) + 'px</span>' : '';
    div.innerHTML =
      '<img src="'+src+'" alt="'+img.image+'" onclick="openLightbox(\\x27'+src+'\\x27)">'+
      '<div class="info">'+
        '<span class="tag tag-ripe">ripe '+img.ripe+'</span>'+
        '<span class="tag tag-overripe">overripe '+img.overripe+'</span>'+
        '<span class="tag tag-unripe">unripe '+img.unripe+'</span>'+
        pathTag+
      '</div>'+
      '<div class="img-name">'+img.image+'</div>';
    grid.appendChild(div);
  }
}
applyLang(currentLang);
load();
</script>
</body>
</html>"""

EVALUATION_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>模型性能评估</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',sans-serif; background:#1a1a2e; color:#eee; padding:24px; }
h1 { text-align:center; padding:14px; background:#e94560; border-radius:8px; margin-bottom:20px; font-size:1.4em; letter-spacing:1px; }
.nav-bar { text-align:center; margin-bottom:18px; }
.nav-bar a { color:#e94560; text-decoration:none; margin:0 12px; font-size:0.95em; border-bottom:2px solid transparent; padding-bottom:2px; }
.nav-bar a:hover, .nav-bar a.active { border-bottom-color:#e94560; }
#lang-toggle { position:fixed; top:16px; right:24px; z-index:50; background:#0f3460; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85em; color:#eee; border:1px solid #e94560; }
#lang-toggle:hover { background:#16213e; }
.tabs { display:flex; justify-content:center; gap:0; margin-bottom:24px; }
.tab-btn { background:#0f3460; color:#aaa; border:none; padding:12px 32px; cursor:pointer; font-size:0.95em; }
.tab-btn:first-child { border-radius:8px 0 0 8px; }
.tab-btn:last-child { border-radius:0 8px 8px 0; }
.tab-btn.active { background:#e94560; color:#fff; }
.tab-content { display:none; }
.tab-content.active { display:block; }
.summary { display:flex; gap:14px; justify-content:center; flex-wrap:wrap; margin-bottom:24px; }
.card { background:#0f3460; border-radius:10px; padding:16px 26px; text-align:center; min-width:120px; }
.card .num { font-size:1.8em; font-weight:bold; color:#00ffff; }
.card .lbl { font-size:0.8em; color:#aaa; margin-top:4px; }
.charts-row { display:flex; gap:20px; flex-wrap:wrap; justify-content:center; margin-bottom:28px; }
.chart-box { background:#0f3460; border-radius:10px; padding:20px; width:460px; min-width:300px; }
.chart-box h3 { text-align:center; margin-bottom:12px; font-size:0.95em; color:#ccc; }
h2 { text-align:center; margin-bottom:16px; font-size:1.1em; color:#ccc; }
.grid { display:flex; flex-wrap:wrap; gap:16px; justify-content:center; margin-bottom:28px; }
.img-card { background:#16213e; border-radius:8px; overflow:hidden; width:320px; }
.img-card img { width:100%; display:block; cursor:pointer; }
.img-card .info { padding:8px 10px; font-size:0.8em; color:#aaa; text-align:center; }
#lightbox { display:none; position:fixed; inset:0; background:#000a; z-index:99; align-items:center; justify-content:center; }
#lightbox.show { display:flex; }
#lightbox img { max-width:92vw; max-height:92vh; border-radius:8px; }
#lightbox-close { position:absolute; top:16px; right:24px; font-size:2em; cursor:pointer; color:#fff; line-height:1; }
</style>
</head>
<body>
<div id="lang-toggle" onclick="toggleLang()">EN</div>
<h1 data-i18n="eval_title">模型性能评估</h1>
<div class="nav-bar">
  <a href="/" data-i18n="nav_dashboard">检测面板</a>
  <a href="/evaluation" class="active" data-i18n="nav_evaluation">模型评估</a>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('det')" data-i18n="tab_detection">检测模型</button>
  <button class="tab-btn" onclick="switchTab('cls')" data-i18n="tab_classification">分类模型</button>
</div>

<!-- Detection Tab -->
<div id="tab-det" class="tab-content active">
  <div class="summary" id="det-summary"></div>
  <div class="charts-row">
    <div class="chart-box"><h3 data-i18n="chart_train_loss">训练损失</h3><canvas id="det-loss"></canvas></div>
    <div class="chart-box"><h3 data-i18n="chart_val_loss">验证损失</h3><canvas id="det-val-loss"></canvas></div>
  </div>
  <div class="charts-row">
    <div class="chart-box"><h3 data-i18n="chart_metrics">检测指标</h3><canvas id="det-metrics"></canvas></div>
  </div>
  <h2 data-i18n="vis_title">可视化结果</h2>
  <div class="grid" id="det-images"></div>
</div>

<!-- Classification Tab -->
<div id="tab-cls" class="tab-content">
  <div class="summary" id="cls-summary"></div>
  <div class="charts-row">
    <div class="chart-box"><h3 data-i18n="chart_cls_loss">训练/验证损失</h3><canvas id="cls-loss"></canvas></div>
    <div class="chart-box"><h3 data-i18n="chart_accuracy">准确率</h3><canvas id="cls-acc"></canvas></div>
  </div>
  <h2 data-i18n="vis_title">可视化结果</h2>
  <div class="grid" id="cls-images"></div>
</div>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">&#x2715;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const LANG = {
  zh: {
    eval_title:"模型性能评估", nav_dashboard:"检测面板", nav_evaluation:"模型评估",
    tab_detection:"检测模型", tab_classification:"分类模型",
    chart_train_loss:"训练损失", chart_val_loss:"验证损失", chart_metrics:"检测指标",
    chart_cls_loss:"训练/验证损失", chart_accuracy:"准确率", vis_title:"可视化结果",
  },
  en: {
    eval_title:"Model Performance", nav_dashboard:"Dashboard", nav_evaluation:"Model Evaluation",
    tab_detection:"Detection Model", tab_classification:"Classification Model",
    chart_train_loss:"Training Loss", chart_val_loss:"Validation Loss", chart_metrics:"Detection Metrics",
    chart_cls_loss:"Train/Val Loss", chart_accuracy:"Accuracy", vis_title:"Visualizations",
  }
};
let currentLang = localStorage.getItem('lang') || 'zh';
function applyLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (LANG[lang][k]) el.textContent = LANG[lang][k];
  });
  document.title = LANG[lang].eval_title;
  document.getElementById('lang-toggle').textContent = lang === 'zh' ? 'EN' : '中文';
}
function toggleLang() { applyLang(currentLang === 'zh' ? 'en' : 'zh'); }

function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('show'); }
document.getElementById('lightbox').addEventListener('click', function(e){ if(e.target===this) closeLightbox(); });

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', (tab==='det'?i===0:i===1)));
  document.getElementById('tab-det').classList.toggle('active', tab==='det');
  document.getElementById('tab-cls').classList.toggle('active', tab==='cls');
}

const chartOpts = {
  responsive:true,
  plugins:{ legend:{ labels:{ color:'#eee', font:{size:11} }}},
  scales:{ x:{ ticks:{color:'#aaa'}, grid:{color:'#1a1a2e'} }, y:{ ticks:{color:'#aaa'}, grid:{color:'#222'} }}
};

function mkLine(ctx, labels, datasets) {
  return new Chart(ctx, { type:'line', data:{ labels, datasets }, options:chartOpts });
}

function addCard(container, label, value) {
  const d = document.createElement('div');
  d.className = 'card';
  d.innerHTML = '<div class="num">'+value+'</div><div class="lbl">'+label+'</div>';
  container.appendChild(d);
}

function addImage(container, modelType, filename) {
  const src = '/api/training/image/'+modelType+'/'+filename;
  const d = document.createElement('div');
  d.className = 'img-card';
  d.innerHTML = '<img src="'+src+'" onclick="openLightbox(\\x27'+src+'\\x27)"><div class="info">'+filename+'</div>';
  container.appendChild(d);
}

async function init() {
  const [mRes, iRes] = await Promise.all([
    fetch('/api/training/metrics').then(r=>r.json()),
    fetch('/api/training/available-images').then(r=>r.json())
  ]);

  // Detection metrics
  if (mRes.detection) {
    const det = mRes.detection;
    const ep = det.epochs;
    const last = ep.length - 1;
    const ds = document.getElementById('det-summary');
    addCard(ds, 'mAP50', det.mAP50[last].toFixed(3));
    addCard(ds, 'mAP50-95', det.mAP50_95[last].toFixed(3));
    addCard(ds, 'Precision', det.precision[last].toFixed(3));
    addCard(ds, 'Recall', det.recall[last].toFixed(3));
    addCard(ds, 'Epochs', ep.length);

    mkLine(document.getElementById('det-loss'), ep, [
      { label:'box_loss', data:det.train_box_loss, borderColor:'#e94560', fill:false, pointRadius:0 },
      { label:'cls_loss', data:det.train_cls_loss, borderColor:'#00cc44', fill:false, pointRadius:0 },
      { label:'dfl_loss', data:det.train_dfl_loss, borderColor:'#ff8800', fill:false, pointRadius:0 },
    ]);
    mkLine(document.getElementById('det-val-loss'), ep, [
      { label:'val_box', data:det.val_box_loss, borderColor:'#e94560', fill:false, pointRadius:0 },
      { label:'val_cls', data:det.val_cls_loss, borderColor:'#00cc44', fill:false, pointRadius:0 },
      { label:'val_dfl', data:det.val_dfl_loss, borderColor:'#ff8800', fill:false, pointRadius:0 },
    ]);
    mkLine(document.getElementById('det-metrics'), ep, [
      { label:'precision', data:det.precision, borderColor:'#0ff', fill:false, pointRadius:0 },
      { label:'recall', data:det.recall, borderColor:'#ff0', fill:false, pointRadius:0 },
      { label:'mAP50', data:det.mAP50, borderColor:'#0f0', fill:false, pointRadius:0 },
      { label:'mAP50-95', data:det.mAP50_95, borderColor:'#f0f', fill:false, pointRadius:0 },
    ]);
  }

  // Classification metrics
  if (mRes.classification) {
    const cls = mRes.classification;
    const ep = cls.epochs;
    const last = ep.length - 1;
    const cs = document.getElementById('cls-summary');
    addCard(cs, 'Top-1 Acc', (cls.accuracy_top1[last]*100).toFixed(1)+'%');
    addCard(cs, 'Top-5 Acc', (cls.accuracy_top5[last]*100).toFixed(1)+'%');
    addCard(cs, 'Train Loss', cls.train_loss[last].toFixed(3));
    addCard(cs, 'Val Loss', cls.val_loss[last].toFixed(3));
    addCard(cs, 'Epochs', ep.length);

    mkLine(document.getElementById('cls-loss'), ep, [
      { label:'train_loss', data:cls.train_loss, borderColor:'#e94560', fill:false, pointRadius:0 },
      { label:'val_loss', data:cls.val_loss, borderColor:'#00cc44', fill:false, pointRadius:0 },
    ]);
    mkLine(document.getElementById('cls-acc'), ep, [
      { label:'top1', data:cls.accuracy_top1, borderColor:'#0ff', fill:false, pointRadius:0 },
      { label:'top5', data:cls.accuracy_top5, borderColor:'#ff0', fill:false, pointRadius:0 },
    ]);
  }

  // Visualization images
  const detGrid = document.getElementById('det-images');
  for (const f of (iRes.detection||[])) addImage(detGrid, 'detect', f);
  const clsGrid = document.getElementById('cls-images');
  for (const f of (iRes.classification||[])) addImage(clsGrid, 'classify', f);
}
applyLang(currentLang);
init();
</script>
</body>
</html>"""

# ── FastAPI ───────────────────────────────────────────────────────────────────

DET_RUNS_DIR = "runs/detect/strawberry_stem_detection"
CLS_RUNS_DIR = "runs/classify/strawberry_ripeness"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation():
    return EVALUATION_HTML


@app.get("/results")
def get_results():
    return JSONResponse({"done": processing_done, **results_summary})


@app.get("/image/{filename}")
def get_image(filename: str):
    path = Path(SAVE_DIR) / filename
    if not path.exists():
        path = Path(SOURCE_DIR) / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path))


def _parse_csv(filepath: str) -> dict:
    """解析 YOLO results.csv，返回按列名分组的数值列表。"""
    result = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    k = key.strip()
                    if k not in result:
                        result[k] = []
                    try:
                        result[k].append(float(val.strip()))
                    except (ValueError, AttributeError):
                        result[k].append(0.0)
    except FileNotFoundError:
        pass
    return result


@app.get("/api/training/metrics")
def training_metrics():
    det_raw = _parse_csv(f"{DET_RUNS_DIR}/results.csv")
    cls_raw = _parse_csv(f"{CLS_RUNS_DIR}/results.csv")

    det = {}
    if det_raw:
        det = {
            "epochs": [int(e) for e in det_raw.get("epoch", [])],
            "train_box_loss": det_raw.get("train/box_loss", []),
            "train_cls_loss": det_raw.get("train/cls_loss", []),
            "train_dfl_loss": det_raw.get("train/dfl_loss", []),
            "precision": det_raw.get("metrics/precision(B)", []),
            "recall": det_raw.get("metrics/recall(B)", []),
            "mAP50": det_raw.get("metrics/mAP50(B)", []),
            "mAP50_95": det_raw.get("metrics/mAP50-95(B)", []),
            "val_box_loss": det_raw.get("val/box_loss", []),
            "val_cls_loss": det_raw.get("val/cls_loss", []),
            "val_dfl_loss": det_raw.get("val/dfl_loss", []),
        }

    cls = {}
    if cls_raw:
        cls = {
            "epochs": [int(e) for e in cls_raw.get("epoch", [])],
            "train_loss": cls_raw.get("train/loss", []),
            "accuracy_top1": cls_raw.get("metrics/accuracy_top1", []),
            "accuracy_top5": cls_raw.get("metrics/accuracy_top5", []),
            "val_loss": cls_raw.get("val/loss", []),
        }

    return JSONResponse({"detection": det, "classification": cls})


@app.get("/api/training/image/{model_type}/{filename}")
def training_image(model_type: str, filename: str):
    if ".." in filename or "/" in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    base = DET_RUNS_DIR if model_type == "detect" else CLS_RUNS_DIR
    path = Path(base) / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path))


@app.get("/api/training/available-images")
def training_available_images():
    def _list_images(directory: str) -> list:
        d = Path(directory)
        if not d.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        return sorted(f.name for f in d.iterdir() if f.suffix.lower() in exts)

    return JSONResponse({
        "detection": _list_images(DET_RUNS_DIR),
        "classification": _list_images(CLS_RUNS_DIR),
    })


# ── 启动 ──────────────────────────────────────────────────────────────────────

def _kill_port(port: int) -> None:
    try:
        result = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
        for pid in result.stdout.strip().split():
            if pid and int(pid) != os.getpid():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    except Exception:
        return
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.1)


def main():
    global det_model, cls_model
    print("加载模型...")
    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)
    print("模型加载完成")

    threading.Thread(target=run_detection, daemon=True).start()

    _kill_port(PORT)
    print(f"网页: http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
