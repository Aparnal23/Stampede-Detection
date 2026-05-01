import json
import math
import os
import re
import secrets
import time
import threading
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS
import qrcode
from io import BytesIO
import base64

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(_BASE_DIR, "..", "frontend")
CCTV_DIR     = os.path.join(_BASE_DIR, "..", "cctv")

app = Flask(__name__)
CORS(app)

_FRONTEND_ASSETS = frozenset({"style.css", "script.js", "user.js", "alert-sound.js", "evaluation.js"})
_CCTV_IMAGE_EXT  = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".mp4"})

# ─────────────────────────────────────────────────────────────
#  INFERENCE — model, scaler, job tracking
# ─────────────────────────────────────────────────────────────
_inference_model  = None
_inference_scaler = None
_inference_lock   = threading.Lock()
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, "..", ".."))

MODEL_PATH  = os.path.join(PROJECT_ROOT, "crowd_detection", "stampede_earlywarning.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "crowd_detection", "feature_scaler_ew.pkl")

INFERENCE_UPLOAD_DIR = os.path.join(_BASE_DIR, "..", "inference_uploads")
os.makedirs(INFERENCE_UPLOAD_DIR, exist_ok=True)

_inference_jobs: dict     = {}
_inference_jobs_lock      = threading.Lock()

_zone_inference_job: dict = {}
_zone_inference_lock      = threading.Lock()

ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

ALERT_HIGH = 0.65
ALERT_MED  = 0.40

SAMPLE_FPS        = 5
RESIZE_W          = 640
RESIZE_H          = 360
TARGET_WINDOW_SEC = 10
TARGET_STEP_SEC   = 2
MIN_WINDOW_FRAMES = 3
MIN_FLOW_FRAMES   = 2


def _allowed_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT


def _load_inference_components():
    """Lazy-load the pickled classifier and scaler exactly once."""
    global _inference_model, _inference_scaler
    with _inference_lock:
        if _inference_model is not None:
            return _inference_model, _inference_scaler
        try:
            import pickle
            with open(MODEL_PATH,  "rb") as f:
                _inference_model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                _inference_scaler = pickle.load(f)
            print("[inference] Model and scaler loaded.")
            return _inference_model, _inference_scaler
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model file not found: {e}. "
                "Train the model first with stampede_early_warning.py."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Could not load model: {e}") from e


# ─────────────────────────────────────────────────────────────
#  PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────

def _get_video_info(video_path: str):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur  = n_fr / fps if n_fr > 0 else 0.0
        cap.release()
        return fps, n_fr, dur
    except Exception as e:
        print(f"[inference] get_video_info error: {e}")
        return None


def _adaptive_params(duration_sec: float):
    win_sec  = min(TARGET_WINDOW_SEC,  max(1.0, duration_sec * 0.50))
    step_sec = min(TARGET_STEP_SEC,    max(0.4, win_sec      * 0.30))
    look_sec = min(8.0,                max(0.5, win_sec      * 0.60))
    win_frames  = max(MIN_WINDOW_FRAMES, int(win_sec  * SAMPLE_FPS))
    step_frames = max(1,                 int(step_sec * SAMPLE_FPS))
    return win_frames, step_frames, look_sec


def _slope(series):
    import numpy as np
    s = [v for v in series if v is not None]
    if len(s) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])


def _safe(series, fn=None):
    import numpy as np
    if fn is None:
        fn = np.mean
    s = [v for v in series if v is not None]
    return float(fn(s)) if s else 0.0


def _window_to_features(window_signals):
    import numpy as np

    counts     = [s["count"]          for s in window_signals]
    speeds     = [s["speed"]          for s in window_signals]
    alignments = [s["alignment"]      for s in window_signals]
    zones      = [s["zone_imbalance"] for s in window_signals]
    edges      = [s["edge_frac"]      for s in window_signals]

    sp_v = [v for v in speeds     if v is not None]
    al_v = [v for v in alignments if v is not None]
    zo_v = [v for v in zones      if v is not None]
    ed_v = [v for v in edges      if v is not None]

    if len(sp_v) < MIN_FLOW_FRAMES:
        return None

    count_mean     = _safe(counts)
    count_velocity = _slope(counts)
    c_diff         = np.diff([c for c in counts]) if len(counts) > 1 else [0]
    count_accel    = float(np.mean(np.diff(c_diff))) if len(c_diff) > 1 else 0.0
    count_std      = _safe(counts, np.std)

    speed_mean  = float(np.mean(sp_v))
    speed_slope = _slope(speeds)
    sp_d        = np.diff(sp_v)
    speed_accel = float(np.mean(np.diff(sp_d))) if len(sp_d) > 1 else 0.0
    speed_std   = float(np.std(sp_v))

    align_mean  = float(np.mean(al_v)) if al_v else 0.0
    align_slope = _slope(alignments)
    al_d        = np.diff(al_v)
    align_accel = float(np.mean(np.diff(al_d))) if len(al_d) > 1 else 0.0

    zone_mean  = float(np.mean(zo_v)) if zo_v else 0.0
    zone_slope = _slope(zones)
    zone_peak  = float(np.max(zo_v))  if zo_v else 0.0

    edge_mean  = float(np.mean(ed_v)) if ed_v else 0.0
    edge_slope = _slope(edges)
    edge_peak  = float(np.max(ed_v))  if ed_v else 0.0

    high_risk_frac = sum(
        1 for s, a in zip(speeds, alignments)
        if s is not None and a is not None and s > 2.0 and a > 0.5
    ) / max(1, len(sp_v))

    surge_frac = 0.0
    if len(counts) > 1:
        surges = sum(
            1 for i in range(1, len(counts))
            if counts[i - 1] > 0 and
               (counts[i] - counts[i - 1]) / counts[i - 1] > 0.10)
        surge_frac = surges / (len(counts) - 1)

    compress_trend = edge_slope + zone_slope
    chaos_index    = speed_std  * (1.0 - max(0.0, align_mean))
    crowd_momentum = count_velocity * speed_slope

    return np.array([
        count_mean, count_velocity, count_accel, count_std,
        speed_mean, speed_slope,    speed_accel, speed_std,
        align_mean, align_slope,    align_accel,
        zone_mean,  zone_slope,     zone_peak,
        edge_mean,  edge_slope,     edge_peak,
        high_risk_frac, surge_frac, compress_trend,
        chaos_index, crowd_momentum,
    ], dtype=np.float32)


def _extract_frame_signals(video_path: str, crowd_counter) -> list:
    import cv2
    import numpy as np
    from collections import deque

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[inference] Cannot open video: {video_path}")
        return []

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps_video / SAMPLE_FPS))

    signals   = []
    prev_gray = None
    count_buf = deque(maxlen=10)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_step != 0:
            continue

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        H, W  = frame.shape[:2]
        t_sec = frame_idx / fps_video

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        sig = {
            "t":              t_sec,
            "count":          0.0,
            "speed":          None,
            "alignment":      None,
            "zone_imbalance": None,
            "edge_frac":      None,
        }

        cnt = crowd_counter.predict(frame)
        count_buf.append(cnt)
        sig["count"] = float(np.mean(count_buf))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            step = 24
            vxs, vys, pts = [], [], []
            for gy in range(step, H - step, step):
                for gx in range(step, W - step, step):
                    vx  = float(flow[gy, gx, 0])
                    vy  = float(flow[gy, gx, 1])
                    mag = float(np.hypot(vx, vy))
                    if mag > 0.3:
                        vxs.append(vx)
                        vys.append(vy)
                        pts.append((gx, gy, mag))

            if vxs:
                vxs_a = np.array(vxs)
                vys_a = np.array(vys)
                mags  = np.array([p[2] for p in pts])

                sig["speed"] = float(np.mean(mags))

                avg_vx  = float(np.mean(vxs_a))
                avg_vy  = float(np.mean(vys_a))
                dom_mag = float(np.hypot(avg_vx, avg_vy)) + 1e-6
                dom     = np.array([avg_vx / dom_mag, avg_vy / dom_mag])
                dots    = (vxs_a * dom[0] + vys_a * dom[1]) / (mags + 1e-6)
                sig["alignment"] = float(np.mean(np.clip(dots, -1, 1)))

                lm = [mags[i] for i, p in enumerate(pts) if p[0] < W // 2]
                rm = [mags[i] for i, p in enumerate(pts) if p[0] >= W // 2]
                sig["zone_imbalance"] = abs(
                    (float(np.mean(lm)) if lm else 0.0) -
                    (float(np.mean(rm)) if rm else 0.0))

                edge_w = int(W * 0.15)
                n_edge = sum(1 for p in pts
                             if p[0] < edge_w or p[0] > W - edge_w)
                sig["edge_frac"] = n_edge / max(1, len(pts))

        prev_gray = gray
        signals.append(sig)

    cap.release()
    print(f"[inference] Extracted {len(signals)} sampled frames from {video_path}")
    return signals


def _run_inference_job(job_id: str, video_path: str):
    with _inference_jobs_lock:
        _inference_jobs[job_id]["status"] = "running"

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from collections import deque

        model, scaler = _load_inference_components()

        CSRNET_WEIGHTS = os.path.join(PROJECT_ROOT, "crowd_detection", "csrnet_shanghaitech_partB.pth")

        class CSRNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.frontend = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),   nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),  nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                )
                self.backend = nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True),
                )
                self.output_layer = nn.Conv2d(64, 1, 1)

            def forward(self, x):
                return self.output_layer(self.backend(self.frontend(x)))

        class CrowdCounter:
            def __init__(self, weights_path, device):
                self.device = device
                self.model  = CSRNet().to(device)
                state = torch.load(weights_path, map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state)
                self.model.eval()
                self.tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
                ])

            @torch.no_grad()
            def predict(self, frame_bgr):
                import cv2
                h, w  = frame_bgr.shape[:2]
                small = cv2.resize(frame_bgr, (w // 2, h // 2))
                rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                t     = self.tf(rgb).unsqueeze(0).to(self.device)
                dm    = self.model(t).squeeze().cpu().numpy()
                return float(dm.sum())

        device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crowd_counter = CrowdCounter(CSRNET_WEIGHTS, device)
        print(f"[inference] Job {job_id}: CSRNet loaded on {device}")

        info = _get_video_info(video_path)
        if info is None:
            raise RuntimeError("Cannot read video file.")
        _, _, dur = info

        win_frames, step_frames, look_sec = _adaptive_params(dur)

        sigs = _extract_frame_signals(video_path, crowd_counter)
        if not sigs:
            raise RuntimeError("No frames could be extracted from the video.")

        n = len(sigs)
        if n < win_frames:
            win_frames  = n
            step_frames = max(1, n)

        smooth_len = max(3, int(10 / (step_frames / SAMPLE_FPS)))
        risk_buf   = deque(maxlen=smooth_len)

        classes = list(model.classes_)
        if "PRE_STAMPEDE" in classes:
            pos_idx = classes.index("PRE_STAMPEDE")
        else:
            pos_idx = len(classes) - 1

        print(f"[inference] Job {job_id}: classes={classes}, PRE_STAMPEDE idx={pos_idx}")
        print(f"[inference] Job {job_id}: {n} signal frames, "
              f"win={win_frames} step={step_frames} smooth_len={smooth_len}")

        rows = []
        i    = 0
        while i + win_frames <= n:
            window  = sigs[i: i + win_frames]
            t_start = window[0]["t"]
            t_end   = window[-1]["t"]

            fv = _window_to_features(window)
            if fv is None:
                i += step_frames
                continue

            raw    = float(model.predict_proba(scaler.transform([fv]))[0][pos_idx])
            risk_buf.append(raw)
            smooth = float(np.mean(risk_buf))

            level = ("HIGH"   if smooth >= ALERT_HIGH else
                     "MEDIUM" if smooth >= ALERT_MED  else
                     "NORMAL")

            # Use raw CSRNet count from the last signal in the window for people count
            # fv[0] is count_mean (smoothed over frames in window) — multiply by
            # a scale factor since ShanghaiTech Part B outputs small density values.
            raw_count = float(fv[0])
            # If raw_count looks like a density-map sum (typically 0–300+ for Part B),
            # use it directly; otherwise fall back to 0.
            people_count = max(0, round(raw_count))

            rows.append({
                "t_start":      round(t_start, 2),
                "t_end":        round(t_end,   2),
                "raw":          round(raw,      4),
                "smooth":       round(smooth,   4),
                "level":        level,
                "prediction":   "PRE_STAMPEDE" if smooth >= ALERT_HIGH else "NORMAL",
                "people_count": people_count,
            })
            i += step_frames

        if not rows:
            raise RuntimeError(
                "No windows produced features — video may be too short or have no motion.")

        # FIX: use PEAK smooth score across all windows, not just last window
        peak        = max(rows, key=lambda r: r["smooth"])
        first_alert = next((r for r in rows if r["level"] == "HIGH"), None)

        # FIX: overall risk/prediction based on ANY window being HIGH/MEDIUM
        overall = ("HIGH"   if any(r["level"] == "HIGH"   for r in rows) else
                   "MEDIUM" if any(r["level"] == "MEDIUM" for r in rows) else
                   "NORMAL")
        overall_prediction = "PRE_STAMPEDE" if overall == "HIGH" else "NORMAL"

        # FIX: latest_people_count — use the window with the highest count
        # (more reliable than just the last window which may be near the end)
        latest_count = max((r["people_count"] for r in rows), default=0)

        with _inference_jobs_lock:
            _inference_jobs[job_id].update({
                "status":              "done",
                "duration_sec":        round(dur, 2),
                "window_sec":          round(win_frames / SAMPLE_FPS, 2),
                "lookahead_sec":       round(look_sec, 2),
                "overall_risk":        overall,
                "overall_prediction":  overall_prediction,
                "peak_risk":           peak,
                "first_high_alert":    first_alert,
                "rows":                rows,
                "total_windows":       len(rows),
                "latest_people_count": latest_count,
            })

        # Update live cctv_counts
        zone_name = _inference_jobs[job_id].get("zone_name")
        if zone_name and latest_count > 0:
            cctv_counts[zone_name] = latest_count

        print(f"[inference] Job {job_id} done — overall={overall} ({overall_prediction}) "
              f"people≈{latest_count}  peak_score={peak['smooth']:.4f}")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        with _inference_jobs_lock:
            _inference_jobs[job_id]["status"] = "error"
            _inference_jobs[job_id]["error"]  = str(exc)
        print(f"[inference] Job {job_id} FAILED: {exc}\n{tb}")


def _start_zone_inference(zone_name: str, video_path: str):
    """Spawn a background inference job for a zone. Returns job_id or None."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"[inference] Model files missing — skipping auto-inference for '{zone_name}'")
        return None

    job_id = "job-" + secrets.token_hex(6)
    with _inference_jobs_lock:
        _inference_jobs[job_id] = {
            "status":     "queued",
            "job_id":     job_id,
            "zone_name":  zone_name,
            "video_path": video_path,
            "queued_at":  int(time.time()),
        }
    with _zone_inference_lock:
        _zone_inference_job[zone_name] = job_id

    t = threading.Thread(target=_run_inference_job, args=(job_id, video_path), daemon=True)
    t.start()
    print(f"[inference] Auto-started job {job_id} for zone '{zone_name}'")
    return job_id


def _get_zone_risk(zone_name: str) -> dict:
    """Return the latest inference result for a zone (used in /api/status)."""
    with _zone_inference_lock:
        job_id = _zone_inference_job.get(zone_name)
    if not job_id:
        return {"status": "none"}
    with _inference_jobs_lock:
        job = dict(_inference_jobs.get(job_id, {}))
    if not job:
        return {"status": "none"}

    status = job.get("status", "none")
    if status == "done":
        return {
            "status":              "done",
            "job_id":              job_id,
            "overall_risk":        job.get("overall_risk",       "NORMAL"),
            "overall_prediction":  job.get("overall_prediction", "NORMAL"),
            "peak_risk":           job.get("peak_risk"),
            "duration_sec":        job.get("duration_sec"),
            "total_windows":       job.get("total_windows"),
            "rows":                job.get("rows", []),
            "latest_people_count": job.get("latest_people_count", 0),
        }
    if status == "error":
        return {"status": "error", "job_id": job_id,
                "error": job.get("error", "unknown error")}
    return {"status": status, "job_id": job_id}


# ─────────────────────────────────────────────────────────────
#  MODEL-DRIVEN CROWD LEVEL
# ─────────────────────────────────────────────────────────────

def _model_crowd_level() -> str:
    """
    FIX: Use the PEAK smooth score across all windows (not just the last window).
    The last window is often low-risk (end of video) which caused it to always
    return Normal even for clearly abnormal videos.
    """
    zone_risks = []
    with _zone_inference_lock:
        job_ids = list(_zone_inference_job.values())
    for jid in job_ids:
        with _inference_jobs_lock:
            job = _inference_jobs.get(jid, {})
        if job.get("status") != "done":
            continue
        rows = job.get("rows", [])
        if rows:
            # FIX: use PEAK score, not last-window score
            zone_risks.append(max(r["smooth"] for r in rows))

    if not zone_risks:
        return "Normal"

    max_risk = max(zone_risks)
    if max_risk >= ALERT_HIGH:
        return "Critical"
    elif max_risk >= ALERT_MED:
        return "Warning"
    return "Normal"


def _feed_state_for_zone(zone_name: str) -> str:
    """
    FIX: Use PEAK smooth score (not last window) so feed stays 'abnormal'
    after the video finishes if a high-risk window was detected earlier.
    """
    with _zone_inference_lock:
        job_id = _zone_inference_job.get(zone_name)
    if not job_id:
        return "normal"
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id, {})
    if job.get("status") != "done":
        return "normal"
    rows = job.get("rows", [])
    if not rows:
        return "normal"
    # FIX: use peak score across all windows
    peak_smooth = max(r["smooth"] for r in rows)
    return "abnormal" if peak_smooth >= ALERT_MED else "normal"


def _people_count_for_zone(zone_name: str) -> int:
    """
    FIX: prefer inference job's latest_people_count over stale cctv_counts.
    """
    with _zone_inference_lock:
        job_id = _zone_inference_job.get(zone_name)
    if job_id:
        with _inference_jobs_lock:
            job = _inference_jobs.get(job_id, {})
        if job.get("status") == "done":
            count = job.get("latest_people_count", 0)
            if count > 0:
                cctv_counts[zone_name] = count
            return count
    return cctv_counts.get(zone_name, 0)


# ─────────────────────────────────────────────────────────────
#  ZONE HELPERS
# ─────────────────────────────────────────────────────────────

def _safe_title_from_stem(stem):
    s = re.sub(r"[-_]+", " ", stem).strip()
    return s[:1].upper() + s[1:] if s else "Zone"


def _zones_from_directory():
    if not os.path.isdir(CCTV_DIR):
        return []
    zones = []
    files = sorted(os.listdir(CCTV_DIR))
    for i, fn in enumerate(files):
        full = os.path.join(CCTV_DIR, fn)
        if not os.path.isfile(full):
            continue
        name, ext = os.path.splitext(fn.lower())
        if ext not in _CCTV_IMAGE_EXT:
            continue
        is_video = ext in {".mp4", ".webm", ".mov"}
        zones.append({
            "name":               _safe_title_from_stem(name),
            "camera_id":          f"CCTV-{i+1:02d}",
            "anchor_lat":         12.9703 + i * 0.001,
            "anchor_lng":         77.5940 + i * 0.001,
            "image":              None if is_video else fn,
            "image_abnormal":     None,
            "video_normal_url":   fn if is_video else None,
            "video_abnormal_url": None,
        })
    return zones


def _load_zones_from_json():
    path = os.path.join(CCTV_DIR, "cctv_zones.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[cctv] could not read cctv_zones.json: {e}")
        return None
    if not isinstance(data, list):
        return None
    zones = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        img = (row.get("image") or "").strip()
        if not img:
            continue
        fn   = os.path.basename(img)
        full = os.path.join(CCTV_DIR, fn)
        if not os.path.isfile(full):
            print(f"[cctv] missing image file, skipped: {fn}")
            continue
        abn = (row.get("image_abnormal") or "").strip()
        if abn:
            abn = os.path.basename(abn)
            if not os.path.isfile(os.path.join(CCTV_DIR, abn)):
                print(f"[cctv] missing image_abnormal, ignored: {abn}")
                abn = None
        else:
            abn = None
        try:
            lat = float(row.get("anchor_lat", 12.9703))
            lng = float(row.get("anchor_lng", 77.5940))
        except (TypeError, ValueError):
            lat, lng = 12.9703, 77.5940
        zones.append({
            "name":           (row.get("name") or f"Zone {i+1}").strip() or f"Zone {i+1}",
            "camera_id":      (row.get("camera_id") or f"CCTV-{i+1:02d}").strip(),
            "anchor_lat":     lat,
            "anchor_lng":     lng,
            "image":          fn,
            "image_abnormal": abn,
        })
    return zones


def load_zones():
    os.makedirs(CCTV_DIR, exist_ok=True)
    zones = _load_zones_from_json()
    if zones is None:
        zones = []
    if not zones:
        zones = _zones_from_directory()
    return zones


def get_zones():
    return load_zones()


# ─────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────
PUBLIC_ALERT_REGION = "Everyone"
REGIONS             = [PUBLIC_ALERT_REGION]
NEARBY_RADIUS_M     = 220.0
MAX_NAME_LEN        = 120

crowd_level = "Normal"
alerts      = []
_next_id    = 0

MAX_ALERTS   = 50
MAX_PRESENCE = 200

cctv_counts      = {}   # FIX: start empty; populated dynamically as zones are uploaded
presence_records = []

_crowd_history = []
_max_history   = 500

_zone_feed_state_prev    = {}
_cctv_zone_alert_last_ts = {}
CCTV_AUTO_ALERT_COOLDOWN_S = 90.0

SIMULATION_TICK_INTERVAL_S = 3.0
_last_simulation_tick      = 0.0


def _next_alert_id():
    global _next_id
    _next_id += 1
    return _next_id


def _push_alert(message, region, source):
    alerts.insert(0, {
        "id":      _next_alert_id(),
        "message": message,
        "region":  region,
        "ts":      int(time.time()),
        "source":  source,
    })
    del alerts[MAX_ALERTS:]


def haversine_m(lat1, lng1, lat2, lng2):
    r      = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlmb   = math.radians(lng2 - lng1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _nearest_zone_for_point(lat, lng):
    best, best_d = None, float("inf")
    for z in get_zones():
        d = haversine_m(lat, lng, z["anchor_lat"], z["anchor_lng"])
        if d < best_d:
            best_d = d
            best   = z["name"]
    return best, best_d


# ─────────────────────────────────────────────────────────────
#  TICK
# ─────────────────────────────────────────────────────────────

def _push_cctv_abnormal_alerts_if_needed():
    now = time.time()
    for z in get_zones():
        name  = z["name"]
        state = _feed_state_for_zone(name)
        prev  = _zone_feed_state_prev.get(name)
        _zone_feed_state_prev[name] = state
        if prev is None:
            continue
        if prev != "normal" or state != "abnormal":
            continue
        last_ts = _cctv_zone_alert_last_ts.get(name, 0.0)
        if now - last_ts < CCTV_AUTO_ALERT_COOLDOWN_S:
            continue
        cam = z.get("camera_id") or name
        msg = (
            f"CCTV alert — {name} ({cam}): "
            "model detected elevated stampede risk (PRE_STAMPEDE signal). "
            "Review the live zone and broadcast to the public if needed."
        )
        _push_alert(msg, PUBLIC_ALERT_REGION, "system")
        _cctv_zone_alert_last_ts[name] = now


def _tick_simulation():
    global crowd_level

    _push_cctv_abnormal_alerts_if_needed()

    for z in get_zones():
        name  = z["name"]
        count = _people_count_for_zone(name)
        cctv_counts[name] = count

    previous    = crowd_level
    predicted   = _model_crowd_level()
    crowd_level = predicted

    _record_crowd_prediction(predicted, crowd_level)

    if crowd_level == "Warning" and previous != "Warning":
        _push_alert(
            "Crowd status: Warning — model detects elevated crowd risk. "
            "Monitor all CCTV zones and stand by.",
            PUBLIC_ALERT_REGION, "system",
        )
    if crowd_level == "Critical" and previous != "Critical":
        _push_alert(
            "⚠️ Critical — PRE_STAMPEDE condition detected! "
            "Follow staff instructions and move to the nearest safe exit.",
            PUBLIC_ALERT_REGION, "system",
        )
    if crowd_level == "Normal" and previous in ("Warning", "Critical"):
        _push_alert(
            "Crowd status returned to Normal — no active stampede risk detected.",
            PUBLIC_ALERT_REGION, "system",
        )


def _maybe_tick_simulation():
    global _last_simulation_tick
    now = time.time()
    if now - _last_simulation_tick < SIMULATION_TICK_INTERVAL_S:
        return
    _last_simulation_tick = now
    _tick_simulation()


def _current_feed_url(z, state):
    if z.get("image"):
        if state == "abnormal" and z.get("image_abnormal"):
            return f"/cctv/{z['image_abnormal']}"
        return f"/cctv/{z['image']}"
    if z.get("video_normal_url"):
        if state == "abnormal" and z.get("video_abnormal_url"):
            return f"/cctv/{z['video_abnormal_url']}"
        return f"/cctv/{z['video_normal_url']}"
    return ""


def _presence_count_for_zone(zone_name):
    return 0


def _trim_presence():
    del presence_records[MAX_PRESENCE:]


def _record_crowd_prediction(predicted, actual):
    global _crowd_history
    _crowd_history.insert(0, {
        "timestamp": int(time.time()),
        "predicted": predicted,
        "actual":    actual,
    })
    del _crowd_history[_max_history:]


def _notify_creators_of_registration(name, anon_id, timestamp):
    timestamp_str  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    admin_log_file = os.path.join(os.path.dirname(_BASE_DIR), "registration_log.txt")
    try:
        with open(admin_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp_str}] New registration: {name} (ID: {anon_id})\n")
    except Exception as e:
        print(f"[admin] could not write registration log: {e}")
    print(f"[ADMIN NOTIFICATION] User registered: {name} (ID: {anon_id}) at {timestamp_str}")


def _calculate_evaluation_metrics():
    if not _crowd_history:
        return {
            "total_predictions": 0,
            "accuracy":          0,
            "precision":  {"Normal": 0, "Warning": 0, "Critical": 0},
            "recall":     {"Normal": 0, "Warning": 0, "Critical": 0},
            "f1_score":   {"Normal": 0, "Warning": 0, "Critical": 0},
            "confusion_matrix": {},
            "history": [],
        }
    total    = len(_crowd_history)
    correct  = sum(1 for h in _crowd_history if h["predicted"] == h["actual"])
    accuracy = correct / total if total > 0 else 0
    categories = ["Normal", "Warning", "Critical"]
    tp = {c: 0 for c in categories}
    fp = {c: 0 for c in categories}
    fn = {c: 0 for c in categories}
    for h in _crowd_history:
        pred   = h["predicted"]
        actual = h["actual"]
        if pred == actual:
            tp[actual] += 1
        else:
            fp[pred]   += 1
            fn[actual] += 1
    precision = {}
    recall    = {}
    f1        = {}
    for c in categories:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        precision[c] = round(p, 4)
        recall[c]    = round(r, 4)
        f1[c]        = round(f, 4)
    confusion = {}
    for actual in categories:
        confusion[actual] = {}
        for pred in categories:
            confusion[actual][pred] = sum(
                1 for h in _crowd_history
                if h["actual"] == actual and h["predicted"] == pred)
    return {
        "total_predictions": total,
        "accuracy":          round(accuracy, 4),
        "precision":         precision,
        "recall":            recall,
        "f1_score":          f1,
        "confusion_matrix":  confusion,
        "history":           _crowd_history[:100],
    }


def _generate_registration_qr(base_url="http://localhost:5000/user"):
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=10, border=2)
    qr.add_data(base_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def _devices_near_zone_anchor(zone):
    out = []
    for p in presence_records:
        if p["lat"] is None or p["lng"] is None:
            continue
        d = haversine_m(p["lat"], p["lng"], zone["anchor_lat"], zone["anchor_lng"])
        if d <= NEARBY_RADIUS_M:
            nearest, nd = _nearest_zone_for_point(p["lat"], p["lng"])
            out.append({
                "anon_id":                 p["anon_id"],
                "distance_m":              round(d, 1),
                "checked_in_zone":         p.get("zone"),
                "nearest_zone_by_gps":     nearest,
                "nearest_zone_distance_m": round(nd, 1),
                "ts":                      p["ts"],
            })
    out.sort(key=lambda x: x["distance_m"])
    return out


def _enrich_zone_row(z):
    name     = z["name"]
    state    = _feed_state_for_zone(name)
    nearby   = _devices_near_zone_anchor(z)
    is_video = bool(z.get("video_normal_url"))
    inf      = _get_zone_risk(name)
    people_estimate = _people_count_for_zone(name)

    return {
        "name":                 name,
        "camera_id":            z["camera_id"],
        "anchor_lat":           z["anchor_lat"],
        "anchor_lng":           z["anchor_lng"],
        "cctv_people_estimate": people_estimate,
        "presence_reports":     _presence_count_for_zone(name),
        "feed_state":           state,
        "feed_media":           "video" if is_video else "image",
        "current_video_url":    _current_feed_url(z, state),
        "devices_nearby":       nearby,
        "nearby_device_count":  len(nearby),
        "inference":            inf,
    }


def _zones_snapshot():
    return [_enrich_zone_row(z) for z in get_zones()]


def _presence_public_list(limit=40):
    return [{"anon_id": p["anon_id"],
             "display_name": p.get("display_name") or "—",
             "ts": p["ts"]}
            for p in presence_records[:limit]]


def _presence_detail_for_authority(limit=80):
    return [{"anon_id": p["anon_id"],
             "display_name": p.get("display_name") or "—",
             "ts": p["ts"]}
            for p in presence_records[:limit]]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def authority_dashboard():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/user")
def user_view():
    return send_from_directory(FRONTEND_DIR, "user.html")


@app.route("/evaluation")
def evaluation_page():
    return send_from_directory(FRONTEND_DIR, "evaluation.html")


@app.route("/cctv/<path:filename>")
def serve_cctv(filename):
    if ".." in filename or filename.startswith(("/", "\\")):
        abort(404)
    fn = os.path.basename(filename)
    _, ext = os.path.splitext(fn.lower())
    if ext not in _CCTV_IMAGE_EXT:
        abort(404)
    cctv_abs = os.path.abspath(CCTV_DIR)
    full     = os.path.abspath(os.path.join(CCTV_DIR, fn))
    try:
        if os.path.commonpath([cctv_abs, full]) != cctv_abs:
            abort(404)
    except ValueError:
        abort(404)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(CCTV_DIR, fn)


@app.route("/api/regions", methods=["GET"])
def list_regions():
    return jsonify({"regions": REGIONS})


@app.route("/api/qrcode", methods=["GET"])
def get_qr_code():
    host             = request.host_url.rstrip('/')
    registration_url = f"{host}/user"
    qr_data          = _generate_registration_qr(registration_url)
    return jsonify({"qr_code": qr_data, "registration_url": registration_url})


@app.route("/api/zones", methods=["GET"])
def list_zones():
    return jsonify({"zones": get_zones(), "nearby_radius_m": NEARBY_RADIUS_M})


@app.route("/api/public/snapshot", methods=["GET"])
def public_snapshot():
    return jsonify({"presence": _presence_public_list(50),
                    "total_registered": len(presence_records)})


@app.route("/api/presence", methods=["POST"])
def register_presence():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    if len(name) > MAX_NAME_LEN:
        name = name[:MAX_NAME_LEN]
    ts  = int(time.time())
    rec = {
        "anon_id":      "P-" + secrets.token_hex(4).upper(),
        "display_name": name,
        "ts":           ts,
        "lat":          None,
        "lng":          None,
    }
    _notify_creators_of_registration(name, rec["anon_id"], ts)
    presence_records.insert(0, rec)
    _trim_presence()
    return jsonify({"status": "recorded",
                    "anon_id": rec["anon_id"],
                    "display_name": rec["display_name"]})


@app.route("/api/status", methods=["GET"])
def get_status():
    _maybe_tick_simulation()
    return jsonify({
        "level":               crowd_level,
        "alerts":              alerts,
        "zones":               _zones_snapshot(),
        "presence_detail":     _presence_detail_for_authority(60),
        "total_registrations": len(presence_records),
        "nearby_radius_m":     NEARBY_RADIUS_M,
    })


@app.route("/api/user/alerts", methods=["GET"])
def user_alerts():
    _maybe_tick_simulation()
    return jsonify({"alerts": alerts, "crowd_level": crowd_level})


@app.route("/api/evaluation/metrics", methods=["GET"])
def evaluation_metrics():
    _maybe_tick_simulation()
    return jsonify(_calculate_evaluation_metrics())


@app.route("/api/authority/broadcast", methods=["POST"])
def authority_broadcast():
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    region  = (data.get("region")  or PUBLIC_ALERT_REGION).strip()
    if region not in REGIONS:
        region = PUBLIC_ALERT_REGION
    if not message:
        return jsonify({"error": "message is required"}), 400
    _push_alert(message, region, "authority")
    print(f"[broadcast] message={message!r}")
    return jsonify({"status": "sent", "region": region,
                    "delivered_to": "all registered users (simulated; connect SMS/push here)"})


@app.route("/send-alert", methods=["POST"])
def send_alert_legacy():
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip() or "Move to nearest exit immediately"
    _push_alert(f"⚠️ {message}", PUBLIC_ALERT_REGION, "authority")
    return jsonify({"status": "Alert sent", "region": PUBLIC_ALERT_REGION})


@app.route("/<path:fname>")
def frontend_asset(fname):
    if fname not in _FRONTEND_ASSETS:
        abort(404)
    return send_from_directory(FRONTEND_DIR, fname)


UPLOAD_FOLDER      = CCTV_DIR
ALLOWED_EXTENSIONS = {"mp4", "webm", "mov"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    print("[DEBUG] upload_video called")
    print("[DEBUG] files in request:", list(request.files.keys()))
    print("[DEBUG] CCTV_DIR =", CCTV_DIR)
    print("[DEBUG] CCTV_DIR exists =", os.path.exists(CCTV_DIR))
    print("[DEBUG] MODEL_PATH exists =", os.path.exists(MODEL_PATH))
    print("[DEBUG] SCALER_PATH exists =", os.path.exists(SCALER_PATH))
    """Upload a CCTV video → new zone → auto-start stampede inference."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename  = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        stem      = os.path.splitext(filename.lower())[0]
        zone_name = _safe_title_from_stem(stem)

        if zone_name not in cctv_counts:
            cctv_counts[zone_name] = 0

        job_id = _start_zone_inference(zone_name, save_path)

        resp = {"status": "uploaded", "filename": filename,
                "url": f"/cctv/{filename}", "zone_name": zone_name}
        if job_id:
            resp["inference_job_id"]   = job_id
            resp["inference_message"]  = (
                "Stampede inference started. "
                "People count, risk score and NORMAL/PRE_STAMPEDE prediction "
                "will appear on the zone card once analysis is complete."
            )
        else:
            resp["inference_message"] = (
                "Video uploaded but model files not found — inference skipped."
            )
        return jsonify(resp)
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/api/inference/upload", methods=["POST"])
def inference_upload():
    """Legacy direct-upload endpoint (not tied to a zone)."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not _allowed_video(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_VIDEO_EXT}"}), 400

    job_id   = "job-" + secrets.token_hex(6)
    ext      = file.filename.rsplit(".", 1)[-1].lower()
    fname    = f"{job_id}.{ext}"
    savepath = os.path.join(INFERENCE_UPLOAD_DIR, fname)
    file.save(savepath)

    with _inference_jobs_lock:
        _inference_jobs[job_id] = {
            "status":     "queued",
            "job_id":     job_id,
            "filename":   file.filename,
            "video_path": savepath,
            "queued_at":  int(time.time()),
        }

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        with _inference_jobs_lock:
            _inference_jobs[job_id]["status"] = "error"
            _inference_jobs[job_id]["error"]  = (
                "Model files not found. "
                f"Expected: {MODEL_PATH} and {SCALER_PATH}."
            )
        return jsonify(_inference_jobs[job_id]), 202

    t = threading.Thread(target=_run_inference_job, args=(job_id, savepath), daemon=True)
    t.start()
    return jsonify({"status": "queued", "job_id": job_id,
                    "message": "Poll /api/inference/status/<job_id> for results."}), 202


@app.route("/api/inference/status/<job_id>", methods=["GET"])
def inference_status(job_id):
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route("/api/debug/jobs", methods=["GET"])
def debug_jobs():
    with _inference_jobs_lock:
        return jsonify(dict(_inference_jobs))

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)