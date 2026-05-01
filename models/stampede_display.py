"""
stampede_display.py
===================
Visual real-time stampede detector with full feature overlay.

Displays the video with:
  - Risk level bar (colour-coded)
  - All 5 live signals: count, speed, alignment, zone imbalance, edge frac
  - Mini risk history graph
  - Large status banner (NORMAL / MEDIUM RISK / HIGH RISK - STAMPEDE ALERT)
  - Optical flow arrows overlaid on video
  - All 22 window features listed live

Usage
-----
  python stampede_display.py --video path/to/video.mp4
  python stampede_display.py --camera 0
  python stampede_display.py --video path/to/video.mp4 --no-flow
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
import os, pickle, argparse, warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIG  (must match stampede_early_warning.py)
# ─────────────────────────────────────────────────────────────
CSRNET_WEIGHTS = r"D:\mm\crowd_detection\crowd_detection (2)\crowd_detection\csrnet_shanghaitech_partB.pth"
MODEL_OUT      = "stampede_earlywarning.pkl"
SCALER_OUT     = "feature_scaler_ew.pkl"

SAMPLE_FPS        = 5
RESIZE_W          = 640
RESIZE_H          = 360
TARGET_WINDOW_SEC = 10
TARGET_STEP_SEC   = 2
MIN_WINDOW_FRAMES = 3
MIN_FLOW_FRAMES   = 2

ALERT_HIGH = 0.65
ALERT_MED  = 0.40

# Display layout
PANEL_W   = 320
GRAPH_H   = 100
HISTORY_N = 60

# Colours (BGR)
COL_GREEN  = (50,  200,  50)
COL_YELLOW = (0,   200, 220)
COL_RED    = (50,   50, 240)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0,     0,   0)
COL_BLUE   = (200, 120,  50)


# ─────────────────────────────────────────────────────────────
#  CSRNET
# ─────────────────────────────────────────────────────────────
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),   nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=2,dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,64, 3,padding=2,dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64,1,1)

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
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def predict(self, frame_bgr):
        h, w  = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (w//2, h//2))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t     = self.tf(rgb).unsqueeze(0).to(self.device)
        dm    = self.model(t).squeeze().cpu().numpy()
        return float(dm.sum())


# ─────────────────────────────────────────────────────────────
#  ADAPTIVE PARAMS
# ─────────────────────────────────────────────────────────────
def adaptive_params(duration_sec):
    win_sec  = min(TARGET_WINDOW_SEC,  max(1.0, duration_sec * 0.50))
    step_sec = min(TARGET_STEP_SEC,    max(0.4, win_sec      * 0.30))
    look_sec = min(8.0,                max(0.5, win_sec      * 0.60))
    win_frames  = max(MIN_WINDOW_FRAMES, int(win_sec  * SAMPLE_FPS))
    step_frames = max(1,                 int(step_sec * SAMPLE_FPS))
    return win_frames, step_frames, look_sec


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def _slope(series):
    s = [v for v in series if v is not None]
    if len(s) < 2: return 0.0
    return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])

def _safe(series, fn=np.mean):
    s = [v for v in series if v is not None]
    return float(fn(s)) if s else 0.0

def window_to_features(window_signals):
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
            if counts[i-1] > 0 and
               (counts[i] - counts[i-1]) / counts[i-1] > 0.10)
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


FEATURE_NAMES = [
    "count_mean","count_velocity","count_accel","count_std",
    "speed_mean","speed_slope","speed_accel","speed_std",
    "align_mean","align_slope","align_accel",
    "zone_mean","zone_slope","zone_peak",
    "edge_mean","edge_slope","edge_peak",
    "high_risk_frac","surge_frac","compress_trend",
    "chaos_index","crowd_momentum",
]


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def draw_text(img, text, pos, scale=0.55, color=COL_WHITE,
              thickness=1, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, scale, COL_BLACK, thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color,     thickness,   cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_val, color, label, val_str=None):
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,60,60), -1)
    fill = int(w * min(1.0, max(0.0, value / max(max_val, 1e-6))))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (100,100,100), 1)
    if label:
        draw_text(img, label, (x, y-5), scale=0.38, color=COL_WHITE)
    vs = val_str if val_str else f"{value:.2f}"
    draw_text(img, vs, (x+w+6, y+h-2), scale=0.38, color=color)


def draw_risk_graph(img, x, y, w, h, history):
    cv2.rectangle(img, (x, y), (x+w, y+h), (40,40,40), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (80,80,80), 1)
    for thr, col in [(ALERT_HIGH, COL_RED), (ALERT_MED, COL_YELLOW)]:
        ly = y + h - int(thr * h)
        cv2.line(img, (x, ly), (x+w, ly), col, 1)
    if len(history) < 2:
        return
    pts = list(history)
    n   = len(pts)
    coords = [(x + int(i * w / max(n-1,1)),
               y + h - int(v * h)) for i, v in enumerate(pts)]
    for i in range(1, len(coords)):
        v     = pts[i]
        color = (COL_RED if v >= ALERT_HIGH else
                 COL_YELLOW if v >= ALERT_MED else COL_GREEN)
        cv2.line(img, coords[i-1], coords[i], color, 2, cv2.LINE_AA)
    draw_text(img, "Risk History", (x+4, y+12), scale=0.38, color=(180,180,180))
    draw_text(img, "1.0", (x+w+4, y+10),  scale=0.32, color=(140,140,140))
    draw_text(img, "0.0", (x+w+4, y+h-2), scale=0.32, color=(140,140,140))


def draw_flow_arrows(frame, flow, step=32):
    H, W = frame.shape[:2]
    for gy in range(step, H-step, step):
        for gx in range(step, W-step, step):
            vx  = float(flow[gy, gx, 0])
            vy  = float(flow[gy, gx, 1])
            mag = float(np.hypot(vx, vy))
            if mag < 0.5:
                continue
            scale = min(mag * 3, 20)
            ex    = int(gx + vx / mag * scale)
            ey    = int(gy + vy / mag * scale)
            alpha = min(1.0, mag / 5.0)
            color = (int(50*alpha), int(200*alpha), int(50*alpha))
            cv2.arrowedLine(frame, (gx, gy), (ex, ey),
                            color, 1, tipLength=0.35)


def draw_status_banner(frame, smooth_risk, alert_count):
    H, W = frame.shape[:2]
    if smooth_risk >= ALERT_HIGH:
        bg, text, col = (0,0,180), "  !! STAMPEDE ALERT !!  HIGH RISK DETECTED", COL_WHITE
    elif smooth_risk >= ALERT_MED:
        bg, text, col = (0,140,180), "  MEDIUM RISK — Monitor Crowd Closely", COL_BLACK
    else:
        bg, text, col = (30,120,30), "  NORMAL — Crowd Behaviour OK", COL_WHITE

    cv2.rectangle(frame, (0, 0), (W, 36), bg, -1)
    cv2.putText(frame, text, (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.68, col, 2, cv2.LINE_AA)

    # Flashing red border on HIGH alert
    if smooth_risk >= ALERT_HIGH and alert_count % 10 < 5:
        cv2.rectangle(frame, (0, 0), (W-1, H-1), COL_RED, 5)


# ─────────────────────────────────────────────────────────────
#  INFO PANEL
# ─────────────────────────────────────────────────────────────
_last_fv = None   # global so it persists across frames

def build_info_panel(panel_h, signals_buf, smooth_risk,
                     raw_risk, risk_history, t_now):
    global _last_fv
    panel = np.full((panel_h, PANEL_W, 3), 20, dtype=np.uint8)
    x0    = 12
    y     = 14
    bw    = PANEL_W - x0*2 - 44     # bar width

    # ── Title ──
    draw_text(panel, "STAMPEDE DETECTOR", (x0, y),
              scale=0.50, color=(200,200,255), bold=True)
    y += 20
    draw_text(panel, f"t = {t_now:6.1f}s", (x0, y),
              scale=0.40, color=(160,160,160))
    y += 16
    cv2.line(panel, (x0, y), (PANEL_W-x0, y), (60,60,60), 1)
    y += 8

    # ── Risk score ──
    risk_col = (COL_RED if smooth_risk >= ALERT_HIGH else
                COL_YELLOW if smooth_risk >= ALERT_MED else COL_GREEN)

    draw_text(panel, "RISK SCORE", (x0, y), scale=0.43, color=(180,180,180))
    y += 16
    draw_bar(panel, x0, y, bw, 16, smooth_risk, 1.0,
             risk_col, "", f"{smooth_risk:.3f}")
    y += 26
    draw_text(panel, f"Raw: {raw_risk:.3f}   Smooth: {smooth_risk:.3f}",
              (x0, y), scale=0.38, color=risk_col)
    y += 18
    cv2.line(panel, (x0, y), (PANEL_W-x0, y), (60,60,60), 1)
    y += 8

    # ── Live signals (5 bars) ──
    draw_text(panel, "LIVE SIGNALS", (x0, y), scale=0.43, color=(180,180,180))
    y += 16

    if signals_buf:
        last  = signals_buf[-1]
        count = last["count"]
        speed = last["speed"] or 0.0
        align = last["alignment"] or 0.0
        zone  = last["zone_imbalance"] or 0.0
        edge  = last["edge_frac"] or 0.0

        draw_bar(panel, x0, y, bw, 12, min(count,300), 300,
                 COL_BLUE,   "People Count",   f"{count:.0f}")
        y += 24
        draw_bar(panel, x0, y, bw, 12, min(speed,8), 8,
                 COL_YELLOW, "Speed",           f"{speed:.2f}")
        y += 24
        draw_bar(panel, x0, y, bw, 12, (align+1)/2, 1.0,
                 COL_GREEN,  "Alignment",       f"{align:.2f}")
        y += 24
        draw_bar(panel, x0, y, bw, 12, min(zone,5), 5,
                 COL_RED,    "Zone Imbalance",  f"{zone:.2f}")
        y += 24
        draw_bar(panel, x0, y, bw, 12, edge, 1.0,
                 COL_RED,    "Edge Fraction",   f"{edge:.2f}")
        y += 24
    else:
        draw_text(panel, "  (warming up...)", (x0, y),
                  scale=0.40, color=(120,120,120))
        y += 24*5

    cv2.line(panel, (x0, y), (PANEL_W-x0, y), (60,60,60), 1)
    y += 8

    # ── 22 Window features ──
    draw_text(panel, "WINDOW FEATURES (22)", (x0, y),
              scale=0.43, color=(180,180,180))
    y += 16

    graph_top = panel_h - GRAPH_H - 14
    max_y_for_features = graph_top - 8

    if _last_fv is not None:
        for name, val in zip(FEATURE_NAMES, _last_fv):
            if y >= max_y_for_features:
                break
            col = (COL_RED    if abs(val) > 2   else
                   COL_YELLOW if abs(val) > 0.5 else (140,200,140))
            draw_text(panel, f"{name:<18} {val:+.3f}",
                      (x0, y), scale=0.34, color=col)
            y += 12
    else:
        draw_text(panel, "  (building window...)", (x0, y),
                  scale=0.38, color=(120,120,120))

    # ── Risk graph (bottom) ──
    cv2.line(panel, (x0, graph_top-4), (PANEL_W-x0, graph_top-4),
             (60,60,60), 1)
    draw_risk_graph(panel, x0, graph_top,
                    PANEL_W - x0*2 - 10, GRAPH_H - 10, risk_history)

    return panel


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
def run_display(source, show_flow=True):
    global _last_fv

    print("=" * 64)
    print("  STAMPEDE DISPLAY — loading model & weights …")
    print("=" * 64)

    if not os.path.exists(MODEL_OUT) or not os.path.exists(SCALER_OUT):
        print(f"[ERROR] Model files not found.\n"
              f"        Run:  python stampede_early_warning.py  first.")
        return

    with open(MODEL_OUT,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_OUT, "rb") as f: scaler = pickle.load(f)
    pos_idx = list(model.classes_).index("PRE_STAMPEDE")
    print(f"[OK]  Classifier : {model.__class__.__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crowd_counter = CrowdCounter(CSRNET_WEIGHTS, device)
    print(f"[OK]  CSRNet loaded  |  device={device}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f    = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration   = total_f / fps_video if total_f > 0 else 9999.0
    frame_step = max(1, int(fps_video / SAMPLE_FPS))

    win_frames, step_frames, look_sec = adaptive_params(duration)
    smooth_len = max(3, int(10 / (step_frames / SAMPLE_FPS)))

    print(f"[OK]  Source opened  fps={fps_video:.1f}  "
          f"win={win_frames/SAMPLE_FPS:.1f}s  look={look_sec:.1f}s")
    print("\nControls:  Q=quit   P=pause/resume   S=screenshot\n")

    signals_buf  = deque(maxlen=win_frames + 10)
    risk_buf     = deque(maxlen=smooth_len)
    risk_history = deque(maxlen=HISTORY_N)
    count_buf    = deque(maxlen=10)
    prev_gray    = None
    last_flow    = None

    raw_risk    = 0.0
    smooth_risk = 0.0
    alert_count = 0
    frame_idx   = 0
    paused      = False
    display_frame = None

    cv2.namedWindow("Stampede Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stampede Detector", RESIZE_W + PANEL_W, RESIZE_H)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('s') and display_frame is not None:
            fname = f"screenshot_{frame_idx:05d}.png"
            cv2.imwrite(fname, display_frame)
            print(f"[INFO] Screenshot → {fname}")

        if paused:
            if display_frame is not None:
                cv2.imshow("Stampede Detector", display_frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break
        frame_idx += 1

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        H, W  = frame.shape[:2]
        t_sec = frame_idx / fps_video

        draw_frame = frame.copy()

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)

        sig = {
            "t": t_sec, "count": 0.0,
            "speed": None, "alignment": None,
            "zone_imbalance": None, "edge_frac": None,
        }

        if frame_idx % frame_step == 0:
            # Crowd count
            cnt = crowd_counter.predict(frame)
            count_buf.append(cnt)
            sig["count"] = float(np.mean(count_buf))

            # Optical flow
            if prev_gray is not None:
                flow      = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                last_flow = flow

                step = 24
                vxs, vys, pts = [], [], []
                for gy in range(step, H-step, step):
                    for gx in range(step, W-step, step):
                        vx  = float(flow[gy,gx,0])
                        vy  = float(flow[gy,gx,1])
                        mag = float(np.hypot(vx, vy))
                        if mag > 0.3:
                            vxs.append(vx); vys.append(vy)
                            pts.append((gx, gy, mag))

                if vxs:
                    vxs_a   = np.array(vxs)
                    vys_a   = np.array(vys)
                    mags    = np.array([p[2] for p in pts])
                    sig["speed"] = float(np.mean(mags))

                    avg_vx  = float(np.mean(vxs_a))
                    avg_vy  = float(np.mean(vys_a))
                    dom_mag = float(np.hypot(avg_vx, avg_vy)) + 1e-6
                    dom     = np.array([avg_vx/dom_mag, avg_vy/dom_mag])
                    dots    = (vxs_a*dom[0] + vys_a*dom[1]) / (mags+1e-6)
                    sig["alignment"] = float(np.mean(np.clip(dots,-1,1)))

                    lm = [mags[i] for i,p in enumerate(pts) if p[0] < W//2]
                    rm = [mags[i] for i,p in enumerate(pts) if p[0] >= W//2]
                    sig["zone_imbalance"] = abs(
                        (float(np.mean(lm)) if lm else 0.0) -
                        (float(np.mean(rm)) if rm else 0.0))

                    edge_w = int(W * 0.15)
                    n_edge = sum(1 for p in pts
                                 if p[0] < edge_w or p[0] > W - edge_w)
                    sig["edge_frac"] = n_edge / max(1, len(pts))

            prev_gray = gray
            signals_buf.append(sig)

            # Predict
            if len(signals_buf) >= win_frames:
                window = list(signals_buf)[-win_frames:]
                fv     = window_to_features(window)
                if fv is not None:
                    _last_fv    = fv
                    raw_risk    = float(
                        model.predict_proba(
                            scaler.transform([fv]))[0][pos_idx])
                    risk_buf.append(raw_risk)
                    smooth_risk = float(np.mean(risk_buf))
                    risk_history.append(smooth_risk)
                    if smooth_risk >= ALERT_HIGH:
                        alert_count += 1

        # Draw flow arrows on video
        if show_flow and last_flow is not None:
            draw_flow_arrows(draw_frame, last_flow, step=32)

        # Status banner
        draw_status_banner(draw_frame, smooth_risk, alert_count)

        # Timestamp
        draw_text(draw_frame, f"t={t_sec:.1f}s  frame={frame_idx}",
                  (8, H-8), scale=0.38, color=(160,160,160))

        # Info panel
        panel = build_info_panel(
            H, list(signals_buf), smooth_risk,
            raw_risk, risk_history, t_sec)

        # Combine video + panel side by side
        display_frame = np.hstack([draw_frame, panel])
        cv2.imshow("Stampede Detector", display_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stampede visual display")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  metavar="PATH",
                       help="Path to a video file")
    group.add_argument("--camera", metavar="INDEX", type=int,
                       help="Camera index (0 = default webcam)")
    parser.add_argument("--no-flow", action="store_true",
                        help="Disable optical flow arrow overlay")
    args = parser.parse_args()

    source    = args.video if args.video else args.camera
    show_flow = not args.no_flow
    run_display(source, show_flow=show_flow)