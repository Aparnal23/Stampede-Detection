import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
from pathlib import Path
import os, pickle, warnings, argparse
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
NORMAL_DIR     = r"D:\mm\crowd_detection\crowd_detection (2)\crowd_detection\Normal Crowds"
ABNORMAL_DIR   = r"D:\mm\crowd_detection\crowd_detection (2)\crowd_detection\Abnormal Crowds"
CSRNET_WEIGHTS = r"D:\mm\crowd_detection\crowd_detection (2)\crowd_detection\csrnet_shanghaitech_partB.pth"

MODEL_OUT      = "stampede_earlywarning.pkl"
SCALER_OUT     = "feature_scaler_ew.pkl"

SAMPLE_FPS     = 5       # frames sampled per second
RESIZE_W       = 640
RESIZE_H       = 360

TARGET_WINDOW_SEC    = 10
TARGET_STEP_SEC      = 2
TARGET_LOOKAHEAD_SEC = 8

# Absolute minimums
MIN_WINDOW_FRAMES = 3    # minimum frames required in a window
MIN_FLOW_FRAMES   = 2    # minimum frames that must have optical-flow data

# For abnormal videos without an annotation: onset placed at this fraction
# of video duration. Used only as a last resort if auto-detection also fails.
DEFAULT_ONSET_RATIO = 0.60

# Per-video onset annotations (seconds from start). Add yours here.
# "filename.mp4": onset_second
# If left empty, auto-detection will be used automatically.
ONSET_TIMESTAMPS = {
    # "stampede_001.mp4": 47,
}

# ─────────────────────────────────────────────────────────────
#  AUTO ONSET DETECTION
# ─────────────────────────────────────────────────────────────
def auto_detect_onset_speed(signals, threshold_multiplier=2.0):
    """
    Detect onset by finding the first frame where speed spikes
    significantly above the baseline (first third of video = calm period).
    Returns onset time in seconds, or None if not detected.
    """
    speeds = [s["speed"] for s in signals if s["speed"] is not None]
    if len(speeds) < 6:
        return None

    baseline_portion = speeds[:max(1, len(speeds) // 3)]
    baseline_mean = float(np.mean(baseline_portion))
    baseline_std  = float(np.std(baseline_portion))

    # Threshold: baseline + N * std (default N=2.0)
    threshold = baseline_mean + threshold_multiplier * baseline_std

    # Walk through signals (not just speed list) to get timestamps
    for s in signals:
        if s["speed"] is not None and s["speed"] > threshold:
            return s["t"]

    return None


def auto_detect_onset_count(signals, surge_threshold=0.15):
    """
    Detect onset by finding the first frame where crowd count jumps
    by more than surge_threshold (default 15%) in a single step.
    Returns onset time in seconds, or None if not detected.
    """
    count_signals = [(s["t"], s["count"]) for s in signals]
    if len(count_signals) < 4:
        return None

    for i in range(1, len(count_signals)):
        prev_count = count_signals[i - 1][1]
        curr_count = count_signals[i][1]
        if prev_count > 0:
            delta_ratio = (curr_count - prev_count) / prev_count
            if delta_ratio > surge_threshold:
                return count_signals[i][0]

    return None


def auto_detect_onset_combined(signals, duration_sec):
    """
    Combine speed spike and count surge detection.
    Returns the EARLIER of the two detected onsets, giving the widest
    possible pre-stampede window.
    Falls back to DEFAULT_ONSET_RATIO if neither fires.
    """
    speed_onset = auto_detect_onset_speed(signals)
    count_onset = auto_detect_onset_count(signals)

    if speed_onset is not None and count_onset is not None:
        onset = min(speed_onset, count_onset)
        src   = f"auto (speed={speed_onset:.1f}s, count={count_onset:.1f}s → earlier={onset:.1f}s)"
    elif speed_onset is not None:
        onset = speed_onset
        src   = f"auto-speed ({onset:.1f}s)"
    elif count_onset is not None:
        onset = count_onset
        src   = f"auto-count ({onset:.1f}s)"
    else:
        onset = duration_sec * DEFAULT_ONSET_RATIO
        src   = f"fallback {DEFAULT_ONSET_RATIO*100:.0f}% of {duration_sec:.1f}s"

    return onset, src


# ─────────────────────────────────────────────────────────────
#  VIDEO METADATA
# ─────────────────────────────────────────────────────────────
def get_video_info(video_path):
    """Return (fps, total_frames, duration_sec) or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nf  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = nf / fps if fps > 0 and nf > 0 else 0.0
    cap.release()
    return fps, int(nf), dur


def adaptive_params(duration_sec):
    """
    Return (window_frames, step_frames, lookahead_sec) scaled to the
    video duration so that at least 2 windows can be produced.
    """
    win_sec  = min(TARGET_WINDOW_SEC,    max(1.0, duration_sec * 0.50))
    step_sec = min(TARGET_STEP_SEC,      max(0.4, win_sec      * 0.30))
    look_sec = min(TARGET_LOOKAHEAD_SEC, max(0.5, win_sec      * 0.60))

    win_frames  = max(MIN_WINDOW_FRAMES, int(win_sec  * SAMPLE_FPS))
    step_frames = max(1,                 int(step_sec * SAMPLE_FPS))

    return win_frames, step_frames, look_sec


# ─────────────────────────────────────────────────────────────
#  CSRNET
# ─────────────────────────────────────────────────────────────
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),  nn.ReLU(inplace=True),
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
#  PER-FRAME SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_frame_signals(video_path, crowd_counter):
    """
    Returns list of per-sampled-frame dicts:
      { t, count, speed, alignment, zone_imbalance, edge_frac }
    speed/alignment/zone_imbalance/edge_frac are None for the first frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps_video / SAMPLE_FPS))

    prev_gray = None
    frame_idx = 0
    signals   = []
    count_buf = deque(maxlen=10)

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

        cnt = crowd_counter.predict(frame)
        count_buf.append(cnt)
        smooth_cnt = float(np.mean(count_buf))

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)

        sig = {
            "t": t_sec, "count": smooth_cnt,
            "speed": None, "alignment": None,
            "zone_imbalance": None, "edge_frac": None,
        }

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

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
                vxs_a = np.array(vxs)
                vys_a = np.array(vys)
                mags  = np.array([p[2] for p in pts])

                sig["speed"] = float(np.mean(mags))

                avg_vx  = float(np.mean(vxs_a))
                avg_vy  = float(np.mean(vys_a))
                dom_mag = float(np.hypot(avg_vx, avg_vy)) + 1e-6
                dom     = np.array([avg_vx/dom_mag, avg_vy/dom_mag])
                dots    = (vxs_a*dom[0] + vys_a*dom[1]) / (mags + 1e-6)
                sig["alignment"] = float(np.mean(np.clip(dots, -1, 1)))

                lm = [mags[i] for i,p in enumerate(pts) if p[0] <  W//2]
                rm = [mags[i] for i,p in enumerate(pts) if p[0] >= W//2]
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
    return signals


# ─────────────────────────────────────────────────────────────
#  WINDOW → FEATURE VECTOR  (22 temporal features)
# ─────────────────────────────────────────────────────────────
def _slope(series):
    s = [v for v in series if v is not None]
    if len(s) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])


def _safe(series, fn=np.mean):
    s = [v for v in series if v is not None]
    return float(fn(s)) if s else 0.0


def window_to_features(window_signals):
    """Convert a list of per-frame dicts into a 22-dim feature vector."""
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

    # Count
    count_mean     = _safe(counts)
    count_velocity = _slope(counts)
    c_diff         = np.diff([c for c in counts]) if len(counts) > 1 else [0]
    count_accel    = float(np.mean(np.diff(c_diff))) if len(c_diff) > 1 else 0.0
    count_std      = _safe(counts, np.std)

    # Speed
    speed_mean  = float(np.mean(sp_v))
    speed_slope = _slope(speeds)
    sp_d        = np.diff(sp_v)
    speed_accel = float(np.mean(np.diff(sp_d))) if len(sp_d) > 1 else 0.0
    speed_std   = float(np.std(sp_v))

    # Alignment
    align_mean  = float(np.mean(al_v)) if al_v else 0.0
    align_slope = _slope(alignments)
    al_d        = np.diff(al_v)
    align_accel = float(np.mean(np.diff(al_d))) if len(al_d) > 1 else 0.0

    # Zone imbalance
    zone_mean  = float(np.mean(zo_v)) if zo_v else 0.0
    zone_slope = _slope(zones)
    zone_peak  = float(np.max(zo_v))  if zo_v else 0.0

    # Edge compression
    edge_mean  = float(np.mean(ed_v)) if ed_v else 0.0
    edge_slope = _slope(edges)
    edge_peak  = float(np.max(ed_v))  if ed_v else 0.0

    # Composite
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
#  SLIDE WINDOWS — adaptive to video duration
# ─────────────────────────────────────────────────────────────
def slide_windows(signals, is_abnormal, onset_sec, duration_sec):
    """
    Yields (feature_vector, label) for all valid windows.
    """
    win_frames, step_frames, look_sec = adaptive_params(duration_sec)
    n = len(signals)
    results = []

    # Whole-video fallback
    if n < win_frames:
        fv = window_to_features(signals)
        if fv is None:
            return results
        label = "PRE_STAMPEDE" if is_abnormal else "NORMAL"
        results.append((fv, label))
        return results

    # Clamp onset for abnormal so at least one PRE_STAMPEDE window fits
    if is_abnormal:
        first_win_end = signals[min(win_frames - 1, n - 1)]["t"]
        onset_sec     = max(onset_sec, first_win_end + 0.01)

    i = 0
    while i + win_frames <= n:
        window = signals[i: i + win_frames]
        t_end  = window[-1]["t"]

        fv = window_to_features(window)
        if fv is None:
            i += step_frames
            continue

        if is_abnormal:
            pre_start = onset_sec - look_sec
            if t_end > onset_sec:
                i += step_frames
                continue                    # post-event — skip
            label = "PRE_STAMPEDE" if t_end >= pre_start else "NORMAL"
        else:
            label = "NORMAL"

        results.append((fv, label))
        i += step_frames

    # Safety net for abnormal videos
    if is_abnormal and results and \
       not any(lbl == "PRE_STAMPEDE" for _, lbl in results):
        fv_last      = results[-1][0]
        results[-1]  = (fv_last, "PRE_STAMPEDE")

    return results


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def get_videos(folder):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(str(f) for f in Path(folder).iterdir()
                  if f.suffix.lower() in exts)


def print_metrics(y_true, y_pred, y_prob=None, title=""):
    pos = "PRE_STAMPEDE"
    print(f"\n{'─'*58}")
    if title:
        print(f"  {title}")
        print(f"{'─'*58}")
    cm     = confusion_matrix(y_true, y_pred, labels=[pos, "NORMAL"])
    TP, FN = cm[0,0], cm[0,1]
    FP, TN = cm[1,0], cm[1,1]
    print(f"  Confusion matrix  (positive = PRE_STAMPEDE):")
    print(f"                    Predicted")
    print(f"               PRE_STAMPEDE   NORMAL")
    print(f"  Actual  PRE     {TP:>5}      {FN:>5}")
    print(f"          NOR     {FP:>5}      {TN:>5}")
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=pos, zero_division=0)
    print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score : {f1:.4f}")
    if y_prob is not None:
        try:
            auc = roc_auc_score(
                [1 if v == pos else 0 for v in y_true], y_prob)
            print(f"  ROC-AUC  : {auc:.4f}")
        except Exception:
            pass
    print()
    print(classification_report(y_true, y_pred,
          target_names=[pos, "NORMAL"], zero_division=0))
    print(f"{'─'*58}")


# ─────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────
def train(_args):
    print("=" * 64)
    print("  STAMPEDE EARLY-WARNING — TRAINING  (adaptive windows)")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Target window     : {TARGET_WINDOW_SEC}s  "
          f"(auto-reduced for short videos)")
    print(f"[INFO] Target look-ahead : {TARGET_LOOKAHEAD_SEC}s\n")

    crowd_counter = CrowdCounter(CSRNET_WEIGHTS, device)

    normal_vids   = get_videos(NORMAL_DIR)
    abnormal_vids = get_videos(ABNORMAL_DIR)
    print(f"[INFO] Normal   videos : {len(normal_vids)}")
    print(f"[INFO] Abnormal videos : {len(abnormal_vids)}\n")

    X, y = [], []

    # ── Normal videos ──────────────────────────────────────
    print("── Normal videos ────────────────────────────────────────────")
    for path in normal_vids:
        name = os.path.basename(path)
        info = get_video_info(path)
        if info is None:
            print(f"  {name:<42} SKIPPED (unreadable)")
            continue
        _, _, dur = info
        w, s, lk  = adaptive_params(dur)
        print(f"  {name:<42} dur={dur:5.1f}s  win={w/SAMPLE_FPS:.1f}s",
              end="  ", flush=True)

        sigs  = extract_frame_signals(path, crowd_counter)
        pairs = slide_windows(sigs, is_abnormal=False,
                               onset_sec=None, duration_sec=dur)
        for fv, lbl in pairs:
            X.append(fv); y.append(lbl)
        print(f"{len(pairs)} windows")

    # ── Abnormal videos ────────────────────────────────────
    print("\n── Abnormal videos ──────────────────────────────────────────")
    for path in abnormal_vids:
        name = os.path.basename(path)
        info = get_video_info(path)
        if info is None:
            print(f"  {name:<42} SKIPPED (unreadable)")
            continue
        _, _, dur = info

        # ── Onset resolution (priority order) ─────────────
        # 1. Manual annotation in ONSET_TIMESTAMPS  → most accurate
        # 2. Auto-detection from signals            → good approximation
        # 3. DEFAULT_ONSET_RATIO fallback            → last resort only
        if name in ONSET_TIMESTAMPS:
            onset = float(ONSET_TIMESTAMPS[name])
            src   = "annotated"
            # Extract signals normally (onset already known)
            sigs  = extract_frame_signals(path, crowd_counter)
        else:
            # Extract signals first so auto-detection can analyse them
            sigs  = extract_frame_signals(path, crowd_counter)
            onset, src = auto_detect_onset_combined(sigs, dur)

        w, s, lk = adaptive_params(dur)
        print(f"  {name:<42} dur={dur:5.1f}s  onset={onset:.1f}s [{src}]  "
              f"win={w/SAMPLE_FPS:.1f}s  look={lk:.1f}s",
              end="  ", flush=True)

        pairs = slide_windows(sigs, is_abnormal=True,
                               onset_sec=onset, duration_sec=dur)
        pre   = sum(1 for _, l in pairs if l == "PRE_STAMPEDE")
        nor   = sum(1 for _, l in pairs if l == "NORMAL")
        for fv, lbl in pairs:
            X.append(fv); y.append(lbl)
        print(f"{len(pairs)} windows  ({pre} PRE / {nor} NOR)")

    # ── Sanity check ───────────────────────────────────────
    if not X:
        print("\n[ERROR] No windows extracted. Check your video paths.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    n_pre = (y == "PRE_STAMPEDE").sum()
    n_nor = (y == "NORMAL").sum()
    print(f"\n[INFO] Feature matrix   : {X.shape}")
    print(f"[INFO] PRE_STAMPEDE     : {n_pre}")
    print(f"[INFO] NORMAL           : {n_nor}")

    if n_pre == 0:
        print("\n[ERROR] No PRE_STAMPEDE windows found.")
        print("        Add onset timestamps to ONSET_TIMESTAMPS or "
              "raise DEFAULT_ONSET_RATIO.")
        return

    if n_pre < 3:
        print(f"\n[WARN] Only {n_pre} PRE_STAMPEDE windows — "
              f"results will be unreliable. Add more annotated videos.")

    # ── Scale ──────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Cross-validation strategy ─────────────────────────
    n_total = len(y)
    if n_total < 20:
        cv      = LeaveOneOut()
        cv_name = "Leave-One-Out"
    elif n_total < 60:
        cv      = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_name = "3-fold Stratified"
    else:
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-fold Stratified"

    print(f"\n[INFO] Cross-validation : {cv_name}  ({n_total} total windows)")

    # ── Gradient Boosting ─────────────────────────────────
    print("\n── Gradient Boosting ────────────────────────────────────────")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gb_preds = cross_val_predict(gb, X_scaled, y, cv=cv, method="predict")
    gb.fit(X_scaled, y)
    pos_idx = list(gb.classes_).index("PRE_STAMPEDE")
    gb_prob = cross_val_predict(gb, X_scaled, y, cv=cv,
                                 method="predict_proba")[:, pos_idx]
    gb.fit(X_scaled, y)
    print_metrics(y, gb_preds, gb_prob, "Gradient Boosting — " + cv_name)

    # ── SVM ───────────────────────────────────────────────
    print("\n── SVM ──────────────────────────────────────────────────────")
    svm = SVC(kernel="rbf", C=10, gamma="scale",
               probability=True, class_weight="balanced")
    svm_preds = cross_val_predict(svm, X_scaled, y, cv=cv, method="predict")
    svm.fit(X_scaled, y)
    svm_pos  = list(svm.classes_).index("PRE_STAMPEDE")
    svm_prob = cross_val_predict(svm, X_scaled, y, cv=cv,
                                  method="predict_proba")[:, svm_pos]
    svm.fit(X_scaled, y)
    print_metrics(y, svm_preds, svm_prob, "SVM — " + cv_name)

    # ── Feature importance ────────────────────────────────
    print("\n── Feature importance (Gradient Boosting) ───────────────────")
    imp     = gb.feature_importances_
    indices = np.argsort(imp)[::-1]
    for i in indices:
        bar = "█" * int(imp[i] * 50)
        print(f"  {FEATURE_NAMES[i]:<22} {imp[i]:.3f}  {bar}")

    # ── Save best model ────────────────────────────────────
    gb_f1  = f1_score(y, gb_preds,  pos_label="PRE_STAMPEDE", zero_division=0)
    svm_f1 = f1_score(y, svm_preds, pos_label="PRE_STAMPEDE", zero_division=0)

    if gb_f1 >= svm_f1:
        best_model, best_name, best_f1 = gb,  "GradientBoosting", gb_f1
    else:
        best_model, best_name, best_f1 = svm, "SVM",              svm_f1

    with open(MODEL_OUT,  "wb") as f: pickle.dump(best_model, f)
    with open(SCALER_OUT, "wb") as f: pickle.dump(scaler,     f)

    print(f"\n[INFO] Best model : {best_name}  (F1={best_f1:.4f})")
    print(f"[INFO] Saved      : {MODEL_OUT}")
    print(f"[INFO] Saved      : {SCALER_OUT}")
    print("\nDone.  Run with --infer <video> to score a new video.\n")


# ─────────────────────────────────────────────────────────────
#  INFER
# ─────────────────────────────────────────────────────────────
def infer(video_path):
    print("=" * 64)
    print("  STAMPEDE EARLY-WARNING — INFERENCE")
    print("=" * 64)
    print(f"  Video : {video_path}\n")

    if not os.path.exists(MODEL_OUT) or not os.path.exists(SCALER_OUT):
        print("[ERROR] Model not found — run training first.")
        return

    with open(MODEL_OUT,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_OUT, "rb") as f: scaler = pickle.load(f)
    pos_idx = list(model.classes_).index("PRE_STAMPEDE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crowd_counter = CrowdCounter(CSRNET_WEIGHTS, device)

    info = get_video_info(video_path)
    if info is None:
        print("[ERROR] Cannot read video."); return
    _, _, dur = info
    win_frames, step_frames, look_sec = adaptive_params(dur)

    print(f"[INFO] Duration   : {dur:.1f}s")
    print(f"[INFO] Window     : {win_frames/SAMPLE_FPS:.1f}s  "
          f"({win_frames} sampled frames)  "
          f"step={step_frames/SAMPLE_FPS:.1f}s\n")

    sigs = extract_frame_signals(video_path, crowd_counter)
    if not sigs:
        print("[ERROR] No frames extracted."); return

    n = len(sigs)
    if n < win_frames:
        win_frames  = n
        step_frames = n

    smooth_len = max(3, int(10 / (step_frames / SAMPLE_FPS)))
    risk_buf   = deque(maxlen=smooth_len)
    rows       = []
    ALERT_HIGH = 0.65
    ALERT_MED  = 0.40

    print(f"  {'t_start':>8}  {'t_end':>6}  {'raw':>6}  {'smooth':>7}  status")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*22}")

    i = 0
    while i + win_frames <= n:
        window  = sigs[i: i + win_frames]
        t_start = window[0]["t"]
        t_end   = window[-1]["t"]

        fv = window_to_features(window)
        if fv is None:
            i += step_frames; continue

        raw    = float(model.predict_proba(scaler.transform([fv]))[0][pos_idx])
        risk_buf.append(raw)
        smooth = float(np.mean(risk_buf))

        status = ("⚠  HIGH RISK — ALERT" if smooth >= ALERT_HIGH else
                  "⚡ MEDIUM RISK"        if smooth >= ALERT_MED  else
                  "   normal")

        print(f"  {t_start:>7.1f}s  {t_end:>5.1f}s  "
              f"{raw:>6.3f}  {smooth:>7.3f}  {status}")
        rows.append({"t_start": t_start, "t_end": t_end,
                     "raw": raw, "smooth": smooth})
        i += step_frames

    csv_path = "risk_scores.csv"
    with open(csv_path, "w") as f:
        f.write("t_start,t_end,raw_risk,smooth_risk\n")
        for r in rows:
            f.write(f"{r['t_start']:.2f},{r['t_end']:.2f},"
                    f"{r['raw']:.4f},{r['smooth']:.4f}\n")

    if rows:
        peak = max(rows, key=lambda r: r["smooth"])
        fa   = next((r for r in rows if r["smooth"] >= ALERT_HIGH), None)
        print(f"\n[INFO] Peak risk  : {peak['smooth']:.3f} "
              f"at t={peak['t_start']:.1f}s")
        if fa:
            print(f"[INFO] First HIGH alert at t={fa['t_start']:.1f}s  "
                  f"(trained to fire ~{look_sec:.0f}s before onset)")
        else:
            print("[INFO] No HIGH-risk alert triggered.")
    print(f"[INFO] Scores saved → {csv_path}")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stampede early-warning predictor")
    parser.add_argument("--infer", metavar="VIDEO_PATH",
                        help="Score a single video (skip training)")
    args = parser.parse_args()
    if args.infer:
        infer(args.infer)
    else:
        train(args)