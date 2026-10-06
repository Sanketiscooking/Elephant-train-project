# -*- coding: utf-8 -*-
"""
Elephant Detection with MSP430 LED Control
      + Independent Analytics & Prediction window
      + Fully resizable (ribbon-style) windows
      + Proportionally scaled, card-style analytics
      + 30-min P(presence) window
      + Hour-of-day chart from log detection data
      + History line 2 = past + present combined            (v16)
Author: Ami
Date: Sep 20, 2026
"""

import cv2
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta

import numpy as np
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ============================================================
#  CONFIGURATION
# ============================================================
MODEL_PATH          = "yolov8l.pt"
CONF_THRESHOLD      = 0.50
CAMERA_INDICES      = (0,1,2)          # force index 1 (external/USB camera)

LOG_DIR             = "elephant_detections"
TEXT_LOG            = os.path.join(LOG_DIR, "detections.log")
CSV_LOG             = os.path.join(LOG_DIR, "events.csv")
REPORT_TXT          = os.path.join(LOG_DIR, "analytics_report.txt")
STATE_JSON          = os.path.join(LOG_DIR, "analytics_state.json")
PLOT_PNG            = os.path.join(LOG_DIR, "session_summary.png")

REQUIRE_MSP430           = True
SEND_PREDICTIVE_CAUTION  = False
SERIAL_BAUD              = 9600

# --- Analytics tuning ---
MIN_DETECTION_S     = 0.5
SAMPLE_INTERVAL_S   = 0.25
RATE_WINDOW_S       = 300
CONF_WINDOW_N       = 200
PROB_HISTORY_N      = 1800          # 30 min at 1 sample/sec
DECAY_TAU_S         = 120.0
PRIOR_SMOOTHING_S   = 60.0
W_RECENCY, W_RATE, W_PRIOR = 0.45, 0.25, 0.30
REPORT_INTERVAL_S   = 5.0

RUN_ID = time.strftime("%Y%m%d-%H%M%S")

# ============================================================
#  WINDOWS
# ============================================================
CAMERA_WINDOW        = "Elephant Detection Feed"
ANALYTICS_WINDOW     = "Elephant Detection — Analytics"
CAMERA_DEFAULT_SIZE  = (960, 540)
ANALYTICS_DEFAULT_SZ = (1000, 800)

# ============================================================
#  ANALYTICS SCALING REFERENCE
#     s = min(W/900, H/700), clamped to [0.55, 2.00]
# ============================================================
AN_BASE_W, AN_BASE_H = 900, 700
AN_S_MIN,  AN_S_MAX  = 0.55, 2.00

AN_PAD          = 14
AN_TITLE_H      = 34
AN_FOOTER_H     = 18
AN_CARD_PAD     = 12
AN_ROW_H        = 20
AN_SEC_H        = 18
AN_GAP          = 8
AN_CHART_MIN_H  = 70

AN_F_TITLE      = 0.60
AN_F_SECTION    = 0.48
AN_F_BODY       = 0.44
AN_F_SMALL      = 0.36
AN_F_TICK       = 0.32

# ============================================================
#  COLORS (BGR)
# ============================================================
C_BG            = (24, 24, 24)
C_TITLE_BG      = (20, 20, 20)
C_CARD_BG       = (34, 34, 34)
C_CARD_BORDER   = (56, 56, 56)
C_DIVIDER       = (70, 70, 70)
C_ACCENT        = (170, 150, 100)
C_LABEL         = (140, 140, 140)
C_VALUE         = (225, 225, 225)
C_VALUE_DIM     = (185, 185, 185)
C_CHART_BG      = (30, 30, 30)
C_CHART_GRID    = (52, 52, 52)

# hour-chart bar colors
C_HOUR_OBS      = (85, 85, 85)      # gray — observed time
C_HOUR_OBS_CUR  = (120, 120, 120)   # lighter gray for current hour
C_HOUR_DET      = (90, 60, 170)     # red — detected time
C_HOUR_DET_CUR  = (60, 60, 220)     # brighter red for current hour

RISK_BGR = {
    "LOW":      (110, 200, 110),
    "MEDIUM":   (0, 205, 235),
    "HIGH":     (0, 140, 240),
    "CRITICAL": (60, 60, 220),
}

# ============================================================
#  HELPERS
# ============================================================
def iso(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))

def hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

def fmt_dur_short(sec):
    """Compact duration for chart axis: 45s, 12m, 3h20, ..."""
    sec = int(max(0, sec))
    if sec < 60:   return f"{sec}s"
    if sec < 3600: return f"{sec // 60}m"
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h}h{m:02d}" if m else f"{h}h"

def split_by_hour(start_epoch, end_epoch):
    cur = start_epoch
    while cur < end_epoch - 1e-9:
        dt = datetime.fromtimestamp(cur)
        nxt = (dt.replace(minute=0, second=0, microsecond=0)
               + timedelta(hours=1)).timestamp()
        chunk_end = min(end_epoch, nxt)
        yield dt.hour, chunk_end - cur
        cur = chunk_end

def get_window_size(name, default):
    try:
        rect = cv2.getWindowImageRect(name)
        if rect and len(rect) == 4:
            _, _, w, h = rect
            if w > 1 and h > 1:
                return int(w), int(h)
    except Exception:
        pass
    return default

# ============================================================
#  HISTORY LOADING
# ============================================================
def load_history(csv_path):
    hist = {
        "n_runs": 0,
        "run_intervals": [],
        "det_intervals": [],
        "hour_observed": defaultdict(float),
        "hour_detected": defaultdict(float),
    }
    if not os.path.exists(csv_path):
        return hist
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ev = (row.get("event") or "").strip().upper()
                try:
                    s = float(row["start_epoch"]); e = float(row["end_epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                if e <= s:
                    continue
                if ev == "RUN":
                    hist["n_runs"] += 1
                    hist["run_intervals"].append((s, e))
                    for h, secs in split_by_hour(s, e):
                        hist["hour_observed"][h] += secs
                elif ev == "DETECTION":
                    hist["det_intervals"].append((s, e))
                    for h, secs in split_by_hour(s, e):
                        hist["hour_detected"][h] += secs
    except Exception as exc:
        print(f"⚠️  Could not parse history ({exc}); starting fresh.")
    return hist

def derive_gaps(hist):
    gaps = []
    for rs, re in hist["run_intervals"]:
        dets = sorted(d for d in hist["det_intervals"] if d[0] >= rs and d[1] <= re)
        prev = rs
        for ds, de in dets:
            if ds > prev:
                gaps.append(ds - prev)
            prev = max(prev, de)
        if re > prev:
            gaps.append(re - prev)
    return gaps

# ============================================================
#  STRUCTURED EVENT LOGGER
# ============================================================
class EventLogger:
    HEADER = ["run_id", "event", "start_epoch", "end_epoch", "duration_s",
              "peak_confidence", "mean_confidence", "start_iso", "end_iso"]

    def __init__(self, csv_path, text_path, run_id, model_path):
        self.csv_path, self.text_path, self.run_id = csv_path, text_path, run_id
        fresh = not os.path.exists(csv_path)
        self.fh = open(csv_path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if fresh:
            self.w.writerow(self.HEADER)
            self.fh.flush()
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Run {run_id} started {iso(time.time())} | "
                    f"model={model_path} ===\n")

    def _row(self, event, s, e, peak=None, mean=None):
        self.w.writerow([self.run_id, event, f"{s:.3f}", f"{e:.3f}", f"{e - s:.3f}",
                         "" if peak is None else f"{peak:.3f}",
                         "" if mean is None else f"{mean:.3f}",
                         iso(s), iso(e)])
        self.fh.flush()

    def detection(self, s, e, peak, mean):
        self._row("DETECTION", s, e, peak, mean)
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(f"{iso(s)} -> {iso(e)} | ELEPHANT {e - s:.1f}s | "
                    f"peak {peak:.2f} | mean {mean:.2f}\n")

    def run(self, s, e):
        self._row("RUN", s, e)

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass

# ============================================================
#  LIVE ANALYTICS
# ============================================================
class LiveAnalytics:
    def __init__(self, run_id, history, logger):
        self.run_id = run_id
        self.history = history
        self.logger = logger
        self.t0 = time.time()

        self.streak_start = None
        self.streak_conf_sum = 0.0
        self.streak_conf_n = 0
        self.streak_conf_peak = 0.0

        self.detections = []
        self.gaps = []
        self.last_det_end = None
        self.gap_start = self.t0

        self.samples = deque()
        self.conf_hist = deque(maxlen=CONF_WINDOW_N)
        self.prob_history = deque(maxlen=PROB_HISTORY_N)
        self._last_push = 0.0

    def update(self, detected, conf, now):
        if now - self._last_push >= SAMPLE_INTERVAL_S:
            self.samples.append((now, 1 if detected else 0))
            self._last_push = now
        cutoff = now - RATE_WINDOW_S
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        if detected:
            if self.streak_start is None:
                self.streak_start = now
                self.streak_conf_sum = 0.0
                self.streak_conf_n = 0
                self.streak_conf_peak = 0.0
                if self.gap_start is not None and now > self.gap_start:
                    self.gaps.append((self.gap_start, now))
                self.gap_start = None
            self.streak_conf_sum += conf
            self.streak_conf_n += 1
            self.streak_conf_peak = max(self.streak_conf_peak, conf)
            self.conf_hist.append(conf)
            self.last_det_end = now
        else:
            if self.streak_start is not None:
                self._close_streak(now)
                self.gap_start = now

    def _close_streak(self, now):
        dur = now - self.streak_start
        if dur >= MIN_DETECTION_S and self.streak_conf_n > 0:
            rec = (self.streak_start, now, self.streak_conf_peak,
                   self.streak_conf_sum / self.streak_conf_n)
            self.detections.append(rec)
            self.logger.detection(*rec)
        self.streak_start = None

    def flush(self):
        if self.streak_start is not None:
            self._close_streak(time.time())
            self.streak_start = None

    @property
    def recent_rate(self):
        if not self.samples:
            return 0.0
        return sum(v for _, v in self.samples) / len(self.samples)

    def session_detected_s(self, now=None):
        total = sum(e - s for s, e, _, _ in self.detections)
        if self.streak_start is not None:
            total += (now or time.time()) - self.streak_start
        return total

    def session_duration_s(self, now=None):
        return (now or time.time()) - self.t0

    def since_last_detection(self, now=None):
        if self.streak_start is not None:
            return 0.0
        if self.last_det_end is None:
            return None
        return (now or time.time()) - self.last_det_end

    def mean_conf(self):
        return statistics.fmean(self.conf_hist) if self.conf_hist else 0.0

    def peak_conf(self):
        return max(self.conf_hist) if self.conf_hist else 0.0

# ============================================================
#  PRESENCE PREDICTOR
# ============================================================
class PresencePredictor:
    def __init__(self, history):
        self.history = history
        obs_total = sum(history["hour_observed"].values())
        det_total = sum(history["hour_detected"].values())
        self.base_rate = (det_total / obs_total) if obs_total > 1e-6 else 0.05
        self.hour_prior = {}
        for h in range(24):
            o = history["hour_observed"].get(h, 0.0)
            d = history["hour_detected"].get(h, 0.0)
            self.hour_prior[h] = (d + PRIOR_SMOOTHING_S * self.base_rate) / \
                                 (o + PRIOR_SMOOTHING_S)
        gaps = derive_gaps(history)
        self.median_gap = statistics.median(gaps) if gaps else None
        self.peak_hour = max(self.hour_prior, key=self.hour_prior.get) \
                         if obs_total > 0 else None

    def predict(self, now, detected, last_det_end, recent_rate):
        if detected:
            return 0.99, "CRITICAL", 1.0, self.hour_prior.get(
                datetime.fromtimestamp(now).hour, self.base_rate)
        hour = datetime.fromtimestamp(now).hour
        p_hour = self.hour_prior.get(hour, self.base_rate)
        p_rec = 0.0 if last_det_end is None else \
                math.exp(-max(0.0, now - last_det_end) / DECAY_TAU_S)
        p = W_RECENCY * p_rec + W_RATE * recent_rate + W_PRIOR * p_hour
        p = min(max(p, 0.0), 0.99)
        return p, risk_label(p), p_rec, p_hour

def risk_label(p):
    if p >= 0.70: return "CRITICAL"
    if p >= 0.45: return "HIGH"
    if p >= 0.20: return "MEDIUM"
    return "LOW"

# ============================================================
#  MSP430 SERIAL
# ============================================================
class NullSerial:
    def write(self, *_): pass
    def close(self): pass

'''
def find_msp430_uart():
  
    for p in serial.tools.list_ports.comports():
        print(f"Found port: {p.device} - {p.description}")
        if "Application UART" in p.description or "MSP430" in p.description:
            return p.device
    return None
'''

def find_msp430_uart():
    """
    Find the MSP430 Application UART.

    On this desktop, Windows identifies both MSP430 interfaces
    as 'USB Serial Device'. COM3 has been experimentally
    confirmed as the Application UART.
    """

    ports = serial.tools.list_ports.comports()

    # First try the original description-based detection
    for p in ports:
        print(f"Found port: {p.device} - {p.description}")

        if "Application UART" in p.description or "MSP430" in p.description:
            print(f"✓ MSP430 Application UART found: {p.device}")
            return p.device

    # Desktop fallback:
    # COM3 has been experimentally confirmed to control
    # the MSP430 application LEDs.
    for p in ports:
        if p.device == "COM3":
            print("✓ MSP430 Application UART found: COM3")
            return p.device

    return None

class SignalController:
    def __init__(self, ser):
        self.ser = ser
        self.state = None

    def set(self, state):
        if state == self.state:
            return
        self.state = state
        self.ser.write((state + "\n").encode("ascii"))
        if state == "ELEPHANT":
            print("➡️ Sent ELEPHANT   → RED   ■■■")
        elif state == "CAUTION":
            print("➡️ Sent CAUTION    → AMBER ■■■  (predictive)")
        else:
            print("➡️ Sent NOELEPHANT → GREEN ■■■")

# ============================================================
#  CAMERA AUTO-DETECT
# ============================================================
def open_camera(indices=CAMERA_INDICES):
    backends = (("DSHOW", cv2.CAP_DSHOW),
                ("MSMF",  cv2.CAP_MSMF),
                ("ANY",   cv2.CAP_ANY))
    print(f"🎥 Probing camera indices {indices} (forced order) …")
    for idx in indices:
        for name, flag in backends:
            cap = cv2.VideoCapture(idx, flag)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✅ Camera opened: index={idx} backend={name} "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            print(f"⚠️  index={idx} backend={name} opened but no frame")
            cap.release()
    return None

# ============================================================
#  CAMERA-WINDOW RENDERER
# ============================================================
def prepare_camera_canvas(frame, results, model_names, win_w, win_h):
    h, w = frame.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    detected = False
    best_conf = 0.0

    box_thickness = max(2, int(round(2 * scale)))
    font_box      = max(0.5, min(1.4, 0.6 * scale))
    font_status   = max(0.8, min(2.0, 1.0 * scale))

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if model_names[cls_id] == "elephant" and conf > CONF_THRESHOLD:
            detected = True
            best_conf = max(best_conf, conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * scale); y1 = int(y1 * scale)
            x2 = int(x2 * scale); y2 = int(y2 * scale)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            cv2.putText(resized, f"Elephant {conf:.2f}",
                        (x1, max(int(15 * scale), y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_box, (0, 0, 255),
                        box_thickness, cv2.LINE_AA)

    status_text = "Elephant Detected!" if detected else "No Elephant"
    color = (0, 255, 0) if detected else (255, 255, 255)
    cv2.putText(resized, status_text,
                (int(20 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_status, color,
                max(2, int(round(2 * scale))), cv2.LINE_AA)

    if new_w == win_w and new_h == win_h:
        return resized, detected, best_conf

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_off = (win_w - new_w) // 2
    y_off = (win_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas, detected, best_conf

# ============================================================
#  ANALYTICS-WINDOW RENDERER
# ============================================================
def render_analytics_window(analytics, predictor, p, risk, p_rec, p_hour,
                            detected, now, W, H):
    # ---------------- scale factor ----------------
    s = min(W / AN_BASE_W, H / AN_BASE_H)
    s = max(AN_S_MIN, min(s, AN_S_MAX))

    def S(v):   return max(1, int(round(v * s)))
    def F(v):   return max(0.20, v * s)

    PAD         = S(AN_PAD)
    TITLE_H     = S(AN_TITLE_H)
    FOOTER_H    = S(AN_FOOTER_H)
    CARD_PAD    = S(AN_CARD_PAD)
    ROW_H       = S(AN_ROW_H)
    SEC_H       = S(AN_SEC_H)
    GAP         = S(AN_GAP)

    F_TITLE     = F(AN_F_TITLE)
    F_SECTION   = F(AN_F_SECTION)
    F_BODY      = F(AN_F_BODY)
    F_SMALL     = F(AN_F_SMALL)
    F_TICK      = F(AN_F_TICK)

    THIN        = 1
    THICK       = max(1, int(round(s)))

    canvas = np.full((H, W, 3), C_BG, dtype="uint8")

    # ---------------- title bar ----------------
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), C_TITLE_BG, -1)
    cv2.line(canvas, (0, TITLE_H), (W, TITLE_H), C_DIVIDER, THIN)
    baseline = TITLE_H - max(2, S(9))
    cv2.putText(canvas, "ELEPHANT DETECTION ANALYTICS",
                (PAD, baseline),
                cv2.FONT_HERSHEY_SIMPLEX, F_TITLE, C_VALUE, THICK, cv2.LINE_AA)
    run_txt = f"Run {analytics.run_id}"
    (tw, _), _ = cv2.getTextSize(run_txt, cv2.FONT_HERSHEY_SIMPLEX, F_SMALL, THIN)
    cv2.putText(canvas, run_txt, (W - tw - PAD, baseline),
                cv2.FONT_HERSHEY_SIMPLEX, F_SMALL, C_LABEL, THIN, cv2.LINE_AA)

    # ---------------- footer ----------------
    cv2.putText(canvas, f"updated {iso(now)}   ·   q to quit",
                (PAD, H - max(3, S(5))),
                cv2.FONT_HERSHEY_SIMPLEX, F_SMALL, C_LABEL, THIN, cv2.LINE_AA)

    # ---------------- derived content ----------------
    sess  = analytics.session_duration_s(now)
    det_s = analytics.session_detected_s(now)
    pct   = (det_s / sess * 100) if sess > 0 else 0.0
    mg    = predictor.median_gap
    risk_c = RISK_BGR[risk]

    if detected and analytics.streak_start is not None:
        state_txt = f"PRESENT  ({now - analytics.streak_start:.1f}s)"
        st_col = RISK_BGR["CRITICAL"]
    else:
        sld = analytics.since_last_detection(now)
        state_txt = "clear" if sld is None else f"clear  ({sld:.0f}s)"
        st_col = RISK_BGR["LOW"]

    tot_obs = sum(analytics.history["hour_observed"].values())
    tot_det = sum(analytics.history["hour_detected"].values())
    n_runs  = analytics.history["n_runs"]

    # combined (past + present) values for history card line 2
    comb_runs = n_runs + 1
    comb_obs  = tot_obs + sess
    comb_det  = tot_det + det_s
    comb_rate = (comb_det / comb_obs * 100) if comb_obs > 0 else 0.0

    # ---------------- small helpers ----------------
    def text_w(txt, font, thick=THIN):
        (tw_, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        return tw_

    def draw_card(x, y, w, h, title=None):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CARD_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CARD_BORDER, THIN)
        if title:
            ty = y + CARD_PAD + S(10)
            cv2.putText(canvas, title, (x + CARD_PAD, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, F_SECTION, C_ACCENT,
                        THIN, cv2.LINE_AA)

    def kv_right(x_card, y_card, w_card, y, label, value, vcol=C_VALUE):
        cv2.putText(canvas, label, (x_card + CARD_PAD, y + S(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BODY, C_LABEL,
                    THIN, cv2.LINE_AA)
        vx = x_card + w_card - CARD_PAD - text_w(value, F_BODY)
        cv2.putText(canvas, value, (vx, y + S(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BODY, vcol,
                    THIN, cv2.LINE_AA)

    def kv_bar_right(x_card, y_card, w_card, y, label, pval, bar_color,
                     label_col=C_LABEL):
        cv2.putText(canvas, label, (x_card + CARD_PAD, y + S(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BODY, label_col,
                    THIN, cv2.LINE_AA)
        val_txt = f"{pval:.2f}"
        vx = x_card + w_card - CARD_PAD - text_w(val_txt, F_BODY)
        cv2.putText(canvas, val_txt, (vx, y + S(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_BODY, bar_color,
                    THIN, cv2.LINE_AA)
        bar_x = x_card + CARD_PAD + text_w(label, F_BODY) + S(8)
        bar_y = y + S(4)
        bar_w = max(S(20), vx - S(8) - bar_x)
        bar_h = max(S(6), S(9))
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w * pval), bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), C_CARD_BORDER, THIN)

    # ---------------- hour-of-day chart (from log data) ----------------
    def draw_hour_chart(x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CHART_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CARD_BORDER, THIN)

        # title (left)
        cv2.putText(canvas, "Hour-of-day detection from log",
                    (x + CARD_PAD, y + CARD_PAD + S(9)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_SECTION, C_ACCENT,
                    THIN, cv2.LINE_AA)

        # legend (right)
        lx = x + w - CARD_PAD
        for label, col in (("det", C_HOUR_DET), ("obs", C_HOUR_OBS)):
            tw_ = text_w(label, F_TICK)
            lx -= tw_
            cv2.putText(canvas, label,
                        (lx, y + CARD_PAD + S(9)),
                        cv2.FONT_HERSHEY_SIMPLEX, F_TICK, C_LABEL,
                        THIN, cv2.LINE_AA)
            lx -= S(6)
            cv2.rectangle(canvas,
                          (lx - S(8), y + CARD_PAD + S(2)),
                          (lx, y + CARD_PAD + S(9)),
                          col, -1)
            lx -= S(12)

        # ----- pull per-hour durations from the log -----
        obs_h = [0.0] * 24
        det_h = [0.0] * 24

        # 1) all previously-logged runs
        for hr, secs in analytics.history["hour_observed"].items():
            obs_h[hr] += secs
        for hr, secs in analytics.history["hour_detected"].items():
            det_h[hr] += secs

        # 2) current run observation time
        for hr, secs in split_by_hour(analytics.t0, now):
            obs_h[hr] += secs

        # 3) current run closed detections
        for s0, e0, _pk, _mn in analytics.detections:
            for hr, secs in split_by_hour(s0, e0):
                det_h[hr] += secs

        # 4) current run ongoing streak (not yet closed)
        if analytics.streak_start is not None and analytics.streak_start < now:
            for hr, secs in split_by_hour(analytics.streak_start, now):
                det_h[hr] += secs

        max_v = max(max(obs_h), 1.0)

        inner_x = x + CARD_PAD + S(34)
        inner_y = y + CARD_PAD + SEC_H + S(6)
        inner_w = w - CARD_PAD * 2 - S(34)
        inner_h = h - (inner_y - y) - CARD_PAD - S(14)
        if inner_w < S(40) or inner_h < S(16):
            return

        # grid + y-axis labels (durations)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            gy = int(inner_y + inner_h - frac * inner_h)
            cv2.line(canvas, (inner_x, gy), (inner_x + inner_w, gy),
                     C_CHART_GRID, THIN)
            lbl = fmt_dur_short(frac * max_v)
            (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                         F_TICK, THIN)
            cv2.putText(canvas, lbl,
                        (inner_x - S(6) - lw, gy + S(3)),
                        cv2.FONT_HERSHEY_SIMPLEX, F_TICK, C_LABEL,
                        THIN, cv2.LINE_AA)

        bar_w = inner_w / 24.0
        cur_h = datetime.now().hour
        for hr in range(24):
            bx = int(inner_x + hr * bar_w)
            bw = max(1, int(bar_w) - 1)

            # observed (background)
            if obs_h[hr] > 0:
                bh_obs = int((obs_h[hr] / max_v) * inner_h)
                by = inner_y + inner_h - bh_obs
                col_obs = C_HOUR_OBS_CUR if hr == cur_h else C_HOUR_OBS
                cv2.rectangle(canvas, (bx + 1, by),
                              (bx + bw, inner_y + inner_h), col_obs, -1)

            # detected (foreground overlay)
            if det_h[hr] > 0:
                bh_det = int((det_h[hr] / max_v) * inner_h)
                by = inner_y + inner_h - bh_det
                col_det = C_HOUR_DET_CUR if hr == cur_h else C_HOUR_DET
                cv2.rectangle(canvas, (bx + 1, by),
                              (bx + bw, inner_y + inner_h), col_det, -1)

        # hour ticks
        for hr in range(0, 24, 6):
            tx = int(inner_x + hr * bar_w)
            cv2.putText(canvas, f"{hr:02d}h",
                        (tx, inner_y + inner_h + S(11)),
                        cv2.FONT_HERSHEY_SIMPLEX, F_TICK, C_LABEL,
                        THIN, cv2.LINE_AA)

    def draw_sparkline(x, y, w, h, probs, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CHART_BG, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), C_CARD_BORDER, THIN)

        cv2.putText(canvas, "P(presence) last 30 min",
                    (x + CARD_PAD, y + CARD_PAD + S(9)),
                    cv2.FONT_HERSHEY_SIMPLEX, F_SECTION, C_ACCENT,
                    THIN, cv2.LINE_AA)

        inner_x = x + CARD_PAD + S(28)
        inner_y = y + CARD_PAD + SEC_H + S(6)
        inner_w = w - CARD_PAD * 2 - S(28)
        inner_h = h - (inner_y - y) - CARD_PAD - S(4)
        if inner_w < S(40) or inner_h < S(12):
            return

        for frac in (0.0, 0.5, 1.0):
            gy = int(inner_y + inner_h - frac * inner_h)
            cv2.line(canvas, (inner_x, gy), (inner_x + inner_w, gy),
                     C_CHART_GRID, THIN)
            lbl = f"{frac:.1f}"
            (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                         F_TICK, THIN)
            cv2.putText(canvas, lbl,
                        (inner_x - S(6) - lw, gy + S(3)),
                        cv2.FONT_HERSHEY_SIMPLEX, F_TICK, C_LABEL,
                        THIN, cv2.LINE_AA)

        n = len(probs)
        if n >= 2:
            step = inner_w / (PROB_HISTORY_N - 1)
            pts = []
            for i, pv in enumerate(probs):
                px = int(inner_x + (PROB_HISTORY_N - n + i) * step)
                py = int(inner_y + inner_h - pv * inner_h)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color,
                         max(1, int(round(s * 1.5))), cv2.LINE_AA)

    # ---------------- content layout ----------------
    y_top = TITLE_H + PAD
    y_bot = H - FOOTER_H - PAD

    det_rows  = 6
    pred_rows = 5
    rows_max  = max(det_rows, pred_rows)
    card_h    = CARD_PAD + SEC_H + rows_max * ROW_H + CARD_PAD

    HIST_ROWS = 2
    hist_h    = CARD_PAD + SEC_H + HIST_ROWS * ROW_H + CARD_PAD

    min_card_w = S(200)
    side_by_side = (W - 3 * PAD) >= 2 * min_card_w

    if side_by_side:
        card_w = (W - 3 * PAD) // 2

        dx = PAD
        draw_card(dx, y_top, card_w, card_h, "DETECTION")
        y = y_top + CARD_PAD + SEC_H
        kv_right(dx, y_top, card_w, y, "State",        state_txt, st_col);       y += ROW_H
        kv_right(dx, y_top, card_w, y, "Detections",   f"{len(analytics.detections)}"); y += ROW_H
        kv_right(dx, y_top, card_w, y, "Session",      hms(sess));               y += ROW_H
        kv_right(dx, y_top, card_w, y, "Detected",
                 f"{hms(det_s)}  ({pct:.1f}%)");                                 y += ROW_H
        kv_right(dx, y_top, card_w, y, "5-min rate",
                 f"{analytics.recent_rate * 100:.1f}%");                         y += ROW_H
        kv_right(dx, y_top, card_w, y, "Confidence",
                 f"{analytics.mean_conf():.2f} / {analytics.peak_conf():.2f}")

        px = 2 * PAD + card_w
        draw_card(px, y_top, card_w, card_h, "PREDICTION in NEXT 60 s")
        y = y_top + CARD_PAD + SEC_H
        kv_right(px, y_top, card_w, y, "Risk", risk, risk_c);                    y += ROW_H
        kv_bar_right(px, y_top, card_w, y, "P(presence)", p, risk_c);            y += ROW_H
        kv_right(px, y_top, card_w, y, "Recently seen (Recency)",
                 f"{p_rec:.2f} (Decay time const = {DECAY_TAU_S:.0f}s)");                        y += ROW_H
        kv_right(px, y_top, card_w, y, "Hour prior",
                 f"{p_hour:.2f}" +
                 (f"  (peak {predictor.peak_hour:02d}h)"
                  if predictor.peak_hour is not None else ""));                  y += ROW_H
        kv_right(px, y_top, card_w, y, "Median gap",
                 f"{mg:.0f} s" if mg else "n/a")

        y_cursor = y_top + card_h + GAP
    else:
        dx = PAD
        card_w = W - 2 * PAD
        draw_card(dx, y_top, card_w, card_h, "DETECTION")
        y = y_top + CARD_PAD + SEC_H
        kv_right(dx, y_top, card_w, y, "State",        state_txt, st_col);       y += ROW_H
        kv_right(dx, y_top, card_w, y, "Detections",   f"{len(analytics.detections)}"); y += ROW_H
        kv_right(dx, y_top, card_w, y, "Session",      hms(sess));               y += ROW_H
        kv_right(dx, y_top, card_w, y, "Detected",
                 f"{hms(det_s)}  ({pct:.1f}%)");                                 y += ROW_H
        kv_right(dx, y_top, card_w, y, "5-min rate",
                 f"{analytics.recent_rate * 100:.1f}%");                         y += ROW_H
        kv_right(dx, y_top, card_w, y, "Confidence",
                 f"{analytics.mean_conf():.2f} / {analytics.peak_conf():.2f}")

        y_top2 = y_top + card_h + GAP
        draw_card(dx, y_top2, card_w, card_h, "PREDICTION in NEXT 60 s")
        y = y_top2 + CARD_PAD + SEC_H
        kv_right(dx, y_top2, card_w, y, "Risk", risk, risk_c);                   y += ROW_H
        kv_bar_right(dx, y_top2, card_w, y, "P(presence)", p, risk_c);           y += ROW_H
        kv_right(dx, y_top2, card_w, y, "Recency",
                 f"{p_rec:.2f}  (τ={DECAY_TAU_S:.0f}s)");                        y += ROW_H
        kv_right(dx, y_top2, card_w, y, "Hour prior",
                 f"{p_hour:.2f}" +
                 (f"  (peak {predictor.peak_hour:02d}h)"
                  if predictor.peak_hour is not None else ""));                  y += ROW_H
        kv_right(dx, y_top2, card_w, y, "Median gap",
                 f"{mg:.0f} s" if mg else "n/a")

        y_cursor = y_top2 + card_h + GAP

    # ---------------- HISTORY card (past + combined) ----------------
    draw_card(PAD, y_cursor, W - 2 * PAD, hist_h, "HISTORY")

    hist_x  = PAD + CARD_PAD
    hist_y0 = y_cursor + CARD_PAD + SEC_H + S(4)

    # line 1 — previous runs only
    line1 = (f"Previous runs {n_runs}, Observed Duration (Hours) = {hms(tot_obs)}, "
             f"Detected Duration (Hours) = {hms(tot_det)} "
             f"({(tot_det / tot_obs * 100) if tot_obs > 0 else 0:.1f}%)")
    cv2.putText(canvas, line1,
                (hist_x, hist_y0 + S(14)),
                cv2.FONT_HERSHEY_SIMPLEX, F_BODY, C_VALUE_DIM,
                THIN, cv2.LINE_AA)

    # line 2 — previous + current combined
    line2 = (f"Past + current:  {comb_runs} runs, Observed Duration (Hours) = {hms(comb_obs)} "
             f"Detected Duration (Hours) = {hms(comb_det)}  ({comb_rate:.1f}%)")
    cv2.putText(canvas, line2,
                (hist_x, hist_y0 + ROW_H + S(14)),
                cv2.FONT_HERSHEY_SIMPLEX, F_BODY, C_VALUE,
                THIN, cv2.LINE_AA)

    y_cursor += hist_h + GAP

    # ---------------- charts ----------------
    remaining = y_bot - y_cursor
    min_chart_h = S(AN_CHART_MIN_H)
    if remaining >= 2 * min_chart_h + GAP:
        chart_h = (remaining - GAP) // 2
        chart_w = W - 2 * PAD
        draw_hour_chart(PAD, y_cursor, chart_w, chart_h)
        y_cursor += chart_h + GAP
        draw_sparkline(PAD, y_cursor, chart_w, chart_h,
                       list(analytics.prob_history), risk_c)

    return canvas

# ============================================================
#  REPORTS
# ============================================================
def write_report(path, run_id, a, predictor, p, risk, p_rec, p_hour,
                 detected, now):
    sess = a.session_duration_s(now)
    det_s = a.session_detected_s(now)
    pct = (det_s / sess * 100) if sess > 0 else 0.0
    sld = a.since_last_detection(now)

    tot_obs = sum(predictor.history["hour_observed"].values())
    tot_det = sum(predictor.history["hour_detected"].values())
    n_runs  = predictor.history["n_runs"]
    comb_obs = tot_obs + sess
    comb_det = tot_det + det_s
    comb_rate = (comb_det / comb_obs * 100) if comb_obs > 0 else 0.0

    lines = [
        "================ LIVE ANALYTICS ================",
        f"Run ID      : {run_id}",
        f"Updated     : {iso(now)}",
        f"Session     : {hms(sess)}",
        "",
        "DETECTION",
        f"  Current state        : "
        f"{'ELEPHANT PRESENT (' + format(now - a.streak_start, '.1f') + ' s)' if detected else 'clear'}",
        f"  Detections this run  : {len(a.detections)}",
        f"  Time detected        : {hms(det_s)} ({pct:.1f} % of session)",
        f"  Last detection       : {'n/a' if sld is None else f'{sld:.1f} s ago'}",
        f"  Confidence           : mean {a.mean_conf():.2f} | peak {a.peak_conf():.2f}",
        f"  5-min activity rate  : {a.recent_rate * 100:.1f} %",
        "",
        "PREDICTION in next 60 s",
        f"  P(presence)          : {p:.2f}   [{risk}]",
        f"  Hour-of-day prior    : {p_hour:.2f}"
        + (f"   (peak hour {predictor.peak_hour:02d}:00)"
           if predictor.peak_hour is not None else ""),
        f"  Recency component    : {p_rec:.2f}   (tau = {DECAY_TAU_S:.0f} s)",
        f"  Historical median gap: "
        f"{'n/a' if predictor.median_gap is None else f'{predictor.median_gap:.0f} s'}",
        "",
        "HISTORY",
        f"  Number of Previous runs: {n_runs}",
        f"  Previous Observed Duration: {hms(tot_obs)}",
        f"  Previous Elephant detected Duration: {hms(tot_det)}"
        f"  ({(tot_det/tot_obs*100) if tot_obs > 0 else 0:.1f} %)",
        f"  Past + current runs: {n_runs + 1}",
        f"  Combined Observed Duration: {hms(comb_obs)}",
        f"  Combined Detected Duration: {hms(comb_det)}  ({comb_rate:.1f} %)",
        "================================================",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_state_json(path, run_id, a, predictor, p, risk, detected, now):
    state = {
        "run_id": run_id,
        "updated": iso(now),
        "elephant_detected": bool(detected),
        "prediction": round(p, 3),
        "risk": risk,
        "recent_rate_5min": round(a.recent_rate, 3),
        "detections_this_run": len(a.detections),
        "session_seconds": round(a.session_duration_s(now), 1),
        "detected_seconds": round(a.session_detected_s(now), 1),
        "mean_confidence": round(a.mean_conf(), 3),
        "peak_confidence": round(a.peak_conf(), 3),
        "history": {
            "runs": predictor.history["n_runs"],
            "base_rate": round(predictor.base_rate, 4),
            "median_gap_s": (round(predictor.median_gap, 1)
                             if predictor.median_gap else None),
            "peak_hour": predictor.peak_hour,
            "hour_prior": {str(h): round(v, 4)
                           for h, v in predictor.hour_prior.items()},
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

def save_session_plot(path, predictor, a, now):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hours = list(range(24))
    priors = [predictor.hour_prior.get(h, predictor.base_rate) * 100 for h in hours]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(hours, priors, color="#c0392b")
    axes[0].set_title("Historical detection risk by hour of day")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("% time with elephant")
    axes[0].set_xticks(hours); axes[0].grid(axis="y", alpha=.3)
    dets = a.detections
    if dets:
        t0 = a.t0
        for s, e, pk, _ in dets:
            axes[1].barh(0, e - s, left=s - t0, height=0.5,
                         color="#c0392b", alpha=0.8)
        axes[1].set_ylim(-1, 1)
        axes[1].set_xlim(0, max(1, a.session_duration_s(now)))
        axes[1].set_yticks([])
    axes[1].set_title("This session — detection timeline")
    axes[1].set_xlabel("Seconds since session start")
    axes[1].grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"📊 Session chart saved: {path}")

# ============================================================
#  MAIN
# ============================================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    model_names = model.names

    history = load_history(CSV_LOG)
    logger = EventLogger(CSV_LOG, TEXT_LOG, RUN_ID, MODEL_PATH)
    analytics = LiveAnalytics(RUN_ID, history, logger)
    predictor = PresencePredictor(history)

    print(f"📚 History loaded: {history['n_runs']} previous run(s), "
          f"base rate {predictor.base_rate * 100:.1f} %")
    

    port = find_msp430_uart()
    if port:
        print(f"✅ Using {port} for MSP430 communication")
        ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
    elif REQUIRE_MSP430:
        raise Exception("MSP430 Application UART not found")
    else:
        print("⚠️  MSP430 not found — running in dry-run mode")
        ser = NullSerial()
    signal = SignalController(ser)

    cap = open_camera()
    if cap is None:
        print(f"❌ Could not open any of camera indices {CAMERA_INDICES}.")
        print("   • Is your USB camera plugged in and recognized by Windows?")
        print("   • Is another app using it? (Teams / Zoom / Camera / browser)")
        logger.close()
        sys.exit(1)

    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW, *CAMERA_DEFAULT_SIZE)
    cv2.moveWindow(CAMERA_WINDOW, 60, 60)

    cv2.namedWindow(ANALYTICS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ANALYTICS_WINDOW, *ANALYTICS_DEFAULT_SZ)
    cv2.moveWindow(ANALYTICS_WINDOW,
                   60 + CAMERA_DEFAULT_SIZE[0] + 40, 60)

    print("✅ Camera + analytics windows started (both resizable). "
          "Drag any edge to resize. Press 'q' to quit.")

    last_report = 0.0
    last_prob_push = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()

            results = model(frame, verbose=False)

            cam_w, cam_h = get_window_size(CAMERA_WINDOW, CAMERA_DEFAULT_SIZE)
            ana_w, ana_h = get_window_size(ANALYTICS_WINDOW, ANALYTICS_DEFAULT_SZ)

            cam_canvas, detected, best_conf = prepare_camera_canvas(
                frame, results, model_names, cam_w, cam_h)

            analytics.update(detected, best_conf, now)
            p, risk, p_rec, p_hour = predictor.predict(
                now, detected, analytics.last_det_end, analytics.recent_rate)

            if now - last_prob_push >= 1.0:
                analytics.prob_history.append(p)
                last_prob_push = now

            if detected:
                target = "ELEPHANT"
            elif SEND_PREDICTIVE_CAUTION and risk in ("HIGH", "CRITICAL"):
                target = "CAUTION"
            else:
                target = "NOELEPHANT"
            signal.set(target)

            cv2.imshow(CAMERA_WINDOW, cam_canvas)

            analytics_canvas = render_analytics_window(
                analytics, predictor, p, risk, p_rec, p_hour,
                detected, now, ana_w, ana_h)
            cv2.imshow(ANALYTICS_WINDOW, analytics_canvas)

            if now - last_report >= REPORT_INTERVAL_S:
                write_report(REPORT_TXT, RUN_ID, analytics, predictor,
                             p, risk, p_rec, p_hour, detected, now)
                write_state_json(STATE_JSON, RUN_ID, analytics, predictor,
                                 p, risk, detected, now)
                print(f"[{iso(now)}] P={p:.2f} [{risk}] | "
                      f"rate={analytics.recent_rate * 100:4.1f}% | "
                      f"dets={len(analytics.detections):3d} | "
                      f"{'ELEPHANT' if detected else 'clear'}")
                last_report = now

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        end = time.time()
        analytics.flush()
        if analytics.gap_start is not None and end > analytics.gap_start:
            analytics.gaps.append((analytics.gap_start, end))
        logger.run(analytics.t0, end)

        p, risk, p_rec, p_hour = predictor.predict(
            end, False, analytics.last_det_end, analytics.recent_rate)
        write_report(REPORT_TXT, RUN_ID, analytics, predictor,
                     p, risk, p_rec, p_hour, False, end)
        write_state_json(STATE_JSON, RUN_ID, analytics, predictor,
                         p, risk, False, end)
        save_session_plot(PLOT_PNG, predictor, analytics, end)

        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        logger.close()

        print("\n📂 Session summary")
        print(f"   Structured log : {CSV_LOG}")
        print(f"   Human log      : {TEXT_LOG}")
        print(f"   Live report    : {REPORT_TXT}")
        print(f"   JSON state     : {STATE_JSON}")
        print(f"   Runs logged    : {predictor.history['n_runs'] + 1}")


if __name__ == "__main__":
    main()