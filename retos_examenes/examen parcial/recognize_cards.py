#!/usr/bin/env python3
"""
recognize_cards.py (actualizado con color detection)
Muestra por encima de cada carta detectada: rank, suit y color (red/black).
"""

import cv2
import numpy as np
import os
import glob
import argparse
import pandas as pd
import time

# ----------------------------
# Utilidades generales
# ----------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def order_points(pts):
    pts = pts.astype("float32")
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts, W=200, H=300):
    rect = order_points(pts)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped

# ----------------------------
# Cargar plantillas (grises binarizadas)
# ----------------------------
def load_templates(folder):
    templates = {}
    if not os.path.isdir(folder):
        return templates
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # if RGBA or color, convert to gray
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        templates[name] = th
    return templates

# ----------------------------
# Matching multi-scale (template matching)
# ----------------------------
def match_templates(search_gray, templates, threshold=0.55, scales=(0.8, 1.0, 1.2)):
    best_name = ""
    best_score = 0.0
    if not templates:
        return best_name, best_score
    h_s, w_s = search_gray.shape[:2]
    for name, tpl in templates.items():
        th_h, th_w = tpl.shape[:2]
        for s in scales:
            neww = max(6, int(th_w * s))
            newh = max(6, int(th_h * s))
            if newh >= h_s or neww >= w_s:
                continue
            try:
                tpl_rs = cv2.resize(tpl, (neww, newh), interpolation=cv2.INTER_AREA)
            except Exception:
                continue
            res = cv2.matchTemplate(search_gray, tpl_rs, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, _ = cv2.minMaxLoc(res)
            if maxv > best_score:
                best_score = float(maxv)
                best_name = name
    if best_score < threshold:
        return "", best_score
    return best_name, best_score

# ----------------------------
# Green mask (calibrable)
# ----------------------------
def default_green_mask_bgr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def sample_green_hsv(img, sample_rect=None, margin=30):
    h, w = img.shape[:2]
    if sample_rect is None:
        cx, cy = w//2, h//2
        sw, sh = max(20, w//10), max(20, h//10)
        x1 = max(0, cx - sw//2)
        y1 = max(0, cy - sh//2)
        x2 = min(w, cx + sw//2)
        y2 = min(h, cy + sh//2)
    else:
        x1,y1,sw,sh = sample_rect
        x2 = x1 + sw
        y2 = y1 + sh
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = int(np.mean(hsv[:,:,0]))
    s_mean = int(np.mean(hsv[:,:,1]))
    v_mean = int(np.mean(hsv[:,:,2]))
    lower = np.array([max(0,h_mean-margin), max(20, s_mean - margin), max(20, v_mean - margin)])
    upper = np.array([min(179, h_mean+margin), 255, 255])
    return lower, upper

def green_mask_from_hsv(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# ----------------------------
# Find card candidates
# ----------------------------
def find_card_candidates(img, mask_green, min_area=1500):
    inv = cv2.bitwise_not(mask_green)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2)
            candidates.append(('quad', pts, area, cnt))
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            rect_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
            candidates.append(('rect', rect_pts, area, cnt))
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates

# ----------------------------
# Extract corner ROI (top-left)
# ----------------------------
def extract_corner_region(warped_card, corner_h_ratio=0.22, corner_w_ratio=0.18):
    H, W = warped_card.shape[:2]
    h = max(6, int(corner_h_ratio * H))
    w = max(6, int(corner_w_ratio * W))
    corner = warped_card[0:h, 0:w]
    return corner

# ----------------------------
# Detect color (red or black) from corner ROI
# ----------------------------
def detect_color_from_corner(corner_bgr):
    """
    Try to determine if the suit/rank in corner is red or black.
    Heuristics:
      - compute mean of R,G,B in a mask of non-white pixels
      - compute mean hue in HSV; red tends to have hue near 0 (or >160)
    Returns "red" or "black"
    """
    if corner_bgr is None or corner_bgr.size == 0:
        return "unknown"
    # remove near-white background
    gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # non-white regions
    if cv2.countNonZero(mask) < 20:
        # nothing detected; fallback to simple mean BGR
        b_mean, g_mean, r_mean = np.mean(corner_bgr.reshape(-1,3), axis=0)
        if r_mean > g_mean * 1.1 and r_mean > b_mean * 1.1 and r_mean > 80:
            return "red"
        else:
            return "black"
    # compute mean color on masked region
    masked = cv2.bitwise_and(corner_bgr, corner_bgr, mask=mask)
    b_mean = np.sum(masked[:,:,0]) / (cv2.countNonZero(mask) + 1e-6)
    g_mean = np.sum(masked[:,:,1]) / (cv2.countNonZero(mask) + 1e-6)
    r_mean = np.sum(masked[:,:,2]) / (cv2.countNonZero(mask) + 1e-6)
    # HSV check for red
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:,:,0][mask>0]
    if len(h_vals) > 0:
        h_mean = int(np.mean(h_vals))
        # red wraps near 0; consider both low and high hue values
        if (h_mean <= 10) or (h_mean >= 160):
            return "red"
    # fallback RGB ratio
    if (r_mean > g_mean * 1.1) and (r_mean > b_mean * 1.1) and (r_mean > 80):
        return "red"
    return "black"

# ----------------------------
# process single frame: suit-first approach + color detection
# ----------------------------
def process_frame(img, rank_templates_black, rank_templates_red, suit_templates,
                  hsv_range=None, params=None):
    if params is None:
        params = {}
    blur = cv2.GaussianBlur(img, (5,5), 0)
    if hsv_range is None:
        mask_green = default_green_mask_bgr(blur)
    else:
        mask_green = green_mask_from_hsv(blur, hsv_range[0], hsv_range[1])
    candidates = find_card_candidates(img, mask_green, min_area=params.get('min_area',1500))
    detections = []
    for kind, pts, area, cnt in candidates:
        try:
            warped = four_point_transform(img, pts, W=params.get('warpW',200), H=params.get('warpH',300))
        except Exception:
            x,y,w,h = cv2.boundingRect(cnt)
            rect_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
            warped = four_point_transform(img, rect_pts, W=params.get('warpW',200), H=params.get('warpH',300))

        corner = extract_corner_region(warped, corner_h_ratio=params.get('corner_h',0.22),
                                       corner_w_ratio=params.get('corner_w',0.18))
        if corner.size == 0:
            continue

        gray_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray_corner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = cv2.bitwise_not(th)

        # 1) Detect suit first
        suit_name, suit_conf = match_templates(th, suit_templates, threshold=params.get('suit_th',0.45))
        if suit_name == "":
            suit_name, suit_conf = match_templates(th_inv, suit_templates, threshold=params.get('suit_th',0.45))

        # determine color from suit if possible
        color = ""
        if suit_name in ['heart', 'diamond']:
            color = "red"
        elif suit_name in ['spade', 'club']:
            color = "black"
        else:
            # fallback: detect color from corner image
            color = detect_color_from_corner(corner)

        # choose rank templates bank based on suit
        if suit_name in ['heart', 'diamond']:
            rank_bank = rank_templates_red
        elif suit_name in ['spade', 'club']:
            rank_bank = rank_templates_black
        else:
            rank_bank = None

        # 2) Match rank
        rank_name = ""
        rank_conf = 0.0
        if rank_bank is not None and len(rank_bank) > 0:
            rank_name, rank_conf = match_templates(th, rank_bank, threshold=params.get('rank_th',0.50))
            if rank_name == "":
                rank_name, rank_conf = match_templates(th_inv, rank_bank, threshold=params.get('rank_th',0.50))
        else:
            rb1_name, rb1_conf = ("", 0.0)
            rb2_name, rb2_conf = ("", 0.0)
            if rank_templates_black:
                rb1_name, rb1_conf = match_templates(th, rank_templates_black, threshold=params.get('rank_th',0.50))
                if rb1_name == "":
                    rb1_name, rb1_conf = match_templates(th_inv, rank_templates_black, threshold=params.get('rank_th',0.50))
            if rank_templates_red:
                rb2_name, rb2_conf = match_templates(th, rank_templates_red, threshold=params.get('rank_th',0.50))
                if rb2_name == "":
                    rb2_name, rb2_conf = match_templates(th_inv, rank_templates_red, threshold=params.get('rank_th',0.50))
            if rb1_conf >= rb2_conf:
                rank_name, rank_conf = rb1_name, rb1_conf
            else:
                rank_name, rank_conf = rb2_name, rb2_conf
            if suit_name == "" and rank_name != "":
                if (rb2_conf > rb1_conf) and (rb2_name != ""):
                    suit_name = "heart_or_diamond"

        detection = {
            'pts': pts.tolist(),
            'area': float(area),
            'warped': warped,
            'corner': corner,
            'rank': rank_name,
            'rank_conf': float(rank_conf),
            'suit': suit_name,
            'suit_conf': float(suit_conf),
            'color': color
        }
        detections.append(detection)
    return detections, mask_green

# ----------------------------
# main CLI and orchestration
# ----------------------------
def main(args):
    rank_templates_black = load_templates(args.ranks_black)
    rank_templates_red   = load_templates(args.ranks_red)
    suit_templates = load_templates(args.suits)

    print(f"Loaded rank templates (black): {len(rank_templates_black)}")
    print(f"Loaded rank templates (red): {len(rank_templates_red)}")
    print(f"Loaded suit templates: {len(suit_templates)} -> {list(suit_templates.keys())}")

    source = args.source
    try:
        src_int = int(source)
        cap = cv2.VideoCapture(src_int)
    except Exception:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: No se puede abrir la fuente de vídeo. Revisa --source")
        return

    params = {
        'warpW': args.warp_w,
        'warpH': args.warp_h,
        'corner_h': args.corner_h,
        'corner_w': args.corner_w,
        'min_area': args.min_area,
        'rank_th': args.rank_th,
        'suit_th': args.suit_th
    }

    ensure_dir(args.out_dir)
    results = []
    frame_count = 0
    hsv_range = None

    print("Presiona 'c' para calibrar rango verde muestreando el centro de la imagen.")
    print("Presiona 's' para guardar warped images de las detecciones.")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del stream o error de lectura.")
            break
        frame_count += 1

        if args.resize_width and frame.shape[1] > args.resize_width:
            scale = args.resize_width / frame.shape[1]
            frame = cv2.resize(frame, (args.resize_width, int(frame.shape[0]*scale)))

        display = frame.copy()

        if hsv_range is None and args.auto_calibrate:
            lower, upper = sample_green_hsv(frame)
            hsv_range = (lower, upper)
            print(f"Auto-calibrated HSV lower={lower}, upper={upper}")

        detections, mask_green = process_frame(frame, rank_templates_black, rank_templates_red,
                                               suit_templates, hsv_range=hsv_range, params=params)

        # draw detecciones + label with color
        for i, det in enumerate(detections):
            pts = np.array(det['pts'], dtype=np.int32).reshape(-1,2)
            cv2.polylines(display, [pts], True, (0,255,0), 2)
            cx = int(np.mean(pts[:,0])); cy = int(np.mean(pts[:,1]))
            # build label: rank suit (color)
            rank = det.get('rank', '') or "?"
            suit = det.get('suit', '') or "?"
            color = det.get('color', 'unknown')
            label = f"{rank} {suit} ({color})"
            # put label above contour (try to avoid going off-image)
            text_x = max(10, cx - 80)
            text_y = max(20, cy - 10)
            cv2.putText(display, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # save detection row
            results.append({
                'frame': frame_count,
                'rank': det['rank'],
                'rank_conf': det['rank_conf'],
                'suit': det['suit'],
                'suit_conf': det['suit_conf'],
                'color': det['color'],
                'area': det['area']
            })

        mask_vis = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
        combine = np.hstack([cv2.resize(display, (display.shape[1]//2, display.shape[0]//2)),
                             cv2.resize(mask_vis, (mask_vis.shape[1]//2, mask_vis.shape[0]//2))])
        cv2.imshow("Detecciones (left)  |  Mascara verde (right)", combine)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            lower, upper = sample_green_hsv(frame)
            hsv_range = (lower, upper)
            print(f"Calibrado manual: HSV lower={lower}, upper={upper}")
        elif key == ord('s'):
            ts = int(time.time())
            for ii, det in enumerate(detections):
                fname = os.path.join(args.out_dir, f"warp_{frame_count}_{ii}_{ts}.png")
                cv2.imwrite(fname, det['warped'])
            print(f"Saved {len(detections)} warped card images to {args.out_dir}")

    cap.release()
    cv2.destroyAllWindows()

    # save CSV
    if len(results) > 0:
        df = pd.DataFrame(results)
        out_csv = os.path.join(args.out_dir, args.out_csv)
        df.to_csv(out_csv, index=False)
        print(f"Resultados guardados en {out_csv}")
    else:
        print("No se guardaron resultados (ninguna detección).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="índice de webcam o URL (ej: 0 o http://192.168.x.y:8080/video)")
    ap.add_argument("--ranks_black", default="templates/ranks/ranks_black", help="carpeta con ranks negros")
    ap.add_argument("--ranks_red", default="templates/ranks/ranks_red", help="carpeta con ranks rojos")
    ap.add_argument("--suits", default="templates/suits", help="carpeta con suits (spade,club,heart,diamond)")
    ap.add_argument("--out_dir", default="out", help="carpeta para guardar outputs")
    ap.add_argument("--out_csv", default="results.csv", help="nombre CSV de resultados")
    ap.add_argument("--warp_w", type=int, default=200, help="anchura de la carta rectificada")
    ap.add_argument("--warp_h", type=int, default=300, help="altura de la carta rectificada")
    ap.add_argument("--corner_h", type=float, default=0.22, help="proporcion altura corner")
    ap.add_argument("--corner_w", type=float, default=0.18, help="proporcion ancho corner")
    ap.add_argument("--min_area", type=int, default=1500, help="área mínima para contornos candidatos")
    ap.add_argument("--rank_th", type=float, default=0.50, help="umbral confianza rank")
    ap.add_argument("--suit_th", type=float, default=0.45, help="umbral confianza suit")
    ap.add_argument("--resize_width", type=int, default=1280, help="si la imagen es muy grande, redimensionar ancho")
    ap.add_argument("--auto_calibrate", action='store_true', help="calibrar HSV automáticamente al inicio")
    args = ap.parse_args()
    main(args)
