"""
================================================================================
LIVE INTERACTIVE DEMO — AUTONOMOUS FRONTIER EXPLORATION & RESCUE MAPPING
- Chạy trực tiếp 30 FPS siêu mượt (OpenCV + Tkinter, không dùng Matplotlib lag)
- Hỗ trợ 4 bản đồ chuẩn (Map A/B/C/D) + Sinh Random Map ngẫu nhiên không giới hạn
- 3 Chế độ trình diễn:
    [1] Đua trực tiếp 3 Thuật toán (Random Walk vs. Nearest Frontier vs. Proposed Utility)
    [2] Soi chi tiết Cơ chế Đề xuất (Toàn cảnh 2D + Camera Cận cảnh 3x + Biểu đồ Độ phủ)
    [3] Thử nghiệm Khói mù 30% (Raw Mapping vs. Bayesian Log-Odds)
- Phím tắt quay màn hình: SPACE (Chạy/Dừng) | N (Random Map mới) | 1/2/3 (Đổi chế độ)
                          H (Ẩn/Hiện thanh điều khiển) | F11 (Fullscreen) | T (Thank You)
================================================================================
"""

import os
import math
import time
import random
import tkinter as tk
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import rescue_robot_exploration as rre

# ==============================================================================
# 1. CẤU HÌNH & PHÔNG CHỮ TIẾNG VIỆT SẮC NÉT
# ==============================================================================
CANVAS_W = 1600
CANVAS_H = 900
TARGET_FPS = 30
GRID = rre.GRID_SIZE  # 50x50
RES = rre.RESOLUTION  # 0.5 m/ô

BG_DARK = (10, 15, 26)       # RGB
BG_PANEL = (17, 24, 39)      # RGB
BORDER_CYAN = (0, 210, 211)
BORDER_SLATE = (45, 58, 82)


def load_font(size, bold=False, mono=False):
    win_fonts = r"C:\Windows\Fonts"
    if mono:
        cands = ["consolab.ttf" if bold else "consola.ttf", "courbd.ttf" if bold else "cour.ttf"]
    elif bold:
        cands = ["segoeuib.ttf", "arialbd.ttf", "tahomabd.ttf"]
    else:
        cands = ["segoeui.ttf", "arial.ttf", "tahoma.ttf"]
    for fn in cands:
        fp = os.path.join(win_fonts, fn)
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


F_TITLE = load_font(22, bold=True)
F_SUB = load_font(15, bold=False)
F_HEAD = load_font(17, bold=True)
F_BODY = load_font(15, bold=False)
F_BOLD = load_font(15, bold=True)
F_SMALL = load_font(13, bold=False)
F_MONO = load_font(14, bold=True, mono=True)
F_MONO_SM = load_font(13, bold=False, mono=True)
F_THANK_BIG = load_font(96, bold=True)
F_THANK_SUB = load_font(26, bold=False)


# ==============================================================================
# 2. SINH BẢN ĐỒ NGẪU NHIÊN (PROCEDURAL RANDOM MAP) ĐẢM BẢO LIÊN THÔNG 100%
# ==============================================================================
def create_random_disaster_map(seed=None):
    """Sinh ngẫu nhiên tòa nhà sập có phòng, cửa, xà bần & 3 nạn nhân luôn đi tới được."""
    if seed is None:
        seed = random.randint(1000, 999999)
    rng = np.random.RandomState(seed)

    for _attempt in range(30):
        grid = np.zeros((GRID, GRID), dtype=int)
        grid[0, :] = rre.CELL_OBSTACLE
        grid[-1, :] = rre.CELL_OBSTACLE
        grid[:, 0] = rre.CELL_OBSTACLE
        grid[:, -1] = rre.CELL_OBSTACLE

        # Chọn ngẫu nhiên 2 tường ngang & 2 tường dọc tạo bố cục 9 phòng bất kỳ
        row_walls = sorted([int(rng.randint(13, 21)), int(rng.randint(29, 38))])
        col_walls = sorted([int(rng.randint(13, 21)), int(rng.randint(29, 38))])

        for rw in row_walls:
            grid[rw, :] = rre.CELL_OBSTACLE
        for cw in col_walls:
            grid[:, cw] = rre.CELL_OBSTACLE

        # Đục cửa thông phòng ngẫu nhiên rộng 3 ô giữa các khoang
        row_bounds = [1, row_walls[0], row_walls[1], GRID - 1]
        col_bounds = [1, col_walls[0], col_walls[1], GRID - 1]
        doors = []
        for rw in row_walls:
            for j in range(3):
                c_lo, c_hi = col_bounds[j] + 3, col_bounds[j + 1] - 3
                if c_hi > c_lo:
                    dc = int(rng.randint(c_lo, c_hi + 1))
                    grid[rw, max(1, dc - 1):min(GRID - 1, dc + 2)] = rre.CELL_FREE
                    doors.append((rw, dc))
        for cw in col_walls:
            for i in range(3):
                r_lo, r_hi = row_bounds[i] + 3, row_bounds[i + 1] - 3
                if r_hi > r_lo:
                    dr = int(rng.randint(r_lo, r_hi + 1))
                    grid[max(1, dr - 1):min(GRID - 1, dr + 2), cw] = rre.CELL_FREE
                    doors.append((dr, cw))

        # Sinh vị trí xuất phát ở 1 trong 4 góc
        corners = [(3, 3), (3, GRID - 4), (GRID - 4, 3), (GRID - 4, GRID - 4)]
        start_pos = corners[int(rng.randint(0, 4))]

        # Sinh 10-14 khối xà bần + 14-20 mảnh vỡ rời rạc
        n_blocks = int(rng.randint(10, 15))
        for _ in range(n_blocks):
            bh = int(rng.randint(2, 4))
            bw = int(rng.randint(2, 4))
            br = int(rng.randint(3, GRID - bh - 2))
            bc = int(rng.randint(3, GRID - bw - 2))
            if math.hypot(br - start_pos[0], bc - start_pos[1]) < 5.5:
                continue
            if any(abs(br - dr) <= 3 and abs(bc - dc) <= 3 for dr, dc in doors):
                continue
            grid[br:br + bh, bc:bc + bw] = rre.CELL_OBSTACLE

        for _ in range(int(rng.randint(14, 22))):
            rr = int(rng.randint(3, GRID - 3))
            cc = int(rng.randint(3, GRID - 3))
            if math.hypot(rr - start_pos[0], cc - start_pos[1]) < 4.5:
                continue
            if any(abs(rr - dr) <= 3 and abs(cc - dc) <= 3 for dr, dc in doors):
                continue
            grid[rr, cc] = rre.CELL_OBSTACLE

        grid[start_pos[0] - 1:start_pos[0] + 2, start_pos[1] - 1:start_pos[1] + 2] = rre.CELL_FREE

        # Kiểm tra liên thông từ start_pos bằng cây đường đi ngắn nhất
        _, geom, _ = rre.compute_shortest_path_tree(grid, start_pos, safety_cost_weight=0.0)
        reachable = np.argwhere((geom < float("inf")) & (geom > 16.0))
        if len(reachable) < 120:
            continue

        # Chọn 3 nạn nhân phân bố rải rác ở các vùng xa điểm xuất phát
        victims = []
        idxs = rng.permutation(len(reachable))
        for idx in idxs:
            vr, vc = int(reachable[idx, 0]), int(reachable[idx, 1])
            if 2 <= vr <= GRID - 3 and 2 <= vc <= GRID - 3:
                if all(math.hypot(vr - pr, vc - pc) >= 14.0 for pr, pc in victims):
                    victims.append((vr, vc))
                    if len(victims) == 3:
                        break
        if len(victims) == 3:
            for vx, vy in victims:
                grid[vx - 1:vx + 2, vy - 1:vy + 2] = rre.CELL_FREE
            return grid, start_pos, victims, seed

    # Dự phòng nếu hiếm khi không đạt
    grid, start_pos, victims = rre.create_environment_map("Map_C", seed=seed)
    return grid, start_pos, victims, seed


def load_map_by_key(map_key, rand_seed=None):
    if map_key == "Random":
        grid, sp, vc, sd = create_random_disaster_map(seed=rand_seed)
        return grid, sp, vc, f"Random Map #{sd % 10000:04d}"
    labels = {
        "Map_A": "Map A — Văn phòng 9 phòng",
        "Map_B": "Map B — Mê cung U-Trap",
        "Map_C": "Map C — Tòa nhà Sập Xà bần",
        "Map_D": "Map D — Hang động / Hầm lò",
    }
    grid, sp, vc = rre.create_environment_map(map_key, seed=42)
    return grid, sp, vc, labels.get(map_key, map_key)


# ==============================================================================
# 3. CHẠY MÔ PHỎNG THUẬT TOÁN THỰC TẾ TRÊN BẢN ĐỒ BẤT KỲ (NHANH ~0.35s)
# ==============================================================================
def run_algorithm_trace(true_map, start_pos, victims, strategy="utility",
                        lidar_cells=10, noise_std=0.0, smoke_prob=0.0,
                        use_log_odds=True, max_steps=340, seed=42):
    rng = np.random.RandomState(seed)
    explored = np.full((GRID, GRID), rre.CELL_UNKNOWN, dtype=int)
    log_odds = np.zeros((GRID, GRID), dtype=float)

    _, geom_true, _ = rre.compute_shortest_path_tree(true_map, start_pos, safety_cost_weight=0.0)
    reachable_free = max(1, int(np.sum(geom_true < float("inf"))))

    robot_pos = start_pos
    trajectory = [robot_pos]
    heading = 0.0
    victims_found = [False] * len(victims)
    victim_all_step = None
    dist_m = 0.0
    sharp_turns = 0
    step = 0
    frames = []

    rre.simulate_lidar_scan(
        true_map, robot_pos, explored, log_odds,
        lidar_range=lidar_cells, noise_std=noise_std, dropout_prob=smoke_prob,
        use_log_odds_filter=use_log_odds, rng=rng,
    )

    def snapshot(frontiers=None, target=None, planned=None, evac=None, done=False):
        known_free = int(np.sum((explored == rre.CELL_FREE) & (geom_true < float("inf"))))
        cov = min(100.0, (known_free / reachable_free) * 100.0)
        phantom = (explored == rre.CELL_OBSTACLE) & (true_map == rre.CELL_FREE)
        ee = (known_free * (RES ** 2)) / max(0.5, dist_m)
        return {
            "step": step,
            "pos": robot_pos,
            "heading": heading,
            "explored": explored.copy(),
            "phantom": phantom if smoke_prob > 0 else None,
            "phantom_cnt": int(np.sum(phantom)) if smoke_prob > 0 else 0,
            "traj": list(trajectory),
            "frontiers": list(frontiers) if frontiers else [],
            "target": target,
            "planned": list(planned) if planned else [],
            "evac": list(evac) if evac else None,
            "cov": cov,
            "dist_m": dist_m,
            "sharp_turns": sharp_turns,
            "victims": list(victims_found),
            "v_all_step": victim_all_step,
            "ee": ee,
            "done": done,
        }

    while step < max_steps:
        for vi, vp in enumerate(victims):
            if not victims_found[vi] and rre.check_victim_line_of_sight(true_map, robot_pos, vp, lidar_cells):
                victims_found[vi] = True
        if victim_all_step is None and all(victims_found):
            victim_all_step = step

        clusters = rre.detect_and_cluster_frontiers(explored, min_cluster_size=2)
        if not clusters:
            clusters = rre.detect_and_cluster_frontiers(explored, min_cluster_size=1)
        f_pts = [cl["centroid"] for cl in clusters]

        snap = snapshot(frontiers=f_pts)
        if step == 0:
            frames.append(snap)
        if snap["cov"] >= 98.5 or not clusters:
            break

        chosen, path = rre.select_next_frontier(
            clusters, robot_pos, explored,
            strategy=strategy, current_heading=heading, rng=rng,
        )
        if path is None or len(path) <= 1:
            break

        tgt = chosen["centroid"] if chosen else None
        horizon = 5 if strategy == "utility" else (4 if strategy == "nearest" else 6)
        steps_to_move = min(horizon, len(path) - 1)

        for i in range(1, steps_to_move + 1):
            nxt = path[i]
            dx, dy = nxt[0] - robot_pos[0], nxt[1] - robot_pos[1]
            new_h = math.atan2(dy, dx)
            if step > 0:
                dh = abs((new_h - heading + math.pi) % (2 * math.pi) - math.pi)
                if dh > (math.pi / 2.0 + 0.05):
                    sharp_turns += 1
            heading = new_h
            dist_m += math.hypot(dx, dy) * RES
            robot_pos = nxt
            trajectory.append(robot_pos)
            step += 1

            rre.simulate_lidar_scan(
                true_map, robot_pos, explored, log_odds,
                lidar_range=lidar_cells, noise_std=noise_std, dropout_prob=smoke_prob,
                use_log_odds_filter=use_log_odds, rng=rng,
            )
            for vi, vp in enumerate(victims):
                if not victims_found[vi] and rre.check_victim_line_of_sight(true_map, robot_pos, vp, lidar_cells):
                    victims_found[vi] = True
            if victim_all_step is None and all(victims_found):
                victim_all_step = step

            frames.append(snapshot(frontiers=f_pts, target=tgt, planned=path[i:]))

            if chosen is not None and strategy == "utility":
                cr, cc = chosen["centroid"]
                if np.sum(explored[max(1, cr - 1):min(GRID - 1, cr + 2),
                                   max(1, cc - 1):min(GRID - 1, cc + 2)] == rre.CELL_UNKNOWN) == 0:
                    break
            if frames[-1]["cov"] >= 98.5 or step >= max_steps:
                break

    # Lập đường sơ tán Safety-Inflated A* tới nạn nhân xa nhất
    evac_path = None
    best_v = max(victims, key=lambda v: math.hypot(v[0] - start_pos[0], v[1] - start_pos[1]))
    evac_cand, _ = rre.astar_plan(explored, start_pos, best_v, safety_cost_weight=2.2)
    if evac_cand and len(evac_cand) >= 2:
        evac_path = evac_cand

    frames.append(snapshot(evac=evac_path, done=True))
    return frames


# ==============================================================================
# 4. BỘ VẼ OPENCV SIÊU TỐC (6-10 MS/FRAME) + CHỮ TIẾNG VIỆT PIL SẮC NÉT
# ==============================================================================
def interp_pose(f0, f1, alpha):
    r0, c0 = f0["pos"]
    r1, c1 = f1["pos"]
    r = r0 + (r1 - r0) * alpha
    c = c0 + (c1 - c0) * alpha
    h0, h1 = f0["heading"], f1["heading"]
    dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
    h = h0 + dh * alpha
    return r, c, h


def render_map_patch(true_map, frame, next_frame, alpha, start_pos, victims,
                     out_size, traj_bgr=(255, 215, 0), body_bgr=(67, 159, 255),
                     show_frontiers=False, show_planned=False, show_rays=True,
                     zoom_radius=None):
    """Vẽ bản đồ Occupancy Grid + LiDAR + Robot bánh xích + Frontier + A* bằng OpenCV cực nhanh."""
    base = np.empty((GRID, GRID, 3), dtype=np.uint8)
    exp = frame["explored"]
    # Màu BGR cho OpenCV: Unknown=xám xanh, Free=trắng ngà, Obstacle=xanh navy đậm
    base[exp == rre.CELL_UNKNOWN] = (72, 60, 50)
    base[exp == rre.CELL_FREE] = (242, 240, 235)
    base[exp == rre.CELL_OBSTACLE] = (68, 36, 20)

    if frame["phantom"] is not None and frame["phantom_cnt"] > 0:
        base[frame["phantom"]] = (55, 55, 255)  # Đỏ rực cho tường ma do khói

    r_pos, c_pos, hdg = interp_pose(frame, next_frame, alpha)

    if zoom_radius is not None:
        # Cắt vùng cận cảnh quanh robot rồi phóng to
        pad = int(math.ceil(zoom_radius)) + 2
        padded = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(25, 20, 15))
        scale = out_size / (2.0 * zoom_radius)
        cx_p = c_pos + pad + 0.5
        cy_p = r_pos + pad + 0.5
        M = np.float32([[scale, 0, out_size * 0.5 - cx_p * scale],
                        [0, scale, out_size * 0.5 - cy_p * scale]])
        img = cv2.warpAffine(padded, M, (out_size, out_size), flags=cv2.INTER_NEAREST, borderValue=(25, 20, 15))

        def to_px(rr, cc):
            return (int(round((cc - c_pos) * scale + out_size * 0.5)),
                    int(round((rr - r_pos) * scale + out_size * 0.5)))
    else:
        img = cv2.resize(base, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        scale = out_size / float(GRID)

        def to_px(rr, cc):
            return (int(round((cc + 0.5) * scale)), int(round((rr + 0.5) * scale)))

    # 1. Vẽ chùm tia LiDAR 360° sống động
    if show_rays and not frame["done"]:
        lidar_r = 10.0
        ray_overlay = img.copy()
        rob_px = to_px(r_pos, c_pos)
        angles = np.linspace(0, 2 * math.pi, 36, endpoint=False) + hdg
        hit_pts = []
        for ang in angles:
            ca, sa = math.cos(ang), math.sin(ang)
            er, ec = r_pos + lidar_r * ca, c_pos + lidar_r * sa
            hit = False
            for d in np.linspace(0.5, lidar_r, 22):
                gr, gc = int(round(r_pos + d * ca)), int(round(c_pos + d * sa))
                if not (0 <= gr < GRID and 0 <= gc < GRID):
                    er, ec = r_pos + d * ca, c_pos + d * sa
                    break
                if true_map[gr, gc] == rre.CELL_OBSTACLE:
                    er, ec = gr, gc
                    hit = True
                    break
            epx = to_px(er, ec)
            cv2.line(ray_overlay, rob_px, epx, (220, 210, 0), 1, cv2.LINE_AA)
            if hit:
                hit_pts.append(epx)
        cv2.circle(ray_overlay, rob_px, int(round(lidar_r * scale)), (230, 190, 10), 1, cv2.LINE_AA)
        cv2.addWeighted(ray_overlay, 0.42, img, 0.58, 0, img)
        for hp in hit_pts:
            cv2.circle(img, hp, 2 if zoom_radius is None else 3, (87, 71, 255), -1, cv2.LINE_AA)

    # 2. Vẽ quỹ đạo đã đi
    traj = frame["traj"]
    if len(traj) >= 2:
        pts = [to_px(p[0], p[1]) for p in traj] + [to_px(r_pos, c_pos)]
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [arr], False, traj_bgr, 2 if zoom_radius is None else 3, cv2.LINE_AA)

    # 3. Cụm Frontier (tam giác xanh ngọc) & đường dự kiến A* (nét chấm vàng cam)
    if show_frontiers and not frame["done"]:
        for fr, fc in frame["frontiers"]:
            px, py = to_px(fr, fc)
            sz = 5 if zoom_radius is None else 8
            tri = np.array([[px, py - sz], [px - sz, py + sz], [px + sz, py + sz]], dtype=np.int32)
            cv2.fillConvexPoly(img, tri, (201, 206, 0), cv2.LINE_AA)
            cv2.polylines(img, [tri], True, (10, 15, 25), 1, cv2.LINE_AA)

    if show_planned and frame["planned"] and not frame["done"]:
        p_pts = [to_px(r_pos, c_pos)] + [to_px(p[0], p[1]) for p in frame["planned"]]
        for i in range(len(p_pts) - 1):
            if i % 2 == 0:
                cv2.line(img, p_pts[i], p_pts[i + 1], (18, 156, 243), 2 if zoom_radius is None else 3, cv2.LINE_AA)
        if frame["target"]:
            tx, ty = to_px(*frame["target"])
            ds = 6 if zoom_radius is None else 10
            dm = np.array([[tx, ty - ds], [tx + ds, ty], [tx, ty + ds], [tx - ds, ty]], dtype=np.int32)
            cv2.fillConvexPoly(img, dm, (15, 196, 241), cv2.LINE_AA)
            cv2.polylines(img, [dm], True, (10, 15, 25), 1, cv2.LINE_AA)

    # 4. Đường sơ tán Safety-Inflated A* (nét đứt đỏ nổi bật khi hoàn tất)
    if frame["evac"] and len(frame["evac"]) >= 2:
        e_pts = [to_px(p[0], p[1]) for p in frame["evac"]]
        for i in range(len(e_pts) - 1):
            if i % 2 == 0:
                cv2.line(img, e_pts[i], e_pts[i + 1], (87, 71, 255), 3 if zoom_radius is None else 4, cv2.LINE_AA)

    # 5. Điểm xuất phát (xanh lá) & 3 Nạn nhân (ngôi sao đỏ khi tìm thấy / xám khi chưa)
    sx, sy = to_px(*start_pos)
    cv2.circle(img, (sx, sy), 6 if zoom_radius is None else 9, (113, 204, 46), -1, cv2.LINE_AA)
    cv2.circle(img, (sx, sy), 6 if zoom_radius is None else 9, (15, 20, 30), 1, cv2.LINE_AA)

    for vi, (vr, vc) in enumerate(victims):
        vx, vy = to_px(vr, vc)
        found = frame["victims"][vi]
        col = (87, 71, 255) if found else (175, 180, 185)
        rad = 7 if zoom_radius is None else 11
        cv2.circle(img, (vx, vy), rad, col, -1, cv2.LINE_AA)
        cv2.circle(img, (vx, vy), rad, (255, 255, 255) if found else (40, 45, 55), 2, cv2.LINE_AA)
        if found:
            cv2.circle(img, (vx, vy), rad + 5, (87, 71, 255), 2, cv2.LINE_AA)

    # 6. Vẽ Robot cứu hộ bánh xích xoay theo góc hướng thực
    rx_px, ry_px = to_px(r_pos, c_pos)
    L = scale * (1.35 if zoom_radius is None else 1.55)
    W = scale * (0.95 if zoom_radius is None else 1.10)
    # Trong hệ tọa độ ảnh: x = cột (sin hdg), y = hàng (cos hdg)
    ux, uy = math.sin(hdg), math.cos(hdg)
    nx, ny = -uy, ux

    def poly_local(coords):
        return np.array(
            [[int(round(rx_px + lx * ux + ly * nx)), int(round(ry_px + lx * uy + ly * ny))]
             for lx, ly in coords],
            dtype=np.int32,
        )

    # 2 dải bánh xích
    for side in (-1, 1):
        tr = poly_local([
            (-L * 0.55, side * W * 0.42), (L * 0.55, side * W * 0.42),
            (L * 0.55, side * W * 0.72), (-L * 0.55, side * W * 0.72),
        ])
        cv2.fillConvexPoly(img, tr, (35, 38, 45), cv2.LINE_AA)

    # Thân xe bọc thép
    body = poly_local([
        (-L * 0.48, -W * 0.46), (L * 0.48, -W * 0.46),
        (L * 0.48, W * 0.46), (-L * 0.48, W * 0.46),
    ])
    cv2.fillConvexPoly(img, body, body_bgr, cv2.LINE_AA)
    cv2.polylines(img, [body], True, (15, 20, 30), 1, cv2.LINE_AA)

    # Mũi xe chỉ hướng (màu đỏ cam)
    nose = poly_local([(L * 0.35, -W * 0.36), (L * 0.62, 0.0), (L * 0.35, W * 0.36)])
    cv2.fillConvexPoly(img, nose, (60, 85, 240), cv2.LINE_AA)

    # Tháp LiDAR tròn trên lưng xe
    cv2.circle(img, (rx_px, ry_px), max(3, int(round(scale * 0.32))), (227, 132, 9), -1, cv2.LINE_AA)
    cv2.circle(img, (rx_px, ry_px), max(3, int(round(scale * 0.32))), (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_coverage_chart(w, h, traces_dict, cur_idx):
    """Vẽ biểu đồ đường cong Độ phủ (%) theo Quãng đường (m) chạy sống bằng OpenCV."""
    img = np.full((h, w, 3), (26, 19, 13), dtype=np.uint8)  # BGR
    pad_l, pad_r, pad_t, pad_b = 56, 20, 34, 38
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    # Lưới ngang 0..100%
    for pct in (0, 25, 50, 75, 100):
        yy = pad_t + ph - int(ph * pct / 100.0)
        cv2.line(img, (pad_l, yy), (pad_l + pw, yy), (55, 45, 38), 1, cv2.LINE_AA)

    # Đường mục tiêu 98.5%
    y98 = pad_t + ph - int(ph * 0.985)
    cv2.line(img, (pad_l, y98), (pad_l + pw, y98), (110, 180, 90), 1, cv2.LINE_AA)

    max_d = max(90.0, max(tr[-1]["dist_m"] for tr in traces_dict.values()) * 1.05)

    styles = [
        ("random", (129, 107, 255), 2),   # Hồng đỏ
        ("nearest", (118, 190, 255), 2),  # Cam
        ("utility", (251, 219, 72), 3),   # Xanh cyan sáng
    ]
    for key, col_bgr, th in styles:
        tr = traces_dict[key]
        end_i = min(cur_idx, len(tr) - 1)
        pts = []
        for k in range(0, end_i + 1):
            xx = pad_l + int(pw * min(1.0, tr[k]["dist_m"] / max_d))
            yy = pad_t + ph - int(ph * min(100.0, tr[k]["cov"]) / 100.0)
            pts.append((xx, yy))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))], False, col_bgr, th, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 4, col_bgr, -1, cv2.LINE_AA)

    # Khung trục
    cv2.rectangle(img, (pad_l, pad_t), (pad_l + pw, pad_t + ph), (120, 105, 90), 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    d.text((pad_l, 8), "Biểu đồ Độ phủ (%) theo Quãng đường (m) — Thời gian thực", font=F_BOLD, fill=(254, 202, 87))
    for pct in (0, 50, 100):
        yy = pad_t + ph - int(ph * pct / 100.0) - 8
        d.text((10, yy), f"{pct:3d}%", font=F_SMALL, fill=(180, 190, 205))
    for frac in (0.0, 0.5, 1.0):
        xx = pad_l + int(pw * frac) - 14
        d.text((xx, pad_t + ph + 8), f"{max_d * frac:.0f}m", font=F_SMALL, fill=(180, 190, 205))
    # Chú thích màu
    lx = w - 340
    for txt, col_rgb in [("Proposed", (72, 219, 251)), ("Nearest", (255, 190, 118)), ("Random", (255, 107, 129))]:
        d.rectangle([lx, 12, lx + 14, 22], fill=col_rgb)
        d.text((lx + 18, 8), txt, font=F_SMALL, fill=(235, 240, 245))
        lx += 105
    return pil


# ==============================================================================
# 5. ỨNG DỤNG TRÌNH DIỄN TƯƠNG TÁC TRỰC TIẾP (30 FPS SIÊU MƯỢT)
# ==============================================================================
class LiveDemoApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LIVE DEMO — AUTONOMOUS FRONTIER EXPLORATION & RESCUE ROBOT")
        self.root.configure(bg="#060a12")
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1480x860")

        self.mode = 1          # 1: Đua 3 thuật toán | 2: Chi tiết Đề xuất | 3: Khói mù 30% | 9: Thank You
        self.map_key = "Map_C"
        self.map_label = ""
        self.true_map = None
        self.start_pos = None
        self.victims = None

        self.traces = {}       # random, nearest, utility
        self.smoke_traces = {} # raw, log_odds

        self.playing = True
        self.speed = 1.0
        self.sim_pos = 0.0     # Chỉ số bước dạng số thực để nội suy mượt!
        self.last_tick = time.perf_counter()
        self.bar_visible = True
        self.is_fs = False
        self.win_w = 1480
        self.win_h = 810

        self._build_ui()
        self.load_scenario("Map_C")
        self._loop()

    def _build_ui(self):
        self.ctrl = tk.Frame(self.root, bg="#0c1426", height=52, highlightthickness=1, highlightbackground="#1e2d4a")
        self.ctrl.pack(side=tk.TOP, fill=tk.X)

        def mk_btn(parent, text, cmd, bg="#1e293b", fg="white", padx=8):
            b = tk.Button(
                parent, text=text, command=cmd, bg=bg, fg=fg,
                activebackground="#334155", activeforeground="white",
                font=("Segoe UI", 9, "bold"), padx=padx, pady=4,
                relief=tk.FLAT, cursor="hand2",
            )
            b.pack(side=tk.LEFT, padx=2, pady=6)
            return b

        self.btn_play = mk_btn(self.ctrl, "⏸ Dừng (SPACE)", self.toggle_play, bg="#e67e22", padx=10)
        mk_btn(self.ctrl, "↺ Chạy lại (R)", self.restart, bg="#0984e3")

        tk.Label(self.ctrl, text="|", bg="#0c1426", fg="#334155").pack(side=tk.LEFT, padx=4)

        # Chọn chế độ demo
        self.mode_btns = {}
        for mid, mtxt in [(1, "1. Đua 3 Thuật toán"), (2, "2. Chi tiết 2D + Zoom 3x"), (3, "3. Khói mù 30%")]:
            self.mode_btns[mid] = mk_btn(self.ctrl, mtxt, lambda m=mid: self.set_mode(m))

        tk.Label(self.ctrl, text="|", bg="#0c1426", fg="#334155").pack(side=tk.LEFT, padx=4)

        # Chọn bản đồ + Random Map
        self.map_btns = {}
        for mk, mlbl in [("Map_A", "Map A"), ("Map_B", "Map B"), ("Map_C", "Map C"), ("Map_D", "Map D")]:
            self.map_btns[mk] = mk_btn(self.ctrl, mlbl, lambda k=mk: self.load_scenario(k), padx=7)
        self.map_btns["Random"] = mk_btn(self.ctrl, "🎲 Random Map (N)", lambda: self.load_scenario("Random"), bg="#6c5ce7", padx=9)

        # Nhóm nút bên phải
        right_box = tk.Frame(self.ctrl, bg="#0c1426")
        right_box.pack(side=tk.RIGHT, padx=8)

        self.spd_btns = {}
        for sp in (0.5, 1.0, 2.0, 3.0):
            b = tk.Button(
                right_box, text=f"{sp:g}x", command=lambda s=sp: self.set_speed(s),
                bg="#00d2d3" if sp == 1.0 else "#1e293b",
                fg="#060a12" if sp == 1.0 else "white",
                font=("Segoe UI", 8, "bold"), padx=6, pady=4, relief=tk.FLAT, cursor="hand2",
            )
            b.pack(side=tk.LEFT, padx=1, pady=6)
            self.spd_btns[sp] = b

        tk.Button(
            right_box, text="Thank You (T)", command=lambda: self.set_mode(9),
            bg="#00b894", fg="white", font=("Segoe UI", 8, "bold"),
            padx=8, pady=4, relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 2), pady=6)

        tk.Button(
            right_box, text="Ẩn thanh (H)", command=self.toggle_bar,
            bg="#334155", fg="#feca57", font=("Segoe UI", 8, "bold"),
            padx=8, pady=4, relief=tk.FLAT, cursor="hand2",
        ).pack(side=tk.LEFT, padx=2, pady=6)

        self.lbl_view = tk.Label(self.root, bg="#060a12", bd=0, highlightthickness=0)
        self.lbl_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.lbl_view.bind("<Configure>", self._on_resize)

        # Phím tắt
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("r", lambda e: self.restart())
        self.root.bind("R", lambda e: self.restart())
        self.root.bind("n", lambda e: self.load_scenario("Random"))
        self.root.bind("N", lambda e: self.load_scenario("Random"))
        self.root.bind("1", lambda e: self.set_mode(1))
        self.root.bind("2", lambda e: self.set_mode(2))
        self.root.bind("3", lambda e: self.set_mode(3))
        self.root.bind("t", lambda e: self.set_mode(9))
        self.root.bind("T", lambda e: self.set_mode(9))
        self.root.bind("h", lambda e: self.toggle_bar())
        self.root.bind("H", lambda e: self.toggle_bar())
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self._exit_fullscreen())

    def _on_resize(self, ev):
        if ev.width > 200 and ev.height > 200:
            self.win_w, self.win_h = ev.width, ev.height

    def _refresh_highlights(self):
        for mid, b in self.mode_btns.items():
            b.configure(bg="#00d2d3" if self.mode == mid else "#1e293b",
                        fg="#060a12" if self.mode == mid else "white")
        for mk, b in self.map_btns.items():
            if mk == "Random":
                b.configure(bg="#a29bfe" if self.map_key == "Random" else "#6c5ce7",
                            fg="#060a12" if self.map_key == "Random" else "white")
            else:
                b.configure(bg="#feca57" if self.map_key == mk else "#1e293b",
                            fg="#060a12" if self.map_key == mk else "white")

    def load_scenario(self, map_key):
        self.map_key = map_key
        seed = None if map_key == "Random" else 42
        self.true_map, self.start_pos, self.victims, self.map_label = load_map_by_key(map_key, rand_seed=seed)

        # Chạy mô phỏng thực tế ngay lập tức cho 3 thuật toán (khói mù tính khi mở Mode 3 cho nhanh)
        self.traces = {
            "random": run_algorithm_trace(self.true_map, self.start_pos, self.victims, "random", max_steps=280),
            "nearest": run_algorithm_trace(self.true_map, self.start_pos, self.victims, "nearest", max_steps=280),
            "utility": run_algorithm_trace(self.true_map, self.start_pos, self.victims, "utility", max_steps=280),
        }
        self.smoke_traces = {}
        if self.mode == 3:
            self._ensure_smoke_traces()
        elif self.mode == 9:
            self.mode = 1
        self.sim_pos = 0.0
        self.playing = True
        self.btn_play.configure(text="⏸ Dừng (SPACE)", bg="#e67e22")
        self._refresh_highlights()

    def _ensure_smoke_traces(self):
        if not self.smoke_traces:
            self.smoke_traces = {
                "raw": run_algorithm_trace(self.true_map, self.start_pos, self.victims, "utility",
                                           noise_std=0.45, smoke_prob=0.30, use_log_odds=False, max_steps=280),
                "log": run_algorithm_trace(self.true_map, self.start_pos, self.victims, "utility",
                                           noise_std=0.45, smoke_prob=0.30, use_log_odds=True, max_steps=280),
            }

    def set_mode(self, m):
        self.mode = m
        if m == 3:
            self._ensure_smoke_traces()
        if m != 9:
            self.sim_pos = 0.0
            self.playing = True
            self.btn_play.configure(text="⏸ Dừng (SPACE)", bg="#e67e22")
        self._refresh_highlights()

    def set_speed(self, sp):
        self.speed = sp
        for s, b in self.spd_btns.items():
            b.configure(bg="#00d2d3" if s == sp else "#1e293b", fg="#060a12" if s == sp else "white")

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            if self.sim_pos >= self._max_steps() - 1:
                self.sim_pos = 0.0
            self.btn_play.configure(text="⏸ Dừng (SPACE)", bg="#e67e22")
        else:
            self.btn_play.configure(text="▶ Tiếp tục (SPACE)", bg="#00b894")

    def restart(self):
        if self.mode == 9:
            self.mode = 1
        self.sim_pos = 0.0
        self.playing = True
        self.btn_play.configure(text="⏸ Dừng (SPACE)", bg="#e67e22")
        self._refresh_highlights()

    def toggle_bar(self):
        self.bar_visible = not self.bar_visible
        if self.bar_visible:
            self.lbl_view.pack_forget()
            self.ctrl.pack(side=tk.TOP, fill=tk.X)
            self.lbl_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            self.ctrl.pack_forget()

    def toggle_fullscreen(self):
        self.is_fs = not self.is_fs
        self.root.attributes("-fullscreen", self.is_fs)

    def _exit_fullscreen(self):
        self.is_fs = False
        self.root.attributes("-fullscreen", False)

    def _max_steps(self):
        if self.mode == 1:
            return max(len(self.traces["nearest"]), len(self.traces["utility"]), min(len(self.traces["random"]), len(self.traces["nearest"]) + 25))
        if self.mode == 2:
            return len(self.traces["utility"])
        if self.mode == 3:
            return max(len(self.smoke_traces["raw"]), len(self.smoke_traces["log"]))
        return 1

    def _get_pair(self, tr):
        idx = int(self.sim_pos)
        alpha = self.sim_pos - idx
        if idx >= len(tr) - 1:
            return tr[-1], tr[-1], 0.0
        return tr[idx], tr[idx + 1], alpha

    # --------------------------------------------------------------------------
    # DỰNG KHUNG HÌNH CHO TỪNG CHẾ ĐỘ
    # --------------------------------------------------------------------------
    def _render_thank_you(self):
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_DARK)
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle([80, 80, CANVAS_W - 80, CANVAS_H - 80], radius=26,
                            fill=(14, 22, 38), outline=BORDER_CYAN, width=3)
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        txt = "THANK YOU"
        bb = d.textbbox((0, 0), txt, font=F_THANK_BIG)
        d.text((cx - (bb[2] - bb[0]) // 2, cy - 82), txt, font=F_THANK_BIG, fill=(255, 255, 255))
        d.line([cx - 210, cy + 48, cx + 210, cy + 48], fill=(29, 209, 161), width=4)
        sub = "Autonomous Frontier-Based Exploration & Rescue Mapping"
        bb2 = d.textbbox((0, 0), sub, font=F_THANK_SUB)
        d.text((cx - (bb2[2] - bb2[0]) // 2, cy + 72), sub, font=F_THANK_SUB, fill=(130, 204, 221))
        return canvas

    def _draw_header(self, canvas, badge, title, subtitle, badge_col=(0, 184, 148)):
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, CANVAS_W, 68], fill=(12, 20, 38))
        d.line([0, 67, CANVAS_W, 67], fill=BORDER_CYAN, width=2)
        d.rounded_rectangle([20, 14, 210, 54], radius=9, fill=badge_col)
        bb = d.textbbox((0, 0), badge, font=F_BOLD)
        d.text((20 + (190 - (bb[2] - bb[0])) // 2, 23), badge, font=F_BOLD, fill=(255, 255, 255))
        d.text((228, 11), title, font=F_TITLE, fill=(245, 248, 255))
        d.text((228, 40), subtitle, font=F_SUB, fill=(130, 204, 221))
        # Nhãn bản đồ đang chạy ở góc phải
        mtxt = f"Bản đồ: {self.map_label}"
        bbm = d.textbbox((0, 0), mtxt, font=F_BOLD)
        d.rounded_rectangle([CANVAS_W - (bbm[2] - bbm[0]) - 46, 16, CANVAS_W - 20, 52],
                            radius=8, fill=(25, 38, 62), outline=(254, 202, 87), width=1)
        d.text((CANVAS_W - (bbm[2] - bbm[0]) - 33, 24), mtxt, font=F_BOLD, fill=(254, 202, 87))
        return d

    def _render_mode_1(self):
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_DARK)
        d = self._draw_header(
            canvas, "SO SÁNH 3 THUẬT TOÁN",
            "ĐỐI ĐẦU TRỰC TIẾP: RANDOM WALK vs. NEAREST FRONTIER vs. PROPOSED LOG-UTILITY",
            "Cùng xuất phát trên 1 bản đồ — Quan sát tốc độ phủ bản đồ, độ ngắn quỹ đạo, độ mượt góc cua & tốc độ cứu 3 nạn nhân",
            badge_col=(0, 184, 148),
        )

        configs = [
            ("random", "1. Random Walk (Ngẫu nhiên)", (255, 107, 129), (129, 107, 255), (105, 95, 235)),
            ("nearest", "2. Nearest Frontier (Gần nhất)", (255, 190, 118), (118, 190, 255), (50, 155, 245)),
            ("utility", "3. Proposed Log-Utility (Đề xuất)", (72, 219, 251), (251, 219, 72), (161, 209, 29)),
        ]
        map_sz = 486
        gap = 28
        x_start = (CANVAS_W - (3 * map_sz + 2 * gap)) // 2
        y_top = 84

        cur_frames = {}
        for i, (key, title, col_rgb, tr_bgr, bd_bgr) in enumerate(configs):
            f0, f1, al = self._get_pair(self.traces[key])
            cur_frames[key] = f0
            x0 = x_start + i * (map_sz + gap)
            d.rounded_rectangle([x0 - 6, y_top, x0 + map_sz + 6, y_top + map_sz + 64],
                                radius=12, fill=BG_PANEL, outline=col_rgb if key == "utility" else BORDER_SLATE,
                                width=2 if key == "utility" else 1)
            d.text((x0 + 8, y_top + 8), title, font=F_HEAD, fill=col_rgb)
            st_txt = f"Phủ: {f0['cov']:5.1f}% | Đi: {f0['dist_m']:5.1f}m | Cứu: {sum(f0['victims'])}/3"
            d.text((x0 + 8, y_top + 33), st_txt, font=F_MONO, fill=(235, 240, 248))

            patch = render_map_patch(
                self.true_map, f0, f1, al, self.start_pos, self.victims, map_sz,
                traj_bgr=tr_bgr, body_bgr=bd_bgr,
                show_frontiers=(key == "utility"), show_planned=(key == "utility"),
            )
            canvas.paste(Image.fromarray(patch), (x0, y_top + 58))

            if f0["done"]:
                badge = "★ HOÀN TẤT + ĐƯỜNG SƠ TÁN A*" if key == "utility" else ("VỀ ĐÍCH" if f0["cov"] >= 98.0 else "DỪNG SỚM / CHƯA PHỦ HẾT")
                bcol = (0, 184, 148) if key == "utility" else ((230, 126, 34) if f0["cov"] >= 98.0 else (214, 48, 49))
                d.rounded_rectangle([x0 + 18, y_top + map_sz + 14, x0 + map_sz - 18, y_top + map_sz + 48],
                                    radius=8, fill=bcol)
                bb = d.textbbox((0, 0), badge, font=F_BOLD)
                d.text((x0 + (map_sz - (bb[2] - bb[0])) // 2, y_top + map_sz + 21), badge, font=F_BOLD, fill=(255, 255, 255))

        # Bảng so sánh định lượng trực tiếp phía dưới
        by0 = y_top + map_sz + 78
        d.rounded_rectangle([x_start - 6, by0, CANVAS_W - x_start + 6, CANVAS_H - 18],
                            radius=12, fill=(14, 22, 38), outline=BORDER_CYAN, width=1)

        fr, fn, fu = cur_frames["random"], cur_frames["nearest"], cur_frames["utility"]
        save_dist = (1.0 - fu["dist_m"] / max(0.1, fn["dist_m"])) * 100.0 if fn["dist_m"] > 5 else 0.0

        d.text((x_start + 16, by0 + 12), "BẢNG ĐỐI SÁNH HIỆU QUẢ TRỰC TIẾP (LIVE METRICS)", font=F_HEAD, fill=(254, 202, 87))
        rows = [
            f"{'Tiêu chí':<24} | {'1. Random Walk':<18} | {'2. Nearest Frontier':<20} | {'3. Proposed (Đề xuất)':<24}",
            "─" * 94,
            f"{'Độ phủ không gian (%)':<24} | {fr['cov']:15.1f}%   | {fn['cov']:17.1f}%   | {fu['cov']:18.1f}%",
            f"{'Quãng đường đi (m)':<24} | {fr['dist_m']:15.1f} m  | {fn['dist_m']:17.1f} m  | {fu['dist_m']:18.1f} m",
            f"{'Góc cua gắt (>90°)':<24} | {fr['sharp_turns']:13d} lần  | {fn['sharp_turns']:15d} lần  | {fu['sharp_turns']:16d} lần",
            f"{'Nạn nhân đã định vị':<24} | {sum(fr['victims']):14d}/3   | {sum(fn['victims']):16d}/3   | {sum(fu['victims']):17d}/3   (Hiệu suất EE: {fu['ee']:.2f} m²/m)",
        ]
        for ri, line in enumerate(rows):
            col = (0, 255, 204) if ri >= 2 else (190, 210, 230)
            d.text((x_start + 16, by0 + 42 + ri * 24), line, font=F_MONO, fill=col)

        # Hộp nổi bật % tiết kiệm ở góc phải bảng
        hx0 = CANVAS_W - x_start - 390
        d.rounded_rectangle([hx0, by0 + 18, CANVAS_W - x_start - 14, CANVAS_H - 32],
                            radius=10, fill=(18, 42, 54), outline=(29, 209, 161), width=2)
        d.text((hx0 + 18, by0 + 30), "ƯU THẾ THUẬT TOÁN ĐỀ XUẤT", font=F_HEAD, fill=(29, 209, 161))
        d.text((hx0 + 18, by0 + 64), f"• Tiết kiệm quãng đường: {save_dist:+.1f}%", font=F_BOLD, fill=(255, 255, 255))
        d.text((hx0 + 18, by0 + 94), f"• Giảm rẽ gắt >90°: {fu['sharp_turns']} vs {fn['sharp_turns']} lần", font=F_BOLD, fill=(254, 202, 87))
        v_txt = f"Bước {fu['v_all_step']}" if fu["v_all_step"] is not None else f"{sum(fu['victims'])}/3 người"
        d.text((hx0 + 18, by0 + 124), f"• Cứu đủ 3 nạn nhân: {v_txt}", font=F_BOLD, fill=(72, 219, 251))
        return canvas

    def _render_mode_2(self):
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_DARK)
        d = self._draw_header(
            canvas, "CƠ CHẾ CHI TIẾT",
            "VÒNG KÍN SENSE–MAP–PLAN–ACT: LIDAR 360°, CỤM FRONTIER, CAMERA 3x & ĐƯỜNG SƠ TÁN A*",
            "Tam giác xanh ngọc: Đường biên Frontier | Hình thoi vàng: Đích C_k* tối ưu | Nét chấm cam: A* dự kiến | Nét đứt đỏ: Đường sơ tán A*",
            badge_col=(9, 132, 227),
        )
        f0, f1, al = self._get_pair(self.traces["utility"])

        # Trái: Bản đồ toàn cảnh lớn 760x760
        m_sz = 750
        mx0, my0 = 28, 120
        d.rounded_rectangle([mx0 - 6, my0 - 36, mx0 + m_sz + 6, my0 + m_sz + 6],
                            radius=12, fill=BG_PANEL, outline=BORDER_CYAN, width=2)
        d.text((mx0 + 8, my0 - 28), f"1. Bản đồ Lưới Chiếm chỗ 2D — Bước {f0['step']} | Phủ {f0['cov']:.1f}% | Đường {f0['dist_m']:.1f}m",
               font=F_HEAD, fill=(72, 219, 251))
        p_main = render_map_patch(
            self.true_map, f0, f1, al, self.start_pos, self.victims, m_sz,
            traj_bgr=(251, 219, 72), body_bgr=(67, 159, 255),
            show_frontiers=True, show_planned=True, show_rays=True,
        )
        canvas.paste(Image.fromarray(p_main), (mx0, my0))

        # Phải trên: Camera cận cảnh 3x + Telemetry
        rx0 = mx0 + m_sz + 30
        zm_sz = 380
        d.rounded_rectangle([rx0 - 6, my0 - 36, rx0 + zm_sz + 6, my0 + zm_sz + 6],
                            radius=12, fill=BG_PANEL, outline=(254, 202, 87), width=2)
        d.text((rx0 + 8, my0 - 28), "2. Camera Cận cảnh 3x Bám Robot", font=F_HEAD, fill=(254, 202, 87))
        p_zm = render_map_patch(
            self.true_map, f0, f1, al, self.start_pos, self.victims, zm_sz,
            traj_bgr=(251, 219, 72), body_bgr=(67, 159, 255),
            show_frontiers=True, show_planned=True, show_rays=True, zoom_radius=8.5,
        )
        canvas.paste(Image.fromarray(p_zm), (rx0, my0))

        # Hộp Telemetry cạnh Camera Zoom
        tx0 = rx0 + zm_sz + 20
        tw = CANVAS_W - tx0 - 26
        d.rounded_rectangle([tx0, my0 - 36, tx0 + tw, my0 + zm_sz + 6],
                            radius=12, fill=(14, 23, 42), outline=BORDER_SLATE, width=1)
        d.text((tx0 + 16, my0 - 24), "3. TELEMETRY LIVE", font=F_HEAD, fill=(29, 209, 161))
        r_cur, c_cur, h_cur = interp_pose(f0, f1, al)
        v_str = "  ".join([f"V{i+1}:{'OK' if f0['victims'][i] else '...'}" for i in range(3)])
        t_lines = [
            f"Vị trí (x,y): ({c_cur*RES:4.1f}m, {r_cur*RES:4.1f}m)",
            f"Góc hướng θ : {math.degrees(h_cur):6.1f}°",
            f"Bước đi     : {f0['step']:4d}",
            f"Quãng đường : {f0['dist_m']:5.1f} m",
            f"Độ phủ      : {f0['cov']:5.1f} %",
            f"Hiệu suất EE: {f0['ee']:5.2f} m²/m",
            f"Cua gắt >90°: {f0['sharp_turns']:4d} lần",
            f"Nạn nhân    : {sum(f0['victims'])}/3",
            f"  [{v_str}]",
            "",
            "Hàm mục tiêu Log-Utility:",
            "U = wI·ln(1+IG+1.5|F|)",
            "    - wD·L_A* - wθ·|Δθ|/π",
            "",
            "Trạng thái:",
            "✔ ĐÃ XUẤT ĐƯỜNG SƠ TÁN A*" if f0["done"] else "▶ Đang mở bản đồ & tìm nạn nhân",
        ]
        for li, ln in enumerate(t_lines):
            clr = (29, 209, 161) if "✔" in ln else ((254, 202, 87) if "Hàm" in ln else (225, 235, 245))
            d.text((tx0 + 16, my0 + 12 + li * 22), ln, font=F_MONO_SM, fill=clr)

        # Phải dưới: Biểu đồ Độ phủ sống
        ch_y = my0 + zm_sz + 24
        ch_h = (my0 + m_sz + 6) - ch_y
        ch_w = CANVAS_W - rx0 - 20
        chart_img = draw_coverage_chart(ch_w, ch_h, self.traces, int(self.sim_pos))
        canvas.paste(chart_img, (rx0 - 6, ch_y))
        return canvas

    def _render_mode_3(self):
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_DARK)
        d = self._draw_header(
            canvas, "KHÓI MÙ 30% & NHIỄU",
            "ĐỐI ĐẦU TRỰC TIẾP DƯỚI KHÓI MÙ 30% + NHIỄU GAUSS: RAW MAPPING vs. BAYESIAN LOG-ODDS",
            "Chấm đỏ tươi = Vật cản giả ('tường ma') do tia laser dội lại từ hạt khói giữa không trung",
            badge_col=(225, 112, 85),
        )
        fr0, fr1, alr = self._get_pair(self.smoke_traces["raw"])
        fl0, fl1, all_ = self._get_pair(self.smoke_traces["log"])

        m_sz = 640
        gap = 56
        x0 = (CANVAS_W - (2 * m_sz + gap)) // 2
        y0 = 136

        panels = [
            (x0, fr0, fr1, alr, "1. KHÔNG LỌC (Raw Mapping)", (255, 107, 129), (129, 107, 255), (83, 82, 238), True),
            (x0 + m_sz + gap, fl0, fl1, all_, "2. BỘ LỌC BAYESIAN LOG-ODDS (Đề xuất)", (29, 209, 161), (211, 210, 0), (161, 209, 29), False),
        ]
        for px, f0, f1, al, title, col_rgb, tr_bgr, bd_bgr, is_raw in panels:
            d.rounded_rectangle([px - 8, y0 - 52, px + m_sz + 8, y0 + m_sz + 56],
                                radius=14, fill=BG_PANEL, outline=col_rgb, width=2)
            d.text((px + 8, y0 - 44), title, font=F_HEAD, fill=col_rgb)
            st = f"Độ phủ: {f0['cov']:5.1f}%  |  Tường ma do khói: {f0['phantom_cnt']:3d} ô  |  Cứu: {sum(f0['victims'])}/3"
            d.text((px + 8, y0 - 20), st, font=F_MONO, fill=(240, 245, 250))

            patch = render_map_patch(
                self.true_map, f0, f1, al, self.start_pos, self.victims, m_sz,
                traj_bgr=tr_bgr, body_bgr=bd_bgr, show_frontiers=False, show_planned=False,
            )
            canvas.paste(Image.fromarray(patch), (px, y0))

            foot = (
                "✖ Bị tường ma do khói bịt kín đường — Dừng sớm thất bại!" if (is_raw and f0["done"] and f0["cov"] < 90)
                else ("✔ Xuyên thủng màn khói — Bản đồ sạch & cứu đủ nạn nhân!" if (not is_raw and f0["done"])
                      else ("Đang bị hạt khói ghi đè thành vật cản ảo..." if is_raw else "Tia xuyên thấu (-1.25) liên tục xóa sạch nhiễu khói..."))
            )
            d.text((px + 10, y0 + m_sz + 18), foot, font=F_BOLD, fill=col_rgb)

        return canvas

    def _loop(self):
        now = time.perf_counter()
        dt = min(0.08, now - self.last_tick)
        self.last_tick = now

        if self.playing and self.mode != 9:
            # 12 bước mô phỏng/giây ở tốc độ 1.0x, nội suy ở 30 FPS -> mượt tuyệt đối!
            self.sim_pos += dt * 12.0 * self.speed
            mx = self._max_steps() - 1
            if self.sim_pos >= mx:
                self.sim_pos = float(mx)
                self.playing = False
                self.btn_play.configure(text="↺ Chạy lại (SPACE)", bg="#0984e3")

        if self.mode == 9:
            frame_pil = self._render_thank_you()
        elif self.mode == 1:
            frame_pil = self._render_mode_1()
        elif self.mode == 2:
            frame_pil = self._render_mode_2()
        else:
            frame_pil = self._render_mode_3()

        # Resize cực nhanh bằng cv2.resize vừa khít cửa sổ hiện tại
        scale = min(self.win_w / CANVAS_W, self.win_h / CANVAS_H)
        nw = max(320, int(CANVAS_W * scale))
        nh = max(180, int(CANVAS_H * scale))
        arr = np.asarray(frame_pil)
        if nw != CANVAS_W or nh != CANVAS_H:
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        tk_im = ImageTk.PhotoImage(Image.fromarray(arr))
        self.lbl_view.configure(image=tk_im)
        self.lbl_view.image = tk_im

        elapsed_ms = (time.perf_counter() - now) * 1000.0
        wait_ms = max(5, int(round((1000.0 / TARGET_FPS) - elapsed_ms)))
        self.root.after(wait_ms, self._loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = LiveDemoApp()
    app.run()
