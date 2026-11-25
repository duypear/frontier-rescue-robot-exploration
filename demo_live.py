"""
================================================================================
CHƯƠNG TRÌNH DEMO MÔ PHỎNG ROBOT CỨU HỘ TRỰC QUAN (RVIZ-STYLE 2D SIMULATOR)
ĐỀ TÀI: Ứng dụng thuật toán thám hiểm đường biên để định vị nạn nhân và
        lập bản đồ cho robot cứu hộ trong tòa nhà sập đổ

ĐẶC ĐIỂM HIỂN THỊ ROBOT TRONG DEMO:
  - Hiển thị đầy đủ hình dáng Robot cứu hộ bánh xích (Thân xe bọc thép, 2 dải
    bánh xích trái/phải, tháp cảm biến 2D LiDAR trên lưng và mũi tên hướng đầu xe)
    xoay chính xác theo góc hướng động học θ(t).
  - Hiển thị chùm tia quét Laser 2D thực tế bắn ra 360° chạm vào vách tường/xà bần.
  - Nội suy chuyển động mượt mà (Smooth Pose Interpolation), bảng thông số động học
    trực tiếp (v, ω, θ, Độ phủ %, Số nạn nhân) và đường thoát hiểm khẩn cấp A*.
================================================================================
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter

import rescue_robot_exploration as rre

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. HÀM VẼ MÔ HÌNH CHI TIẾT ROBOT CỨU HỘ BÁNH XÍCH & CHÙM TIA LASER 2D
# ==============================================================================
def draw_rescue_robot(ax, rx, ry, theta, true_map, lidar_range_cells=10,
                      body_color='#ff9f43', draw_laser_rays=True, scale=1.35):
    """
    Vẽ mô hình hình học chi tiết của Robot cứu hộ dẫn động vi sai / bánh xích
    tại tọa độ lưới (rx: hàng/trục dọc, ry: cột/trục ngang) với góc hướng theta.
    Lưu ý trên imshow: trục hoành X của đồ thị là cột (ry), trục tung Y là hàng (rx).
    """
    px = float(ry)
    py = float(rx)
    # Góc xoay trên hệ trục đồ thị (X = cột = dy, Y = hàng = dx)
    plot_angle_rad = math.atan2(math.cos(theta), math.sin(theta))
    plot_angle_deg = math.degrees(plot_angle_rad)

    # 1. Vẽ chùm tia quét Laser 2D thực tế bắn từ tháp LiDAR của Robot
    if draw_laser_rays and true_map is not None:
        ray_angles = np.linspace(0, 2 * np.pi, 36, endpoint=False) + theta
        steps = np.linspace(0.4, lidar_range_cells, int(lidar_range_cells * 2.5))
        for ang in ray_angles:
            ca = math.cos(ang)
            sa = math.sin(ang)
            end_r, end_c = rx + lidar_range_cells * ca, ry + lidar_range_cells * sa
            hit = False
            for d in steps:
                cr = int(round(rx + d * ca))
                cc = int(round(ry + d * sa))
                if not (0 <= cr < rre.GRID_SIZE and 0 <= cc < rre.GRID_SIZE):
                    end_r, end_c = rx + d * ca, ry + d * sa
                    break
                if true_map[cr, cc] == rre.CELL_OBSTACLE:
                    end_r, end_c = cr, cc
                    hit = True
                    break
            ax.plot([px, end_c], [py, end_r], color='#00d2d3', linewidth=0.65, alpha=0.42, zorder=4)
            if hit:
                ax.scatter(end_c, end_r, c='#ff4757', s=8, marker='o', alpha=0.75, zorder=5)

        # Vòng tròn giới hạn tầm quét 2D LiDAR
        fov_circle = plt.Circle((px, py), lidar_range_cells, color='#0abde3',
                                fill=False, linestyle='--', linewidth=1.1, alpha=0.65, zorder=5)
        ax.add_patch(fov_circle)

    # Phép biến đổi xoay và tịnh tiến thân Robot theo tư thế (px, py, plot_angle_deg)
    trans = Affine2D().rotate_deg(plot_angle_deg).translate(px, py) + ax.transData

    L = 1.30 * scale   # Chiều dài thân robot (ô lưới)
    W = 0.95 * scale   # Chiều rộng thân robot (ô lưới)
    t_w = 0.28 * scale # Bề rộng mỗi dải bánh xích

    # 2. Hai dải bánh xích đen (Left & Right Tracks)
    left_track = mpatches.FancyBboxPatch(
        (-L * 0.52, W * 0.42), L * 1.04, t_w,
        boxstyle="round,pad=0.04", facecolor="#1e272e", edgecolor="black", linewidth=1.0,
        transform=trans, zorder=8
    )
    right_track = mpatches.FancyBboxPatch(
        (-L * 0.52, -W * 0.42 - t_w), L * 1.04, t_w,
        boxstyle="round,pad=0.04", facecolor="#1e272e", edgecolor="black", linewidth=1.0,
        transform=trans, zorder=8
    )
    ax.add_patch(left_track)
    ax.add_patch(right_track)

    # 3. Thân chính bọc thép của Robot cứu hộ (Main Armored Chassis)
    chassis = mpatches.FancyBboxPatch(
        (-L * 0.45, -W * 0.45), L * 0.90, W * 0.90,
        boxstyle="round,pad=0.08", facecolor=body_color, edgecolor="#1e272e", linewidth=1.4,
        transform=trans, zorder=9
    )
    ax.add_patch(chassis)

    # Cản trước nhận diện đầu xe (Front Bumper)
    bumper = mpatches.Polygon(
        [[L * 0.35, -W * 0.38], [L * 0.56, -W * 0.22], [L * 0.56, W * 0.22], [L * 0.35, W * 0.38]],
        closed=True, facecolor="#ee5253", edgecolor="black", linewidth=0.9,
        transform=trans, zorder=10
    )
    ax.add_patch(bumper)

    # 4. Tháp quét 2D LiDAR xoay trên lưng Robot (LiDAR Turret)
    turret = mpatches.Circle(
        (0.0, 0.0), radius=0.32 * scale,
        facecolor="#0984e3", edgecolor="white", linewidth=1.1,
        transform=trans, zorder=11
    )
    ax.add_patch(turret)

    # 5. Mũi tên chỉ hướng di chuyển θ(t) (Heading Vector Arrow)
    hx = px + (L * 0.95) * math.cos(plot_angle_rad)
    hy = py + (L * 0.95) * math.sin(plot_angle_rad)
    ax.annotate("", xy=(hx, hy), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#d63031", mutation_scale=12),
                zorder=12)


# ==============================================================================
# 2. TẠO DỮ LIỆU QUỸ ĐẠO CHI TIẾT TỪNG BƯỚC ĐỂ PHÁT HOẠT ẢNH MƯỢT MÀ
# ==============================================================================
def record_detailed_mission_frames(map_type="Map_C", strategy="utility",
                                   lidar_range_m=5.0, noise_std=0.0, smoke_prob=0.0):
    """Ghi lại trạng thái bản đồ, tư thế (x, y, θ), cụm Frontier và mục tiêu ở từng bước"""
    lidar_cells = int(round(lidar_range_m / rre.RESOLUTION))
    rng = np.random.RandomState(42)
    true_map, start_pos, victims = rre.create_environment_map(map_type, seed=42)
    explored_map = np.full((rre.GRID_SIZE, rre.GRID_SIZE), rre.CELL_UNKNOWN, dtype=int)
    log_odds_map = np.zeros((rre.GRID_SIZE, rre.GRID_SIZE), dtype=float)

    _, geom_true, _ = rre.compute_shortest_path_tree(true_map, start_pos, safety_cost_weight=0.0)
    reachable_free_cells = int(np.sum(geom_true < float('inf')))

    robot_pos = start_pos
    trajectory = [robot_pos]
    current_heading = 0.0
    victims_found = [False] * len(victims)
    frames = []

    rre.simulate_lidar_scan(true_map, robot_pos, explored_map, log_odds_map,
                            lidar_range=lidar_cells, noise_std=noise_std,
                            dropout_prob=smoke_prob, use_log_odds_filter=True, rng=rng)

    step = 0
    max_steps = 320
    min_c_size = 1 if strategy in ["nearest", "euclidean"] else 2
    horizon = 2 if strategy in ["nearest", "euclidean"] else 5
    cov_history = []
    dist_m = 0.0

    while step < max_steps:
        known_free = int(np.sum((explored_map == rre.CELL_FREE) & (geom_true < float('inf'))))
        cov_pct = min(100.0, (known_free / reachable_free_cells) * 100.0)
        cov_history.append(cov_pct)

        for v_idx, vp in enumerate(victims):
            if not victims_found[v_idx]:
                if rre.check_victim_line_of_sight(true_map, robot_pos, vp, lidar_cells):
                    victims_found[v_idx] = True

        clusters = rre.detect_and_cluster_frontiers(explored_map, min_cluster_size=min_c_size)
        if not clusters and min_c_size > 1:
            clusters = rre.detect_and_cluster_frontiers(explored_map, min_cluster_size=1)

        chosen_cl, path = rre.select_next_frontier(
            clusters, robot_pos, explored_map,
            strategy=strategy, current_heading=current_heading, rng=rng
        )

        frames.append({
            'step': step,
            'robot_pos': robot_pos,
            'heading': current_heading,
            'explored_map': explored_map.copy(),
            'trajectory': list(trajectory),
            'frontiers': [c['centroid'] for c in clusters],
            'target_frontier': chosen_cl['centroid'] if chosen_cl else None,
            'planned_path': list(path) if path else None,
            'coverage': cov_pct,
            'cov_history': list(cov_history),
            'dist_m': dist_m,
            'victims_found': list(victims_found),
            'evac_path': None,
            'done': False
        })

        if cov_pct >= 98.5 or path is None or len(path) <= 1:
            break

        steps_to_move = min(horizon, len(path) - 1)
        for i in range(1, steps_to_move + 1):
            next_pos = path[i]
            dx = next_pos[0] - robot_pos[0]
            dy = next_pos[1] - robot_pos[1]
            current_heading = math.atan2(dy, dx)
            dist_m += math.hypot(dx, dy) * rre.RESOLUTION
            robot_pos = next_pos
            trajectory.append(robot_pos)
            step += 1

            rre.simulate_lidar_scan(true_map, robot_pos, explored_map, log_odds_map,
                                    lidar_range=lidar_cells, noise_std=noise_std,
                                    dropout_prob=smoke_prob, use_log_odds_filter=True, rng=rng)

            known_free = int(np.sum((explored_map == rre.CELL_FREE) & (geom_true < float('inf'))))
            cov_pct = min(100.0, (known_free / reachable_free_cells) * 100.0)
            cov_history.append(cov_pct)

            for v_idx, vp in enumerate(victims):
                if not victims_found[v_idx]:
                    if rre.check_victim_line_of_sight(true_map, robot_pos, vp, lidar_cells):
                        victims_found[v_idx] = True

            frames.append({
                'step': step,
                'robot_pos': robot_pos,
                'heading': current_heading,
                'explored_map': explored_map.copy(),
                'trajectory': list(trajectory),
                'frontiers': [c['centroid'] for c in clusters],
                'target_frontier': chosen_cl['centroid'] if chosen_cl else None,
                'planned_path': list(path[i:]) if path else None,
                'coverage': cov_pct,
                'cov_history': list(cov_history),
                'dist_m': dist_m,
                'victims_found': list(victims_found),
                'evac_path': None,
                'done': False
            })

            if horizon <= 5 and chosen_cl is not None:
                cr, cc = chosen_cl['centroid']
                if np.sum(explored_map[max(1, cr-1):min(rre.GRID_SIZE-1, cr+2),
                                       max(1, cc-1):min(rre.GRID_SIZE-1, cc+2)] == rre.CELL_UNKNOWN) == 0:
                    break
            if cov_pct >= 98.5 or step >= max_steps:
                break

    evac_path, evac_len = rre.astar_plan(explored_map, start_pos, victims[-1], safety_cost_weight=2.2)
    if frames:
        final_f = frames[-1].copy()
        final_f['evac_path'] = evac_path
        final_f['evac_len_m'] = evac_len * rre.RESOLUTION if evac_path else 0.0
        final_f['done'] = True
        # Giữ khung hình cuối trong vài nhịp để người xem quan sát rõ đường thoát hiểm
        for _ in range(8):
            frames.append(final_f)

    return frames, true_map, start_pos, victims, lidar_cells


# ==============================================================================
# 3. CHẾ ĐỘ 1: CHẠY DEMO TRỰC TIẾP 1 ROBOT CỨU HỘ (CÓ THỂ LƯU GIF)
# ==============================================================================
def run_single_robot_demo(map_type="Map_C", strategy="utility", lidar_range_m=5.0,
                          noise_std=0.0, smoke_prob=0.0, delay=0.04, save_gif_path=None):
    print(f"[*] Đang khởi tạo mô phỏng Robot cứu hộ trên {map_type} (Thuật toán: {strategy.upper()})...")
    frames, true_map, start_pos, victims, lidar_cells = record_detailed_mission_frames(
        map_type=map_type, strategy=strategy, lidar_range_m=lidar_range_m,
        noise_std=noise_std, smoke_prob=smoke_prob
    )

    cmap_grid = plt.cm.colors.ListedColormap(['#95a5a6', '#ffffff', '#1e272e'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_grid = plt.cm.colors.BoundaryNorm(bounds, cmap_grid.N)

    # Lấy tập con khung hình (mỗi 2 bước) để hoạt ảnh mượt và nhanh
    sampled_frames = frames[::2] + [frames[-1]]

    fig = plt.figure(figsize=(15.5, 6.8), dpi=115)
    try:
        fig.canvas.manager.set_window_title(f"MÔ PHỎNG ROBOT CỨU HỘ BÁNH XÍCH - {map_type} ({strategy.upper()})")
    except Exception:
        pass

    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.05, 1.0], height_ratios=[1.05, 0.95])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[:, 1])
    ax_curve = fig.add_subplot(gs[0, 2])
    ax_hud = fig.add_subplot(gs[1, 2])

    def render_frame(f):
        ax_map.clear()
        ax_zoom.clear()
        ax_curve.clear()
        ax_hud.clear()

        for ax_curr, is_zoom in [(ax_map, False), (ax_zoom, True)]:
            ax_curr.imshow(f['explored_map'], cmap=cmap_grid, norm=norm_grid, origin='upper')
            traj = np.array(f['trajectory'])
            ax_curr.plot(traj[:, 1], traj[:, 0], color='#2980b9', linewidth=2.2 if is_zoom else 1.8, alpha=0.88, label='Quỹ đạo đã đi')

            if f['frontiers'] and not f['done']:
                f_pts = np.array(f['frontiers'])
                ax_curr.scatter(f_pts[:, 1], f_pts[:, 0], c='#00cec9', s=70 if is_zoom else 45, marker='^', edgecolors='black', zorder=5, label='Cụm biên (Frontiers)')

            if f['planned_path'] and not f['done']:
                p_arr = np.array(f['planned_path'])
                ax_curr.plot(p_arr[:, 1], p_arr[:, 0], color='#f39c12', linestyle=':', linewidth=2.6 if is_zoom else 2.0, zorder=6, label='Đường dự kiến A*')

            if f['target_frontier'] and not f['done']:
                tf = f['target_frontier']
                ax_curr.scatter(tf[1], tf[0], c='#f1c40f', s=120 if is_zoom else 95, marker='D', edgecolors='black', zorder=7, label='Đích Frontier đang chọn')

            if f['evac_path']:
                ev = np.array(f['evac_path'])
                ax_curr.plot(ev[:, 1], ev[:, 0], color='#e74c3c', linewidth=3.2, linestyle='--', zorder=12,
                             label=f"Lộ trình thoát hiểm A* ({f.get('evac_len_m', 0):.1f}m)")

            ax_curr.scatter(start_pos[1], start_pos[0], c='#2ecc71', s=115, marker='o', edgecolors='black', zorder=6, label='Cửa vào (Start)')
            for v_idx, vp in enumerate(victims):
                found = f['victims_found'][v_idx]
                col_v = '#e74c3c' if found else '#dfe6e9'
                ax_curr.scatter(vp[1], vp[0], c=col_v, s=160, marker='*', edgecolors='black', zorder=7,
                                label=f"Nạn nhân V{v_idx+1} ({'Đã cứu!' if found else 'Chưa biết'})")
                if found:
                    ring = plt.Circle((vp[1], vp[0]), 1.8, color='#e74c3c', fill=False, linewidth=1.6, linestyle='-')
                    ax_curr.add_patch(ring)

            rx, ry = f['robot_pos']
            draw_rescue_robot(ax_curr, rx, ry, f['heading'], true_map,
                              lidar_range_cells=lidar_cells, body_color='#ff9f43',
                              draw_laser_rays=(not f['done']), scale=1.45 if is_zoom else 1.25)

        rx, ry = f['robot_pos']
        status_txt = "HOÀN TẤT 100% NHIỆM VỤ!" if f['done'] else "ĐANG DÒ ĐƯỜNG THÁM HIỂM..."
        ax_map.set_title(f"1. BẢN ĐỒ TOÀN CẢNH ({map_type})\n[{status_txt}]", fontsize=10.5, fontweight='bold')
        ax_map.set_xlim(-0.5, rre.GRID_SIZE - 0.5)
        ax_map.set_ylim(rre.GRID_SIZE - 0.5, -0.5)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.0)

        # Cửa sổ Zoom cận cảnh bám theo Robot (Follow-Cam bán kính 9 ô quanh Robot)
        zoom_r = 9.5
        ax_zoom.set_xlim(ry - zoom_r, ry + zoom_r)
        ax_zoom.set_ylim(rx + zoom_r, rx - zoom_r)
        ax_zoom.set_title(f"2. CAMERA CẬN CẢNH ROBOT & TIA LASER 2D\n(Góc hướng θ = {math.degrees(f['heading']):.0f}° | Phóng đại 3x)",
                          fontsize=10.5, fontweight='bold', color='#d35400')
        ax_zoom.grid(True, color='#dfe6e9', linestyle=':', linewidth=0.6)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])

        # Đồ thị tốc độ mở bản đồ
        ax_curve.plot(f['cov_history'], color='#27ae60', linewidth=2.5, label='Độ phủ bản đồ (%)')
        ax_curve.axhline(98.5, color='#c0392b', linestyle='--', linewidth=1.2, label='Ngưỡng hoàn thành (98.5%)')
        ax_curve.set_xlim(0, max(160, len(frames)))
        ax_curve.set_ylim(0, 105)
        ax_curve.set_title("ĐỒ THỊ TỐC ĐỘ MỞ BẢN ĐỒ THEO THỜI GIAN THỰC", fontsize=10.5, fontweight='bold')
        ax_curve.set_xlabel("Số bước thời gian (Steps)", fontsize=9)
        ax_curve.set_ylabel("Độ phủ (%)", fontsize=9)
        ax_curve.grid(True, linestyle=':', alpha=0.7)
        ax_curve.legend(loc='lower right', fontsize=8.5)

        # Bảng đồng hồ Telemetry của Robot (HUD Panel)
        ax_hud.set_xlim(0, 1)
        ax_hud.set_ylim(0, 1)
        ax_hud.axis('off')
        hud_box = mpatches.FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.04",
                                          facecolor="#f8f9fa", edgecolor="#2c3e50", linewidth=1.5)
        ax_hud.add_patch(hud_box)
        deg_heading = math.degrees(f['heading'])
        v_found_cnt = sum(f['victims_found'])
        hud_text = (
            f"BẢNG THÔNG SỐ ĐỘNG HỌC & TRẠNG THÁI ROBOT (LIVE TELEMETRY)\n"
            f"──────────────────────────────────────────────────────────\n"
            f"• Mô hình Robot   : Xe cứu hộ bánh xích vi sai (2D LiDAR 360°)\n"
            f"• Tọa độ hiện tại : x = {ry*0.5:5.1f} m ,  y = {rx*0.5:5.1f} m\n"
            f"• Góc hướng (θ)   : {deg_heading:6.1f}°   |   Bán kính LiDAR : {lidar_range_m:.1f} m\n"
            f"• Bước thời gian  : {f['step']:3d} bước  |   Quãng đường đi : {f['dist_m']:5.1f} m\n"
            f"• Độ phủ bản đồ   : {f['coverage']:5.1f} %   |   Nạn nhân cứu   : {v_found_cnt} / 3 người\n"
            f"• Trạng thái      : {'Đã vạch xong đường thoát hiểm A*!' if f['done'] else 'Đang quét tia Laser & bám đường A*'}"
        )
        ax_hud.text(0.06, 0.50, hud_text, va='center', ha='left', fontsize=9.5,
                    fontweight='bold', color='#1e272e')
        plt.tight_layout()

    if save_gif_path:
        print(f"[*] Đang xuất ảnh động GIF chất lượng cao tại: {save_gif_path}...")
        plt.switch_backend('Agg')
        anim = FuncAnimation(fig, lambda idx: render_frame(sampled_frames[idx]),
                             frames=len(sampled_frames), interval=120)
        anim.save(save_gif_path, writer=PillowWriter(fps=8))
        plt.close(fig)
        print(f"[+] Đã lưu thành công file GIF: {os.path.abspath(save_gif_path)}")
    else:
        plt.ion()
        for f in sampled_frames:
            render_frame(f)
            plt.tight_layout()
            plt.pause(delay)
        plt.ioff()
        plt.show()


# ==============================================================================
# 4. CHẾ ĐỘ 2: DEMO ĐỐI ĐẦU 2 MÀN HÌNH (NEAREST FRONTIER vs ĐỀ XUẤT UTILITY)
# ==============================================================================
def run_side_by_side_demo(map_type="Map_C", delay=0.04, save_gif_path=None):
    print(f"[*] Đang chuẩn bị dữ liệu đối đầu 2 thuật toán trên {map_type}...")
    frames_near, true_map, start_pos, victims, lidar_cells = record_detailed_mission_frames(
        map_type=map_type, strategy="nearest"
    )
    frames_util, _, _, _, _ = record_detailed_mission_frames(
        map_type=map_type, strategy="utility"
    )

    cmap_grid = plt.cm.colors.ListedColormap(['#95a5a6', '#ffffff', '#1e272e'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_grid = plt.cm.colors.BoundaryNorm(bounds, cmap_grid.N)

    max_len = max(len(frames_near), len(frames_util))
    indices = list(range(0, max_len, 3)) + [max_len - 1]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=115)
    try:
        fig.canvas.manager.set_window_title(f"ĐỐI ĐẦU TRỰC TIẾP: NEAREST FRONTIER vs ĐỀ XUẤT LOG-UTILITY ({map_type})")
    except Exception:
        pass

    def render_pair(step_idx):
        ax_l.clear()
        ax_r.clear()
        for ax, fr_list, title_name, body_col, line_col in [
            (ax_l, frames_near, "1. Nearest Frontier (Yamauchi - Cơ sở)", "#e67e22", "#d35400"),
            (ax_r, frames_util, "2. Log-Utility Frontier (Đề xuất Tối ưu)", "#2ecc71", "#27ae60")
        ]:
            f = fr_list[min(step_idx, len(fr_list) - 1)]
            ax.imshow(f['explored_map'], cmap=cmap_grid, norm=norm_grid, origin='upper')
            traj = np.array(f['trajectory'])
            ax.plot(traj[:, 1], traj[:, 0], color=line_col, linewidth=2.1, alpha=0.88, label='Quỹ đạo Robot')

            if f['frontiers'] and not f['done']:
                f_pts = np.array(f['frontiers'])
                ax.scatter(f_pts[:, 1], f_pts[:, 0], c='#00cec9', s=45, marker='^', edgecolors='black', zorder=5)

            if f['evac_path']:
                ev = np.array(f['evac_path'])
                ax.plot(ev[:, 1], ev[:, 0], color='#e74c3c', linewidth=3.0, linestyle='--', zorder=12, label='Đường thoát hiểm A*')

            ax.scatter(start_pos[1], start_pos[0], c='#2ecc71', s=100, marker='o', edgecolors='black', zorder=6)
            for v_idx, vp in enumerate(victims):
                col_v = '#e74c3c' if f['victims_found'][v_idx] else '#dfe6e9'
                ax.scatter(vp[1], vp[0], c=col_v, s=145, marker='*', edgecolors='black', zorder=7)

            rx, ry = f['robot_pos']
            draw_rescue_robot(ax, rx, ry, f['heading'], true_map,
                              lidar_range_cells=lidar_cells, body_color=body_col,
                              draw_laser_rays=(not f['done']), scale=1.3)

            st_str = "ĐÃ VỀ ĐÍCH!" if f['done'] else "Đang dò đường..."
            ax.set_title(
                f"{title_name}\nBước: {f['step']} | Quãng đường: {f['dist_m']:.1f}m | Độ phủ: {f['coverage']:.1f}%\n"
                f"Nạn nhân: {sum(f['victims_found'])}/3 ({st_str})",
                fontsize=10.2, fontweight='bold'
            )
            ax.set_xlim(-0.5, rre.GRID_SIZE - 0.5)
            ax.set_ylim(rre.GRID_SIZE - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])

    if save_gif_path:
        print(f"[*] Đang xuất GIF đối đầu 2 màn hình tại: {save_gif_path}...")
        plt.switch_backend('Agg')
        anim = FuncAnimation(fig, lambda i: render_pair(indices[i]),
                             frames=len(indices), interval=120)
        anim.save(save_gif_path, writer=PillowWriter(fps=8))
        plt.close(fig)
        print(f"[+] Đã lưu thành công file GIF đối đầu: {os.path.abspath(save_gif_path)}")
    else:
        plt.ion()
        for idx in indices:
            render_pair(idx)
            plt.tight_layout()
            plt.pause(delay)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Trực quan Robot Cứu hộ Bánh xích & Quét Laser 2D")
    parser.add_argument("--map", type=str, default="Map_C", choices=["Map_A", "Map_B", "Map_C", "Map_D"])
    parser.add_argument("--strategy", type=str, default="utility", choices=["random", "euclidean", "nearest", "utility"])
    parser.add_argument("--range", type=float, default=5.0, help="Bán kính tầm quét LiDAR (m)")
    parser.add_argument("--noise", type=float, default=0.0, help="Nhiễu đo khoảng cách Gauss (m)")
    parser.add_argument("--smoke", type=float, default=0.0, help="Tỷ lệ khói bụi (0.0 -> 0.4)")
    parser.add_argument("--compare", action="store_true", help="Chạy đối đầu 2 màn hình Nearest vs Utility")
    parser.add_argument("--delay", type=float, default=0.04, help="Tốc độ khung hình (giây)")
    parser.add_argument("--save-gif", type=str, default=None, help="Đường dẫn lưu file ảnh động .gif")
    parser.add_argument("--snapshot", type=str, default=None, help="Lưu 1 khung hình giữa chừng ra ảnh PNG để kiểm tra")
    args = parser.parse_args()

    if args.snapshot:
        plt.switch_backend('Agg')
        frames, true_map, start_pos, victims, lidar_cells = record_detailed_mission_frames(
            map_type=args.map, strategy=args.strategy, lidar_range_m=args.range,
            noise_std=args.noise, smoke_prob=args.smoke
        )
        mid_idx = min(len(frames) - 1, 55)
        f = frames[mid_idx]
        cmap_grid = plt.cm.colors.ListedColormap(['#95a5a6', '#ffffff', '#1e272e'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm_grid = plt.cm.colors.BoundaryNorm(bounds, cmap_grid.N)
        fig = plt.figure(figsize=(15.5, 6.8), dpi=180)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.05, 1.0], height_ratios=[1.05, 0.95])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_zoom = fig.add_subplot(gs[:, 1])
        ax_curve = fig.add_subplot(gs[0, 2])
        ax_hud = fig.add_subplot(gs[1, 2])
        for ax_curr, is_zoom in [(ax_map, False), (ax_zoom, True)]:
            ax_curr.imshow(f['explored_map'], cmap=cmap_grid, norm=norm_grid, origin='upper')
            traj = np.array(f['trajectory'])
            ax_curr.plot(traj[:, 1], traj[:, 0], color='#2980b9', linewidth=2.2 if is_zoom else 1.8, alpha=0.88, label='Quỹ đạo đã đi')
            if f['frontiers']:
                f_pts = np.array(f['frontiers'])
                ax_curr.scatter(f_pts[:, 1], f_pts[:, 0], c='#00cec9', s=70 if is_zoom else 45, marker='^', edgecolors='black', zorder=5, label='Cụm biên (Frontiers)')
            if f['planned_path']:
                p_arr = np.array(f['planned_path'])
                ax_curr.plot(p_arr[:, 1], p_arr[:, 0], color='#f39c12', linestyle=':', linewidth=2.6 if is_zoom else 2.0, zorder=6, label='Đường dự kiến A*')
            if f['target_frontier']:
                tf = f['target_frontier']
                ax_curr.scatter(tf[1], tf[0], c='#f1c40f', s=120 if is_zoom else 95, marker='D', edgecolors='black', zorder=7, label='Đích Frontier đang chọn')
            ax_curr.scatter(start_pos[1], start_pos[0], c='#2ecc71', s=115, marker='o', edgecolors='black', zorder=6, label='Cửa vào (Start)')
            for v_idx, vp in enumerate(victims):
                found = f['victims_found'][v_idx]
                col_v = '#e74c3c' if found else '#dfe6e9'
                ax_curr.scatter(vp[1], vp[0], c=col_v, s=160, marker='*', edgecolors='black', zorder=7, label=f"Nạn nhân V{v_idx+1}")
            rx, ry = f['robot_pos']
            draw_rescue_robot(ax_curr, rx, ry, f['heading'], true_map, lidar_range_cells=lidar_cells, scale=1.45 if is_zoom else 1.25)
        rx, ry = f['robot_pos']
        ax_map.set_title(f"1. BẢN ĐỒ TOÀN CẢNH ({args.map})\n[ĐANG DÒ ĐƯỜNG THÁM HIỂM...]", fontsize=10.5, fontweight='bold')
        ax_map.set_xlim(-0.5, rre.GRID_SIZE - 0.5)
        ax_map.set_ylim(rre.GRID_SIZE - 0.5, -0.5)
        ax_map.set_xticks([]); ax_map.set_yticks([])
        ax_map.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.0)
        zoom_r = 9.5
        ax_zoom.set_xlim(ry - zoom_r, ry + zoom_r)
        ax_zoom.set_ylim(rx + zoom_r, rx - zoom_r)
        ax_zoom.set_title(f"2. CAMERA CẬN CẢNH ROBOT & TIA LASER 2D\n(Góc hướng θ = {math.degrees(f['heading']):.0f}° | Phóng đại 3x)", fontsize=10.5, fontweight='bold', color='#d35400')
        ax_zoom.grid(True, color='#dfe6e9', linestyle=':', linewidth=0.6)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        ax_curve.plot(f['cov_history'], color='#27ae60', linewidth=2.5, label='Độ phủ bản đồ (%)')
        ax_curve.axhline(98.5, color='#c0392b', linestyle='--', linewidth=1.2, label='Ngưỡng hoàn thành (98.5%)')
        ax_curve.set_xlim(0, 150); ax_curve.set_ylim(0, 105)
        ax_curve.set_title("3. TỐC ĐỘ MỞ BẢN ĐỒ THEO THỜI GIAN", fontsize=10.5, fontweight='bold')
        ax_curve.grid(True, linestyle=':', alpha=0.7); ax_curve.legend(loc='lower right', fontsize=8.5)
        ax_hud.set_xlim(0, 1); ax_hud.set_ylim(0, 1)
        ax_hud.axis('off')
        hud_box = mpatches.FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.04", facecolor="#f8f9fa", edgecolor="#2c3e50", linewidth=1.5)
        ax_hud.add_patch(hud_box)
        hud_text = (
            f"BẢNG THÔNG SỐ ĐỘNG HỌC & TRẠNG THÁI ROBOT\n"
            f"──────────────────────────────────────────\n"
            f"• Mô hình Robot : Xe cứu hộ bánh xích 2D LiDAR\n"
            f"• Tọa độ (x, y) : x = {ry*0.5:4.1f}m , y = {rx*0.5:4.1f}m\n"
            f"• Góc hướng (θ) : {math.degrees(f['heading']):5.1f}° | Tầm LiDAR: {args.range:.1f}m\n"
            f"• Bước / Quãng  : {f['step']:3d} bước | {f['dist_m']:5.1f} m\n"
            f"• Độ phủ / Cứu  : {f['coverage']:5.1f}%  | {sum(f['victims_found'])}/3 nạn nhân"
        )
        ax_hud.text(0.06, 0.50, hud_text, va='center', ha='left', fontsize=9.2, fontweight='bold', color='#1e272e')
        plt.tight_layout()
        plt.savefig(args.snapshot, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] Đã xuất ảnh preview giao diện Demo: {args.snapshot}")
    elif args.compare:
        run_side_by_side_demo(map_type=args.map, delay=args.delay, save_gif_path=args.save_gif)
    else:
        run_single_robot_demo(map_type=args.map, strategy=args.strategy, lidar_range_m=args.range,
                              noise_std=args.noise, smoke_prob=args.smoke, delay=args.delay,
                              save_gif_path=args.save_gif)
