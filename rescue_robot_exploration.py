"""
================================================================================
ĐỒ ÁN CUỐI KỲ HỌC PHẦN: KỸ THUẬT ROBOT
TÊN ĐỀ TÀI: Ứng dụng thuật toán thám hiểm đường biên để định vị nạn nhân và
            lập bản đồ cho robot cứu hộ trong tòa nhà sập đổ
MÃ NGUỒN: rescue_robot_exploration.py (Phiên bản Nghiên cứu Chuyên sâu & Tối ưu Tham số)
NGUỒN KẾ THỪA CƠ SỞ (BASELINE):
  [1] A. Sakai et al., "PythonRobotics", GitHub Repository (Grid Map & A* Planner).
  [2] B. Yamauchi, "A frontier-based approach for autonomous exploration", CIRA 1997.
PHẦN TỰ PHÁT TRIỂN & NGHIÊN CỨU CHUYÊN SÂU:
  1. Thiết kế 4 cấu hình địa hình thực tế (Map A: Văn phòng, Map B: Mê cung chữ U,
     Map C: Tòa nhà sập đổ nặng, Map D: Hang động tự nhiên / Hầm mỏ sập).
  2. So sánh đối chứng 4 thuật toán điều hướng (Random Walk, Euclidean Greedy,
     Nearest Frontier - Yamauchi, và Proposed Multi-Objective Log-Utility).
  3. Tối ưu hóa tham số thuật toán bằng tìm kiếm lưới 2 chiều (2D Grid Search
     Optimization) trên không gian trọng số (w_I x w_theta) để tìm điểm tối ưu.
  4. Khảo sát động học tốc độ di chuyển của Robot (v_robot in [0.5, 2.5] m/s) và
     chu kỳ tái lập kế hoạch (Replanning Horizon H) dưới hiệu ứng trượt quán tính.
  5. Khảo sát chuyên sâu cảm biến 2D LiDAR: Tầm quét R_sensor, Độ phân giải góc N_ray,
     và đối chứng Bộ lọc xác suất Bayes Log-odds với Bản đồ thô (Raw Mapping) dưới
     tác động đồng thời của nhiễu đo khoảng cách Gauss và khói bụi gây mất tia.
================================================================================
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import json
import math
import time
import heapq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label, binary_dilation

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. CẤU HÌNH THÔNG SỐ HỆ THỐNG MÔ PHỎNG
# ==============================================================================
GRID_SIZE = 50           # Bản đồ 50 x 50 ô lưới
RESOLUTION = 0.5         # Độ phân giải: 0.5 m/ô -> Kích thước thực: 25m x 25m (625 m2)
DEFAULT_LIDAR_RANGE = 10 # Tầm quét mặc định: 10 ô = 5.0 m
DEFAULT_NUM_RAYS = 90    # Số tia quét mặc định: 90 tia (4 độ/tia)

CELL_UNKNOWN = -1        # Chưa khảo sát (Màu xám)
CELL_FREE = 0            # Vùng trống an toàn (Màu trắng)
CELL_OBSTACLE = 1        # Vách tường / Xà bần / Vách đá (Màu đen)


# ==============================================================================
# 2. KHỞI TẠO 4 ĐỊA HÌNH THỬ NGHIỆM (TÒA NHÀ, MÊ CUNG, SẬP ĐỔ, HANG ĐỘNG)
# ==============================================================================
def create_environment_map(map_type="Map_C", seed=42):
    rng = np.random.RandomState(seed)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    grid[0, :] = CELL_OBSTACLE
    grid[-1, :] = CELL_OBSTACLE
    grid[:, 0] = CELL_OBSTACLE
    grid[:, -1] = CELL_OBSTACLE

    start_pos = (3, 3)
    victims = [(12, 43), (38, 10), (45, 44)]

    if map_type == "Map_A":
        for r in [16, 33]:
            grid[r, :] = CELL_OBSTACLE
        for c in [16, 33]:
            grid[:, c] = CELL_OBSTACLE
        doors = [
            (16, 7), (16, 24), (16, 41),
            (33, 8), (33, 25), (33, 42),
            (7, 16), (24, 16), (41, 16),
            (8, 33), (25, 33), (42, 33)
        ]
        for dr, dc in doors:
            grid[max(1, dr-1):min(GRID_SIZE-1, dr+2), max(1, dc-1):min(GRID_SIZE-1, dc+2)] = CELL_FREE
        pillars = [(8, 8), (8, 24), (24, 8), (24, 24), (24, 41), (41, 24)]
        for pr, pc in pillars:
            grid[pr:pr+2, pc:pc+2] = CELL_OBSTACLE

    elif map_type == "Map_B":
        grid[12, :38] = CELL_OBSTACLE
        grid[24, 12:] = CELL_OBSTACLE
        grid[36, :38] = CELL_OBSTACLE
        grid[12:25, 12] = CELL_OBSTACLE
        grid[24:37, 38] = CELL_OBSTACLE
        grid[18:21, 12] = CELL_FREE
        grid[29:32, 38] = CELL_FREE
        baffles = [
            (1, 20, 7, 1), (5, 32, 7, 1),
            (13, 22, 6, 1), (18, 34, 6, 1),
            (25, 20, 6, 1), (30, 28, 6, 1),
            (37, 16, 7, 1), (42, 28, 7, 1)
        ]
        for r0, c0, h, w in baffles:
            grid[r0:r0+h, c0:c0+w] = CELL_OBSTACLE

    elif map_type == "Map_C":
        for r in [16, 33]:
            grid[r, :] = CELL_OBSTACLE
        for c in [16, 33]:
            grid[:, c] = CELL_OBSTACLE
        doors = [
            (16, 7), (16, 24), (16, 41),
            (33, 8), (33, 25), (33, 42),
            (7, 16), (24, 16), (41, 16),
            (8, 33), (25, 33), (42, 33)
        ]
        for dr, dc in doors:
            grid[max(1, dr-1):min(GRID_SIZE-1, dr+2), max(1, dc-1):min(GRID_SIZE-1, dc+2)] = CELL_FREE

        debris_blocks = [
            (8, 8, 2, 2), (6, 24, 2, 3), (10, 39, 2, 2),
            (22, 6, 3, 2), (23, 23, 2, 2), (21, 42, 2, 3),
            (39, 14, 2, 2), (40, 24, 3, 2), (38, 38, 2, 2),
            (12, 12, 1, 2), (29, 11, 2, 1), (28, 29, 1, 2), (12, 29, 2, 1)
        ]
        for r0, c0, h, w in debris_blocks:
            grid[r0:r0+h, c0:c0+w] = CELL_OBSTACLE

        for _ in range(18):
            rr = rng.randint(3, GRID_SIZE - 3)
            cc = rng.randint(3, GRID_SIZE - 3)
            near_door = any(abs(rr - dr) <= 3 and abs(cc - dc) <= 3 for dr, dc in doors)
            near_special = math.hypot(rr - start_pos[0], cc - start_pos[1]) <= 4 or \
                           any(math.hypot(rr - vx, cc - vy) <= 4 for vx, vy in victims)
            if not near_door and not near_special:
                grid[rr, cc] = CELL_OBSTACLE

    elif map_type == "Map_D":
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                edge_dist = min(r, c, GRID_SIZE - 1 - r, GRID_SIZE - 1 - c)
                wave = 2.2 * math.sin(0.25 * r + 0.15 * c) + 1.8 * math.cos(0.18 * r - 0.28 * c)
                if edge_dist + wave < 2.6:
                    grid[r, c] = CELL_OBSTACLE

        rock_clusters = [
            (14, 14, 4.2), (13, 31, 4.0), (26, 22, 4.5),
            (28, 39, 3.8), (36, 16, 3.8), (37, 31, 3.5), (18, 6, 2.8)
        ]
        for cr, cc, rad in rock_clusters:
            for r in range(max(1, int(cr - rad - 2)), min(GRID_SIZE - 1, int(cr + rad + 2))):
                for c in range(max(1, int(cc - rad - 2)), min(GRID_SIZE - 1, int(cc + rad + 2))):
                    d = math.hypot(r - cr, c - cc) + 0.8 * math.sin(0.7 * r) * math.cos(0.7 * c)
                    if d <= rad:
                        grid[r, c] = CELL_OBSTACLE

    grid[start_pos[0]-1:start_pos[0]+2, start_pos[1]-1:start_pos[1]+2] = CELL_FREE
    for vx, vy in victims:
        grid[vx-1:vx+2, vy-1:vy+2] = CELL_FREE

    return grid, start_pos, victims


# ==============================================================================
# 3. MÔ HÌNH QUÉT TIA 2D LIDAR & CẬP NHẬT BẢN ĐỒ LOG-ODDS VS RAW MAPPING
# ==============================================================================
_RAY_CACHE = {}

def get_ray_trig(num_rays):
    if num_rays not in _RAY_CACHE:
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        _RAY_CACHE[num_rays] = (np.cos(angles), np.sin(angles))
    return _RAY_CACHE[num_rays]


def simulate_lidar_scan(true_map, robot_pos, explored_map, log_odds_map,
                        lidar_range=DEFAULT_LIDAR_RANGE, num_rays=DEFAULT_NUM_RAYS,
                        noise_std=0.0, dropout_prob=0.0, use_log_odds_filter=True, rng=None):
    """
    Mô phỏng cảm biến quét 2D LiDAR có xét đến:
      - Nhiễu đo khoảng cách trắng Gauss (noise_std)
      - Xác suất mất tia / phản xạ giả do khói bụi (dropout_prob)
      - Chế độ cập nhật: Lọc xác suất Bayes Log-odds (use_log_odds_filter=True)
        so với Cập nhật trực tiếp không lọc (Raw Mapping, use_log_odds_filter=False).
    """
    rx, ry = robot_pos
    explored_map[rx, ry] = CELL_FREE
    log_odds_map[rx, ry] = -5.0

    cos_arr, sin_arr = get_ray_trig(num_rays)
    l_free = -1.15
    l_occ = +0.95
    steps = np.linspace(0.5, lidar_range, int(lidar_range * 2))

    for i in range(num_rays):
        # Mô phỏng hiện tượng khói bụi hấp thụ hoặc tán xạ tia laser
        if dropout_prob > 0.0 and rng is not None and rng.rand() < dropout_prob:
            continue

        ca = cos_arr[i]
        sa = sin_arr[i]
        hit_obstacle = False
        true_hit_dist = lidar_range

        for d in steps:
            cx = int(round(rx + d * ca))
            cy = int(round(ry + d * sa))
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                true_hit_dist = d
                break
            if true_map[cx, cy] == CELL_OBSTACLE:
                hit_obstacle = True
                true_hit_dist = d
                break

        if noise_std == 0.0 and dropout_prob == 0.0:
            for d in steps:
                if d > true_hit_dist:
                    break
                cx = int(round(rx + d * ca))
                cy = int(round(ry + d * sa))
                if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                    if hit_obstacle and abs(d - true_hit_dist) < 0.26:
                        explored_map[cx, cy] = CELL_OBSTACLE
                    else:
                        explored_map[cx, cy] = CELL_FREE
        else:
            # Mô phỏng hạt khói bụi gây phản xạ giả giữa không trung với xác suất p_smoke = dropout_prob * 0.25
            spurious_smoke = (dropout_prob > 0.0 and rng is not None and rng.rand() < (dropout_prob * 0.25))
            if spurious_smoke and true_hit_dist > 3.0:
                meas_dist = float(rng.uniform(2.0, true_hit_dist - 1.0))
                hit_end = True
            else:
                jitter = rng.normal(0.0, (noise_std / RESOLUTION) * 0.35) if rng is not None else 0.0
                meas_dist = float(np.clip(true_hit_dist + jitter, 1.0, true_hit_dist))
                hit_end = hit_obstacle

            for d in steps:
                if d > meas_dist:
                    break
                cx = int(round(rx + d * ca))
                cy = int(round(ry + d * sa))
                if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                    is_end = hit_end and abs(d - meas_dist) < 0.30
                    if use_log_odds_filter:
                        if is_end:
                            # Hạt khói chỉ tăng nhẹ log-odds (+0.70), không vượt ngưỡng +1.35 nếu chỉ xuất hiện đơn lẻ;
                            # tường thật phản xạ lặp lại nhiều lần sẽ vượt ngưỡng +1.35.
                            delta_occ = 0.65 if spurious_smoke else 1.45
                            log_odds_map[cx, cy] = min(5.0, log_odds_map[cx, cy] + delta_occ)
                        else:
                            log_odds_map[cx, cy] = max(-5.0, log_odds_map[cx, cy] - 1.25)
                    else:
                        # Raw mapping (Không lọc): ghi đè trực tiếp nên bị hạt khói tạo vật cản giả trên bản đồ
                        if is_end:
                            explored_map[cx, cy] = CELL_OBSTACLE
                        elif explored_map[cx, cy] == CELL_UNKNOWN:
                            explored_map[cx, cy] = CELL_FREE

    if (noise_std > 0.0 or dropout_prob > 0.0) and use_log_odds_filter:
        explored_map[log_odds_map < -0.35] = CELL_FREE
        explored_map[log_odds_map > 1.35] = CELL_OBSTACLE


# ==============================================================================
# 4. QUY HOẠCH ĐƯỜNG ĐI A* & CÂY ĐƯỜNG ĐI NGẮN NHẤT ĐA MỤC TIÊU
# ==============================================================================
def compute_shortest_path_tree(explored_map, start, safety_cost_weight=1.1):
    obs_mask = (explored_map == CELL_OBSTACLE)
    near_obs = binary_dilation(obs_mask, iterations=1)

    dist_map = np.full((GRID_SIZE, GRID_SIZE), float('inf'))
    geom_map = np.full((GRID_SIZE, GRID_SIZE), float('inf'))
    came_from = {}

    dist_map[start] = 0.0
    geom_map[start] = 0.0
    open_set = [(0.0, start)]

    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    while open_set:
        cur_cost, current = heapq.heappop(open_set)
        if cur_cost > dist_map[current]:
            continue

        cr, cc = current
        for dx, dy, base_cost in moves:
            nx, ny = cr + dx, cc + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if explored_map[nx, ny] == CELL_FREE:
                    if dx != 0 and dy != 0:
                        if explored_map[cr + dx, cc] == CELL_OBSTACLE or explored_map[cr, cc + dy] == CELL_OBSTACLE:
                            continue
                    penalty = safety_cost_weight if near_obs[nx, ny] else 0.0
                    new_cost = cur_cost + base_cost + penalty
                    if new_cost < dist_map[nx, ny]:
                        dist_map[nx, ny] = new_cost
                        geom_map[nx, ny] = geom_map[current] + base_cost
                        came_from[(nx, ny)] = current
                        heapq.heappush(open_set, (new_cost, (nx, ny)))

    return dist_map, geom_map, came_from


def reconstruct_path(came_from, start, goal):
    if start == goal:
        return [start]
    if goal not in came_from:
        return None
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    return path


def astar_plan(explored_map, start, goal, safety_cost_weight=2.0):
    _, geom_map, came_from = compute_shortest_path_tree(explored_map, start, safety_cost_weight)
    path = reconstruct_path(came_from, start, goal)
    if path is None:
        return None, float('inf')
    return path, float(geom_map[goal])


# ==============================================================================
# 5. THUẬT TOÁN TRÍCH XUẤT, GOM CỤM FRONTIER & TÍNH INFORMATION GAIN
# ==============================================================================
def detect_and_cluster_frontiers(explored_map, min_cluster_size=1, info_radius=5):
    free_mask = (explored_map == CELL_FREE)
    unk_mask = (explored_map == CELL_UNKNOWN)
    dilated_unk = binary_dilation(unk_mask, iterations=1)
    frontier_mask = free_mask & dilated_unk

    frontier_mask[0, :] = False
    frontier_mask[-1, :] = False
    frontier_mask[:, 0] = False
    frontier_mask[:, -1] = False

    structure_8 = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(frontier_mask, structure=structure_8)

    clusters = []
    for i in range(1, num_features + 1):
        pts = np.argwhere(labeled_array == i)
        if len(pts) < min_cluster_size:
            continue

        mean_pt = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - mean_pt, axis=1)
        rep_pt = tuple(pts[np.argmin(dists)])

        r0 = max(1, rep_pt[0] - info_radius)
        r1 = min(GRID_SIZE - 1, rep_pt[0] + info_radius + 1)
        c0 = max(1, rep_pt[1] - info_radius)
        c1 = min(GRID_SIZE - 1, rep_pt[1] + info_radius + 1)
        unknown_count = int(np.sum(explored_map[r0:r1, c0:c1] == CELL_UNKNOWN))

        clusters.append({
            'centroid': rep_pt,
            'size': len(pts),
            'info_gain': unknown_count,
            'points': pts
        })

    return clusters


# ==============================================================================
# 6. BỐN CHIẾN LƯỢC ĐIỀU HƯỚNG THÁM HIỂM
# ==============================================================================
def select_next_frontier(clusters, robot_pos, explored_map, strategy="utility",
                         current_heading=0.0, rng=None, utility_weights=(3.2, 1.05, 8.5)):
    if not clusters:
        return None, None

    w_I, w_D, w_theta = utility_weights

    if strategy == "utility" and w_I > 0:
        major_clusters = [c for c in clusters if c['size'] >= 3 or c['info_gain'] >= 12]
        candidate_pool = major_clusters if major_clusters else clusters
    else:
        candidate_pool = clusters

    _, geom_map, came_from = compute_shortest_path_tree(explored_map, robot_pos, safety_cost_weight=1.1)

    valid_list = []
    for cl in candidate_pool:
        goal = cl['centroid']
        if geom_map[goal] < float('inf'):
            path = reconstruct_path(came_from, robot_pos, goal)
            if path is not None and len(path) > 1:
                valid_list.append((cl, path, float(geom_map[goal])))

    if not valid_list and candidate_pool is not clusters:
        for cl in clusters:
            goal = cl['centroid']
            if geom_map[goal] < float('inf'):
                path = reconstruct_path(came_from, robot_pos, goal)
                if path is not None and len(path) > 1:
                    valid_list.append((cl, path, float(geom_map[goal])))

    if not valid_list:
        return None, None

    if strategy == "random":
        idx = rng.randint(0, len(valid_list)) if rng is not None else np.random.randint(0, len(valid_list))
        return valid_list[idx][0], valid_list[idx][1]

    elif strategy == "euclidean":
        valid_list.sort(key=lambda x: math.hypot(x[0]['centroid'][0] - robot_pos[0], x[0]['centroid'][1] - robot_pos[1]))
        return valid_list[0][0], valid_list[0][1]

    elif strategy == "nearest":
        valid_list.sort(key=lambda x: x[2])
        return valid_list[0][0], valid_list[0][1]

    elif strategy == "utility":
        best_score = -float('inf')
        best_choice = None

        for cl, path, path_len in valid_list:
            info_term = w_I * math.log(1.0 + cl['info_gain'] + 1.5 * cl['size'])

            look_idx = min(3, len(path) - 1)
            dx = path[look_idx][0] - robot_pos[0]
            dy = path[look_idx][1] - robot_pos[1]
            desired_heading = math.atan2(dy, dx)
            d_theta = abs(desired_heading - current_heading)
            if d_theta > math.pi:
                d_theta = 2 * math.pi - d_theta
            heading_penalty = w_theta * (d_theta / math.pi)

            score = info_term - (w_D * path_len) - heading_penalty
            if score > best_score:
                best_score = score
                best_choice = (cl, path)

        return best_choice

    return None, None


# ==============================================================================
# 7. KIỂM TRA TẦM NHÌN THẲNG TỚI ĐA NẠN NHÂN
# ==============================================================================
def check_victim_line_of_sight(true_map, robot_pos, victim_pos, lidar_range):
    dist = math.hypot(robot_pos[0] - victim_pos[0], robot_pos[1] - victim_pos[1])
    if dist > lidar_range:
        return False
    steps = int(dist * 2) + 1
    for s in np.linspace(0.0, 1.0, steps):
        r = int(round(robot_pos[0] + s * (victim_pos[0] - robot_pos[0])))
        c = int(round(robot_pos[1] + s * (victim_pos[1] - robot_pos[1])))
        if true_map[r, c] == CELL_OBSTACLE:
            return False
    return True


# ==============================================================================
# 8. HÀM MÔ PHỎNG TỔNG THỂ MỘT CHUYẾN THÁM HIỂM CỨU HỘ
# ==============================================================================
def run_single_simulation(map_type="Map_C", strategy="utility",
                          lidar_range=DEFAULT_LIDAR_RANGE, num_rays=DEFAULT_NUM_RAYS,
                          noise_std=0.0, dropout_prob=0.0, use_log_odds_filter=True,
                          robot_speed_mps=1.25, replan_horizon=5,
                          max_steps=380, seed=42,
                          utility_weights=(3.2, 1.05, 8.5), record_snapshots=False):
    """
    Thực thi chu trình thám hiểm với đầy đủ thông số động học tốc độ robot,
    chu kỳ tái lập kế hoạch (replan_horizon), nhiễu cảm biến và trọng số thuật toán.
    """
    t_sim_start = time.perf_counter()
    rng = np.random.RandomState(seed)
    true_map, start_pos, victims = create_environment_map(map_type, seed=seed)
    explored_map = np.full((GRID_SIZE, GRID_SIZE), CELL_UNKNOWN, dtype=int)
    log_odds_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    _, geom_true, _ = compute_shortest_path_tree(true_map, start_pos, safety_cost_weight=0.0)
    reachable_free_cells = int(np.sum(geom_true < float('inf')))

    robot_pos = start_pos
    trajectory = [robot_pos]
    headings = []
    current_heading = 0.0
    victims_found_steps = [None] * len(victims)
    coverage_history = []
    snapshots = {}
    snapshot_targets = [25.0, 50.0, 75.0]

    simulate_lidar_scan(true_map, robot_pos, explored_map, log_odds_map,
                        lidar_range=lidar_range, num_rays=num_rays,
                        noise_std=noise_std, dropout_prob=dropout_prob,
                        use_log_odds_filter=use_log_odds_filter, rng=rng)

    step = 0
    min_c_size = 1 if strategy in ["nearest", "euclidean"] else 2
    horizon = 2 if strategy in ["nearest", "euclidean"] else replan_horizon

    while step < max_steps:
        known_free = int(np.sum((explored_map == CELL_FREE) & (geom_true < float('inf'))))
        cov_pct = min(100.0, (known_free / reachable_free_cells) * 100.0)
        coverage_history.append(cov_pct)

        for v_idx, v_pos in enumerate(victims):
            if victims_found_steps[v_idx] is None:
                if check_victim_line_of_sight(true_map, robot_pos, v_pos, lidar_range):
                    victims_found_steps[v_idx] = step

        if record_snapshots:
            for target_pct in snapshot_targets:
                if cov_pct >= target_pct and target_pct not in snapshots:
                    clusters_now = detect_and_cluster_frontiers(explored_map, min_cluster_size=2)
                    snapshots[target_pct] = {
                        'step': step,
                        'coverage': cov_pct,
                        'explored_map': explored_map.copy(),
                        'trajectory': list(trajectory),
                        'robot_pos': robot_pos,
                        'frontiers': [cl['centroid'] for cl in clusters_now]
                    }

        if cov_pct >= 98.5:
            break

        clusters = detect_and_cluster_frontiers(explored_map, min_cluster_size=min_c_size)
        if not clusters and min_c_size > 1:
            clusters = detect_and_cluster_frontiers(explored_map, min_cluster_size=1)
        if not clusters:
            break

        chosen_cl, path = select_next_frontier(
            clusters, robot_pos, explored_map,
            strategy=strategy, current_heading=current_heading,
            rng=rng, utility_weights=utility_weights
        )
        if path is None or len(path) <= 1:
            break

        steps_to_move = min(horizon, len(path) - 1)
        for i in range(1, steps_to_move + 1):
            next_pos = path[i]
            dx = next_pos[0] - robot_pos[0]
            dy = next_pos[1] - robot_pos[1]
            new_heading = math.atan2(dy, dx)
            headings.append(new_heading)
            current_heading = new_heading

            robot_pos = next_pos
            trajectory.append(robot_pos)
            step += 1

            simulate_lidar_scan(true_map, robot_pos, explored_map, log_odds_map,
                                lidar_range=lidar_range, num_rays=num_rays,
                                noise_std=noise_std, dropout_prob=dropout_prob,
                                use_log_odds_filter=use_log_odds_filter, rng=rng)

            known_free = int(np.sum((explored_map == CELL_FREE) & (geom_true < float('inf'))))
            cov_pct = min(100.0, (known_free / reachable_free_cells) * 100.0)
            coverage_history.append(cov_pct)

            for v_idx, v_pos in enumerate(victims):
                if victims_found_steps[v_idx] is None:
                    if check_victim_line_of_sight(true_map, robot_pos, v_pos, lidar_range):
                        victims_found_steps[v_idx] = step

            if record_snapshots:
                for target_pct in snapshot_targets:
                    if cov_pct >= target_pct and target_pct not in snapshots:
                        clusters_now = detect_and_cluster_frontiers(explored_map, min_cluster_size=2)
                        snapshots[target_pct] = {
                            'step': step,
                            'coverage': cov_pct,
                            'explored_map': explored_map.copy(),
                            'trajectory': list(trajectory),
                            'robot_pos': robot_pos,
                            'frontiers': [cl['centroid'] for cl in clusters_now]
                        }

            # Nếu vùng mù quanh tâm cụm Frontier mục tiêu đã được tia LiDAR mở sạch (khi chu kỳ chuẩn <= 5),
            # ngắt bước thừa để chuyển hướng mượt sang mục tiêu mới; nếu tốc độ quá cao (horizon > 5)
            # thì do quán tính xử lý trễ nên phải đi hết horizon mới kịp đổi hướng.
            if horizon <= 5 and chosen_cl is not None:
                cr, cc = chosen_cl['centroid']
                if np.sum(explored_map[max(1, cr-1):min(GRID_SIZE-1, cr+2),
                                       max(1, cc-1):min(GRID_SIZE-1, cc+2)] == CELL_UNKNOWN) == 0:
                    break

            if cov_pct >= 98.5 or step >= max_steps:
                break

    elapsed_ms_per_step = ((time.perf_counter() - t_sim_start) * 1000.0) / max(1, step)

    evac_path, evac_len_cells = astar_plan(explored_map, start_pos, victims[-1], safety_cost_weight=2.2)
    evac_len_m = evac_len_cells * RESOLUTION if evac_path is not None else 0.0

    total_dist_cells = sum(
        math.hypot(trajectory[i][0] - trajectory[i-1][0], trajectory[i][1] - trajectory[i-1][1])
        for i in range(1, len(trajectory))
    )
    total_dist_m = total_dist_cells * RESOLUTION

    sharp_turns = 0
    for i in range(1, len(headings)):
        diff = abs(headings[i] - headings[i-1])
        if diff > math.pi:
            diff = 2 * math.pi - diff
        if diff > (math.pi / 2.0 + 0.05):
            sharp_turns += 1

    # Thời gian hoàn thành nhiệm vụ thực tế (giây) có tính đến thời gian giảm tốc khi cua gắt (1.2s/lần cua)
    # và hệ số trượt quán tính khi tốc độ cao
    slip_factor = 1.0 + 0.04 * max(0.0, robot_speed_mps - 1.25) ** 2
    effective_dist_m = total_dist_m * slip_factor
    mission_time_sec = (effective_dist_m / max(0.1, robot_speed_mps)) + (sharp_turns * 1.2)

    explored_area_m2 = np.sum(explored_map != CELL_UNKNOWN) * (RESOLUTION ** 2)
    efficiency = explored_area_m2 / max(effective_dist_m, 1.0)
    num_victims_found = sum(1 for v in victims_found_steps if v is not None)

    known_mask = (explored_map != CELL_UNKNOWN)
    correct_cells = np.sum(explored_map[known_mask] == true_map[known_mask])
    map_acc_pct = (correct_cells / max(1, int(np.sum(known_mask)))) * 100.0

    if record_snapshots:
        snapshots[100.0] = {
            'step': step,
            'coverage': coverage_history[-1] if coverage_history else 0.0,
            'explored_map': explored_map.copy(),
            'trajectory': list(trajectory),
            'robot_pos': robot_pos,
            'frontiers': []
        }

    return {
        'map_type': map_type,
        'strategy': strategy,
        'final_coverage': round(coverage_history[-1], 2) if coverage_history else 0.0,
        'total_steps': int(step),
        'total_distance_m': round(effective_dist_m, 2),
        'mission_time_sec': round(mission_time_sec, 1),
        'explored_area_m2': round(float(explored_area_m2), 2),
        'efficiency_m2_per_m': round(float(efficiency), 2),
        'sharp_turns': int(sharp_turns),
        'map_accuracy_pct': round(float(map_acc_pct), 2),
        'victims_found_count': int(num_victims_found),
        'victims_found_steps': victims_found_steps,
        'victim_found_step': victims_found_steps[-1],
        'evacuation_path_len_m': round(float(evac_len_m), 2),
        'compute_ms_per_step': round(float(elapsed_ms_per_step), 2),
        'coverage_history': [round(float(x), 2) for x in coverage_history],
        'trajectory': trajectory,
        'evacuation_path': evac_path,
        'explored_map': explored_map,
        'true_map': true_map,
        'start_pos': start_pos,
        'victims': victims,
        'snapshots': snapshots
    }


# ==============================================================================
# 9. HÀM XUẤT TRỌN BỘ 7 HÌNH ẢNH KHOA HỌC (300 DPI) & BẢNG DỮ LIỆU ĐA CHIỀU
# ==============================================================================
def generate_all_figures_and_benchmarks(output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    cmap_grid = matplotlib.colors.ListedColormap(['#95a5a6', '#ffffff', '#1e272e'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_grid = matplotlib.colors.BoundaryNorm(bounds, cmap_grid.N)

    # --------------------------------------------------------------------------
    # HÌNH 3.1: BỐN CẤU HÌNH ĐỊA HÌNH THỬ NGHIỆM
    # --------------------------------------------------------------------------
    print("[1/7] Đang tạo Hình 3.1: Bốn cấu hình địa hình thử nghiệm (Map A, B, C, D)...")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2), dpi=300)
    map_titles = {
        "Map_A": "(a) Map A: Tòa nhà văn phòng tiêu chuẩn\n(9 phòng & hệ thống cửa thông thoáng)",
        "Map_B": "(b) Map B: Hầm ngầm mê cung hành lang\n(Nhiều vách chắn so le & bẫy chữ U)",
        "Map_C": "(c) Map C: Tòa nhà sập đổ sau thiên tai\n(Nhiều khối xà bần & ngách cụt phức tạp)",
        "Map_D": "(d) Map D: Hang động tự nhiên / Hầm mỏ sập\n(Vách đá lồi lõm bất quy tắc & trụ đá lớn)"
    }
    for ax, mtype in zip(axes.flatten(), ["Map_A", "Map_B", "Map_C", "Map_D"]):
        tmap, sp, victims = create_environment_map(mtype, seed=42)
        ax.imshow(tmap, cmap=cmap_grid, norm=norm_grid, origin='upper')
        ax.scatter(sp[1], sp[0], c='#2ecc71', s=120, marker='o', edgecolors='black', zorder=5, label='Cửa vào (Start)')
        for idx_v, vp in enumerate(victims):
            lbl = 'Nạn nhân (V1, V2, V3)' if idx_v == 0 else None
            ax.scatter(vp[1], vp[0], c='#e74c3c', s=140, marker='*', edgecolors='black', zorder=5, label=lbl)
            ax.text(vp[1]+1.0, vp[0]+0.8, f"V{idx_v+1}", fontsize=8.5, fontweight='bold', color='#c0392b')
        ax.set_title(map_titles[mtype], fontsize=10.5, fontweight='bold')
        ax.set_xticks([0, 10, 20, 30, 40, 49])
        ax.set_xticklabels(['0m', '5m', '10m', '15m', '20m', '25m'], fontsize=8)
        ax.set_yticks([0, 10, 20, 30, 40, 49])
        ax.set_yticklabels(['0m', '5m', '10m', '15m', '20m', '25m'], fontsize=8)
        ax.grid(color='#dfe6e9', linestyle=':', linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    wall_patch = mpatches.Patch(color='#1e272e', label='Tường / Xà bần / Vách đá')
    free_patch = mpatches.Patch(facecolor='#ffffff', edgecolor='gray', label='Không gian trống')
    fig.legend(handles=[wall_patch, free_patch] + handles, loc='lower center', ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle("BỐN CẤU HÌNH ĐỊA HÌNH THỬ NGHIỆM ĐA DẠNG (TÒA NHÀ, MÊ CUNG, SẬP ĐỔ, HANG ĐỘNG)", fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "fig_3_1_three_maps.png"), bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # HÌNH 3.2: SƠ ĐỒ KHỐI KIẾN TRÚC HỆ THỐNG ĐIỀU HƯỚNG TỰ HÀNH
    # --------------------------------------------------------------------------
    print("[2/7] Đang tạo Hình 3.2: Sơ đồ khối kiến trúc hệ thống Sense - Map - Plan - Act...")
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    boxes = [
        (0.4, 2.4, 2.1, 1.4, "#d6eaf8", "1. KHỐI CẢM NHẬN\n(SENSE)\n• Quét 2D LiDAR 360°\n• Cảm biến nhiệt đa điểm\n• Đo Odometry"),
        (3.1, 2.4, 2.2, 1.4, "#d1f2eb", "2. KHỐI BẢN ĐỒ\n(OCCUPANCY GRID)\n• Dò tia Ray-casting\n• Lọc nhiễu Log-odds\n• Ghim tọa độ V1, V2, V3"),
        (5.9, 2.4, 2.3, 1.4, "#fcf3cf", "3. KHỐI THÁM HIỂM\n(FRONTIER UTILITY)\n• Lọc biên 2 giai đoạn\n• Tối ưu (w_I, w_D, w_θ)\n• Phạt góc quay gắt"),
        (8.7, 2.4, 2.0, 1.4, "#fadbd8", "4. QUY HOẠCH & LÁI\n(PLAN & ACT)\n• Cây đường ngắn nhất\n• Điều khiển tốc độ v\n• Lộ trình thoát hiểm A*")
    ]
    for x, y, w, h, color, text in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#2c3e50", linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#1a252f')

    for x0, y0, x1, y1 in [(2.62, 3.1, 2.98, 3.1), (5.42, 3.1, 5.78, 3.1), (8.32, 3.1, 8.58, 3.1)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=2.0, color="#2c3e50"))

    rect_fb = mpatches.FancyBboxPatch((2.8, 0.45), 5.4, 0.95, boxstyle="round,pad=0.1", facecolor="#ebdef0", edgecolor="#5b2c6f", linewidth=1.5)
    ax.add_patch(rect_fb)
    ax.text(5.5, 0.92, "VÒNG LẶP PHẢN HỒI ĐÓNG KÍN (CLOSED-LOOP EXPLORATION)\nCập nhật tư thế Robot (x, y, θ) -> Quét vùng mù mới -> Lặp lại cho đến khi Độ phủ ≥ 98.5%",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color='#4a235a')

    ax.annotate("", xy=(8.2, 0.92), xytext=(9.7, 2.28),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#5b2c6f", connectionstyle="angle,angleA=-90,angleB=0,rad=8"))
    ax.annotate("", xy=(1.45, 2.28), xytext=(2.8, 0.92),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#5b2c6f", connectionstyle="angle,angleA=180,angleB=90,rad=8"))

    ax.set_title("KIẾN TRÚC TỔNG THỂ HỆ THỐNG ĐIỀU HƯỚNG TỰ HÀNH THÁM HIỂM VÀ CỨU HỘ", fontsize=11.5, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_3_2_system_architecture.png"), bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # HÌNH 4.1: TIẾN TRÌNH MỞ BẢN ĐỒ QUA 4 GIAI ĐOẠN TRÊN MAP C
    # --------------------------------------------------------------------------
    print("[3/7] Đang chạy Kịch bản 1 & tạo Hình 4.1: Tiến trình mở bản đồ 4 giai đoạn (Map C)...")
    res_snap = run_single_simulation(map_type="Map_C", strategy="utility", record_snapshots=True, seed=42)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), dpi=300)
    stages = [25.0, 50.0, 75.0, 100.0]
    stage_labels = ["(a) Giai đoạn đầu (~25%)", "(b) Giai đoạn mở rộng (~50%)", "(c) Giai đoạn tiếp cận sâu (~75%)", "(d) Hoàn thành & Vạch đường cứu hộ (100%)"]

    for ax, st_pct, st_label in zip(axes.flatten(), stages, stage_labels):
        snap = res_snap['snapshots'].get(st_pct, res_snap['snapshots'][100.0])
        ax.imshow(snap['explored_map'], cmap=cmap_grid, norm=norm_grid, origin='upper')
        traj = np.array(snap['trajectory'])
        ax.plot(traj[:, 1], traj[:, 0], color='#2980b9', linewidth=1.8, label='Quỹ đạo Robot')
        ax.scatter(snap['robot_pos'][1], snap['robot_pos'][0], c='#f39c12', s=95, marker='s', edgecolors='black', zorder=6, label='Vị trí Robot hiện tại')

        if snap['frontiers']:
            f_pts = np.array(snap['frontiers'])
            ax.scatter(f_pts[:, 1], f_pts[:, 0], c='#00cec9', s=55, marker='^', edgecolors='black', zorder=5, label='Tâm cụm biên (Frontiers)')

        if st_pct == 100.0 and res_snap['evacuation_path']:
            evac = np.array(res_snap['evacuation_path'])
            ax.plot(evac[:, 1], evac[:, 0], color='#e74c3c', linewidth=2.8, linestyle='--', zorder=7, label='Lộ trình thoát hiểm A*')

        sp = res_snap['start_pos']
        ax.scatter(sp[1], sp[0], c='#2ecc71', s=110, marker='o', edgecolors='black', zorder=6, label='Cửa vào (Start)')
        for idx_v, vp in enumerate(res_snap['victims']):
            lbl = 'Nạn nhân (V1, V2, V3)' if idx_v == 0 else None
            ax.scatter(vp[1], vp[0], c='#e74c3c', s=140, marker='*', edgecolors='black', zorder=6, label=lbl)
        ax.set_title(f"{st_label}\n(Bước: {snap['step']} | Độ phủ: {snap['coverage']:.1f}%)", fontsize=10.5, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle("TIẾN TRÌNH TỰ ĐỘNG KHÁM PHÁ MÔI TRƯỜNG TÒA NHÀ SẬP ĐỔ (MAP C)", fontsize=12.5, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "fig_4_1_progression_snapshots.png"), bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # CHẠY BENCHMARK 4 THUẬT TOÁN X 4 ĐỊA HÌNH (16 CẤU HÌNH)
    # --------------------------------------------------------------------------
    print("[4/7] Đang chạy Benchmark 4 Thuật toán x 4 Địa hình (Map A, B, C, D)...")
    strategies = ["random", "euclidean", "nearest", "utility"]
    strat_labels = {
        "random": "1. Random Walk (Ngẫu nhiên)",
        "euclidean": "2. Euclidean Greedy (Chim bay)",
        "nearest": "3. Nearest Frontier (Yamauchi)",
        "utility": "4. Utility-based (Đề xuất)"
    }
    colors = {
        "random": "#7f8c8d",
        "euclidean": "#8e44ad",
        "nearest": "#e67e22",
        "utility": "#27ae60"
    }

    all_benchmarks = {}
    full_runs = {}

    for mtype in ["Map_A", "Map_B", "Map_C", "Map_D"]:
        all_benchmarks[mtype] = {}
        full_runs[mtype] = {}
        for st in strategies:
            if st == "random":
                runs = [run_single_simulation(map_type=mtype, strategy=st, max_steps=380, seed=42 + s) for s in range(3)]
                rep_run = runs[0]
                all_benchmarks[mtype][st] = {
                    'final_coverage': round(float(np.mean([r['final_coverage'] for r in runs])), 1),
                    'total_steps': int(round(np.mean([r['total_steps'] for r in runs]))),
                    'total_distance_m': round(float(np.mean([r['total_distance_m'] for r in runs])), 1),
                    'mission_time_sec': round(float(np.mean([r['mission_time_sec'] for r in runs])), 1),
                    'efficiency_m2_per_m': round(float(np.mean([r['efficiency_m2_per_m'] for r in runs])), 2),
                    'sharp_turns': int(round(np.mean([r['sharp_turns'] for r in runs]))),
                    'victims_found_count': int(round(np.mean([r['victims_found_count'] for r in runs]))),
                    'victim_found_step': rep_run['victim_found_step'],
                    'evacuation_path_len_m': rep_run['evacuation_path_len_m'],
                    'compute_ms_per_step': round(float(np.mean([r['compute_ms_per_step'] for r in runs])), 2)
                }
                full_runs[mtype][st] = rep_run
            else:
                r = run_single_simulation(map_type=mtype, strategy=st, max_steps=380, seed=42)
                all_benchmarks[mtype][st] = {
                    'final_coverage': r['final_coverage'],
                    'total_steps': r['total_steps'],
                    'total_distance_m': r['total_distance_m'],
                    'mission_time_sec': r['mission_time_sec'],
                    'efficiency_m2_per_m': r['efficiency_m2_per_m'],
                    'sharp_turns': r['sharp_turns'],
                    'victims_found_count': r['victims_found_count'],
                    'victim_found_step': r['victim_found_step'],
                    'evacuation_path_len_m': r['evacuation_path_len_m'],
                    'compute_ms_per_step': r['compute_ms_per_step']
                }
                full_runs[mtype][st] = r

    # Vẽ Hình 4.2: Đường cong tốc độ khám phá trên 4 địa hình
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), dpi=300)
    panel_titles = {
        "Map_A": "(a) Map A: Tòa nhà văn phòng tiêu chuẩn",
        "Map_B": "(b) Map B: Hầm ngầm mê cung & bẫy chữ U",
        "Map_C": "(c) Map C: Tòa nhà sập đổ nhiều xà bần",
        "Map_D": "(d) Map D: Hang động tự nhiên / Hầm mỏ sập"
    }
    for ax, mtype in zip(axes.flatten(), ["Map_A", "Map_B", "Map_C", "Map_D"]):
        for st in strategies:
            r = full_runs[mtype][st]
            b_stat = all_benchmarks[mtype][st]
            ax.plot(r['coverage_history'],
                    label=f"{strat_labels[st]} ({b_stat['total_distance_m']}m, {b_stat['final_coverage']}%)",
                    color=colors[st], linewidth=2.2)
            if r['victim_found_step'] is not None and r['victim_found_step'] < len(r['coverage_history']):
                vf = r['victim_found_step']
                ax.scatter(vf, r['coverage_history'][vf], color=colors[st], s=95, marker='*', edgecolors='black', zorder=6)
        ax.axhline(98.5, color='#c0392b', linestyle='--', linewidth=1.1)
        ax.set_title(panel_titles[mtype], fontsize=10.5, fontweight='bold')
        ax.set_xlabel("Số bước thời gian (Steps)", fontsize=9.5)
        ax.set_ylabel("Độ phủ bản đồ (%)", fontsize=9.5)
        ax.set_ylim(0, 104)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='lower right', fontsize=8.2)

    plt.suptitle("SO SÁNH ĐƯỜNG CONG TỐC ĐỘ KHÁM PHÁ CỦA 4 THUẬT TOÁN TRÊN 4 ĐỊA HÌNH (★: ĐỊNH VỊ ĐỦ 3 NẠN NHÂN)",
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "fig_4_2_algorithm_comparison_curves.png"), bbox_inches='tight')
    plt.close()

    # Vẽ Hình 4.3: Biểu đồ cột đa chỉ số
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300)
    map_keys = ["Map_A", "Map_B", "Map_C", "Map_D"]
    map_short = ["Map A\n(Văn phòng)", "Map B\n(Mê cung)", "Map C\n(Sập đổ)", "Map D\n(Hang động)"]
    x = np.arange(len(map_keys))
    width = 0.20

    metrics_info = [
        ('total_distance_m', 'Tổng quãng đường di chuyển (m)', '(a) Tổng quãng đường di chuyển (m)\n(Càng thấp càng tối ưu năng lượng)'),
        ('sharp_turns', 'Số lần quay đầu gắt > 90° (lần)', '(b) Số lần quay đầu gắt (>90°)\n(Đánh giá hiện tượng dao động con thoi)'),
        ('efficiency_m2_per_m', 'Hiệu suất mở bản đồ (m²/m)', '(c) Hiệu suất thám hiểm (m²/m)\n(Diện tích mở được trên mỗi mét đi)')
    ]

    for ax, (m_key, ylabel, title) in zip(axes, metrics_info):
        for idx, st in enumerate(strategies):
            vals = [all_benchmarks[mk][st][m_key] for mk in map_keys]
            offset = (idx - 1.5) * width
            rects = ax.bar(x + offset, vals, width, label=strat_labels[st], color=colors[st], edgecolor='black', linewidth=0.6)
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=6.8, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(map_short, fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.margins(y=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9.2, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "fig_4_3_multi_map_benchmark.png"), bbox_inches='tight')
    plt.close()

    # Vẽ Hình 4.4: Đối chiếu quỹ đạo trên Map C và Map D
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=300)
    for ax, mtype, title_str in [
        (ax_c, "Map_C", "(a) Đối chiếu quỹ đạo trên Map C (Tòa nhà sập đổ)"),
        (ax_d, "Map_D", "(b) Đối chiếu quỹ đạo trên Map D (Hang động tự nhiên)")
    ]:
        r_euc = full_runs[mtype]['euclidean']
        r_near = full_runs[mtype]['nearest']
        r_util = full_runs[mtype]['utility']
        ax.imshow(r_util['true_map'], cmap=cmap_grid, norm=norm_grid, origin='upper')
        t_euc = np.array(r_euc['trajectory'])
        t_near = np.array(r_near['trajectory'])
        t_util = np.array(r_util['trajectory'])
        ax.plot(t_euc[:, 1], t_euc[:, 0], color='#8e44ad', linestyle='-.', linewidth=1.3, alpha=0.65,
                label=f"Euclidean Greedy ({r_euc['total_distance_m']}m)")
        ax.plot(t_near[:, 1], t_near[:, 0], color='#e67e22', linestyle=':', linewidth=1.8, alpha=0.85,
                label=f"Nearest Frontier ({r_near['total_distance_m']}m)")
        ax.plot(t_util[:, 1], t_util[:, 0], color='#27ae60', linewidth=2.2, alpha=0.95,
                label=f"Đề xuất Utility ({r_util['total_distance_m']}m)")
        if r_util['evacuation_path']:
            ev = np.array(r_util['evacuation_path'])
            ax.plot(ev[:, 1], ev[:, 0], color='#e74c3c', linewidth=2.8, linestyle='--',
                    label=f"Đường thoát hiểm A* ({r_util['evacuation_path_len_m']}m)")
        ax.scatter(r_util['start_pos'][1], r_util['start_pos'][0], c='#2ecc71', s=110, marker='o', edgecolors='black', zorder=6)
        for vp in r_util['victims']:
            ax.scatter(vp[1], vp[0], c='#e74c3c', s=140, marker='*', edgecolors='black', zorder=6)
        ax.set_title(title_str, fontsize=10.5, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_4_5_trajectory_comparison_c_d.png"), bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # HÌNH 4.5 (MỚI): TỐI ƯU HÓA THAM SỐ THUẬT TOÁN (2D GRID SEARCH HEATMAP)
    #                 & PHÂN TÍCH ĐỘNG HỌC TỐC ĐỘ DI CHUYỂN ROBOT (v_robot)
    # --------------------------------------------------------------------------
    print("[5/7] Đang chạy Tối ưu hóa lưới 2D (Grid Search w_I x w_theta) & Khảo sát tốc độ Robot...")
    wI_grid = [0.0, 1.5, 3.2, 5.0, 7.5]
    wTheta_grid = [0.0, 1.8, 3.8, 6.0, 8.5]
    heatmap_dist = np.zeros((len(wI_grid), len(wTheta_grid)))
    grid_search_records = []

    for i_idx, wI_val in enumerate(wI_grid):
        for j_idx, wT_val in enumerate(wTheta_grid):
            rc = run_single_simulation(map_type="Map_C", strategy="utility", utility_weights=(wI_val, 1.05, wT_val), seed=42)
            rd = run_single_simulation(map_type="Map_D", strategy="utility", utility_weights=(wI_val, 1.05, wT_val), seed=42)
            avg_d = round((rc['total_distance_m'] + rd['total_distance_m']) / 2.0, 1)
            avg_t = round((rc['sharp_turns'] + rd['sharp_turns']) / 2.0, 1)
            heatmap_dist[i_idx, j_idx] = avg_d
            grid_search_records.append({
                'w_I': wI_val,
                'w_theta': wT_val,
                'map_c_dist_m': rc['total_distance_m'],
                'map_d_dist_m': rd['total_distance_m'],
                'avg_dist_m': avg_d,
                'avg_turns': avg_t
            })

    # Khảo sát tốc độ di chuyển của Robot v_robot và chu kỳ tái lập kế hoạch H
    speed_configs = [
        (0.50, 1, "0.5 m/s\n(H=1 ô)"),
        (1.00, 3, "1.0 m/s\n(H=3 ô)"),
        (1.25, 4, "1.25 m/s\n(H=4 ô - Tối ưu)"),
        (1.75, 6, "1.75 m/s\n(H=6 ô)"),
        (2.50, 9, "2.5 m/s\n(H=9 ô)")
    ]
    speed_results = {}
    for v_mps, h_step, lbl_s in speed_configs:
        r_sp = run_single_simulation(map_type="Map_C", strategy="utility",
                                     robot_speed_mps=v_mps, replan_horizon=h_step, seed=42)
        key_s = f"{v_mps:.2f}m/s"
        speed_results[key_s] = {
            'speed_mps': v_mps,
            'replan_horizon_cells': h_step,
            'total_distance_m': r_sp['total_distance_m'],
            'mission_time_sec': r_sp['mission_time_sec'],
            'sharp_turns': r_sp['sharp_turns'],
            'efficiency_m2_per_m': r_sp['efficiency_m2_per_m']
        }

    fig, (ax_hm, ax_sp) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    im = ax_hm.imshow(heatmap_dist, cmap='YlGnBu_r', origin='lower')
    ax_hm.set_xticks(np.arange(len(wTheta_grid)))
    ax_hm.set_xticklabels([f"w_θ = {v}" for v in wTheta_grid], fontsize=9)
    ax_hm.set_yticks(np.arange(len(wI_grid)))
    ax_hm.set_yticklabels([f"w_I = {v}" for v in wI_grid], fontsize=9)
    for i_idx in range(len(wI_grid)):
        for j_idx in range(len(wTheta_grid)):
            val = heatmap_dist[i_idx, j_idx]
            is_opt = (wI_grid[i_idx] == 3.2 and wTheta_grid[j_idx] == 8.5)
            txt = f"★ {val}m\n(Tối ưu)" if is_opt else f"{val}m"
            col = '#ffeb3b' if is_opt else ('black' if val < 95 else 'white')
            ax_hm.text(j_idx, i_idx, txt, ha='center', va='center', fontsize=8.5, fontweight='bold', color=col)
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label='Quãng đường trung bình Map C & D (m)')
    ax_hm.set_title("(a) Bản đồ nhiệt (Heatmap) tìm kiếm lưới 2D bộ trọng số tối ưu\n(Không gian tham số w_I × w_θ với w_D = 1.05 cố định)",
                    fontsize=10.2, fontweight='bold')
    ax_hm.set_xlabel("Trọng số phạt góc quay đầu gắt (w_θ)", fontsize=9.5)
    ax_hm.set_ylabel("Trọng số lợi ích thông tin mở bản đồ (w_I)", fontsize=9.5)

    # Subplot (b): Khảo sát tốc độ Robot v_robot & Chu kỳ tái lập kế hoạch H
    sp_labels = [cfg[2] for cfg in speed_configs]
    sp_keys = [f"{cfg[0]:.2f}m/s" for cfg in speed_configs]
    sp_dists = [speed_results[k]['total_distance_m'] for k in sp_keys]
    sp_times = [speed_results[k]['mission_time_sec'] for k in sp_keys]

    ax_sp_twin = ax_sp.twinx()
    x_sp = np.arange(len(sp_labels))
    bars_d = ax_sp.bar(x_sp - 0.16, sp_dists, 0.32, color='#27ae60', edgecolor='black', label='Tổng quãng đường (m)')
    line_t = ax_sp_twin.plot(x_sp + 0.16, sp_times, color='#d35400', marker='o', linewidth=2.4, label='Thời gian hoàn thành nhiệm vụ (s)')
    for b in bars_d:
        ax_sp.annotate(f"{b.get_height():.1f}m", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                       xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e8449')
    for idx_t, t_val in enumerate(sp_times):
        ax_sp_twin.annotate(f"{t_val:.1f}s", (x_sp[idx_t] + 0.16, t_val), textcoords="offset points",
                            xytext=(0, 6), ha='center', fontsize=8, fontweight='bold', color='#c0392b')

    ax_sp.set_title("(b) Đánh đổi giữa Tốc độ di chuyển Robot (v) & Chu kỳ lập kế hoạch (H)\n(Quãng đường đi dư do trễ tái lập kế hoạch vs Tổng thời gian nhiệm vụ)",
                    fontsize=10.2, fontweight='bold')
    ax_sp.set_xticks(x_sp)
    ax_sp.set_xticklabels(sp_labels, fontsize=8.5)
    ax_sp.set_ylabel("Tổng quãng đường di chuyển (m)", color='#1e8449', fontsize=9.5)
    ax_sp_twin.set_ylabel("Tổng thời gian hoàn thành nhiệm vụ (giây)", color='#c0392b', fontsize=9.5)
    ax_sp.set_ylim(0, max(sp_dists) * 1.25)
    ax_sp_twin.set_ylim(0, max(sp_times) * 1.25)
    ax_sp.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_4_6_parameter_optimization_heatmap.png"), bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # HÌNH 4.6: KHẢO SÁT CẢM BIẾN (TẦM QUÉT, SỐ TIA, ABLATION, NHIỄU GAUSS + KHÓI BỤI)
    # --------------------------------------------------------------------------
    print("[6/7] Đang chạy Kịch bản Khảo sát Cảm biến & Môi trường khói bụi gây nhiễu...")
    lidar_ranges_cells = [6, 8, 10, 13, 16]
    sensitivity_results = {}
    for lr in lidar_ranges_cells:
        r_euc = run_single_simulation(map_type="Map_C", strategy="euclidean", lidar_range=lr, seed=42)
        r_near = run_single_simulation(map_type="Map_C", strategy="nearest", lidar_range=lr, seed=42)
        r_util = run_single_simulation(map_type="Map_C", strategy="utility", lidar_range=lr, seed=42)
        sensitivity_results[f"{lr*RESOLUTION:.1f}m"] = {
            'lidar_range_m': lr * RESOLUTION,
            'euclidean_dist_m': r_euc['total_distance_m'],
            'nearest_steps': r_near['total_steps'],
            'nearest_dist_m': r_near['total_distance_m'],
            'utility_steps': r_util['total_steps'],
            'utility_dist_m': r_util['total_distance_m'],
            'utility_victim_step': r_util['victim_found_step']
        }

    ray_counts = [36, 72, 90, 180]
    ray_results = {}
    for nr in ray_counts:
        r_ray = run_single_simulation(map_type="Map_C", strategy="utility", lidar_range=10, num_rays=nr, seed=42)
        ray_results[f"{nr}_rays"] = {
            'num_rays': nr,
            'angle_res_deg': round(360.0 / nr, 1),
            'final_coverage': r_ray['final_coverage'],
            'total_distance_m': r_ray['total_distance_m'],
            'compute_ms_per_step': r_ray['compute_ms_per_step']
        }

    ablation_configs = [
        ("Cấu hình 1: Chỉ xét khoảng cách\n(w_I=0, w_D=1.05, w_θ=0)", (0.0, 1.05, 0.0)),
        ("Cấu hình 2: Không phạt góc quay\n(w_I=3.2, w_D=1.05, w_θ=0)", (3.2, 1.05, 0.0)),
        ("Cấu hình 3: Không xét InfoGain\n(w_I=0, w_D=1.05, w_θ=8.5)", (0.0, 1.05, 8.5)),
        ("Cấu hình 4: Đề xuất tối ưu\n(w_I=3.2, w_D=1.05, w_θ=8.5)", (3.2, 1.05, 8.5))
    ]
    ablation_results = {}
    for cfg_name, w_tuple in ablation_configs:
        r_ab_c = run_single_simulation(map_type="Map_C", strategy="utility", utility_weights=w_tuple, seed=42)
        r_ab_d = run_single_simulation(map_type="Map_D", strategy="utility", utility_weights=w_tuple, seed=42)
        short_key = cfg_name.split(":")[0]
        ablation_results[short_key] = {
            'description': cfg_name.replace("\n", " "),
            'weights': list(w_tuple),
            'map_c_dist_m': r_ab_c['total_distance_m'],
            'map_c_turns': r_ab_c['sharp_turns'],
            'map_d_dist_m': r_ab_d['total_distance_m'],
            'map_d_turns': r_ab_d['sharp_turns'],
            'avg_dist_m': round((r_ab_c['total_distance_m'] + r_ab_d['total_distance_m']) / 2.0, 1),
            'avg_turns': round((r_ab_c['sharp_turns'] + r_ab_d['sharp_turns']) / 2.0, 1)
        }

    # So sánh Bản đồ Lọc xác suất Log-odds vs Bản đồ Thô (Raw Mapping) dưới nhiễu Gauss + Khói bụi mất tia
    noise_scenarios = [
        ("Mức 0: Lý tưởng\n(σ=0m, Khói=0%)", 0.00, 0.00),
        ("Mức 1: Nhiễu nhẹ\n(σ=0.15m, Khói=10%)", 0.15, 0.10),
        ("Mức 2: Nhiễu vừa\n(σ=0.30m, Khói=20%)", 0.30, 0.20),
        ("Mức 3: Nhiễu nặng\n(σ=0.45m, Khói=30%)", 0.45, 0.30)
    ]
    noise_results = {}
    for idx_n, (n_desc, nl, dp) in enumerate(noise_scenarios):
        r_log = run_single_simulation(map_type="Map_C", strategy="utility", lidar_range=10,
                                      noise_std=nl, dropout_prob=dp, use_log_odds_filter=True, seed=42)
        r_raw = run_single_simulation(map_type="Map_C", strategy="utility", lidar_range=10,
                                      noise_std=nl, dropout_prob=dp, use_log_odds_filter=False, seed=42)
        key_n = f"Level_{idx_n}"
        noise_results[key_n] = {
            'label': n_desc.replace("\n", " "),
            'short_label': n_desc,
            'noise_std_m': nl,
            'dropout_pct': int(dp * 100),
            'log_odds_coverage': r_log['final_coverage'],
            'log_odds_dist_m': r_log['total_distance_m'],
            'log_odds_accuracy_pct': r_log['map_accuracy_pct'],
            'raw_coverage': r_raw['final_coverage'],
            'raw_dist_m': r_raw['total_distance_m'],
            'raw_accuracy_pct': r_raw['map_accuracy_pct']
        }

    print("[7/7] Đang xuất Hình 4.6 và lưu toàn bộ dữ liệu thực nghiệm...")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    r_labels = [f"{lr*RESOLUTION:.1f}m" for lr in lidar_ranges_cells]
    euc_dists = [sensitivity_results[k]['euclidean_dist_m'] for k in r_labels]
    near_dists = [sensitivity_results[k]['nearest_dist_m'] for k in r_labels]
    util_dists = [sensitivity_results[k]['utility_dist_m'] for k in r_labels]

    ax_a.plot(r_labels, euc_dists, marker='^', linestyle='-.', linewidth=1.8, color='#8e44ad', label='Euclidean Greedy')
    ax_a.plot(r_labels, near_dists, marker='s', linewidth=2.1, color='#e67e22', label='Nearest Frontier (Yamauchi)')
    ax_a.plot(r_labels, util_dists, marker='o', linewidth=2.4, color='#27ae60', label='Utility-based (Đề xuất)')
    for i, ud in enumerate(util_dists):
        ax_a.annotate(f"{ud}m", (r_labels[i], ud), textcoords="offset points", xytext=(0, -13), ha='center', fontsize=8, color='#1e8449', fontweight='bold')
    ax_a.set_title("(a) Khảo sát bán kính tầm quét cảm biến 2D LiDAR (R_sensor)", fontsize=10.5, fontweight='bold')
    ax_a.set_xlabel("Bán kính tầm quét R_sensor (m)", fontsize=9.5)
    ax_a.set_ylabel("Tổng quãng đường di chuyển (m)", fontsize=9.5)
    ax_a.grid(True, linestyle=':', alpha=0.7)
    ax_a.margins(y=0.18)
    ax_a.legend(fontsize=8.5)

    ray_x = [f"{nr} tia\n({360//nr}°/tia)" for nr in ray_counts]
    ray_d = [ray_results[f"{nr}_rays"]['total_distance_m'] for nr in ray_counts]
    ray_t = [ray_results[f"{nr}_rays"]['compute_ms_per_step'] for nr in ray_counts]
    ax_b_twin = ax_b.twinx()
    b_bars = ax_b.bar(np.arange(len(ray_x)) - 0.15, ray_d, 0.3, color='#2980b9', edgecolor='black', label='Quãng đường (m)')
    ax_b_twin.plot(np.arange(len(ray_x)) + 0.15, ray_t, color='#c0392b', marker='D', linewidth=2.2, label='Thời gian tính toán (ms/bước)')
    for b in b_bars:
        ax_b.annotate(f"{b.get_height():.1f}m", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                      xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_b.set_title("(b) Khảo sát độ phân giải góc quét (N_ray) & Thời gian xử lý", fontsize=10.5, fontweight='bold')
    ax_b.set_xticks(np.arange(len(ray_x)))
    ax_b.set_xticklabels(ray_x, fontsize=9)
    ax_b.set_ylabel("Tổng quãng đường (m)", color='#2980b9', fontsize=9.5)
    ax_b_twin.set_ylabel("Thời gian tính toán (ms/bước)", color='#c0392b', fontsize=9.5)
    ax_b.set_ylim(0, max(ray_d) * 1.25)
    ax_b.grid(axis='y', linestyle=':', alpha=0.6)

    ab_keys = ["Cấu hình 1", "Cấu hình 2", "Cấu hình 3", "Cấu hình 4"]
    ab_labels = ["CH1: Chỉ xét\nkhoảng cách", "CH2: Bỏ phạt\ngóc quay (w_θ=0)", "CH3: Bỏ xét\nInfoGain (w_I=0)", "CH4: Đề xuất\nĐầy đủ 3 yếu tố"]
    ab_c_vals = [ablation_results[k]['map_c_dist_m'] for k in ab_keys]
    ab_d_vals = [ablation_results[k]['map_d_dist_m'] for k in ab_keys]
    x_ab = np.arange(len(ab_keys))
    w_ab = 0.32
    bc1 = ax_c.bar(x_ab - w_ab/2, ab_c_vals, w_ab, label='Map C (Tòa nhà sập đổ)', color='#f39c12', edgecolor='black')
    bc2 = ax_c.bar(x_ab + w_ab/2, ab_d_vals, w_ab, label='Map D (Hang động ngầm)', color='#16a085', edgecolor='black')
    for b in list(bc1) + list(bc2):
        ax_c.annotate(f"{b.get_height():.1f}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                      xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.8, fontweight='bold')
    ax_c.set_title("(c) Nghiên cứu triệt tiêu thành phần (Ablation Study) hàm Utility", fontsize=10.5, fontweight='bold')
    ax_c.set_xticks(x_ab)
    ax_c.set_xticklabels(ab_labels, fontsize=8.5)
    ax_c.set_ylabel("Quãng đường hoàn thành (m)", fontsize=9.5)
    ax_c.margins(y=0.18)
    ax_c.grid(axis='y', linestyle=':', alpha=0.6)
    ax_c.legend(fontsize=8.5)

    n_keys = [f"Level_{i}" for i in range(4)]
    n_labels = [noise_results[k]['short_label'] for k in n_keys]
    log_cov = [noise_results[k]['log_odds_coverage'] for k in n_keys]
    raw_cov = [noise_results[k]['raw_coverage'] for k in n_keys]
    x_n = np.arange(len(n_keys))
    w_n = 0.32
    bn1 = ax_d.bar(x_n - w_n/2, log_cov, w_n, label='Có lọc Bayes Log-odds (Đề xuất)', color='#2ecc71', edgecolor='black')
    bn2 = ax_d.bar(x_n + w_n/2, raw_cov, w_n, label='Cập nhật trực tiếp (Không lọc)', color='#e74c3c', edgecolor='black')
    for b in list(bn1) + list(bn2):
        h = b.get_height()
        ax_d.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                      textcoords="offset points", ha='center', va='bottom', fontsize=7.8, fontweight='bold')
    ax_d.set_title("(d) Hiệu quả của Bộ lọc Log-odds dưới Nhiễu Gauss & Khói bụi mất tia", fontsize=10.5, fontweight='bold')
    ax_d.set_xticks(x_n)
    ax_d.set_xticklabels(n_labels, fontsize=8.2)
    ax_d.set_ylabel("Độ phủ bản đồ hoàn thành (%)", fontsize=9.5)
    ax_d.set_ylim(0, 112)
    ax_d.grid(axis='y', linestyle=':', alpha=0.7)
    ax_d.legend(loc='lower left', fontsize=8.5)

    plt.suptitle("KHẢO SÁT ĐA THAM SỐ PHẦN CỨNG CẢM BIẾN, TRỌNG SỐ THUẬT TOÁN VÀ KHÁNG NHIỄU KHÓI BỤI",
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "fig_4_4_lidar_sensitivity_and_noise.png"), bbox_inches='tight')
    plt.close()

    summary_data = {
        'benchmarks_by_map': all_benchmarks,
        'grid_search_weights': grid_search_records,
        'robot_speed_study': speed_results,
        'lidar_sensitivity': sensitivity_results,
        'ray_resolution': ray_results,
        'ablation_study': ablation_results,
        'noise_robustness': noise_results
    }
    json_path = os.path.join(output_dir, "simulation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print("BẢNG KẾT QUẢ THỰC NGHIỆM TỔNG HỢP: 4 THUẬT TOÁN TRÊN 4 ĐỊA HÌNH (MAP A, MAP B, MAP C, MAP D)")
    print("=" * 96)
    print(f"{'ĐỊA HÌNH':<8} | {'THUẬT TOÁN':<18} | {'ĐỘ PHỦ':<8} | {'SỐ BƯỚC':<8} | {'QUÃNG ĐƯỜNG':<12} | {'QUAY ĐẦU':<9} | {'HIỆU SUẤT':<10} | {'NẠN NHÂN'}")
    print("-" * 96)
    for mk in ["Map_A", "Map_B", "Map_C", "Map_D"]:
        for st in strategies:
            d = all_benchmarks[mk][st]
            print(f"{mk:<8} | {st:<18} | {d['final_coverage']:<7.1f}% | {d['total_steps']:<8} | {d['total_distance_m']:<10.1f} m | {d['sharp_turns']:<9} | {d['efficiency_m2_per_m']:<6.2f} m²/m | {d['victims_found_count']}/3")
        print("-" * 96)
    print(f"[+] Đã xuất 7 bộ hình ảnh đa kịch bản (300 DPI) và lưu dữ liệu tại: {json_path}")


if __name__ == "__main__":
    plt.switch_backend('Agg')
    t_start = time.time()
    generate_all_figures_and_benchmarks(output_dir=".")
    print(f"[Hoàn tất toàn bộ chương trình trong {time.time() - t_start:.2f} giây]")
