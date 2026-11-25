"""
================================================================================
MONTE CARLO GENERALIZATION BENCHMARK ON N = 30 UNSEEN RANDOM DISASTER MAPS
Compares Random Walk, Classical Nearest Frontier, and Proposed Log-Utility
Outputs:
  - random_maps_benchmark_30.json
  - fig_random_maps_monte_carlo_30.png (300 DPI publication-ready figure)
================================================================================
"""

import os
import json
import time
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demo_live_interactive as dli

N_MAPS = 30
SEEDS = [1001 + i for i in range(N_MAPS)]
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_maps_benchmark_30.json")
OUT_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_random_maps_monte_carlo_30.png")


def run_benchmark():
    t0 = time.perf_counter()
    print(f"=== Running Monte Carlo Benchmark on {N_MAPS} Unseen Random Disaster Maps ===")

    per_map = []
    for idx, sd in enumerate(SEEDS, 1):
        grid, sp, vc, actual_seed = dli.create_random_disaster_map(seed=sd)
        rec = {"map_id": idx, "seed": actual_seed}
        for strat in ("random", "nearest", "utility"):
            tr = dli.run_algorithm_trace(grid, sp, vc, strategy=strat, max_steps=320, seed=sd)
            last = tr[-1]
            rec[strat] = {
                "coverage_pct": round(float(last["cov"]), 2),
                "distance_m": round(float(last["dist_m"]), 2),
                "steps": int(last["step"]),
                "sharp_turns": int(last["sharp_turns"]),
                "ee_m2_per_m": round(float(last["ee"]), 2),
                "victims_found": int(sum(last["victims"])),
                "all_victims_step": int(last["v_all_step"]) if last["v_all_step"] is not None else None,
            }
        sav_pct = (1.0 - rec["utility"]["distance_m"] / max(0.1, rec["nearest"]["distance_m"])) * 100.0
        rec["distance_saving_vs_nearest_pct"] = round(sav_pct, 2)
        per_map.append(rec)
        print(
            f"[{idx:02d}/{N_MAPS}] seed={actual_seed} | "
            f"Rand: {rec['random']['coverage_pct']:5.1f}%/{rec['random']['distance_m']:5.1f}m | "
            f"Near: {rec['nearest']['coverage_pct']:5.1f}%/{rec['nearest']['distance_m']:5.1f}m ({rec['nearest']['sharp_turns']:2d} turns) | "
            f"Prop: {rec['utility']['coverage_pct']:5.1f}%/{rec['utility']['distance_m']:5.1f}m ({rec['utility']['sharp_turns']:2d} turns) | "
            f"Saving: {sav_pct:+5.1f}%",
            flush=True,
        )

    # Aggregate statistics
    def agg(strat, key):
        vals = [r[strat][key] for r in per_map if r[strat][key] is not None]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    d_rand = np.array([r["random"]["distance_m"] for r in per_map])
    d_near = np.array([r["nearest"]["distance_m"] for r in per_map])
    d_util = np.array([r["utility"]["distance_m"] for r in per_map])
    t_near = np.array([r["nearest"]["sharp_turns"] for r in per_map])
    t_util = np.array([r["utility"]["sharp_turns"] for r in per_map])
    c_rand = np.array([r["random"]["coverage_pct"] for r in per_map])
    c_near = np.array([r["nearest"]["coverage_pct"] for r in per_map])
    c_util = np.array([r["utility"]["coverage_pct"] for r in per_map])
    savings = np.array([r["distance_saving_vs_nearest_pct"] for r in per_map])

    t_stat, p_ttest = stats.ttest_rel(d_near, d_util)
    try:
        w_stat, p_wilcox = stats.wilcoxon(d_near, d_util)
    except Exception:
        w_stat, p_wilcox = 0.0, float(p_ttest)

    win_dist = int(np.sum(d_util < d_near))
    win_turns = int(np.sum(t_util <= t_near))

    summary = {
        "n_maps": N_MAPS,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "win_rate_distance_vs_nearest": f"{win_dist}/{N_MAPS} ({win_dist / N_MAPS * 100:.1f}%)",
        "win_rate_turns_vs_nearest": f"{win_turns}/{N_MAPS} ({win_turns / N_MAPS * 100:.1f}%)",
        "mean_distance_saving_pct": round(float(np.mean(savings)), 2),
        "std_distance_saving_pct": round(float(np.std(savings, ddof=1)), 2),
        "paired_ttest_t": round(float(t_stat), 4),
        "paired_ttest_p": float(p_ttest),
        "wilcoxon_p": float(p_wilcox),
        "algorithms": {},
    }

    for strat in ("random", "nearest", "utility"):
        cov_m, cov_s = agg(strat, "coverage_pct")
        dst_m, dst_s = agg(strat, "distance_m")
        trn_m, trn_s = agg(strat, "sharp_turns")
        ee_m, ee_s = agg(strat, "ee_m2_per_m")
        v_steps = [r[strat]["all_victims_step"] for r in per_map if r[strat]["all_victims_step"] is not None]
        v_full = sum(1 for r in per_map if r[strat]["victims_found"] == 3)
        summary["algorithms"][strat] = {
            "coverage_mean_std": f"{cov_m:.1f} ± {cov_s:.1f}",
            "distance_mean_std": f"{dst_m:.1f} ± {dst_s:.1f}",
            "sharp_turns_mean_std": f"{trn_m:.1f} ± {trn_s:.1f}",
            "ee_mean_std": f"{ee_m:.2f} ± {ee_s:.2f}",
            "all_3_victims_rate": f"{v_full}/{N_MAPS} ({v_full / N_MAPS * 100:.1f}%)",
            "all_victims_step_mean": round(float(np.mean(v_steps)), 1) if v_steps else None,
        }

    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump({"summary": summary, "per_map": per_map}, jf, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # Publication-ready 3-panel figure (300 DPI)
    # --------------------------------------------------------------------------
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=300, facecolor="white")

    # Panel (a): Total Distance Box Plot + Individual Map Points
    ax = axes[0]
    bp = ax.boxplot(
        [d_rand, d_near, d_util],
        tick_labels=["Random\nWalk", "Nearest\nFrontier", "Proposed\nLog-Utility"],
        patch_artist=True, widths=0.52,
        medianprops=dict(color="#111827", linewidth=2.0),
        whiskerprops=dict(color="#4b5563", linewidth=1.3),
        capprops=dict(color="#4b5563", linewidth=1.3),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    cols = ["#ff7675", "#fdcb6e", "#00b894"]
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1f2937")
    rng_j = np.random.RandomState(0)
    for xi, arr in enumerate([d_rand, d_near, d_util], 1):
        jx = xi + rng_j.uniform(-0.14, 0.14, size=len(arr))
        ax.scatter(jx, arr, s=16, color="#1f2937", alpha=0.45, zorder=4)
    ax.set_ylabel("Total Travel Distance (m)", fontsize=10.5, fontweight="bold")
    ax.set_title(f"(a) Travel Distance across {N_MAPS} Random Maps\n(Paired t-test: p = {p_ttest:.2e})",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel (b): Per-Map Distance Saving % vs Nearest Frontier
    ax = axes[1]
    x_ids = np.arange(1, N_MAPS + 1)
    bar_cols = ["#00b894" if v > 0 else "#d63031" for v in savings]
    ax.bar(x_ids, savings, color=bar_cols, edgecolor="#1f2937", linewidth=0.5, alpha=0.88)
    ax.axhline(0, color="#1f2937", linewidth=1.0)
    ax.axhline(np.mean(savings), color="#0984e3", linestyle="--", linewidth=1.8,
               label=f"Mean Saving: {np.mean(savings):+.1f}%")
    ax.set_xlabel("Unseen Random Map Index (1..30)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Distance Reduction vs. Nearest (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(f"(b) Per-Map Distance Reduction\n(Win Rate: {win_dist}/{N_MAPS} maps = {win_dist/N_MAPS*100:.1f}%)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel (c): Sharp Turns (>90 deg) & Coverage Summary
    ax = axes[2]
    bp2 = ax.boxplot(
        [
            [r["random"]["sharp_turns"] for r in per_map],
            t_near,
            t_util,
        ],
        tick_labels=["Random\nWalk", "Nearest\nFrontier", "Proposed\nLog-Utility"],
        patch_artist=True, widths=0.52,
        medianprops=dict(color="#111827", linewidth=2.0),
    )
    for patch, col in zip(bp2["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1f2937")
    ax.set_ylabel("Sharp Reversals / Turns > 90° (count)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        f"(c) Sharp Turns (>90°) across {N_MAPS} Random Maps\n"
        f"(Proposed: {np.mean(t_util):.1f} vs. Nearest: {np.mean(t_near):.1f})",
        fontsize=10.5, fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        f"Monte Carlo Generalization Benchmark on N = {N_MAPS} Procedurally Generated Unseen Disaster Maps",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_FIG, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)
    print(f"Saved JSON : {OUT_JSON}")
    print(f"Saved PNG  : {OUT_FIG}")


if __name__ == "__main__":
    run_benchmark()
