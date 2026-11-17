"""Run every experiment and save the figures and summary.csv to results/.

    python experiments.py            # full run (about a minute)
    python experiments.py --quick    # fewer trials, for a smoke test

Every network sees the *same* noisy images under both pipelines:
  threshold: noisy -> threshold at 0.5
  fuzzy:     noisy -> fuzzification -> INT -> fuzzy rule filter -> threshold at 0.5
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import fuzzy  # noqa: E402
import models as M  # noqa: E402
from chars import CHARSET, from_bipolar, glyphs, to_bipolar  # noqa: E402
from noise import add_noise  # noqa: E402
from subset import fixed_point_count, greedy_order  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
EVAL_SEED = 11  # different from the training (2026) and tuning (123) seeds

SWEEPS = {
    "gaussian": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "saltpepper": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "faded": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
}
SCENARIO_TITLE = {
    "gaussian": "Gaussian noise",
    "saltpepper": "Salt-and-pepper noise",
    "faded": "Faded scan + noise",
}
LEVEL_LABEL = {
    "gaussian": "noise std σ",
    "saltpepper": "fraction of pixels corrupted",
    "faded": "noise std σ (paper 0.3, ink 0.6)",
}
# One representative level per scenario for the image figures and ink bars.
SHOWCASE = {"gaussian": 0.4, "saltpepper": 0.3, "faded": 0.25}

# --- Visual style (light report surface; colours validated with the dataviz checker) ---
SURFACE, INK, INK_2, MUTED, GRID_C, AXIS_C = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PIPE_COLOR = {"fuzzy": "#2a78d6", "threshold": "#eb6834"}
PIPE_STYLE = {"fuzzy": dict(ls="-", marker="o"), "threshold": dict(ls="--", marker="s")}
PIPE_NAME = {"fuzzy": "Fuzzy enhancement", "threshold": "Threshold only"}
RULE_COLOR = {"hebbian": "#1baf7a", "projection": "#6250d6"}
RAMP = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}  # ordinal blue steps
INK_CMAP = LinearSegmentedColormap.from_list("ink", [SURFACE, INK])
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.edgecolor": AXIS_C, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved results/{name}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(models: M.Models, trials: int, mem_trials: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(EVAL_SEED)
    full_clean = np.repeat(glyphs(models.chars), trials, axis=0)
    full_lab = np.repeat(np.arange(len(models.chars)), trials)
    mem_clean = np.repeat(glyphs(models.memory_chars), mem_trials, axis=0)
    mem_lab = np.repeat(np.arange(len(models.memory_chars)), mem_trials)
    full_ink = full_clean >= 0.5

    acc_rows, img_rows = [], []
    for kind, levels in SWEEPS.items():
        for level in levels:
            t0 = time.time()
            noisy_full = add_noise(full_clean, kind, level, rng)
            noisy_mem = add_noise(mem_clean, kind, level, rng)
            for pipe in M.PIPELINES:
                xf = M.preprocess(noisy_full, pipe)
                xm = M.preprocess(noisy_mem, pipe)
                for net, pred in M.predict_full(models, xf, pipe).items():
                    acc_rows.append(dict(scenario=kind, level=level, network=net, pipeline=pipe,
                                         chars=len(models.chars), accuracy=float((pred == full_lab).mean()), n=len(full_lab)))
                for net, pred in M.predict_memory(models, xm).items():
                    acc_rows.append(dict(scenario=kind, level=level, network=net, pipeline=pipe,
                                         chars=len(models.memory_chars), accuracy=float((pred == mem_lab).mean()), n=len(mem_lab)))
                out_ink = from_bipolar(xf) > 0
                img_rows.append(dict(
                    scenario=kind, level=level, pipeline=pipe,
                    pixel_error=float((out_ink != full_ink).mean()),
                    ink_lost=float((full_ink & ~out_ink).sum(axis=(1, 2)).mean()),
                    ink_added=float((~full_ink & out_ink).sum(axis=(1, 2)).mean()),
                ))
            print(f"  {kind:>10} {level:<5} done in {time.time() - t0:.1f}s")
    return acc_rows, img_rows


def write_csv(rows: list[dict], name: str) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved results/{name}")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
NETWORK_ORDER = ["Perceptron", "Hamming+Maxnet", "Hopfield (projection)", "Hopfield (Hebbian)", "BAM"]


def plot_accuracy(rows: list[dict], memory_chars: str) -> None:
    for kind in SWEEPS:
        fig, axes = plt.subplots(1, len(NETWORK_ORDER), figsize=(16, 3.5), sharey=True)
        for ax, net in zip(axes, NETWORK_ORDER):
            for pipe in ("threshold", "fuzzy"):
                pts = sorted((r["level"], r["accuracy"]) for r in rows
                             if r["scenario"] == kind and r["network"] == net and r["pipeline"] == pipe)
                x, y = zip(*pts)
                ax.plot(x, y, color=PIPE_COLOR[pipe], label=PIPE_NAME[pipe], **PIPE_STYLE[pipe])
            n_chars = len(memory_chars) if net in ("Hopfield (Hebbian)", "BAM") else len(CHARSET)
            ax.set_title(f"{net}\n{n_chars} characters", fontsize=10)
            ax.set_xlabel(LEVEL_LABEL[kind], fontsize=9)
            ax.set_ylim(-0.02, 1.04)
        axes[0].set_ylabel("recognition accuracy")
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
        fig.suptitle(f"Recognition accuracy: {SCENARIO_TITLE[kind]}", y=1.2, fontsize=12, fontweight="semibold")
        save(fig, f"accuracy_{kind}.png")


def plot_pixel_error(img_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, kind in zip(axes, SWEEPS):
        for pipe in ("threshold", "fuzzy"):
            pts = sorted((r["level"], r["pixel_error"]) for r in img_rows if r["scenario"] == kind and r["pipeline"] == pipe)
            x, y = zip(*pts)
            ax.plot(x, y, color=PIPE_COLOR[pipe], label=PIPE_NAME[pipe], **PIPE_STYLE[pipe])
            ax.annotate(f"{y[-1]:.0%}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(SCENARIO_TITLE[kind])
        ax.set_xlabel(LEVEL_LABEL[kind], fontsize=9)
    axes[0].set_ylabel("pixels wrong after binarising")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(loc="upper left")
    fig.suptitle("Image quality: pixel error rate", y=1.03, fontsize=12, fontweight="semibold")
    save(fig, "pixel_error.png")


def plot_ink_changes(img_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    width = 0.36
    for ax, kind in zip(axes, SWEEPS):
        level = SHOWCASE[kind]
        for j, pipe in enumerate(("threshold", "fuzzy")):
            r = next(r for r in img_rows if r["scenario"] == kind and r["level"] == level and r["pipeline"] == pipe)
            vals = [r["ink_lost"], r["ink_added"]]
            xs = np.arange(2) + (j - 0.5) * (width + 0.02)
            bars = ax.bar(xs, vals, width=width, color=PIPE_COLOR[pipe], label=PIPE_NAME[pipe])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
        ax.set_xticks([0, 1], ["ink lost\n(holes in strokes)", "ink added\n(specks on paper)"])
        ax.set_title(f"{SCENARIO_TITLE[kind]} ({level})")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("pixels per 24×24 image")
    axes[0].legend(loc="upper left")
    fig.suptitle("Where the errors are: stroke pixels lost vs background pixels added", y=1.04, fontsize=12, fontweight="semibold")
    save(fig, "ink_changes.png")


def plot_membership_functions() -> None:
    x = np.linspace(0, 1, 401)
    sets = [("Input 1: pixel intensity p", fuzzy.INPUT_MF),
            ("Input 2: neighbour mean m", fuzzy.NEIGHBOUR_MF),
            ("Output: filtered intensity", fuzzy.OUTPUT_MF)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.1), sharey=True)
    for ax, (title, mf) in zip(axes, sets):
        for lbl in fuzzy.LABELS:
            y = fuzzy.trapmf(x, *mf[lbl])
            ax.plot(x, y, color=RAMP[lbl], label=lbl)
            peak = x[y >= 0.999].mean()
            ax.annotate(lbl, (peak, 1.0), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("value (0 = paper, 1 = ink)", fontsize=9)
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("membership degree μ")
    fig.suptitle("Fuzzification: membership functions", y=1.04, fontsize=12, fontweight="semibold")
    save(fig, "membership_functions.png")


def plot_rule_surface() -> None:
    P, Mm, Z = fuzzy.control_surface(61)
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(P, Mm, Z, cmap=BLUES, linewidth=0, antialiased=True)
    ax.set_xlabel("pixel p", color=INK_2)
    ax.set_ylabel("neighbour mean m", color=INK_2)
    ax.set_zlabel("output", color=INK_2)
    ax.set_title("Control surface of the fuzzy filter")
    ax.view_init(elev=28, azim=-130)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.99, 0.99, 0.98, 1.0))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Rule base (IF p AND m THEN output)")
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)
    ax2.set_aspect("equal")
    ax2.grid(False)
    shade = {"low": "#e9f1fc", "medium": "#9ec5f4", "high": "#256abf"}
    for i, p in enumerate(fuzzy.LABELS):
        for j, m in enumerate(fuzzy.LABELS):
            out = fuzzy.RULES[p][m]
            ax2.add_patch(plt.Rectangle((j + 0.03, 2 - i + 0.03), 0.94, 0.94, color=shade[out]))
            ax2.text(j + 0.5, 2.5 - i, out, ha="center", va="center", fontsize=11,
                     color="#ffffff" if out == "high" else INK)
    ax2.set_xticks([0.5, 1.5, 2.5], [f"m {lbl}" for lbl in fuzzy.LABELS])
    ax2.set_yticks([2.5, 1.5, 0.5], [f"p {lbl}" for lbl in fuzzy.LABELS])
    ax2.tick_params(length=0)
    for side in ("left", "bottom"):
        ax2.spines[side].set_visible(False)
    save(fig, "rule_surface.png")


def plot_fuzzy_stages() -> None:
    rng = np.random.default_rng(EVAL_SEED + 1)
    char = "R"
    clean = glyphs(char)[0]
    cols = ["clean", "noisy input", "fuzzified", "after INT", "after rule filter", "threshold only", "fuzzy → binary"]
    fig, axes = plt.subplots(3, len(cols), figsize=(12, 5.6))
    for row, kind in enumerate(SWEEPS):
        noisy = add_noise(clean, kind, SHOWCASE[kind], rng)
        st = fuzzy.enhance_stages(noisy)
        imgs = [clean, noisy, st["fuzzified"], st["intensified"], st["filtered"],
                (noisy >= 0.5).astype(float), (st["filtered"] >= 0.5).astype(float)]
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img, cmap=INK_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
        axes[row, 0].set_ylabel(f"{SCENARIO_TITLE[kind]}\n({SHOWCASE[kind]})", fontsize=9)
    fig.suptitle("Fuzzy enhancement, stage by stage", y=0.99, fontsize=12, fontweight="semibold")
    save(fig, "fuzzy_stages.png")


def plot_recall_samples(models: M.Models) -> None:
    rng = np.random.default_rng(EVAL_SEED + 2)
    chars = "AK7SG3"
    level = SHOWCASE["faded"]
    fig, axes = plt.subplots(len(chars), 6, figsize=(8.6, 1.5 * len(chars)))
    cols = ["clean", "faded + noisy", "threshold input", "→ recall", "fuzzy input", "→ recall"]
    for row, ch in enumerate(chars):
        clean = glyphs(ch)[0]
        noisy = add_noise(clean, "faded", level, rng)
        xt, xf = M.preprocess(noisy, "threshold"), M.preprocess(noisy, "fuzzy")
        rt = models.hopfield.recall(xt, seed=row)
        rf = models.hopfield.recall(xf, seed=row)
        imgs = [clean, noisy, from_bipolar(xt), from_bipolar(rt), from_bipolar(xf), from_bipolar(rf)]
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img, cmap=INK_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
        for col, rec in ((3, rt), (5, rf)):
            idx = models.hopfield.match(rec)[0]
            ok = idx == CHARSET.index(ch)
            label = CHARSET[idx] if idx >= 0 else "spurious"
            axes[row, col].set_xlabel(("✓ " if ok else "✗ ") + label, fontsize=9, color=INK if ok else "#d03b3b")
    fig.suptitle(f"Hopfield (projection rule) recall from a faded, noisy scan (σ = {level})",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, "recall_samples.png")


def plot_hopfield_energy(models: M.Models) -> None:
    rng = np.random.default_rng(EVAL_SEED + 3)
    noisy = add_noise(glyphs("K")[0], "gaussian", 0.6, rng)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for pipe in ("threshold", "fuzzy"):
        _, hist = models.hopfield.recall(M.preprocess(noisy, pipe), seed=0, record=True)
        energy = [e for _, e in hist]
        ax.plot(range(len(energy)), energy, color=PIPE_COLOR[pipe], label=PIPE_NAME[pipe], ls=PIPE_STYLE[pipe]["ls"])
        ax.annotate(f"{energy[-1]:.0f}", (len(energy) - 1, energy[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("single-neuron updates (asynchronous)")
    ax.set_ylabel("energy E = −½ sᵀWs")
    ax.set_title("Hopfield energy never increases during recall (letter K, σ = 0.6)")
    ax.legend()
    save(fig, "hopfield_energy.png")


def plot_hopfield_capacity() -> None:
    order = greedy_order()
    ks = list(range(1, len(order) + 1))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for rule in ("hebbian", "projection"):
        frac = [fixed_point_count(order[:k], rule) / k for k in ks]
        ax.plot(ks, frac, color=RULE_COLOR[rule], marker="o", label=f"{rule.capitalize()} rule")
    ax.set_xlabel("number of characters stored")
    ax.set_ylabel("stored patterns that are stable")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Hopfield capacity on 24×24 glyphs")
    ax.legend(loc="center right")
    save(fig, "hopfield_capacity.png")


def plot_maxnet(models: M.Models) -> None:
    rng = np.random.default_rng(EVAL_SEED + 4)
    x = M.preprocess(add_noise(glyphs("8")[0], "gaussian", 0.7, rng), "fuzzy")
    a = models.hamming.agreements(x)[0]
    net = models.hamming.maxnet
    A = np.maximum(a - np.arange(net.m) * 1e-9 * (a.max() + 1.0), 0.0)
    history = [A.copy()]
    while (A > 0).sum() > 1:
        A = np.maximum(A - net.eps * (A.sum() - A), 0.0)
        history.append(A.copy())
    H = np.array(history)
    winner = int(H[-1].argmax())
    runner = int(np.argsort(a)[-2])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for j in range(net.m):
        if j not in (winner, runner):
            ax.plot(H[:, j], color=AXIS_C, lw=1.0)
    ax.plot(H[:, runner], color=PIPE_COLOR["threshold"], label=f"runner-up '{CHARSET[runner]}'")
    ax.plot(H[:, winner], color=PIPE_COLOR["fuzzy"], label=f"winner '{CHARSET[winner]}'")
    ax.set_xlabel("Maxnet iteration")
    ax.set_ylabel("activation")
    ax.set_title(f"Maxnet picks one winner from 36 Hamming scores ({len(H) - 1} iterations)")
    ax.legend()
    save(fig, "maxnet.png")


def print_headlines(acc_rows: list[dict], img_rows: list[dict]) -> None:
    print("\nHeadline numbers (threshold -> fuzzy):")
    for kind, level in SHOWCASE.items():
        r = {p: next(x for x in img_rows if x["scenario"] == kind and x["level"] == level and x["pipeline"] == p)
             for p in ("threshold", "fuzzy")}
        print(f"  {SCENARIO_TITLE[kind]} {level}: pixel error {r['threshold']['pixel_error']:.1%} -> {r['fuzzy']['pixel_error']:.1%}")
    for net in NETWORK_ORDER:
        for kind in SWEEPS:
            vals = {p: [x["accuracy"] for x in acc_rows if x["scenario"] == kind and x["network"] == net and x["pipeline"] == p]
                    for p in ("threshold", "fuzzy")}
            print(f"  {net:<22} {kind:<10} mean accuracy over sweep {np.mean(vals['threshold']):.1%} -> {np.mean(vals['fuzzy']):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer trials (smoke test)")
    args = parser.parse_args()
    trials, mem_trials = (3, 10) if args.quick else (20, 100)

    OUT.mkdir(exist_ok=True)
    style()
    t0 = time.time()
    print("Training / storing networks ...")
    models = M.build()
    print(f"  memory set for Hebbian Hopfield + BAM: {models.memory_chars}")
    print("Evaluating ...")
    acc_rows, img_rows = evaluate(models, trials, mem_trials)
    write_csv(acc_rows, "summary.csv")
    write_csv(img_rows, "image_quality.csv")
    print("Plotting ...")
    plot_accuracy(acc_rows, models.memory_chars)
    plot_pixel_error(img_rows)
    plot_ink_changes(img_rows)
    plot_membership_functions()
    plot_rule_surface()
    plot_fuzzy_stages()
    plot_recall_samples(models)
    plot_hopfield_energy(models)
    plot_hopfield_capacity()
    plot_maxnet(models)
    print_headlines(acc_rows, img_rows)
    print(f"\nDone in {time.time() - t0:.0f}s. Figures are in {OUT}")


if __name__ == "__main__":
    main()
