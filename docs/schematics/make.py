import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
from matplotlib.colors import ListedColormap
import os
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
TC = ["#3b6fb6", "#e08a2e", "#4a9a5b", "#8e5bb5"]
TN = ["A", "B", "C", "D"]
HEAD = {"Input": "#eef2f8", "What it does": "#fbf1e4", "Output": "#eaf5ec"}
INK = "#2b2b2b"; MUTED = "#6b6b6b"

def frame(title, n=3):
    fig = plt.figure(figsize=(13, 4.4))
    if title: fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.985)
    xs = [0.01, 0.355, 0.70]; w = 0.29
    axes = []
    for x, lab in zip(xs, ["Input", "What it does", "Output"]):
        bg = fig.add_axes([x, 0.03, w, 0.86]); bg.set_axis_off()
        bg.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc=HEAD[lab], ec="none", transform=bg.transAxes))
        bg.text(0.03, 0.955, lab, fontsize=11, fontweight="bold", color=INK, va="top", transform=bg.transAxes)
        axes.append((x, w))
    for x in [0.302, 0.647]:
        a = fig.add_axes([x, 0.42, 0.05, 0.1]); a.set_axis_off(); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.add_patch(FancyArrowPatch((0.1, 0.5), (0.9, 0.5), arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#555"))
    return fig, axes

def sub(fig, panel, box):
    x, w = panel; l, b, ww, h = box
    return fig.add_axes([x + l * w, 0.03 + b * 0.86, ww * w, h * 0.86])

def mat(ax, M, cmap="RdBu_r", vmin=None, vmax=None, title=None, xl=None, yl=None):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#999"); s.set_linewidth(0.8)
    if title: ax.set_title(title, fontsize=9, color=INK, pad=3)
    if xl: ax.set_xlabel(xl, fontsize=8, color=MUTED, labelpad=2)
    if yl: ax.set_ylabel(yl, fontsize=8, color=MUTED, labelpad=2)

def txt(fig, panel, x, y, s, **k):
    px, w = panel
    k.setdefault("fontsize", 8.5); k.setdefault("color", INK)
    fig.text(px + x * w, 0.03 + y * 0.86, s, **k)

def blocky(n, sizes, strength=0.55, noise=0.12):
    lab = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    C = np.where(lab[:, None] == lab[None, :], strength, 0.08) + rng.normal(0, noise, (n, n))
    C = (C + C.T) / 2; np.fill_diagonal(C, 1)
    return np.clip(C, -0.2, 1), lab

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white"); plt.close(fig)

# ---------------- Step 1 ----------------
fig, P = frame(None)
FS = 11
donors = np.repeat([0, 1, 2, 3], [7, 7, 6, 7])
dc = ["#9e9e9e", "#bdbdbd", "#d64545", "#9e9e9e"]
counts = rng.poisson(0.6, (14, len(donors))) * (rng.random((14, len(donors))) < 0.45)
ax = sub(fig, P[0], (0.10, 0.24, 0.62, 0.44)); mat(ax, counts, cmap="Greys", vmin=0, vmax=3)
ax.set_xlabel("cells (all donors)", fontsize=FS - 1, color=MUTED); ax.set_ylabel("genes", fontsize=FS - 1, color=MUTED)
a2 = sub(fig, P[0], (0.10, 0.69, 0.62, 0.05)); a2.imshow(donors[None], cmap=ListedColormap(dc), aspect="auto"); a2.set_axis_off()
txt(fig, P[0], 0.10, 0.82, "raw counts", fontsize=FS, fontweight="bold")
txt(fig, P[0], 0.10, 0.76, "donor:", fontsize=FS - 1, color=MUTED)
txt(fig, P[0], 0.40, 0.76, "held-out", fontsize=FS - 1, color="#d64545", fontweight="bold")
mt = sub(fig, P[0], (0.77, 0.24, 0.20, 0.44)); mt.set_axis_off(); mt.set_xlim(0, 1); mt.set_ylim(0, 1)
for r in range(6):
    mt.add_patch(Rectangle((0, 0.84 - r * 0.15), 1, 0.13, fc="white", ec="#bbb"))
mt.text(0.5, 0.905, "cell id", ha="center", va="center", fontsize=FS - 2, fontweight="bold")
txt(fig, P[0], 0.77, 0.72, "metadata", fontsize=FS, fontweight="bold")
txt(fig, P[0], 0.10, 0.08, "metadata: one row per cell\n(cell id, donor, cell type)", fontsize=FS - 1, color=MUTED)
# does
x0, w = P[1]
ax = sub(fig, P[1], (0.06, 0.50, 0.52, 0.26)); mat(ax, counts[:, donors != 2], cmap="Greys", vmin=0, vmax=3)
txt(fig, P[1], 0.06, 0.79, "reference = other donors", fontsize=FS, fontweight="bold")
ax = sub(fig, P[1], (0.72, 0.50, 0.20, 0.26)); mat(ax, counts[:, donors == 2], cmap="Greys", vmin=0, vmax=3)
txt(fig, P[1], 0.72, 0.79, "query", fontsize=FS, fontweight="bold", color="#d64545")
bx = sub(fig, P[1], (0.04, 0.06, 0.92, 0.32)); bx.set_axis_off(); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.add_patch(FancyBboxPatch((0.02, 0.35), 0.56, 0.55, boxstyle="round,pad=0.02", fc="white", ec="#e08a2e", lw=1.5))
bx.text(0.30, 0.625, "fit SCTransform\n(keep 3000 HVGs)", ha="center", va="center", fontsize=FS)
bx.add_patch(FancyBboxPatch((0.66, 0.35), 0.32, 0.55, boxstyle="round,pad=0.02", fc="white", ec="#e08a2e", lw=1.5))
bx.text(0.82, 0.625, "apply the\nsame model", ha="center", va="center", fontsize=FS)
bx.add_patch(FancyArrowPatch((0.60, 0.625), (0.65, 0.625), arrowstyle="-|>", mutation_scale=14, color="#e08a2e", lw=1.5))
bx.text(0.5, 0.05, "both end up on one expression scale", ha="center", fontsize=FS - 1, color=MUTED)
fig.patches.append(FancyArrowPatch((x0 + 0.32 * w, 0.03 + 0.49 * 0.86), (x0 + 0.32 * w, 0.03 + 0.36 * 0.86), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="#e08a2e", lw=1.5))
fig.patches.append(FancyArrowPatch((x0 + 0.82 * w, 0.03 + 0.36 * 0.86), (x0 + 0.82 * w, 0.03 + 0.49 * 0.86), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="#e08a2e", lw=1.5))
# output
ax = sub(fig, P[2], (0.10, 0.24, 0.54, 0.46)); mat(ax, rng.normal(0, 1, (14, 20)), vmin=-2.5, vmax=2.5)
ax.set_xlabel("reference cells", fontsize=FS - 1, color=MUTED); ax.set_ylabel("3000 HVGs", fontsize=FS - 1, color=MUTED)
txt(fig, P[2], 0.10, 0.74, "reference SCT", fontsize=FS, fontweight="bold")
ax = sub(fig, P[2], (0.74, 0.24, 0.18, 0.46)); mat(ax, rng.normal(0, 1, (14, 7)), vmin=-2.5, vmax=2.5)
ax.set_xlabel("query cells", fontsize=FS - 1, color=MUTED)
txt(fig, P[2], 0.74, 0.74, "query SCT", fontsize=FS, fontweight="bold", color="#d64545")
txt(fig, P[2], 0.10, 0.08, "same genes, same scale\nred = high, blue = low", fontsize=FS - 1, color=MUTED)
save(fig, "step1.png")

# ---------------- Step 2 ----------------
fig, P = frame(None)
FS = 11
ref_lab = np.repeat([0, 1, 2, 3], [6, 6, 6, 6])
prof = rng.normal(0, 1, (4, 14))
R = (prof[ref_lab] + rng.normal(0, 0.7, (len(ref_lab), 14))).T
qlab = np.array([1, 0, 2, 1, 3, 0])
Q = (prof[qlab] + rng.normal(0, 0.7, (6, 14))).T
ax = sub(fig, P[0], (0.10, 0.22, 0.52, 0.46)); mat(ax, R, vmin=-2.5, vmax=2.5)
ax.set_xlabel("reference cells", fontsize=FS - 1, color=MUTED); ax.set_ylabel("genes (HVGs)", fontsize=FS - 1, color=MUTED)
a2 = sub(fig, P[0], (0.10, 0.69, 0.52, 0.05)); a2.imshow(ref_lab[None], cmap=ListedColormap(TC), aspect="auto"); a2.set_axis_off()
for k in range(4): txt(fig, P[0], 0.10 + (k + 0.5) * 0.13 - 0.015, 0.76, TN[k], fontsize=FS, color=TC[k], fontweight="bold")
txt(fig, P[0], 0.10, 0.82, "reference (labeled)", fontsize=FS, fontweight="bold")
ax = sub(fig, P[0], (0.72, 0.22, 0.18, 0.46)); mat(ax, Q, vmin=-2.5, vmax=2.5)
ax.set_xlabel("query cells", fontsize=FS - 1, color=MUTED)
txt(fig, P[0], 0.72, 0.72, "query", fontsize=FS, fontweight="bold")
txt(fig, P[0], 0.10, 0.08, "SCT expression from the data structure step", fontsize=FS - 1, color=MUTED)
# does
bx = sub(fig, P[1], (0.04, 0.06, 0.92, 0.80)); bx.set_axis_off(); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.text(0.02, 0.97, "For each query cell:", fontsize=FS, fontweight="bold", va="top")
bx.add_patch(Rectangle((0.04, 0.50), 0.10, 0.36, fc=TC[1], alpha=0.25, ec="#555"))
bx.text(0.09, 0.68, "cell\n i", ha="center", va="center", fontsize=FS)
rs = [0.41, 0.62, 0.30, 0.25]
for k in range(4):
    y = 0.84 - k * 0.11
    bx.add_patch(FancyArrowPatch((0.16, 0.68), (0.46, y), arrowstyle="-|>", mutation_scale=10, color="#999", lw=1))
    bx.add_patch(Rectangle((0.47, y - 0.04), 0.17, 0.08, fc=TC[k], ec="none"))
    bx.text(0.555, y, "type " + TN[k], ha="center", va="center", fontsize=FS - 1, color="white", fontweight="bold")
    bx.text(0.67, y, f"mean r = {rs[k]:.2f}", va="center", fontsize=FS - 1, fontweight="bold" if k == 1 else "normal")
bx.text(0.02, 0.36, "1. mean Pearson r to every labeled cell\n    of each type  →  one score per type", fontsize=FS - 1, va="top")
bx.text(0.02, 0.17, "2. Pearson r to every other query cell\n    →  query correlation matrix", fontsize=FS - 1, va="top")
# output
C, _ = blocky(10, [4, 3, 3]); perm = rng.permutation(10)
ax = sub(fig, P[2], (0.08, 0.24, 0.44, 0.46)); mat(ax, C[np.ix_(perm, perm)], cmap="viridis", vmin=0, vmax=1)
ax.set_xlabel("query cells", fontsize=FS - 1, color=MUTED); ax.set_ylabel("query cells", fontsize=FS - 1, color=MUTED)
txt(fig, P[2], 0.08, 0.74, "query correlation\n(n × n) → grouping", fontsize=FS, fontweight="bold")
S = 0.3 + 0.3 * (qlab[:, None] == np.arange(4)[None]) + rng.normal(0, 0.03, (6, 4))
ax = sub(fig, P[2], (0.64, 0.24, 0.30, 0.46)); mat(ax, S, cmap="viridis", vmin=0.2, vmax=0.7)
ax.set_xticks(range(4)); ax.set_xticklabels(TN, fontsize=FS - 1, fontweight="bold")
for t, c in zip(ax.get_xticklabels(), TC): t.set_color(c)
ax.set_ylabel("query cells", fontsize=FS - 1, color=MUTED)
txt(fig, P[2], 0.64, 0.74, "type scores\n(n × K)", fontsize=FS, fontweight="bold")
txt(fig, P[2], 0.08, 0.08, "brighter = more similar", fontsize=FS - 1, color=MUTED)
save(fig, "step2.png")

# ---------------- Step 3 ----------------
fig, P = frame(None)
s = np.array([0.62, 0.58, 0.30])
ax = sub(fig, P[0], (0.12, 0.30, 0.38, 0.45))
ax.bar(range(3), s, color=TC[:3], width=0.65)
for i, v in enumerate(s): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(TN[:3]); ax.set_ylim(0, 0.8); ax.set_yticks([])
ax.set_title("s for one query cell", fontsize=9); [ax.spines[x].set_visible(False) for x in ["top", "right", "left"]]
cal = sub(fig, P[0], (0.60, 0.30, 0.30, 0.45))
CS = 0.3 + 0.25 * (np.repeat(np.arange(4), 3)[:, None] == np.arange(4)[None]) + rng.normal(0, 0.05, (12, 4))
mat(cal, CS, cmap="viridis", title="calibration cells", xl="types")
lab_ax = sub(fig, P[0], (0.92, 0.30, 0.04, 0.45)); lab_ax.imshow(np.repeat(np.arange(4), 3)[:, None], cmap=ListedColormap(TC), aspect="auto"); lab_ax.set_axis_off()
txt(fig, P[0], 0.12, 0.10, "scores differ by only a few hundredths;\nheld-out reference donors have true labels", fontsize=8, color=MUTED)
# does
ax = sub(fig, P[1], (0.12, 0.40, 0.80, 0.38))
logT = np.linspace(np.log(0.005), np.log(1.5), 200)
nll = 0.55 * (logT - np.log(0.05)) ** 2 / 4 + 0.45 + 0.02 * np.exp(-(logT - np.log(0.05)) * 1.2)
ax.plot(np.exp(logT), nll, color="#e08a2e", lw=2); ax.set_xscale("log")
ax.axvline(0.05, ls="--", color=MUTED, lw=1); ax.text(0.055, nll.max() * 0.9, "fitted T", fontsize=8, color=MUTED)
ax.set_xlabel("temperature T (log)", fontsize=8); ax.set_ylabel("NLL on\nreference", fontsize=8); ax.set_yticks([]); ax.tick_params(labelsize=7)
[ax.spines[x].set_visible(False) for x in ["top", "right"]]
txt(fig, P[1], 0.08, 0.14, r"$p^0_{ik} = \mathrm{softmax}_k(s_{ik}/T)$", fontsize=10)
txt(fig, P[1], 0.08, 0.05, "pick T that best predicts known reference labels", fontsize=8, color=MUTED)
# output
for j, (T, lbl) in enumerate([(1, "T = 1"), (0.05, "e.g. fitted T = 0.05"), (0.02, "T = 0.02")]):
    p = np.exp(s / T); p /= p.sum()
    ax = sub(fig, P[2], (0.07 + j * 0.31, 0.32, 0.25, 0.43))
    ax.bar(range(3), p, color=TC[:3], width=0.65, alpha=1 if j == 1 else 0.45)
    for i, v in enumerate(p): ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.1); ax.set_yticks([]); ax.set_xticks(range(3)); ax.set_xticklabels(TN[:3], fontsize=8)
    ax.set_title(lbl, fontsize=8.5, fontweight="bold" if j == 1 else "normal"); [ax.spines[x].set_visible(False) for x in ["top", "right", "left"]]
txt(fig, P[2], 0.07, 0.10, "T (one number) + p$^0$ (n × K); reference-only label = argmax,\nsame for any T", fontsize=8, color=MUTED)
save(fig, "step3.png")

# ---------------- Step 4 ----------------
fig, P = frame(None)
n = 30; sizes = [11, 8, 7, 4]
C, lab = blocky(n, sizes, strength=0.6, noise=0.1)
lab = lab.copy(); lab[26:] = -1
perm = rng.permutation(n)
ax = sub(fig, P[0], (0.14, 0.18, 0.72, 0.60)); mat(ax, C[np.ix_(perm, perm)], cmap="jet", vmin=0, vmax=1, title="query correlation (original order)", xl="query cells", yl="query cells")
txt(fig, P[0], 0.14, 0.06, "structure is hidden in the original order", fontsize=8, color=MUTED)
# does: graph
ax = sub(fig, P[1], (0.05, 0.13, 0.90, 0.63)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cent = [(0.25, 0.62), (0.70, 0.66), (0.48, 0.24)]
pts, grp = [], []
for g, (cx, cy) in enumerate(cent):
    for _ in range(6):
        pts.append((cx + rng.normal(0, 0.07), cy + rng.normal(0, 0.07))); grp.append(g)
pts.append((0.9, 0.2)); grp.append(-1); pts.append((0.1, 0.18)); grp.append(-1)
pts = np.array(pts); grp = np.array(grp)
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        if grp[i] == grp[j] and grp[i] >= 0 and rng.random() < 0.8:
            ax.plot(*pts[[i, j]].T, color="#999", lw=0.7, zorder=1)
        elif rng.random() < 0.02:
            ax.plot(*pts[[i, j]].T, color="#ccc", lw=0.6, ls=":", zorder=1)
cc = ["#d64545", "#3b6fb6", "#4a9a5b"]
for (x, y), g in zip(pts, grp):
    ax.add_patch(Circle((x, y), 0.022, fc=cc[g] if g >= 0 else "white", ec="#444", lw=0.8, zorder=2))
for g, (cx, cy) in enumerate(cent):
    ax.add_patch(Circle((cx, cy), 0.17, fc="none", ec=cc[g], lw=1.3, ls="--"))
txt(fig, P[1], 0.05, 0.79, "keep edges with r > threshold; peel off\nthe densest set, repeat on the rest", fontsize=8)
txt(fig, P[1], 0.05, 0.06, r"density score $w(S)/|S|^{2\lambda}$;  λ, threshold tuned on a grid", fontsize=8, color=MUTED)
# output
order = np.argsort(np.where(lab < 0, 99, lab), kind="stable")
ax = sub(fig, P[2], (0.14, 0.18, 0.66, 0.60)); mat(ax, C[np.ix_(order, order)], cmap="jet", vmin=0, vmax=1, title="reordered by ICONS", xl="query cells, ICONS order")
for b in np.cumsum([11, 8, 7])[:]:
    ax.axvline(b - 0.5, color="white", lw=1.2); ax.axhline(b - 0.5, color="white", lw=1.2)
strip = sub(fig, P[2], (0.82, 0.18, 0.05, 0.60))
ids = np.where(lab[order] < 0, 4, lab[order])
strip.imshow(ids[:, None], cmap=ListedColormap(["#d64545", "#3b6fb6", "#4a9a5b", "#e0b020", "white"]), aspect="auto"); strip.set_xticks([]); strip.set_yticks([])
txt(fig, P[2], 0.885, 0.74, "cluster\nid", fontsize=7.5, color=MUTED)
txt(fig, P[2], 0.885, 0.20, "single-\ntons", fontsize=7.5, color=MUTED)
txt(fig, P[2], 0.14, 0.06, "one cluster id per cell - never changes afterwards", fontsize=8, color=MUTED)
save(fig, "step4.png")

# ---------------- Step 5 ----------------
fig, P = frame(None)
m = 8
p0 = np.array([[0.69, 0.31, 0.0], [0.30, 0.70, 0], [0.25, 0.75, 0], [0.96, 0.04, 0], [0.20, 0.78, 0.02], [0.35, 0.63, 0.02], [0.15, 0.83, 0.02], [0.28, 0.70, 0.02]])
def stacks(ax, M, title):
    bottom = np.zeros(len(M))
    for k in range(M.shape[1]):
        ax.bar(range(len(M)), M[:, k], bottom=bottom, color=TC[k], width=0.8); bottom += M[:, k]
    ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    [s.set_visible(False) for s in ax.spines.values()]
ax = sub(fig, P[0], (0.10, 0.25, 0.80, 0.50)); stacks(ax, p0, "p$^0$ for 8 cells of one ICONS cluster (cells 1-8)")
ax.set_xticks(range(8)); ax.set_xticklabels([str(i+1) for i in range(8)], fontsize=7.5); ax.tick_params(length=0)
txt(fig, P[0], 0.10, 0.10, "+ fixed cluster id per cell (Step 3)", fontsize=8, color=MUTED)
for k in range(3): txt(fig, P[0], 0.10 + k * 0.14, 0.18, "■ " + TN[k], fontsize=8, color=TC[k])
# does
bx = sub(fig, P[1], (0.04, 0.08, 0.92, 0.80)); bx.set_axis_off(); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
pi = p0.mean(0)
pa = sub(fig, P[1], (0.40, 0.58, 0.20, 0.28)); b = 0
for k in range(3): pa.bar(0, pi[k], bottom=b, color=TC[k], width=0.6); b += pi[k]
pa.set_axis_off(); pa.set_title(r"cluster mix $\pi_C$", fontsize=8.5)
bx.add_patch(FancyBboxPatch((0.02, 0.28), 0.34, 0.2, boxstyle="round,pad=0.02", fc="white", ec="#e08a2e"))
bx.text(0.19, 0.38, "M-step\n" + r"$\pi_C$ = mean q in C", ha="center", va="center", fontsize=8)
bx.add_patch(FancyBboxPatch((0.64, 0.28), 0.34, 0.2, boxstyle="round,pad=0.02", fc="white", ec="#e08a2e"))
bx.text(0.81, 0.38, "E-step\n" + r"$q_{ik} \propto e^{s_{ik}/T}\,\pi_{C,k}$", ha="center", va="center", fontsize=8)
bx.add_patch(FancyArrowPatch((0.19, 0.50), (0.40, 0.72), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=12, color="#e08a2e"))
bx.add_patch(FancyArrowPatch((0.60, 0.72), (0.81, 0.50), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=12, color="#e08a2e"))
bx.add_patch(FancyArrowPatch((0.64, 0.30), (0.36, 0.30), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=12, color="#e08a2e"))
bx.text(0.5, 0.06, "cell 1: A×π$_A$ < B×π$_B$  →  moves to B; cell 4 (A 0.96) stays A\nrepeat until change < 1e-8; singletons stay at p$^0$", ha="center", fontsize=7.8, color=MUTED)
# output
q = p0.copy()
for _ in range(30):
    pi = q.mean(0); q = p0 * pi; q /= q.sum(1, keepdims=True)
ax = sub(fig, P[2], (0.10, 0.25, 0.62, 0.50)); stacks(ax, q, "q after EM"); ax.set_xticks(range(8)); ax.set_xticklabels([str(i+1) for i in range(8)], fontsize=7.5); ax.tick_params(length=0)
pa = sub(fig, P[2], (0.80, 0.25, 0.10, 0.50)); b = 0
for k in range(3): pa.bar(0, pi[k], bottom=b, color=TC[k], width=0.8); b += pi[k]
pa.set_ylim(0, 1); pa.set_axis_off(); pa.set_title(r"final $\pi_C$", fontsize=8.5)
txt(fig, P[2], 0.10, 0.10, "q (n × K), EM label = argmax q,\ncluster compositions (initial + final), EM history", fontsize=8, color=MUTED)
save(fig, "step5.png")

# ---------------- Step 6 ----------------
fig, P = frame(None)
nc = 16
truth = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 0, 2])
def noisy(t, flips):
    o = t.copy()
    for i, v in flips: o[i] = v
    return o
refo = noisy(truth, [(3, 0), (6, 0), (9, 1), (12, 2), (14, 1)])
maj = noisy(truth, [(12, 2), (11, 2), (15, 1), (13, 0)])
em = noisy(truth, [(12, 2), (15, 1)])
rows = [("truth (held out)", truth), ("reference-only", refo), ("cluster majority", maj), ("EM", em)]
for r, (name, v) in enumerate(rows[1:] + rows[:1] if False else rows):
    ax = sub(fig, P[0], (0.40, 0.66 - r * 0.15, 0.56, 0.09))
    ax.imshow(v[None], cmap=ListedColormap(TC), aspect="auto", vmin=0, vmax=3); ax.set_xticks([]); ax.set_yticks([])
    txt(fig, P[0], 0.04, 0.685 - r * 0.15, name, fontsize=8)
txt(fig, P[0], 0.04, 0.10, "three label sets per query cell + truth\n(truth read only after inference)", fontsize=8, color=MUTED)
# does
for r, (name, v) in enumerate(rows[1:]):
    ax = sub(fig, P[1], (0.40, 0.66 - r * 0.17, 0.56, 0.09))
    ok = (v == truth)
    ax.imshow(ok[None], cmap=ListedColormap(["#d64545", "#cfe8d3"]), aspect="auto", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    for i in np.where(~ok)[0]: ax.text(i, 0, "×", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    txt(fig, P[1], 0.04, 0.685 - r * 0.17, name + "\nvs truth", fontsize=8)
txt(fig, P[1], 0.04, 0.10, "per type and macro precision / recall / F1;\nwhere EM changed labels; EM π vs true mix", fontsize=8, color=MUTED)
# output
ax = sub(fig, P[2], (0.06, 0.25, 0.88, 0.55)); ax.set_axis_off()
cells = [["method", "precision", "recall", "F1", "% changed"], ["reference-only", "…", "…", "…", "-"], ["cluster majority", "…", "…", "…", "-"], ["EM", "…", "…", "…", "…"]]
tb = ax.table(cellText=cells, loc="center", cellLoc="center", colWidths=[0.34, 0.18, 0.15, 0.12, 0.2])
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.6)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbb")
    if r == 0: cell.set_facecolor("#dfe9e1"); cell.set_text_props(fontweight="bold")
txt(fig, P[2], 0.06, 0.82, "method comparison", fontsize=8.5, fontweight="bold")
txt(fig, P[2], 0.06, 0.10, "+ per-cell predictions, per-type P/R/F1,\nchanges by cluster size, true vs EM composition", fontsize=8, color=MUTED)
save(fig, "step6.png")
print("ok")
