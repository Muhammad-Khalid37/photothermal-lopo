"""Figures 3-9 from the results of scripts 01-07."""
from _common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ptml.validation import lopo_constant
from ptml.stats import mae

args()
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
P, R = load_pairs()
y, g = P.dSTY.values, P.pub.values
pubs = np.unique(g)
save = lambda fig, n: (fig.tight_layout(), fig.savefig(C.FIGURES / n, dpi=300, bbox_inches="tight"), plt.close(fig), print("  ->", n))

# Fig. 3 - response distributions
fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3))
a.hist(y, bins=30, color="#2a6db0"); a.axvline(0, color="k", lw=0.8); a.set_xlabel("ΔSTY (mmol g$^{-1}$ h$^{-1}$)"); a.set_ylabel("Comparisons")
b.hist(R.log2r, bins=20, color="#c0562a"); b.axvline(0, color="k", lw=0.8); b.set_xlabel("log$_2$(STY$_{light}$/STY$_{dark}$)")
save(fig, "fig3_response_distributions.png")

# Fig. 4 - publication-specific MAE
rf = np.load(C.RESULTS / "abs_rf_preds.npy").mean(0); pe = np.load(C.RESULTS / "abs_ridge_ext_preds.npy")
M = {"Predict-zero": lopo_constant(y, g, "zero"), "Training median": lopo_constant(y, g, "median"),
     "Training mean": lopo_constant(y, g, "mean"), "Nested Ridge (extended grid)": pe, "Random Forest (20 seeds)": rf}
E_ = pd.DataFrame({k: np.abs(y - v) for k, v in M.items()}).groupby(g).mean()
fig, ax = plt.subplots(figsize=(10, 4.4))
for j, (k, c) in enumerate(zip(M, ["#8a8f99", "#1aa39a", "#d4a23c", "#2a5d9f", "#c0605a"])):
    ax.scatter(np.arange(len(pubs)) + (j - 2) * 0.14, E_[k].values, s=22, color=c, label=k, zorder=3)
ax.set_xticks(range(len(pubs))); ax.set_xticklabels(pubs); ax.set_xlabel("Held-out publication")
ax.set_ylabel("Publication-specific MAE (mmol g$^{-1}$ h$^{-1}$)"); ax.grid(axis="y", color="#ddd", lw=0.6)
ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
save(fig, "fig4_publication_mae.png")

# Fig. 5 - mixed vs held-out R2
T4 = load_json("03_mixed_vs_lopo.json")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, (key, title) in zip(axs, [("dSTY", "Light-induced change ΔSTY"), ("absolute_STY", "Absolute methane STY")]):
    t = T4[key]; mixed = [t["mixed"][m]["r2_pooled"][0] for m in ["ridge", "rf", "median"]]
    held = [t["held_out"]["ridge"][1], t["held_out"]["rf"][2], t["held_out"]["median"][1]]
    x = np.arange(3)
    ax.bar(x - 0.18, mixed, 0.36, color="#c0562a", label="Mixed 5-fold"); ax.bar(x + 0.18, held, 0.36, color="#2a6db0", label="Publication held-out")
    for i, (m_, h_) in enumerate(zip(mixed, held)):
        ax.text(i - 0.18, m_, f"{m_:.2f}", ha="center", va="bottom" if m_ >= 0 else "top", fontsize=6.5)
        ax.text(i + 0.18, h_, f"{h_:.2f}", ha="center", va="bottom" if h_ >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["Ridge", "Random Forest", "Median"]); ax.set_title(title, loc="left")
axs[0].set_ylabel("Out-of-sample R² (pooled)"); axs[0].legend(frameon=False)
save(fig, "fig5_mixed_vs_lopo.png")

# Fig. 6 - light-intensity range per publication
fig, ax = plt.subplots(figsize=(5, 3.6))
for i, q in enumerate(pubs):
    v = P.LI[P.pub == q].unique(); ax.plot([v.min(), v.max()], [i, i], color="#999", lw=1.5); ax.scatter(v, [i] * len(v), s=14, color="#2a6db0", zorder=3)
ax.set_xscale("log"); ax.set_yticks(range(len(pubs))); ax.set_yticklabels(pubs); ax.set_xlabel("Light intensity (W cm$^{-2}$)")
save(fig, "fig6_light_intensity_ranges.png")

# Fig. 7 - P02 intensity series
d = P[(P.pub == "P02") & (P.support == "Al2O3")].sort_values("LI")
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(d.LI, d.STY_light, "o-", color="#c0562a", label="Illuminated (no external heating)")
ax.axhline(d.STY_dark.iloc[0], color="#2a6db0", ls="--", label="Dark reference (furnace, 350 °C)")
ax.set_xlabel("Light intensity (W cm$^{-2}$)"); ax.set_ylabel("CH$_4$ STY (mmol g$^{-1}$ h$^{-1}$)"); ax.legend(frameon=False, fontsize=7)
save(fig, "fig7_P02_intensity_series.png")

# Fig. 8 - relative enhancement
yr, gr = R.log2r.values, R.pub.values; pr = np.load(C.RESULTS / "rel_preds.npy", allow_pickle=True).item()
gb = pr["Gradient boosting, one-hot"].mean(0); med = pr["Training median"][0]; th = pr["Thermal Arrhenius/dT, 1 parameter"][0]
ab = load_json("05_relative_enhancement.json")["ablation"]; full = float(np.mean([mae(yr, p) for p in pr["Gradient boosting, one-hot"]]))
fig, (a, b, c) = plt.subplots(1, 3, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15, 0.9]))
cmap = plt.get_cmap("tab20"); rp = np.unique(gr)
for i, q in enumerate(rp):
    a.scatter(yr[gr == q], gb[gr == q], s=14, color=cmap(i), label=q)
lim = [-3.8, 3.6]; a.plot(lim, lim, "k--", lw=0.8); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("Observed log$_2$ ratio"); a.set_ylabel("LOPO prediction (GB)"); a.legend(fontsize=5.5, ncol=2, frameon=False); a.set_title("(a)", loc="left")
pe_ = pd.DataFrame({"Median": np.abs(yr - med), "Gradient boosting": np.abs(yr - gb), "Thermal (1 par.)": np.abs(yr - th)}).groupby(gr).mean()
x = np.arange(len(rp))
for j, (k, col) in enumerate(zip(pe_.columns, ["#6b7a90", "#2a6db0", "#c0562a"])):
    b.bar(x + (j - 1) * 0.27, pe_[k], 0.27, color=col, label=k)
b.set_xticks(x); b.set_xticklabels([q[1:] for q in rp]); b.set_xlabel("Held-out publication (P##)"); b.set_ylabel("MAE (log$_2$)"); b.legend(fontsize=6.5, frameon=False); b.set_title("(b)", loc="left")
names = ["catalyst descriptors", "light intensity", "catalyst mass", "temperature", "operating conditions"]
dd = [ab[k][0] - full for k in names]
c.barh(["catalyst", "light int.", "cat. mass", "temperature", "all conditions"], dd, xerr=[ab[k][1] for k in names],
       color=["#6b7a90" if v < 0 else "#c0562a" for v in dd], capsize=2)
c.axvline(0, color="k", lw=0.8); c.set_xlabel("ΔMAE on removal (log$_2$)"); c.set_title("(c)", loc="left")
save(fig, "fig8_relative_enhancement.png")

# Fig. 9 - local calibration and within-publication trends
S = pd.read_csv(C.RESULTS / "calibration_summary.csv"); W = load_json("05_relative_enhancement.json")["within_publication"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
sty = {"global": ("Global median", "#6b7a90", "-"), "zero": ("Predict-zero", "black", ":"),
       "revmed": ("Median of revealed pairs", "#c0562a", "--"), "shrink": ("Shrinkage", "#2a6db0", "-")}
for m, (lab, col, ls) in sty.items():
    s = S[(S.subset == "all") & (S.method == m)].sort_values("k")
    a.plot(s.k, s["global_mae"] if m == "global" else s.mae, color=col, ls=ls, marker="o", ms=3.5, label=lab)
n = S[(S.subset == "all") & (S.method == "shrink")].sort_values("k")
top = S.mae.max() * 1.18
for k, v in zip(n.k, n.n_pubs):
    a.text(k, top * 0.96, f"n = {v}", ha="center", fontsize=6.5)
a.set_ylim(0, top); a.set_xlim(0.6, 5.4); a.set_xticks(C.K_LIST); a.set_xlabel("Revealed pairs (k)")
a.set_ylabel("MAE (mmol g$^{-1}$ h$^{-1}$)"); a.legend(fontsize=6, frameon=False, loc="lower left"); a.set_title("(a)", loc="left")
for m, off in [("shrink", -0.08), ("revmed", 0.08)]:
    col = sty[m][1]
    for subset, mk, fill in [("all", "o", col), ("common", "s", "white")]:
        s = S[(S.subset == subset) & (S.method == m)].sort_values("k"); xx = s.k + off + (0.16 if subset == "common" else 0)
        b.errorbar(xx, s["diff"], yerr=[s["diff"] - s.lo, s.hi - s["diff"]], fmt=mk, color=col, mfc=fill, ms=4, capsize=2, lw=1,
                   label=f"{sty[m][0]}{' (common)' if subset == 'common' else ''}")
b.axhline(0, color="k", lw=0.8); b.set_xticks(C.K_LIST); b.set_xlabel("Revealed pairs (k)"); b.set_ylabel("ΔMAE vs. global median")
b.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); b.set_title("(b)", loc="left")
W = sorted(W, key=lambda d: d["pub"], reverse=True)
c.barh([d["pub"] for d in W], [d["rho"] for d in W], color="#2a6db0")
for i, d in enumerate(W):
    c.text(0.03, i, f"{d['levels']} T levels, n = {d['n']}", va="center", fontsize=6.5)
c.axvline(0, color="k", lw=0.8); c.set_xlim(-1.05, 0.8); c.set_xlabel("Within-paper Spearman ρ"); c.set_title("(c)", loc="left")
save(fig, "fig9_calibration_and_temperature.png")
