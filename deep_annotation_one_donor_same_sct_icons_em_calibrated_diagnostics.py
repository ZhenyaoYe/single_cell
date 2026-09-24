#!/usr/bin/env python3

import os
import sys
import gc
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import minimize
from scipy.special import logsumexp

BICCN_DIR = "/data/brutus_data34a/SCORCH/BICCN"
SEURAT_DIR = "/data/brutus_data34a/SCORCH/08042026/Seurat"
OUTPUT_DIR = "/data/brutus_data34a/SCORCH/DeepAnnotate_sameSCT_ICONS_EM"

TEST_DONOR = "H18.30.001"
CELLTYPE_COL = "within_area_subclass"

SCT_LAYER = "data"

N_HVG = 3000
VST_FLAVOR = "v2"

DENSE_ICONS_DIR = "/data/brutus_data34a/SCORCH/09242026"
LAMBDA_VEC = np.round(np.arange(0.70, 0.86, 0.05), 3)
Q_VEC = np.round(np.arange(0.3, 0.7, 0.1), 3)
TUNING_TOL_PCT = 0.01
TUNING_N_JOBS = 4
TUNING_CRITERION = "block"
ICONS_LEFTOVER_AS_ONE_CLUSTER = False

REBUILD_INPUTS = False

EM_MAX_ITER = 100
EM_TOL = 1e-8

if DENSE_ICONS_DIR not in sys.path:
    sys.path.insert(0, DENSE_ICONS_DIR)

from dense_icons import dense


def offdiag_quantile(corr, q=0.6):
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return np.quantile(corr[mask], q)


def block_criterion_noalpha(S, S_eval, norm_S, Clist, CID):
    K = len(CID)
    n = S.shape[0]
    groups = np.full(n, -1, dtype=np.int64)
    start = 0
    for k, size in enumerate(CID):
        end = start + size
        groups[Clist[start:end]] = k
        start = end
    members = [np.where(groups == a)[0] for a in range(K)]
    Sigma = np.zeros((K, K))
    for a in range(K):
        ia = members[a]
        for b in range(K):
            ib = members[b]
            block = S[np.ix_(ia, ib)]
            if a == b and len(ia) > 1:
                mask = ~np.eye(len(ia), dtype=bool)
                Sigma[a, b] = block[mask].mean()
            else:
                Sigma[a, b] = block.mean()
    frob_sq = 0.0
    for a in range(K):
        ia = members[a]
        for b in range(K):
            ib = members[b]
            block = S_eval[np.ix_(ia, ib)]
            frob_sq += np.sum((block - Sigma[a, b]) ** 2)
    rel_frob = np.sqrt(frob_sq) / norm_S
    return rel_frob, K


def choose_best(df, tol_pct=0.01):
    min_val = df["rel_frob"].min()
    tol = tol_pct * min_val
    cand = df[df["rel_frob"] <= min_val + tol].copy()
    cand = cand[cand["K"] == cand["K"].min()]
    cand = cand[cand["threshold"] == cand["threshold"].max()]
    lambda_mid = (df["lambda"].min() + df["lambda"].max()) / 2
    cand["dist"] = (cand["lambda"] - lambda_mid).abs()
    cand = cand.sort_values("dist")
    return cand.iloc[[0]]


def run_dense_tuning(S, lambda_vec=None, q_vec=None, out_dir="dense_results/", n_jobs=1, tol_pct=0.01, verbose=True):
    if lambda_vec is None:
        lambda_vec = np.round(np.arange(0.5, 1.01, 0.1), 2)
    if q_vec is None:
        q_vec = np.round(np.arange(0.1, 1.0, 0.1), 2)
    os.makedirs(out_dir, exist_ok=True)
    S = np.ascontiguousarray(S, dtype=np.float64)
    p = S.shape[0]
    S_eval = S.copy()
    np.fill_diagonal(S_eval, 0)
    norm_S = np.sqrt(np.sum(S_eval ** 2))
    q_to_theta = {q: offdiag_quantile(S, q) for q in q_vec}
    grid = list(itertools.product(lambda_vec, q_vec))
    n_grid = len(grid)
    print(f"Starting dense tuning grid search: {len(lambda_vec)} lambda x {len(q_vec)} q = {n_grid} combinations, n_jobs={n_jobs}")
    t_start = time.time()
    def _eval_one(lam, q, idx=None, log_success=True):
        thr = q_to_theta[q]
        tag = f"[{idx}/{n_grid}] " if idx is not None else ""
        t0 = time.time()
        try:
            fit = dense(S, threshold=thr, lam=lam, return_matrix=False)
        except Exception as e:
            if verbose:
                print(f"{tag}lambda={lam:.2f}, q={q:.2f} (theta={thr:.4f}): dense() failed ({e}) ({time.time() - t0:.1f}s)")
            return None
        Clist = np.asarray(fit["Clist"], dtype=np.int64)
        CID = np.asarray(fit["CID"], dtype=np.int64)
        if len(fit["sizes"]) == 0:
            if verbose:
                print(f"{tag}lambda={lam:.2f}, q={q:.2f} (theta={thr:.4f}): no clusters ({time.time() - t0:.1f}s)")
            return None
        rel_frob, K = block_criterion_noalpha(S, S_eval, norm_S, Clist, CID)
        if verbose and log_success:
            print(f"{tag}lambda={lam:.2f}, q={q:.2f} (theta={thr:.4f}): K={K} ({len(fit['sizes'])} communities, {fit['n_singletons']} unassigned), rel_frob={rel_frob:.4f} ({time.time() - t0:.1f}s)")
        return {"lambda": lam, "q": q, "threshold": thr, "rel_frob": rel_frob, "K": K, "n_communities": len(fit["sizes"]), "n_unassigned": fit["n_singletons"]}
    if n_jobs == 1:
        results = [_eval_one(lam, q, idx=i + 1) for i, (lam, q) in enumerate(grid)]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10 if verbose else 0)(
            delayed(_eval_one)(lam, q, log_success=False) for lam, q in grid
        )
    print(f"Grid search finished in {time.time() - t_start:.1f}s")
    results = [r for r in results if r is not None]
    print(f"{len(results)}/{len(grid)} grid combinations succeeded")
    if not results:
        raise RuntimeError("All (lambda, q) combinations failed or returned empty clusters. Check the dense() messages above.")
    df = pd.DataFrame(results)
    print(df[["lambda", "q", "threshold", "K", "n_communities", "n_unassigned", "rel_frob"]].describe())
    if df["K"].nunique() == 1 and df["K"].iloc[0] in (1, p):
        print(f"WARNING: every grid point produced K={df['K'].iloc[0]} clusters -- q_vec may span too narrow a range.")
    df.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)
    best = choose_best(df, tol_pct)
    best.to_csv(os.path.join(out_dir, "best_parameters.csv"), index=False)
    print("Best parameters:")
    print(best)
    return {"results": df, "best": best}


def plot_cormx_icons(corr, Clist, CID, title, out_path, tick_step=1000, vmin=0, vmax=1.0):
    corr_reordered = corr[np.ix_(Clist, Clist)]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(corr_reordered, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax)
    for b in np.cumsum(CID)[:-1]:
        ax.axvline(x=b - 0.5, color="white", linewidth=1.0)
        ax.axhline(y=b - 0.5, color="white", linewidth=1.0)
    n = corr_reordered.shape[0]
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=11)
    ax.set_xlabel("Cell index, ICONS order", fontsize=14)
    ax.set_ylabel("Cell index, ICONS order", fontsize=14)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="tiff")
    plt.close(fig)
    print(f"Saved: {out_path} | {len(CID)} blocks, sizes: {[int(x) for x in CID]}")


def run_icons_tuned(corr, sample_label, lam, threshold, output_dir, vmax=0.6, tick_step=1000):
    print(f"\nRunning ICONS/dense on {sample_label} with tuned params: lambda={lam}, threshold={threshold:.4f}")
    res = dense(corr, threshold=threshold, lam=lam, return_matrix=False)
    Clist = np.asarray(res["Clist"], dtype=np.int64)
    CID = np.asarray(res["CID"], dtype=np.int64)
    if len(res["sizes"]) == 0:
        raise RuntimeError(f"dense() returned no clusters for {sample_label} with lambda={lam}, threshold={threshold} -- re-check run_dense_tuning's grid range.")
    print(f"ICONS detected {len(res['sizes'])} communities in {sample_label} (+{res['n_singletons']} unassigned); sizes: {res['sizes'].tolist()}")
    plot_cormx_icons(
        corr, Clist, CID,
        f"{sample_label} cell-cell correlation matrix, ICONS cluster order (tuned)",
        os.path.join(output_dir, f"{sample_label}_icons.tiff"),
        vmax=vmax, tick_step=tick_step,
    )
    return {"Clist": Clist, "CID": CID, "sizes": res["sizes"], "membership": res["membership"], "n_singletons": res["n_singletons"], "scores": res["scores"], "theta": threshold, "lambda": lam, "q": None}


_DONOR_CANDIDATES = [
    "donor_id", "donor", "external_donor_name", "specimen_id",
    "subject_id", "DonorID", "donor_name",
]
_ID_CANDIDATES = ["sample_id", "cell_id", "barcode", "cellname", "cell_name"]


def resolve_column(df, explicit, candidates, kind, match_values=None):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"{kind} column {explicit!r} not found")
        return explicit

    for c in candidates:
        if c not in df.columns:
            continue
        if match_values is not None:
            overlap = len(set(df[c].astype(str)) & set(map(str, match_values)))
            if overlap < 0.90 * len(match_values):
                continue
        return c

    raise ValueError(
        f"Could not detect {kind} column. Available columns: {list(df.columns)}"
    )


def load_reference_metadata(biccn_dir):
    meta_path = os.path.join(biccn_dir, "BICCN_meta_share.csv")
    colnames_path = os.path.join(biccn_dir, "colnames.txt")

    with open(colnames_path) as f:
        cellnames = np.array(f.read().splitlines(), dtype=str)

    meta = pd.read_csv(meta_path)
    donor_col = resolve_column(meta, None, _DONOR_CANDIDATES, "donor")
    id_col = resolve_column(
        meta, None, _ID_CANDIDATES, "cell id", match_values=cellnames
    )

    if CELLTYPE_COL not in meta.columns:
        raise ValueError(f"{CELLTYPE_COL!r} not found in metadata")

    return meta, cellnames, donor_col, id_col


def _read_rds_via_rpy2(path):
    import rpy2.robjects as ro
    return ro.r["readRDS"](path)


def load_r_gene_vector(path):
    try:
        import pyreadr
        res = pyreadr.read_r(path)
        df = list(res.values())[0]
        return [str(x) for x in df.iloc[:, 0].tolist()]
    except Exception:
        obj = _read_rds_via_rpy2(path)
        return [str(x) for x in list(obj)]


def load_rds_matrix(path):
    try:
        import pyreadr
        res = pyreadr.read_r(path)
        df = list(res.values())[0]
        row_names = df.index.astype(str).to_numpy()
        col_names = df.columns.astype(str).to_numpy()
        if not np.array_equal(row_names, col_names):
            raise ValueError("RDS row/column names differ")
        arr = df.to_numpy(dtype=np.float64)
    except Exception:
        import rpy2.robjects as ro
        obj = ro.r["readRDS"](path)
        arr = np.asarray(obj, dtype=np.float64)
        dn = ro.r["dimnames"](obj)
        row_names = np.array([str(x) for x in dn[0]], dtype=str)
        col_names = np.array([str(x) for x in dn[1]], dtype=str)
        if not np.array_equal(row_names, col_names):
            raise ValueError("RDS row/column names differ")

    if arr.shape != (len(row_names), len(row_names)):
        raise ValueError(f"Expected square matrix; got {arr.shape}")

    return row_names, arr


def icons_cluster_ids(icons_result, n_expected, leftover_as_one=ICONS_LEFTOVER_AS_ONE_CLUSTER):
    order = np.asarray(icons_result["Clist"])
    sizes = np.asarray(icons_result["sizes"])
    if len(order) != n_expected or set(order.tolist()) != set(range(n_expected)):
        raise ValueError("Clist is not a permutation of query indices")
    membership = np.asarray(icons_result["membership"])
    labels = membership - 1
    leftover = np.sort(order[sizes.sum():])
    if len(leftover):
        labels[leftover] = len(sizes) if leftover_as_one else len(sizes) + np.arange(len(leftover))
    return labels


def softmax_from_scores(scores, temperature):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError("scores must be a finite 2-D matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")

    logits = s / float(temperature)
    log_norm = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - log_norm)


def fit_temperature_reference_only(calib_scores, calib_true_index):
    scores = np.asarray(calib_scores, dtype=float)
    y = np.asarray(calib_true_index, dtype=int)
    if scores.ndim != 2 or y.shape != (scores.shape[0],):
        raise ValueError("Calibration score/label dimensions do not match")

    valid = y >= 0
    if not np.any(valid):
        raise ValueError("No usable donor-held-out reference cells for temperature calibration")

    scores = scores[valid]
    y = y[valid]

    true_finite = np.array([np.isfinite(scores[i, y[i]]) for i in range(len(y))])
    n_finite = np.isfinite(scores).sum(axis=1)
    keep = true_finite & (n_finite >= 2)
    scores = scores[keep]
    y = y[keep]
    if len(y) == 0:
        raise ValueError("No calibration cells have their true type represented in an inner training fold")

    def objective(log_T_arr):
        T = float(np.exp(log_T_arr[0]))
        logits = scores / T
        logits = np.where(np.isfinite(logits), logits, -np.inf)
        log_den = logsumexp(logits, axis=1)
        log_num = logits[np.arange(len(y)), y]
        return float(np.mean(log_den - log_num))

    fit = minimize(objective, x0=np.array([0.0]), method="BFGS")
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Temperature calibration failed: {fit.message}")

    T = float(np.exp(fit.x[0]))

    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    pred = np.argmax(finite_scores, axis=1)
    acc = float(np.mean(pred == y))

    nll_T1 = float(objective(np.array([0.0])))
    nll_fitted = float(fit.fun)

    return T, {
        "temperature": T,
        "n_calibration_cells": int(len(y)),
        "calibration_accuracy": acc,
        "nll_T1": nll_T1,
        "nll_fitted_T": nll_fitted,
        "nll_improvement": nll_T1 - nll_fitted,
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
    }


def normalize_rows(x):
    x = np.asarray(x, dtype=float)
    sums = x.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
        raise ValueError("Cannot normalize rows with non-positive/non-finite sums")
    return x / sums


def icons_em_refine(score_mat, temperature, cluster_id, max_iter=EM_MAX_ITER, tol=EM_TOL):
    scores = np.asarray(score_mat, dtype=float)
    cluster_id = np.asarray(cluster_id)

    if scores.ndim != 2:
        raise ValueError("score_mat must be a 2-D matrix")
    if len(cluster_id) != scores.shape[0]:
        raise ValueError("cluster_id length does not match number of cells")
    if not np.isfinite(scores).all():
        raise ValueError("score_mat contains non-finite values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")

    log_lik = scores / float(temperature)
    p0 = np.exp(log_lik - logsumexp(log_lik, axis=1, keepdims=True))
    q = p0.copy()
    clusters = np.unique(cluster_id)

    def estimate_pi(q_now):
        pi_now = {}
        for c in clusters:
            idx = np.where(cluster_id == c)[0]
            comp = q_now[idx].mean(axis=0)
            comp_sum = comp.sum()
            if not np.isfinite(comp_sum) or comp_sum <= 0:
                raise ValueError(f"Invalid composition for ICONS cluster {c}")
            pi_now[c] = comp / comp_sum
        return pi_now

    def observed_log_likelihood(pi_now):
        total = 0.0
        for c in clusters:
            idx = np.where(cluster_id == c)[0]
            if len(idx) <= 1:
                continue
            log_pi = np.log(np.maximum(pi_now[c], np.finfo(float).tiny))
            total += float(logsumexp(log_lik[idx] + log_pi[None, :], axis=1).sum())
        return total

    pi = estimate_pi(q)
    initial_pi = {c: v.copy() for c, v in pi.items()}
    history_rows = []
    previous_ll = observed_log_likelihood(pi)

    for iteration in range(1, max_iter + 1):
        q_new = np.empty_like(q)

        for c in clusters:
            idx = np.where(cluster_id == c)[0]

            if len(idx) == 1:
                q_new[idx] = p0[idx]
                continue

            log_pi = np.log(np.maximum(pi[c], np.finfo(float).tiny))
            log_post = log_lik[idx] + log_pi[None, :]
            q_new[idx] = np.exp(
                log_post - logsumexp(log_post, axis=1, keepdims=True)
            )

        pi_new = estimate_pi(q_new)

        max_q_change = float(np.max(np.abs(q_new - q)))
        max_pi_change = float(max(
            np.max(np.abs(pi_new[c] - pi[c])) for c in clusters
        ))

        ll = observed_log_likelihood(pi_new)
        ll_change = ll - previous_ll
        monotone = bool(ll_change >= -1e-10)
        if not monotone:
            print(
                f"WARNING: EM observed log-likelihood decreased by {ll_change:.3e} "
                f"at iteration {iteration}"
            )

        history_rows.append({
            "iteration": iteration,
            "log_likelihood": ll,
            "log_likelihood_change": ll_change,
            "log_likelihood_nondecreasing": monotone,
            "max_cell_probability_change": max_q_change,
            "max_cluster_composition_change": max_pi_change,
        })

        q = q_new
        pi = pi_new
        previous_ll = ll

        if max(max_q_change, max_pi_change) < tol:
            break

    final_pi = {c: v.copy() for c, v in pi.items()}
    return p0, q, pd.DataFrame(history_rows), initial_pi, final_pi


def cluster_composition_table(pi_dict, celltypes, cluster_id):
    rows = []
    for c in sorted(pi_dict):
        n_cells = int(np.sum(cluster_id == c))
        for k, ct in enumerate(celltypes):
            rows.append({
                "cluster_id": int(c),
                "n_cells": n_cells,
                "celltype": str(ct),
                "composition": float(pi_dict[c][k]),
            })
    return pd.DataFrame(rows)


def prf_table(truth, pred):
    truth = pd.Series(truth, dtype="string")
    pred = pd.Series(pred, dtype="string")
    ok = truth.notna() & pred.notna()
    truth = truth[ok].astype(str)
    pred = pred[ok].astype(str)

    labels = sorted(set(truth) | set(pred))
    p, r, f, n = precision_recall_fscore_support(
        truth, pred, labels=labels, average=None, zero_division=0
    )

    out = pd.DataFrame({
        "celltype": labels,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": n,
    })

    macro = precision_recall_fscore_support(
        truth, pred, labels=labels, average="macro", zero_division=0
    )
    return out, {
        "precision": float(macro[0]),
        "recall": float(macro[1]),
        "f1": float(macro[2]),
    }


def build_same_sct_inputs(donor, donor_col, id_col, out_dir):
    try:
        import rpy2.robjects as ro
    except ImportError as e:
        raise ImportError(
            "This one-script workflow requires rpy2 so Python can call Seurat. "
            "Install with: pip install rpy2"
        ) from e

    os.makedirs(out_dir, exist_ok=True)

    env = ro.globalenv
    env["PY_BICCN_DIR"] = BICCN_DIR
    env["PY_SEURAT_DIR"] = SEURAT_DIR
    env["PY_OUT_DIR"] = out_dir
    env["PY_DONOR"] = donor
    env["PY_DONOR_COL"] = donor_col
    env["PY_ID_COL"] = id_col
    env["PY_LABEL_COL"] = CELLTYPE_COL
    env["PY_SCT_LAYER"] = SCT_LAYER
    env["PY_N_HVG"] = int(N_HVG)
    env["PY_VST_FLAVOR"] = VST_FLAVOR

    r_code = r'''
    suppressPackageStartupMessages({
      library(Seurat)
      library(Matrix)
    })

    biccn_dir <- PY_BICCN_DIR
    seurat_dir <- PY_SEURAT_DIR
    out_dir <- PY_OUT_DIR
    donor <- PY_DONOR
    donor_col <- PY_DONOR_COL
    id_col <- PY_ID_COL
    label_col <- PY_LABEL_COL
    sct_layer <- PY_SCT_LAYER
    n_hvg <- as.integer(PY_N_HVG)
    vst_flavor <- PY_VST_FLAVOR

    dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

    counts <- Matrix::readMM(file.path(biccn_dir, "BICCN_mat.mtx"))
    genes <- readLines(file.path(biccn_dir, "rownames.txt"))
    cells <- readLines(file.path(biccn_dir, "colnames.txt"))

    if (!all(dim(counts) == c(length(genes), length(cells)))) {
      stop("BICCN_mat.mtx dimensions do not match rownames/colnames")
    }
    rownames(counts) <- genes
    colnames(counts) <- cells

    meta <- read.csv(
      file.path(biccn_dir, "BICCN_meta_share.csv"),
      stringsAsFactors=FALSE,
      check.names=FALSE
    )

    if (!(id_col %in% colnames(meta))) stop("id_col missing: ", id_col)
    if (!(donor_col %in% colnames(meta))) stop("donor_col missing: ", donor_col)
    if (!(label_col %in% colnames(meta))) stop("label_col missing: ", label_col)

    rownames(meta) <- as.character(meta[[id_col]])
    missing_meta <- setdiff(cells, rownames(meta))
    if (length(missing_meta) > 0) {
      stop(length(missing_meta), " count-matrix cells missing from metadata")
    }
    meta <- meta[cells, , drop=FALSE]

    old_corr_path <- file.path(seurat_dir, paste0("query_corr_", donor, ".RDS"))
    hvg_path <- file.path(seurat_dir, paste0("ref_hvgs_", donor, ".RDS"))

    if (!file.exists(old_corr_path)) stop("Missing: ", old_corr_path)
    if (!file.exists(hvg_path)) stop("Missing: ", hvg_path)

    old_corr <- readRDS(old_corr_path)
    if (is.null(rownames(old_corr)) || is.null(colnames(old_corr))) {
      stop("Existing query_corr has no dimnames")
    }
    if (!identical(rownames(old_corr), colnames(old_corr))) {
      stop("Existing query_corr row/column names differ")
    }

    query_cells <- rownames(old_corr)

    ref_hvgs <- as.character(readRDS(hvg_path))

    if (!all(query_cells %in% cells)) {
      stop("Some query cells are absent from the BICCN count matrix")
    }
    if (!all(as.character(meta[query_cells, donor_col]) == donor)) {
      stop("Recovered query cells are not all from held-out donor ", donor)
    }

    ref_cells <- cells[
      !is.na(meta[[donor_col]]) & as.character(meta[[donor_col]]) != donor
    ]
    if (length(ref_cells) == 0) stop("No reference cells remain")

    message("[fold] query donor = ", donor)
    message("[fold] exact query cells = ", length(query_cells))
    message("[fold] reference cells = ", length(ref_cells))
    message("[fold] exact saved fold HVGs = ", length(ref_hvgs))

    reference <- CreateSeuratObject(
      counts=counts[, ref_cells, drop=FALSE],
      meta.data=meta[ref_cells, , drop=FALSE],
      min.cells=0,
      min.features=0
    )

    query <- CreateSeuratObject(
      counts=counts[, query_cells, drop=FALSE],
      meta.data=meta[query_cells, , drop=FALSE],
      min.cells=0,
      min.features=0
    )

    set.seed(1)
    reference <- SCTransform(
      reference,
      assay="RNA",
      new.assay.name="SCT",
      variable.features.n=n_hvg,
      return.only.var.genes=FALSE,
      vst.flavor=vst_flavor,
      verbose=TRUE
    )

    selected_hvgs <- VariableFeatures(reference, assay="SCT")
    hvg_set_matches <- setequal(as.character(selected_hvgs), ref_hvgs)
    hvg_order_matches <- identical(as.character(selected_hvgs), ref_hvgs)

    message("[check] newly selected HVG set matches saved fold HVGs: ", hvg_set_matches)
    message("[check] newly selected HVG order matches saved fold HVGs: ", hvg_order_matches)
    if (!hvg_set_matches) {
      stop("Reference SCTransform HVG set differs from the saved Seurat fold HVGs")
    }
    if (!hvg_order_matches) {
      warning("Reference SCTransform HVG order differs, but the set matches; downstream matrices are explicitly reordered to saved ref_hvgs")
    }

    ref_model <- reference[["SCT"]]@SCTModel.list[[1]]
    query <- SCTransform(
      query,
      assay="RNA",
      new.assay.name="SCT",
      reference.SCT.model=ref_model,
      return.only.var.genes=FALSE,
      verbose=TRUE
    )

    get_assay_matrix <- function(obj, assay, layer_name) {
      tryCatch(
        GetAssayData(obj, assay=assay, layer=layer_name),
        error=function(e) GetAssayData(obj, assay=assay, slot=layer_name)
      )
    }

    ref_sct_all <- get_assay_matrix(reference, "SCT", sct_layer)
    qry_sct_all <- get_assay_matrix(query, "SCT", sct_layer)

    if (!all(ref_hvgs %in% rownames(ref_sct_all))) {
      stop("Some saved fold HVGs are absent from reference SCT matrix")
    }
    if (!all(ref_hvgs %in% rownames(qry_sct_all))) {
      stop("Some saved fold HVGs are absent from query SCT matrix")
    }

    ref_sct <- ref_sct_all[ref_hvgs, ref_cells, drop=FALSE]
    qry_sct <- qry_sct_all[ref_hvgs, query_cells, drop=FALSE]

    stopifnot(identical(rownames(ref_sct), rownames(qry_sct)))
    stopifnot(identical(colnames(qry_sct), query_cells))

    message("[check] same SCT layer query/reference: ", sct_layer)
    message("[check] same HVGs query/reference: ",
            identical(rownames(ref_sct), rownames(qry_sct)))
    message("[check] exact query-cell order preserved: ",
            identical(colnames(qry_sct), query_cells))

    message("[corr] computing fresh query-query Pearson correlation from matched SCT matrix")
    query_corr <- cor(as.matrix(qry_sct), method="pearson")
    rownames(query_corr) <- query_cells
    colnames(query_corr) <- query_cells

    old_corr_aligned <- as.matrix(old_corr[query_cells, query_cells, drop=FALSE])
    max_abs_corr_diff_vs_saved <- max(abs(query_corr - old_corr_aligned))
    message("[check] max abs query-correlation difference vs saved fold: ",
            format(max_abs_corr_diff_vs_saved, scientific=TRUE))
    rm(old_corr_aligned, old_corr)
    gc()

    query_corr_path <- file.path(out_dir, paste0("query_corr_sameSCT_", donor, ".RDS"))
    saveRDS(query_corr, query_corr_path)

    ref_labels <- as.character(meta[ref_cells, label_col])
    valid_ref <- !is.na(ref_labels) & nzchar(ref_labels)
    if (!all(valid_ref)) {
      message("[reference] dropping ", sum(!valid_ref),
              " unlabeled reference cells from label-evidence calculation")
    }

    ref_labels_valid <- ref_labels[valid_ref]
    ref_donors_valid <- as.character(meta[ref_cells[valid_ref], donor_col])
    ref_sct_valid <- ref_sct[, valid_ref, drop=FALSE]
    celltypes <- sort(unique(ref_labels_valid))

    G <- nrow(qry_sct)
    if (G < 2) stop("Need at least two HVGs for Pearson correlation")

    standardize_cells <- function(mat) {
      x <- as.matrix(mat)
      mu <- colMeans(x)
      sdv <- apply(x, 2, sd)
      bad <- !is.finite(sdv) | sdv <= 0
      if (any(bad)) {
        stop(sum(bad), " cells have zero/non-finite SD across HVGs")
      }
      sweep(sweep(x, 2, mu, "-"), 2, sdv, "/")
    }

    qz <- standardize_cells(qry_sct)
    rz <- standardize_cells(ref_sct_valid)

    type_mean_z <- matrix(
      NA_real_,
      nrow=G,
      ncol=length(celltypes),
      dimnames=list(rownames(qry_sct), celltypes)
    )

    for (kk in seq_along(celltypes)) {
      idx <- which(ref_labels_valid == celltypes[kk])
      type_mean_z[, kk] <- rowMeans(rz[, idx, drop=FALSE])
    }

    type_scores <- crossprod(qz, type_mean_z) / (G - 1)
    rownames(type_scores) <- query_cells
    colnames(type_scores) <- celltypes

    if (any(!is.finite(type_scores))) stop("Non-finite reference type scores")

    score_path <- file.path(
      out_dir, paste0("reference_type_scores_sameSCT_", donor, ".csv.gz")
    )
    score_df <- data.frame(cell=query_cells, type_scores, check.names=FALSE)
    write.csv(score_df, gzfile(score_path), row.names=FALSE)

    calib_rows <- list()
    calib_donors <- sort(unique(ref_donors_valid))
    rr <- 1L
    for (vd in calib_donors) {
      val_idx <- which(ref_donors_valid == vd)
      train_idx <- which(ref_donors_valid != vd)
      if (length(val_idx) == 0 || length(train_idx) == 0) next

      train_labels <- ref_labels_valid[train_idx]
      available_types <- intersect(celltypes, unique(train_labels))
      if (length(available_types) < 2) next

      train_mean_z <- matrix(
        NA_real_, nrow=G, ncol=length(celltypes),
        dimnames=list(rownames(rz), celltypes)
      )
      for (kk in seq_along(celltypes)) {
        idx_local <- train_idx[train_labels == celltypes[kk]]
        if (length(idx_local) > 0) {
          train_mean_z[, kk] <- rowMeans(rz[, idx_local, drop=FALSE])
        }
      }

      val_scores <- crossprod(rz[, val_idx, drop=FALSE], train_mean_z) / (G - 1)
      val_df <- data.frame(
        cell=colnames(rz)[val_idx],
        validation_donor=vd,
        truth=ref_labels_valid[val_idx],
        val_scores,
        check.names=FALSE,
        stringsAsFactors=FALSE
      )
      calib_rows[[rr]] <- val_df
      rr <- rr + 1L
    }

    if (length(calib_rows) == 0) {
      stop("Could not create donor-held-out reference calibration folds")
    }
    calib_df <- do.call(rbind, calib_rows)
    calib_path <- file.path(
      out_dir, paste0("reference_temperature_calibration_", donor, ".csv.gz")
    )
    write.csv(calib_df, gzfile(calib_path), row.names=FALSE)

    truth_path <- file.path(out_dir, paste0("query_truth_", donor, ".csv"))
    truth_df <- data.frame(
      cell=query_cells,
      truth=as.character(meta[query_cells, label_col]),
      stringsAsFactors=FALSE
    )
    write.csv(truth_df, truth_path, row.names=FALSE)

    report_path <- file.path(out_dir, paste0("same_sct_report_", donor, ".csv"))
    report <- data.frame(
      donor=donor,
      n_query=length(query_cells),
      n_reference=length(ref_cells),
      n_hvgs=length(ref_hvgs),
      sct_layer=sct_layer,
      hvg_set_matches_saved_fold=hvg_set_matches,
      hvg_order_matches_saved_fold=hvg_order_matches,
      query_order_matches_saved_fold=identical(colnames(qry_sct), query_cells),
      max_abs_corr_diff_vs_saved=max_abs_corr_diff_vs_saved,
      reference_evidence="mean Pearson correlation to all labeled reference cells of each cell type",
      n_reference_celltypes=length(celltypes),
      stringsAsFactors=FALSE
    )
    write.csv(report, report_path, row.names=FALSE)

    rm(reference, query, counts, ref_sct_all, qry_sct_all,
       ref_sct, qry_sct, ref_sct_valid, query_corr,
       qz, rz, type_mean_z, calib_df)
    gc()

    invisible(TRUE)
    '''

    ro.r(r_code)

    query_corr_path = os.path.join(out_dir, f"query_corr_sameSCT_{donor}.RDS")
    score_path = os.path.join(out_dir, f"reference_type_scores_sameSCT_{donor}.csv.gz")
    calib_path = os.path.join(out_dir, f"reference_temperature_calibration_{donor}.csv.gz")
    truth_path = os.path.join(out_dir, f"query_truth_{donor}.csv")
    report_path = os.path.join(out_dir, f"same_sct_report_{donor}.csv")

    for path in (query_corr_path, score_path, calib_path, truth_path, report_path):
        if not os.path.exists(path):
            raise RuntimeError(f"Expected output was not created: {path}")

    return query_corr_path, score_path, calib_path, truth_path, report_path


def run_one_donor(donor):
    donor_dir = os.path.join(OUTPUT_DIR, f"donor_{donor}")
    os.makedirs(donor_dir, exist_ok=True)

    old_corr_path = os.path.join(SEURAT_DIR, f"query_corr_{donor}.RDS")
    hvg_path = os.path.join(SEURAT_DIR, f"ref_hvgs_{donor}.RDS")
    if not os.path.exists(old_corr_path):
        raise FileNotFoundError(old_corr_path)
    if not os.path.exists(hvg_path):
        raise FileNotFoundError(hvg_path)

    meta, all_cells, donor_col, id_col = load_reference_metadata(BICCN_DIR)
    print(f"donor_col={donor_col!r}, id_col={id_col!r}")

    ref_hvgs = load_r_gene_vector(hvg_path)
    print("\n=== FOLD DEFINITION ===")
    print("query donor:", donor)
    print("saved fold-specific HVGs:", len(ref_hvgs))
    print("reference: all other donors")
    print("SCT matrix type for BOTH correlation sources:", SCT_LAYER)
    print("no PCA / CCA / anchor space / kNN")

    query_corr_path = os.path.join(donor_dir, f"query_corr_sameSCT_{donor}.RDS")
    score_path = os.path.join(donor_dir, f"reference_type_scores_sameSCT_{donor}.csv.gz")
    calib_path = os.path.join(donor_dir, f"reference_temperature_calibration_{donor}.csv.gz")
    truth_path = os.path.join(donor_dir, f"query_truth_{donor}.csv")
    report_path = os.path.join(donor_dir, f"same_sct_report_{donor}.csv")
    needed = (query_corr_path, score_path, calib_path)
    if REBUILD_INPUTS:
        query_corr_path, score_path, calib_path, truth_path, report_path = build_same_sct_inputs(
            donor=donor,
            donor_col=donor_col,
            id_col=id_col,
            out_dir=donor_dir,
        )
    else:
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing inputs (run once with --rebuild to build them): {missing}")
        print("\n=== REUSING EXISTING INPUTS ===")
        for p in needed:
            print(p)

    if os.path.exists(report_path):
        print("\n=== SAME-SCT REPORT ===")
        print(pd.read_csv(report_path).to_string(index=False))

    query_cells, query_corr = load_rds_matrix(query_corr_path)

    id_to_donor = meta.set_index(id_col)[donor_col]
    query_donor = id_to_donor.reindex(query_cells)
    if query_donor.isna().any():
        raise ValueError("Some query cells are missing donor metadata")
    if not (query_donor.astype(str) == str(donor)).all():
        raise ValueError("Query correlation contains cells not belonging to held-out donor")

    scores = pd.read_csv(score_path)
    if "cell" not in scores.columns:
        raise ValueError("Reference score file lacks 'cell' column")
    scores = scores.set_index("cell")
    scores.index = scores.index.astype(str)

    missing = set(query_cells) - set(scores.index)
    if missing:
        raise ValueError(f"{len(missing)} query cells missing from reference scores")

    scores = scores.reindex(query_cells)
    celltypes = scores.columns.astype(str).to_numpy()
    score_mat = scores.to_numpy(dtype=float)

    calib = pd.read_csv(calib_path)
    required = {"cell", "validation_donor", "truth"}
    if not required.issubset(calib.columns):
        raise ValueError(f"Calibration file is missing columns: {required - set(calib.columns)}")

    missing_calib_types = [ct for ct in celltypes if ct not in calib.columns]
    if missing_calib_types:
        raise ValueError(f"Calibration file missing cell types: {missing_calib_types[:5]}")

    calib_scores = calib[celltypes].to_numpy(dtype=float)
    type_to_index = {ct: k for k, ct in enumerate(celltypes)}
    calib_true_index = np.array([type_to_index.get(str(x), -1) for x in calib["truth"]], dtype=int)

    temperature, temp_info = fit_temperature_reference_only(
        calib_scores, calib_true_index
    )
    pd.DataFrame([temp_info]).to_csv(
        os.path.join(donor_dir, f"{donor}_temperature_calibration_summary.csv"),
        index=False,
    )
    print("\n=== REFERENCE-ONLY TEMPERATURE CALIBRATION ===")
    print(pd.DataFrame([temp_info]).to_string(index=False))

    p0_reference = softmax_from_scores(score_mat, temperature)
    pred_reference = celltypes[np.argmax(p0_reference, axis=1)]

    tune_dir = os.path.join(donor_dir, "dense_tuning")
    if TUNING_CRITERION == "block":
        tuning = run_dense_tuning(
            query_corr,
            lambda_vec=LAMBDA_VEC,
            q_vec=Q_VEC,
            out_dir=tune_dir,
            n_jobs=TUNING_N_JOBS,
            tol_pct=TUNING_TOL_PCT,
        )
    elif TUNING_CRITERION == "frobenius":
        from icons_tune_matrix import icons_tune_matrix
        os.makedirs(tune_dir, exist_ok=True)
        tuning = icons_tune_matrix(
            query_corr, probs=Q_VEC, lam=LAMBDA_VEC, n_jobs=TUNING_N_JOBS,
            out_csv=os.path.join(tune_dir, "grid_results_Rcriterion.csv"),
        )
        tuning["best"].to_csv(os.path.join(tune_dir, "best_parameters_Rcriterion.csv"), index=False)
        print(tuning["best"].to_string(index=False))
    else:
        raise ValueError(f"unknown TUNING_CRITERION {TUNING_CRITERION!r}")

    best_lambda = float(tuning["best"]["lambda"].iloc[0])
    best_threshold = float(tuning["best"]["threshold"].iloc[0])
    print(f"best_lambda={best_lambda}, best_threshold={best_threshold}")

    icons_result = run_icons_tuned(
        query_corr,
        f"donor_{donor}",
        lam=best_lambda,
        threshold=best_threshold,
        output_dir=donor_dir,
        vmax=1.0,
        tick_step=500,
    )

    cluster_id = icons_cluster_ids(icons_result, len(query_cells))
    print(f"ICONS: {len(icons_result['sizes'])} communities, {icons_result['n_singletons']} unassigned cells "
          f"({'one pooled cluster' if ICONS_LEFTOVER_AS_ONE_CLUSTER else 'each a singleton cluster'})")

    fixed_clusters = pd.DataFrame({
        "cell": query_cells,
        "icons_cluster_id": cluster_id,
    })
    fixed_clusters.to_csv(
        os.path.join(donor_dir, f"{donor}_fixed_icons_clusters.csv"), index=False
    )

    print("\n=== FIXED ICONS STRUCTURE ===")
    print(pd.Series(cluster_id).value_counts().sort_index().to_string())

    p0_em, q_em, history, initial_pi, final_pi = icons_em_refine(
        score_mat, temperature, cluster_id
    )
    if not np.allclose(p0_em, p0_reference, atol=1e-12, rtol=1e-10):
        raise RuntimeError("Reference-only probabilities disagree between calibration and EM paths")
    pred_em = celltypes[np.argmax(q_em, axis=1)]

    pred_majority = np.empty(len(query_cells), dtype=object)
    for c in np.unique(cluster_id):
        idx = np.where(cluster_id == c)[0]
        cluster_mean_p0 = p0_reference[idx].mean(axis=0)
        pred_majority[idx] = celltypes[int(np.argmax(cluster_mean_p0))]

    pd.DataFrame(score_mat, index=query_cells, columns=celltypes).to_csv(
        os.path.join(donor_dir, f"{donor}_reference_mean_correlations.csv.gz")
    )
    pd.DataFrame(p0_reference, index=query_cells, columns=celltypes).to_csv(
        os.path.join(donor_dir, f"{donor}_reference_only_probabilities.csv.gz")
    )
    pd.DataFrame(q_em, index=query_cells, columns=celltypes).to_csv(
        os.path.join(donor_dir, f"{donor}_em_celltype_probabilities.csv.gz")
    )
    history.to_csv(
        os.path.join(donor_dir, f"{donor}_em_history.csv"), index=False
    )

    cluster_composition_table(initial_pi, celltypes, cluster_id).to_csv(
        os.path.join(donor_dir, f"{donor}_cluster_composition_initial.csv"),
        index=False,
    )
    cluster_composition_table(final_pi, celltypes, cluster_id).to_csv(
        os.path.join(donor_dir, f"{donor}_cluster_composition_final.csv"),
        index=False,
    )

    per_cell = pd.DataFrame({
        "cell": query_cells,
        "icons_cluster_id": cluster_id,
        "reference_only_label": pred_reference,
        "hard_cluster_majority_label": pred_majority,
        "em_label": pred_em,
        "reference_only_max_probability": p0_reference.max(axis=1),
        "em_max_probability": q_em.max(axis=1),
        "label_changed_by_em": pred_reference != pred_em,
        "temperature": temperature,
    })

    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path).set_index("cell").reindex(query_cells)["truth"]
        per_cell["truth"] = truth.to_numpy()

        pr_ref, macro_ref = prf_table(per_cell["truth"], pred_reference)
        pr_majority, macro_majority = prf_table(per_cell["truth"], pred_majority)
        pr_em, macro_em = prf_table(per_cell["truth"], pred_em)

        pr_ref.to_csv(
            os.path.join(donor_dir, f"{donor}_reference_only_pr.csv"), index=False
        )
        pr_majority.to_csv(
            os.path.join(donor_dir, f"{donor}_hard_cluster_majority_pr.csv"), index=False
        )
        pr_em.to_csv(
            os.path.join(donor_dir, f"{donor}_em_pr.csv"), index=False
        )

        comparison = pd.DataFrame([
            {"method": "reference_only", **macro_ref},
            {"method": "hard_fixed_ICONS_cluster_majority", **macro_majority},
            {"method": "fixed_ICONS_EM", **macro_em},
        ])
        comparison["n_query_cells"] = len(query_cells)
        comparison["n_labels_changed_by_em"] = int(
            np.sum(pred_reference != pred_em)
        )
        comparison["pct_labels_changed_by_em"] = (
            100.0 * np.mean(pred_reference != pred_em)
        )
        comparison.to_csv(
            os.path.join(donor_dir, f"{donor}_method_comparison.csv"), index=False
        )

        cluster_change = (
            per_cell.groupby("icons_cluster_id", as_index=False)
            .agg(
                n_cells=("cell", "size"),
                n_labels_changed_by_em=("label_changed_by_em", "sum"),
            )
        )
        cluster_change["pct_labels_changed_by_em"] = (
            100.0 * cluster_change["n_labels_changed_by_em"] / cluster_change["n_cells"]
        )
        cluster_change.to_csv(
            os.path.join(donor_dir, f"{donor}_em_label_changes_by_cluster_size.csv"),
            index=False,
        )

        truth_arr = per_cell["truth"].astype(str).to_numpy()
        mixture_rows = []
        model_type_set = set(celltypes.tolist())
        for c in np.unique(cluster_id):
            idx = np.where(cluster_id == c)[0]
            cluster_truth = truth_arr[idx]
            valid_truth = np.array([x not in ("<NA>", "nan", "None") for x in cluster_truth])
            cluster_truth_valid = cluster_truth[valid_truth]
            n_valid = len(cluster_truth_valid)
            n_unmodeled = int(sum(x not in model_type_set for x in cluster_truth_valid))
            for k, ct in enumerate(celltypes):
                true_prop = (
                    float(np.mean(cluster_truth_valid == ct)) if n_valid > 0 else np.nan
                )
                em_prop = float(final_pi[c][k])
                mixture_rows.append({
                    "icons_cluster_id": int(c),
                    "n_cells": int(len(idx)),
                    "n_truth_available": int(n_valid),
                    "n_truth_unmodeled": n_unmodeled,
                    "celltype": str(ct),
                    "true_composition": true_prop,
                    "em_final_composition": em_prop,
                    "absolute_composition_error": (
                        abs(em_prop - true_prop) if np.isfinite(true_prop) else np.nan
                    ),
                })
        pd.DataFrame(mixture_rows).to_csv(
            os.path.join(donor_dir, f"{donor}_true_vs_em_cluster_composition.csv"),
            index=False,
        )

        print("\n=== METHOD COMPARISON ===")
        print(comparison.to_string(index=False))

    per_cell.to_csv(
        os.path.join(donor_dir, f"{donor}_per_cell_predictions.csv"), index=False
    )

    print("\n=== DONE ===")
    print("query donor:", donor)
    print("n query cells:", len(query_cells))
    print("n reference cell types:", len(celltypes))
    print("n fixed ICONS clusters:", len(np.unique(cluster_id)))
    print("labels changed by EM:", int(np.sum(pred_reference != pred_em)))
    print("EM iterations:", len(history))
    if len(history):
        print("final log-likelihood:", history.iloc[-1]["log_likelihood"])
        print("all EM likelihood steps nondecreasing:", bool(history["log_likelihood_nondecreasing"].all()))
        print("final max cell change:", history.iloc[-1]["max_cell_probability_change"])
        print("final max cluster-composition change:", history.iloc[-1]["max_cluster_composition_change"])
    print("outputs:", donor_dir)

    del query_corr
    gc.collect()


def main():
    global REBUILD_INPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default=TEST_DONOR)
    parser.add_argument("--rebuild", action="store_true", help="rebuild same-SCT inputs with Seurat SCTransform (needed once per donor)")
    args = parser.parse_args()
    if args.rebuild:
        REBUILD_INPUTS = True
    print(f"REBUILD_INPUTS={REBUILD_INPUTS}")
    run_one_donor(args.donor)


if __name__ == "__main__":
    main()
