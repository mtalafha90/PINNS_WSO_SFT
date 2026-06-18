# analyze_nmci.py
# Read PINN outputs (field.npy) and compute NMCI components + diagnostics.
# Works for one or multiple result directories (each containing field.npy).
import os, argparse, json, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def logistic(A, Dmax, k, A0):
    return Dmax / (1.0 + np.exp(-k*(A - A0)))

def shoelace_area(x, y):
    """Signed polygon area; for a loop in (x,y) plane."""
    x = np.asarray(x); y = np.asarray(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def permutation_entropy(x, m=3, tau=1):
    """Simple permutation entropy (base e)."""
    x = np.asarray(x)
    if len(x) < (m-1)*tau + 1:
        return np.nan
    patterns = {}
    for i in range(len(x) - (m-1)*tau):
        window = x[i:(i+(m*tau)):tau]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    p = np.array(list(patterns.values()), dtype=float)
    p /= p.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return H / np.log(math.factorial(m))  # normalize to [0,1]

def lz_entropy_rate_binary(x):
    """Lempel–Ziv complexity (binary)."""
    x = np.asarray(x)
    if len(x) < 10:
        return np.nan
    # Symbolize by median
    med = np.median(x)
    s = ''.join('1' if v >= med else '0' for v in x)
    i, c, l = 0, 1, 1
    n = len(s)
    while True:
        if i + l > n:
            c += 1
            break
        sub = s[i:i+l]
        if s[:i].find(sub) != -1:
            l += 1
        else:
            c += 1
            i += l
            l = 1
        if i + l > n:
            break
    # Normalize by n/log n
    return c * np.log2(n) / n

def simple_ar_rmse(x, p=1):
    """AR(p) by least squares; return one-step RMSE on holdout tail."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20 or N <= p+5:
        return np.nan
    train = x[:-p]
    Y = x[p:]
    X = np.column_stack([x[p-k-1: N-k-1] for k in range(p)])
    # Align lengths
    m = min(len(Y), X.shape[0])
    Y = Y[:m]; X = X[:m]
    # Split 80/20
    split = int(0.8*m)
    Xtr, Ytr = X[:split], Y[:split]
    Xte, Yte = X[split:], Y[split:]
    coef, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    pred = Xte @ coef
    return float(np.sqrt(np.mean((pred - Yte)**2)))

def dfa_alpha(x):
    """Very lightweight DFA-1 alpha (not full MFDFA)."""
    x = np.asarray(x, dtype=float)
    X = np.cumsum(x - np.mean(x))
    Ns = np.unique(np.logspace(1, np.log10(len(x)/4), 10).astype(int))
    Fs = []
    for n in Ns:
        if n < 4: 
            continue
        segs = len(X)//n
        if segs < 2:
            continue
        res = []
        for i in range(segs):
            seg = X[i*n:(i+1)*n]
            t = np.arange(n)
            a, b = np.polyfit(t, seg, 1)
            res.append(np.sqrt(np.mean((seg - (a*t+b))**2)))
        Fs.append(np.mean(res))
    if len(Fs) < 2:
        return np.nan
    Ns = np.array(Ns[:len(Fs)], dtype=float)
    Fs = np.array(Fs, dtype=float)
    coeffs = np.polyfit(np.log(Ns), np.log(Fs+1e-12), 1)
    return float(coeffs[0])  # scaling exponent

def normalize_series(values):
    """Min-max normalize to [0,1] with epsilon guard."""
    v = np.array(values, dtype=float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax-vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)

# ----------------------------
# Core physics-ish helpers
# ----------------------------
def build_lat_grid(lat_points):
    # model grid assumed uniform in degrees [-90,90]
    lat_deg = np.linspace(-90, 90, lat_points)
    lat_rad = np.deg2rad(lat_deg)
    return lat_deg, lat_rad

def dipole_moment_from_B(B, lat_rad):
    """Proxy dipole: ∫ B(λ,t) sinλ cosλ dλ (trapezoid). B shape (T, L)."""
    weight = np.sin(lat_rad)*np.cos(lat_rad)
    dm = _trapz(B*weight[None, :], lat_rad, axis=1)
    return dm

def polar_fields(B, lat_deg, pole_cut=55):
    maskN = lat_deg >= pole_cut
    maskS = lat_deg <= -pole_cut
    BN = np.mean(B[:, maskN], axis=1) if maskN.any() else np.full(B.shape[0], np.nan)
    BS = np.mean(B[:, maskS], axis=1) if maskS.any() else np.full(B.shape[0], np.nan)
    return BN, BS

def active_band_amplitude(B, lat_deg, band=(10, 35)):
    """Amplitude proxy within |lat| in band; 95th percentile of |B| per cycle."""
    mask = (np.abs(lat_deg) >= band[0]) & (np.abs(lat_deg) <= band[1])
    return np.nanpercentile(np.abs(B[:, mask]), 95, axis=1)

def split_cycles(time_years, P_years=11.0):
    """Return list of index ranges (start, end) for each ~11y cycle."""
    t0 = time_years[0]
    t_end = time_years[-1]
    edges = np.arange(t0, t_end + 1e-9, P_years)
    if edges[-1] < t_end - 1e-6:
        edges = np.append(edges, t_end)
    idx_ranges = []
    for i in range(len(edges)-1):
        start = edges[i]; end = edges[i+1]
        idx = np.where((time_years >= start) & (time_years < end))[0]
        if len(idx) > 5:
            idx_ranges.append((idx[0], idx[-1]))
    return idx_ranges

# ----------------------------
# NMCI components for one run
# ----------------------------
def compute_components_for_run(field_path, simul_time_years=11.0, lat_points=181, save_dir=None):
    # Load field
    B = np.load(field_path)  # shape (T, L)
    T, L = B.shape
    lat_deg, lat_rad = build_lat_grid(lat_points=L)  # infer from file
    # Time axis in years [0, simul_time]
    t_years = np.linspace(0, simul_time_years, T)

    # Dipole and polar fields
    dip = dipole_moment_from_B(B, lat_rad)
    BN, BS = polar_fields(B, lat_deg)
    Bpol = 0.5*(BN - BS)  # simple dipole proxy from poles

    # Amplitude proxy A(t)
    A_t = active_band_amplitude(B, lat_deg, band=(10,35))

    # Cycles
    cycles = split_cycles(t_years, P_years=11.0)
    A_cycle = []
    D_cycle = []
    for (i0, i1) in cycles:
        A_cycle.append(float(np.nanmax(A_t[i0:i1+1])))
        # final dipole near end: mean over last 10% of that cycle window
        k = max(5, int(0.1*(i1-i0+1)))
        D_cycle.append(float(np.nanmean(dip[i1-k+1:i1+1])))
    A_cycle = np.array(A_cycle); D_cycle = np.array(D_cycle)

    # ---------- Amplitude–Response nonlinearity  (A) ----------
    A_score_parts = []

    # curvature via spline
    try:
        if len(np.unique(A_cycle)) >= 4:
            sp = UnivariateSpline(A_cycle, D_cycle, s=0.0)
            dd = sp.derivative(n=2)(A_cycle)
            curv = np.mean(dd**2)
        else:
            curv = np.nan
    except Exception:
        curv = np.nan
    A_score_parts.append(curv)

    # logistic saturation
    try:
        p0 = [np.max(D_cycle), 1.0, np.median(A_cycle)]
        popt, _ = curve_fit(logistic, A_cycle, D_cycle, p0=p0, maxfev=20000)
        Dmax, k_sat, A0 = popt
    except Exception:
        Dmax, k_sat, A0 = np.nan, np.nan, np.nan
    A_score_parts.append(k_sat)

    # hysteresis area (across full run using A(t) vs D(t))
    try:
        # Reorder by phase within an 11-year template
        phase = (t_years % 11.0) / 11.0
        order = np.argsort(phase)
        area_loop = shoelace_area(A_t[order], dip[order])
    except Exception:
        area_loop = np.nan
    A_score_parts.append(area_loop)

    # ---------- Spatial asymmetry & coupling  (S) ----------
    try:
        delta_H = np.nanmean(np.abs(BN - (-BS))) / (np.nanmean(np.abs(B)) + 1e-8)
    except Exception:
        delta_H = np.nan

    # butterfly skewness: latitude pdf weighted by |B| in active band
    try:
        mask = (np.abs(lat_deg) >= 10) & (np.abs(lat_deg) <= 35)
        w = np.nanmean(np.abs(B[:, mask]), axis=0)  # mean over time
        lat_pdf = w / (np.sum(w) + 1e-12)
        lat_vals = lat_deg[mask]
        # Skew of pdf using discrete approx (centered)
        mu = np.sum(lat_vals * lat_pdf)
        sig = np.sqrt(np.sum(((lat_vals - mu)**2) * lat_pdf))
        skew_bfly = np.sum(((lat_vals - mu)**3) * lat_pdf) / (sig**3 + 1e-12)
    except Exception:
        skew_bfly = np.nan

    # No direct tilt series available; set MI placeholder
    I_tilt_lat = np.nan
    S_parts = [delta_H, abs(skew_bfly), I_tilt_lat]

    # ---------- Temporal irreversibility / hysteresis  (T) ----------
    dx = np.diff(dip)
    R = np.nanmean(dx**3) if len(dx) > 2 else np.nan
    try:
        Hf = permutation_entropy(dip, m=3, tau=1)
        Hr = permutation_entropy(dip[::-1], m=3, tau=1)
        perm_asym = abs(Hf - Hr)
    except Exception:
        perm_asym = np.nan

    T_parts = [abs(R), perm_asym, area_loop]

    # ---------- Multifractality / Non-Gaussianity  (M) ----------
    alpha = dfa_alpha(dip)  # proxy for scaling; not full MFDFA
    # Residuals vs smooth spline over time (order 3)
    try:
        sp_t = UnivariateSpline(np.arange(T), dip, s=0.1* np.var(dip) * T)
        resid = dip - sp_t(np.arange(T))
        k_excess = max(0.0, kurtosis(resid, fisher=True, bias=False))
    except Exception:
        k_excess = np.nan
    M_parts = [abs(alpha-0.5), k_excess]  # alpha far from 0.5 suggests complexity

    # ---------- Entropic / Predictability  (E) ----------
    Hrate = lz_entropy_rate_binary(dip)
    rmse_ar1 = simple_ar_rmse(dip, p=1)
    rmse_ar5 = simple_ar_rmse(dip, p=5)
    Q = np.nan
    if np.isfinite(rmse_ar1) and np.isfinite(rmse_ar5) and rmse_ar5 > 1e-12:
        Q = rmse_ar1 / rmse_ar5  # >1 ⇒ nonlinear/long-memory advantage
    E_parts = [Hrate, Q]

    # Normalize components within this run (if you analyze multiple runs,
    # you can later re-normalize across runs)
    def nrm_list(lst):
        return normalize_series([v if np.isfinite(v) else np.nan for v in lst])

    A_norm = float(np.nanmean(nrm_list(A_score_parts)))
    S_norm = float(np.nanmean(nrm_list(S_parts)))
    T_norm = float(np.nanmean(nrm_list(T_parts)))
    M_norm = float(np.nanmean(nrm_list(M_parts)))
    E_norm = float(np.nanmean(nrm_list(E_parts)))

    NMCI = np.nanmean([A_norm, S_norm, T_norm, M_norm, E_norm])

    # Save diagnostics
    if save_dir:
        ensure_dir(save_dir)
        # D(A) plot
        if len(A_cycle) >= 2:
            plt.figure(figsize=(5,4))
            plt.scatter(A_cycle, D_cycle, s=30)
            xs = np.linspace(min(A_cycle), max(A_cycle), 200)
            try:
                plt.plot(xs, logistic(xs, *(Dmax, k_sat, A0)), lw=1)
            except Exception:
                pass
            plt.xlabel("Cycle amplitude proxy A")
            plt.ylabel("Final dipole D")
            plt.title("Amplitude → Dipole response")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "D_vs_A.png"), dpi=150)
            plt.close()

        # Dipole & polar fields
        plt.figure(figsize=(8,3))
        plt.plot(t_years, dip, label="Dipole")
        if np.isfinite(BN).all(): plt.plot(t_years, BN, label="Polar N", alpha=0.7)
        if np.isfinite(BS).all(): plt.plot(t_years, -BS, label="(-) Polar S", alpha=0.7)
        plt.xlabel("Time [years]")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dipole_polar_time.png"), dpi=150)
        plt.close()

        # Butterfly (absolute B) for context
        plt.figure(figsize=(8,3))
        plt.contourf(t_years, lat_deg, np.abs(B).T, levels=60, cmap="magma")
        plt.xlabel("Time [years]"); plt.ylabel("Latitude [deg]")
        plt.title("Butterfly |B|")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "butterfly_absB.png"), dpi=150)
        plt.close()

    components = {
        "A_parts": {"curvature_mean_dd2": curv, "logistic_k": k_sat, "hysteresis_area": area_loop},
        "S_parts": {"hemi_asym": delta_H, "butterfly_skew_abs": abs(skew_bfly), "MI_tilt_lat": I_tilt_lat},
        "T_parts": {"R_third_moment": R, "perm_entropy_asym": perm_asym, "loop_area": area_loop},
        "M_parts": {"dfa_alpha_offset": (alpha-0.5), "kurtosis_excess_resid": k_excess},
        "E_parts": {"lz_entropy_rate": Hrate, "Q_ar1_over_ar5": Q},
        "normalized": {"A": A_norm, "S": S_norm, "T": T_norm, "M": M_norm, "E": E_norm, "NMCI": NMCI},
        "per_cycle": {"A_cycle": A_cycle.tolist(), "D_cycle": D_cycle.tolist()},
        "meta": {"T": T, "L": L, "simul_time_years": simul_time_years}
    }
    return components
def to_jsonable(obj):
    """Recursively convert NumPy objects to plain Python JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze PINN field.npy and compute NMCI.")
    ap.add_argument("--results", nargs="+", required=True,
                    help="Result directories each containing field.npy")
    ap.add_argument("--simul-time", type=float, default=11.0, help="Total simulated time in years")
    ap.add_argument("--lat-points", type=int, default=181, help="Latitude points (for labeling only)")
    ap.add_argument("--save-dir", type=str, default="analysis_out", help="Where to save CSV/plots")
    args = ap.parse_args()

    ensure_dir(args.save_dir)
    all_rows = []
    summary = {}

    for res_dir in args.results:
        field_path = os.path.join(res_dir, "field.npy")
        if not os.path.isfile(field_path):
            print(f"[WARN] No field.npy in {res_dir}, skipping.")
            continue
        run_name = os.path.basename(os.path.normpath(res_dir))
        outdir = ensure_dir(os.path.join(args.save_dir, run_name))

        comps = compute_components_for_run(field_path, simul_time_years=args.simul_time,
                                           lat_points=args.lat_points, save_dir=outdir)
        # Save JSON per run
        with open(os.path.join(outdir, "nmci_components.json"), "w") as f:
            json.dump(to_jsonable(comps), f, indent=2)

        # Row for CSV-ish summary
        row = {
            "run": run_name,
            "A": comps["normalized"]["A"],
            "S": comps["normalized"]["S"],
            "T": comps["normalized"]["T"],
            "M": comps["normalized"]["M"],
            "E": comps["normalized"]["E"],
            "NMCI": comps["normalized"]["NMCI"]
        }
        all_rows.append(row)
        summary[run_name] = comps["normalized"]

    # Write a simple CSV
    csv_path = os.path.join(args.save_dir, "nmci_summary.csv")
    with open(csv_path, "w") as f:
        cols = ["run","A","S","T","M","E","NMCI"]
        f.write(",".join(cols)+"\n")
        for r in all_rows:
            f.write(",".join([str(r[c]) for c in cols])+"\n")
    print(f"[OK] Wrote summary to {csv_path}")
    print("[OK] Per-run JSONs and plots saved under:", args.save_dir)

if __name__ == "__main__":
    main()
