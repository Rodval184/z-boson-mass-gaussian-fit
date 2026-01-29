# -*- coding: utf-8 -*-
"""
Z boson mass reconstruction from dimuon events (CSV).

- Computes invariant mass from E, px, py, pz of two muons
- Builds histograms (linear + log)
- Zooms into Z region (default 60–120 GeV)
- Finds peaks and performs Gaussian fits around each detected peak
- Saves figures to ./figures

Author: Benjamín Rodríguez Valdez
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def gauss(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def invariant_mass(df: pd.DataFrame) -> np.ndarray:
    E = df["E1"].to_numpy() + df["E2"].to_numpy()
    px = df["px1"].to_numpy() + df["px2"].to_numpy()
    py = df["py1"].to_numpy() + df["py2"].to_numpy()
    pz = df["pz1"].to_numpy() + df["pz2"].to_numpy()
    m2 = E**2 - (px**2 + py**2 + pz**2)
    m2 = np.maximum(m2, 0.0)  # avoid tiny negative due to floating error
    return np.sqrt(m2)


def save_histogram(m: np.ndarray, bins: int, title: str, outpath: Path, logy: bool = False) -> None:
    plt.figure(figsize=(11, 8))
    plt.hist(m, bins=bins, edgecolor="black", linewidth=0.5)
    if logy:
        plt.yscale("log")
    plt.xlabel(r"Invariant mass $m$ [GeV]")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def fit_peak_gaussian(m: np.ndarray, center_guess: float, window: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (popt, perr) for a Gaussian fit in a window around center_guess."""
    mask = (m >= center_guess - window) & (m <= center_guess + window)
    m_local = m[mask]
    if len(m_local) < 20:
        raise RuntimeError("Not enough events in the selected window to fit.")

    counts, edges = np.histogram(m_local, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    A0 = float(np.max(counts))
    mu0 = float(center_guess)
    sigma0 = 2.0  # GeV, reasonable around Z

    popt, pcov = curve_fit(gauss, centers, counts, p0=[A0, mu0, sigma0])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def main() -> None:
    ap = argparse.ArgumentParser(description="Invariant mass reconstruction + Gaussian fit (Z region).")
    ap.add_argument("--csv", type=str, default="data/MuRun2010B.csv", help="Path to input CSV file.")
    ap.add_argument("--bins", type=int, default=110, help="Bins for full histogram.")
    ap.add_argument("--zmin", type=float, default=60.0, help="Min mass for Z window (GeV).")
    ap.add_argument("--zmax", type=float, default=120.0, help="Max mass for Z window (GeV).")
    ap.add_argument("--zbins", type=int, default=55, help="Bins for Z-window histogram.")
    ap.add_argument("--peak_height_frac", type=float, default=0.2, help="Peak threshold as fraction of max in Z histogram.")
    ap.add_argument("--fit_window", type=float, default=5.0, help="Fit window half-width around peak center (GeV).")
    ap.add_argument("--fit_bins", type=int, default=40, help="Bins for local histogram used in Gaussian fit.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Tip: put your file in ./data/ and run:\n"
            "  python src/main.py --csv data/MuRun2010B.csv"
        )

    df = pd.read_csv(csv_path)
    m = invariant_mass(df)

    # Full histograms
    save_histogram(
        m, args.bins,
        title=f"Invariant mass histogram ({csv_path.name})",
        outpath=Path("figures/hist_full.png"),
        logy=False
    )
    save_histogram(
        m, args.bins,
        title=f"Invariant mass histogram (log scale) ({csv_path.name})",
        outpath=Path("figures/hist_full_log.png"),
        logy=True
    )

    # Z window
    mask_z = (m > args.zmin) & (m < args.zmax)
    m_z = m[mask_z]

    counts_z, edges_z = np.histogram(m_z, bins=args.zbins, range=(args.zmin, args.zmax))
    centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])

    peak_height = np.max(counts_z) * float(args.peak_height_frac)
    peaks, props = find_peaks(counts_z, height=peak_height)

    print(f"Z window: [{args.zmin}, {args.zmax}] GeV")
    print(f"Detected peaks (threshold = {args.peak_height_frac:.2f} of max):")
    if len(peaks) == 0:
        print("  None found. Try lowering --peak_height_frac (e.g., 0.1).")
    else:
        for p in peaks:
            print(f"  peak at ~ {centers_z[p]:.3f} GeV (counts={counts_z[p]})")

    # Plot Z window + peak markers (linear + log)
    for logy, fname in [(False, "hist_z.png"), (True, "hist_z_log.png")]:
        plt.figure(figsize=(11, 8))
        plt.hist(m_z, bins=args.zbins, range=(args.zmin, args.zmax),
                 edgecolor="black", linewidth=0.5, label="Events")
        for i, p in enumerate(peaks):
            label = f"Peak ~ {centers_z[p]:.2f} GeV" if i == 0 else None
            plt.axvline(centers_z[p], linestyle="--", linewidth=2, label=label)
        if logy:
            plt.yscale("log")
        plt.xlabel(r"Invariant mass $m$ [GeV]")
        plt.ylabel("Counts")
        plt.title(f"Z region ({args.zmin:.0f}–{args.zmax:.0f} GeV) — {csv_path.name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        Path("figures").mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(Path("figures") / fname, dpi=200)
        plt.close()

    # Gaussian fits around detected peaks
    for p in peaks:
        center_guess = float(centers_z[p])
        try:
            popt, perr = fit_peak_gaussian(m, center_guess, window=args.fit_window, bins=args.fit_bins)
            A, mu, sigma = popt
            eA, emu, esig = perr
            print(f"\nGaussian fit around {center_guess:.3f} GeV (window ±{args.fit_window} GeV):")
            print(f"  mu    = {mu:.4f} ± {emu:.4f} GeV")
            print(f"  sigma = {sigma:.4f} ± {esig:.4f} GeV")
        except Exception as e:
            print(f"\nFit failed near {center_guess:.3f} GeV: {e}")


if __name__ == "__main__":
    main()
