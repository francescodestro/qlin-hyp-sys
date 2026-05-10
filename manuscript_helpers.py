"""Case study builders, reference solver, I/O, and figure utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerics import (
    CaseStudy,
    analytic_solution_constA,
    interp_ref_to,
    l2_error_total,
    run_case_study,
    solve_mol,
)

# --- Case study builders ---

def build_case_constA3(selector_mode: str = "minimax") -> CaseStudy:
    """Case 1: constant diagonal A (3x3), g=0."""
    L = 2 * np.pi
    K, T = 200, 0.5
    A_const = np.diag([0.2, 1.0, 3.0])

    def A_fun(u, z, t):
        return A_const

    def g_fun(u, z, t):
        return np.zeros(3, dtype=float)

    def u0_fun(z):
        center, sigma = 0.5 * L, 0.25 * L
        envelope = np.exp(-((z - center) / sigma) ** 2)
        return envelope * np.array([np.sin(z), 0.5 * np.sin(2 * z), -0.25 * np.sin(3 * z)])

    return CaseStudy(
        name=f"CS1_{selector_mode}",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode=selector_mode,
        u_in_left=np.zeros(3), u_in_right=np.zeros(3), Nz_ref=50000,
    )


def build_case_smooth_anisotropic() -> CaseStudy:
    """Case 2: smooth anisotropic advection (n=2)."""
    L = 2 * np.pi
    K, T = 200, 1.0
    A_const = np.diag([1.0, 2.0])

    def A_fun(u, z, t):
        return A_const + 0.5 * np.diag([u[1], u[1]])

    def g_fun(u, z, t):
        return np.array([np.sin(u[0]), np.sin(u[0])], dtype=float)

    def u0_fun(z):
        return np.array([np.sin(z), np.sin(z)], dtype=float)

    return CaseStudy(
        name="CS2",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="expected",
        u_in_left=np.zeros(2), u_in_right=np.zeros(2), Nz_ref=50000,
    )


def build_case_state_dependent_eigenvectors() -> CaseStudy:
    """Case 3: state-dependent eigenvectors (n=2)."""
    L = 2 * np.pi
    K, T = 200, 1.0

    def A_fun(u, z, t):
        a11 = 1.0 + 0.5 * float(u[1])
        a22 = 2.0 + 0.5 * float(u[1])
        a12 = 0.6 * np.tanh(float(u[0]))
        return np.array([[a11, a12], [a12, a22]], dtype=float)

    def g_fun(u, z, t):
        return np.array([np.sin(u[0]), np.sin(u[0])], dtype=float)

    def u0_fun(z):
        return np.array([np.sin(z), np.sin(z)], dtype=float)

    return CaseStudy(
        name="CS3",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="expected",
        u_in_left=np.zeros(2), u_in_right=np.zeros(2), Nz_ref=50000,
    )


def build_case_mixed_spectrum_homogeneous_boundaries() -> CaseStudy:
    """Case 4: mixed spectrum with homogeneous inflow (n=2)."""
    L = 2 * np.pi
    K, T = 200, 1.0

    def A_fun(u, _z, _t):
        s = np.tanh(float(u[0]))
        return np.diag([1.2 + 0.15 * s, -0.7 - 0.15 * s])

    def g_fun(u, _z, _t):
        return -0.05 * np.asarray(u, dtype=float).reshape(-1)

    def u0_fun(z):
        center, sigma = 0.5 * L, 0.15 * L
        env = np.exp(-((z - center) / sigma) ** 2)
        return env * np.array([0.4 * np.sin(z), 0.25 * np.cos(2 * z)])

    return CaseStudy(
        name="CS4",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="expected",
        u_in_left=np.zeros(2), u_in_right=np.zeros(2), Nz_ref=50000,
    )


def build_case_local_forcing() -> CaseStudy:
    """Case A1: localized source forcing (n=2)."""
    L = 2 * np.pi
    K, T = 200, 1.0
    A_const = np.diag([2.0, 1.0])

    def A_fun(u, z, t):
        return A_const + np.diag([0.2 * np.linalg.norm(u), 1.0 * np.linalg.norm(u)])

    def g_fun(u, z, t):
        center = 0.55 * L
        spatial = np.exp(-((z - center) / (0.08 * L)) ** 2)
        temporal = np.exp(-((t - 0.4) / 0.12) ** 2)
        pulse = 7.5 * spatial * temporal
        return np.array([pulse, -0.6 * pulse], dtype=float) - np.sin(u[1])

    def u0_fun(z):
        return np.array([0.2 * np.sin(3 * z), 0.1 * np.sin(2 * z)], dtype=float)

    return CaseStudy(
        name="CSA1",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="expected",
        u_in_left=np.zeros(2), u_in_right=np.zeros(2), Nz_ref=50000,
    )


def build_case_coupled_constA3() -> CaseStudy:
    """Case A2: coupled constant coefficients (n=3)."""
    L = 2 * np.pi
    K, T = 200, 0.5
    # A = R diag([0.2,1,3]) inv(R) for R=[[1,1,0],[0,1,1],[1,0,1]]
    A_const = np.array([[0.6, 0.4, -0.4], [-1.0, 2.0, 1.0], [-1.4, 1.4, 1.6]])

    def A_fun(_u, _z, _t):
        return A_const

    def g_fun(_u, _z, _t):
        return np.zeros(3, dtype=float)

    def u0_fun(z):
        center, sigma = 0.5 * L, 0.25 * L
        envelope = np.exp(-((z - center) / sigma) ** 2)
        return envelope * np.array([np.sin(z), 0.5 * np.sin(2 * z), -0.25 * np.sin(3 * z)])

    return CaseStudy(
        name="CSA2",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="minimax",
        u_in_left=np.zeros(3), u_in_right=np.zeros(3), Nz_ref=50000,
    )



def build_case_anisotropic_inflow() -> CaseStudy:
    """Case A3: anisotropic inflow with state-dependent coupling (n=2)."""
    L = 2 * np.pi
    K, T = 300, 1.0

    def A_fun(u, z, t):
        coupling = 0.3 * np.tanh(u[0] - u[1])
        return np.array([
            [2.0 + 0.3 * np.sin(u[0]), coupling],
            [coupling, 1.0 + 0.2 * np.sin(u[1])]
        ])

    def g_fun(u, z, t):
        center = 0.4 * L
        spatial = np.exp(-((z - center) / (0.1 * L)) ** 2)
        temporal = np.sin(2 * np.pi * t)
        return np.array([
            -0.2 * u[0] + 0.4 * u[1] * np.cos(z) + spatial * temporal,
            0.3 * u[0] * np.sin(z) - 0.15 * u[1] + 0.5 * spatial * temporal,
        ])

    def u0_fun(z):
        return np.array([0.3 * np.sin(z) + 0.1 * np.cos(2 * z),
                         0.2 * np.sin(2 * z) - 0.1 * np.sin(z)], dtype=float)

    def left_bc(t):
        return np.array([0.3 * np.sin(2 * np.pi * t),
                         0.2 * np.cos(4 * np.pi * t)])

    return CaseStudy(
        name="CSA3",
        L=L, K=K, T=T, A_fun=A_fun, g_fun=g_fun, u0_fun=u0_fun,
        selector_mode="expected",
        u_in_left=left_bc, u_in_right=np.zeros(2), Nz_ref=50000,
    )


# --- MOL reference solver ---

def mol_reference_uniform_grid(
    case: CaseStudy, *, K_ref: int, atol: float, rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """MOL reference on a uniform grid (generic, any n)."""
    L = float(case.L)
    Z_ref = np.linspace(0.0, L, int(K_ref), endpoint=False, dtype=float)
    u0_ref = np.stack([case.u0_fun(float(z)) for z in Z_ref], axis=1)
    u_in_left = case.u_in_left if callable(case.u_in_left) else (
        np.asarray(case.u_in_left, dtype=float) if case.u_in_left is not None else None)
    u_in_right = case.u_in_right if callable(case.u_in_right) else (
        np.asarray(case.u_in_right, dtype=float) if case.u_in_right is not None else None)
    _ts, sols, _dts, _stats = solve_mol(
        u0_ref, Z_ref, float(case.T), L, case.A_fun, case.g_fun,
        scheme_name="SSPRK33", atol=float(atol), rtol=float(rtol),
        cfl_safety=1.0, record_every=10**9, collect_debug=False,
        u_in_left=u_in_left, u_in_right=u_in_right,
    )
    return Z_ref, np.asarray(sols[-1], dtype=float)


# --- Figure generation ---

def save_case_figure(
    case_dir: Path, *, outpath: Path,
    include_reference: bool = True,
    reference_label: str = r"Ref. ($K_{\rm ref}=50\,000$)",
) -> None:
    """Save a profiles + dt-history figure for one case study."""
    case_dir = Path(case_dir)
    zH, UH = load_profile(case_dir, "hybrid")
    zM, UM = load_profile(case_dir, "mol")
    ref_ok = include_reference and (case_dir / "reference_final_profile.csv").exists()
    if ref_ok:
        zR, UR = load_profile(case_dir, "reference")

    stepH, dtH = load_dt(case_dir, "hybrid")
    stepM, dtM = load_dt(case_dir, "mol")

    n = int(UH.shape[0])
    fig, axes = plt.subplots(n + 1, 1, figsize=(7.5, 2.1 * (n + 1)), sharex=False)

    ax = axes[0]
    if dtH is not None:
        ax.plot(stepH, dtH, label="Hybrid", linewidth=2, color="C0")
    if dtM is not None:
        ax.plot(stepM, dtM, "--", label="MOL", linewidth=2.5, color="k")
    ax.set_yscale("log"); ax.set_xlabel("Accepted step index"); ax.set_ylabel(r"Step size $\Delta t$")
    ax.grid(True, which="both", alpha=0.25)

    for i in range(n):
        ax = axes[i + 1]
        ax.plot(zH, UH[i, :], label="Hybrid" if i == 0 else None, linewidth=2, color="C0")
        ax.plot(zM, UM[i, :], label="MOL" if i == 0 else None, linewidth=2.5, color="k", linestyle="--", alpha=0.85)
        if ref_ok:
            ax.plot(zR, UR[i, :], label=reference_label if i == 0 else None, linewidth=3.5, color="C3", linestyle=":", alpha=0.95)
        ax.set_ylabel(rf"$u_{{{i+1}}}(T)$"); ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"$z$")
    for idx, ax in enumerate(axes):
        ax.text(-0.12, 1.0, chr(97+idx), transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def save_case1_selector_figure(
    case_obj: CaseStudy, minimax_dir: Path, expected_dir: Path, *, outpath: Path,
) -> None:
    """Save the selector-comparison figure for Case 1."""
    zM, UM = load_profile(minimax_dir, "mol")
    stepM, dtM = load_dt(minimax_dir, "mol")
    zH1, UH1 = load_profile(minimax_dir, "hybrid")
    stepH1, dtH1 = load_dt(minimax_dir, "hybrid")
    zH2, UH2 = load_profile(expected_dir, "hybrid")
    stepH2, dtH2 = load_dt(expected_dir, "hybrid")

    n = int(UH1.shape[0])
    A0 = np.asarray(case_obj.A_fun(np.zeros_like(case_obj.u0_fun(0.0)), 0.0, 0.0), dtype=float)
    UR = analytic_solution_constA(A0, zM, float(case_obj.T), case_obj.u0_fun, float(case_obj.L))
    fig, axes = plt.subplots(n + 1, 1, figsize=(7.5, 2.1 * (n + 1)), sharex=False)

    ax = axes[0]
    if dtM is not None:
        ax.plot(stepM, dtM, "--", label="MOL", linewidth=2.5, color="k")
    if dtH1 is not None:
        ax.plot(stepH1, dtH1, label="Hybrid (minimax)", linewidth=2, color="C0")
    if dtH2 is not None:
        ax.plot(stepH2, dtH2, label="Hybrid (mean eigenvalue)", linewidth=2, color="C2")
    ax.set_yscale("log"); ax.set_xlabel("Accepted step index"); ax.set_ylabel(r"Step size $\Delta t$")
    ax.grid(True, which="both", alpha=0.25)

    for i in range(n):
        ax = axes[i + 1]
        ax.plot(zM, UM[i, :], label="MOL" if i == 0 else None, linewidth=2.5, color="k", linestyle="--", alpha=0.85)
        ax.plot(zM, UR[i, :], label="Analytic" if i == 0 else None, linewidth=3.5, color="C3", linestyle=":", alpha=0.95)
        ax.plot(zH1, UH1[i, :], label="Hybrid (minimax)" if i == 0 else None, linewidth=2, color="C0")
        ax.plot(zH2, UH2[i, :], label="Hybrid (mean eigenvalue)" if i == 0 else None, linewidth=2, color="C2")
        ax.set_ylabel(rf"$u_{{{i+1}}}(T)$"); ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"$z$")
    for idx, ax in enumerate(axes):
        ax.text(-0.12, 1.0, chr(97+idx), transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


# --- Error metrics ---

def relative_L2_error_to_reference(
    *, u_num: np.ndarray, Z_num: np.ndarray,
    U_ref_T: np.ndarray, Z_ref: np.ndarray, L: float,
) -> float:
    """Relative L2 error ``||u_num - u_ref|| / ||u_ref||`` on the grid ``Z_num``."""
    Uref_on = interp_ref_to(Z_ref, U_ref_T, Z_num)
    err = float(l2_error_total(u_num, Z_num, Uref_on, L))
    nrm = float(l2_error_total(Uref_on, Z_num, np.zeros_like(Uref_on), L))
    return float(err / nrm) if nrm > 0 else np.nan

def relative_L2_error_to_analytic(
    *, u_num: np.ndarray, Z_num: np.ndarray,
    A_const: np.ndarray, T: float, u0_fun, L: float,
) -> float:
    """Relative L2 error against the analytic solution for constant A, g=0."""
    U_exact = analytic_solution_constA(
        np.asarray(A_const, dtype=float), Z_num, float(T), u0_fun, float(L),
    )
    err = float(l2_error_total(u_num, Z_num, U_exact, float(L)))
    nrm = float(l2_error_total(U_exact, Z_num, np.zeros_like(U_exact), float(L)))
    return float(err / nrm) if nrm > 0 else np.nan


def relative_L2_change(
    *, Z_base: np.ndarray, U_base: np.ndarray,
    Z_fine: np.ndarray, U_fine: np.ndarray, L: float,
) -> float:
    """Relative L2 difference ``||U_base - U_fine|| / ||U_fine||`` on ``Z_base``."""
    Ufine_on_base = interp_ref_to(Z_fine, U_fine, Z_base)
    diff = float(l2_error_total(U_base, Z_base, Ufine_on_base, float(L)))
    nrm = float(l2_error_total(Ufine_on_base, Z_base, np.zeros_like(Ufine_on_base), float(L)))
    return float(diff / nrm) if nrm > 0 else np.nan



# --- I/O utilities ---

def load_profile(case_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ``<label>_final_profile.csv`` and return ``(z, U)`` with ``U`` shape ``(n, K)``."""
    df = pd.read_csv(Path(case_dir) / f"{label}_final_profile.csv")
    z = df["z"].to_numpy(dtype=float)
    cols = [c for c in df.columns if c.startswith(f"{label}_u")]
    U = df[cols].to_numpy(dtype=float).T
    return z, U

def load_dt(case_dir: Path, label: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load ``<label>_dt_history.csv``; returns ``(None, None)`` if absent."""
    path = Path(case_dir) / f"{label}_dt_history.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df["step"].to_numpy(dtype=int), df["dt"].to_numpy(dtype=float)

def load_debug_log(case_dir: Path, label: str) -> pd.DataFrame:
    """Load ``<label>_debug_log.csv``; returns an empty DataFrame if absent."""
    path = Path(case_dir) / f"{label}_debug_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def write_reference_profile(case_dir: Path, *, Z_ref: np.ndarray, U_ref_T: np.ndarray) -> None:
    """Save a reference solution as ``reference_final_profile.csv``."""
    U = np.asarray(U_ref_T, dtype=float)
    if U.ndim == 1:
        U = U[np.newaxis, :]
    z_arr = np.asarray(Z_ref, dtype=float).reshape(-1, 1)
    data = np.concatenate([z_arr, U.T], axis=1)
    header = ",".join(["z"] + [f"reference_u{i}" for i in range(U.shape[0])])
    np.savetxt(
        Path(case_dir) / "reference_final_profile.csv",
        data, delimiter=",", header=header, comments="",
    )


# --- Convenience utilities ---

def safe_case_dir_name(name: str) -> str:
    """Sanitize a case study name into a filesystem-safe directory name."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return safe or "case"

def solve_case(
    case: CaseStudy,
    output_root: Path,
    *,
    compute_reference: bool,
    force_rerun: bool = False,
) -> Path:
    """Run MOL and hybrid solvers for *case* (with caching) and return the output directory.

    If *compute_reference* is True, also computes and writes a refined MOL reference profile.
    Set *force_rerun* to True to ignore cached outputs and recompute everything.
    """
    case_dir = output_root / safe_case_dir_name(case.name)
    required = [
        "hybrid_final_profile.csv", "mol_final_profile.csv",
        "hybrid_dt_history.csv",    "mol_dt_history.csv",
        "hybrid_debug_log.csv",     "mol_debug_log.csv",
    ]
    if compute_reference:
        required.append("reference_final_profile.csv")

    if (not force_rerun) and all((case_dir / f).exists() for f in required):
        print(f"  {case.name}: reusing cached outputs")
        return case_dir

    result = run_case_study(case)
    case_dir = Path(result["profiles_dir"])

    if compute_reference:
        target = int(case.Nz_ref)
        Z_ref, U_ref_T = mol_reference_uniform_grid(case, K_ref=target, atol=1e-9, rtol=1e-6)
        write_reference_profile(case_dir, Z_ref=Z_ref, U_ref_T=U_ref_T)
        print(f"  Wrote reference (Nz_ref={target})")

    return case_dir