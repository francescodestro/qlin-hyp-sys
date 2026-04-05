import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

RK_SCHEMES = {
	        "SSPRK33": {
	            "A": np.array([[0.0, 0.0, 0.0],
	                           [1.0, 0.0, 0.0],
	                           [0.25, 0.25, 0.0]], dtype=float),
	            "b": np.array([1/6, 1/6, 2/3], dtype=float),
	            "c": np.array([0.0, 1.0, 0.5], dtype=float),
	            "alpha_p": 1.0,
	            "order": 3,
	            "b_embedded": np.array([0.5, 0.5, 0.0], dtype=float),
	            "error_order": 3,
	        }
	    }

@dataclass
class CaseStudy:
    name: str
    L: float
    K: int
    T: float
    A_fun: Callable[[np.ndarray, float, float], np.ndarray]
    g_fun: Callable[[np.ndarray, float, float], np.ndarray]
    u0_fun: Callable[[float], np.ndarray]
    selector_mode: str = "expected"
    atol: float = 1e-6
    rtol: float = 1e-3
    cfl_safety: float = 0.9
    nf_theta: float = 0.9
    Nz_ref: int = 40
    u_in_left: Optional[Union[np.ndarray, Callable[[float], np.ndarray]]] = None
    u_in_right: Optional[Union[np.ndarray, Callable[[float], np.ndarray]]] = None


def evaluate_boundary(bc, t, n_components):
    """Return boundary state at time t, supporting callables and constant arrays."""
    if bc is None:
        return np.zeros(n_components, dtype=float)
    if callable(bc):
        return np.asarray(bc(float(t)), dtype=float)
    return np.asarray(bc, dtype=float)


def run_case_study(
    case: CaseStudy,
    *,
    save_outputs: bool = True,
    record_every: Optional[int] = None,
    collect_debug: bool = True,
) -> Dict[str, Any]:
    L, K, T = case.L, case.K, case.T
    Z0 = np.linspace(L / (2 * K), L - L / (2 * K), K, dtype=float)
    u0 = np.stack([case.u0_fun(float(z)) for z in Z0], axis=1)
    rec = int(record_every) if record_every is not None else max(1, case.K // 10)
    _, sols_h, meshes_h, dts_h_raw, stats_h = solve_hybrid(
        u0, Z0, T, L, case.A_fun, case.g_fun,
        selector_mode=case.selector_mode,
        scheme_name="SSPRK33",
        atol=case.atol, rtol=case.rtol,
        cfl_safety=case.cfl_safety, nf_theta=case.nf_theta,
        record_every=rec,
        u_in_left=case.u_in_left, u_in_right=case.u_in_right,
        collect_debug=collect_debug,
    )
    u_h_T = sols_h[-1]
    Z_h_T = meshes_h[-1]
    dts_h = np.asarray(dts_h_raw, dtype=float) # accepted step size H

    _, sols_m, dts_m_raw, stats_m = solve_mol(
        u0, Z0, T, L, case.A_fun, case.g_fun,
        scheme_name="SSPRK33",
        atol=case.atol, rtol=case.rtol,
        cfl_safety=case.cfl_safety,
        record_every=rec,
        u_in_left=case.u_in_left, u_in_right=case.u_in_right,
        collect_debug=collect_debug,
    )
    u_m_T = sols_m[-1]
    Z_m_T = Z0
    dts_m = np.asarray(dts_m_raw, dtype=float) # accepted step size MOL

    if not save_outputs:
        return {"profiles_dir": ""}

    safe_name = re.sub(r"[^\w\-]", "_", case.name.strip()) or "case"
    case_dir = Path("case_study_outputs") / safe_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Final profiles
    for z_vals, profile, label in [(Z_h_T, u_h_T, "hybrid"), (Z_m_T, u_m_T, "mol")]:
        prof = np.asarray(profile, dtype=float)
        data = np.concatenate([np.asarray(z_vals).reshape(-1, 1), prof.T], axis=1)
        header = "z," + ",".join(f"{label}_u{i}" for i in range(prof.shape[0]))
        np.savetxt(case_dir / f"{label}_final_profile.csv", data, delimiter=",", header=header, comments="")

    # Step-size histories
    for dts, label in [(dts_h, "hybrid"), (dts_m, "mol")]:
        dts = np.asarray(dts, dtype=float).reshape(-1)
        if dts.size:
            data = np.column_stack([np.arange(1, dts.size + 1), dts])
            np.savetxt(case_dir / f"{label}_dt_history.csv", data, delimiter=",", header="step,dt", comments="")

    # Debug logs (used for CFL utilization table, Table S2)
    for entries, filename in [(stats_h.get("debug_log", []), "hybrid_debug_log.csv"),
                               (stats_m.get("debug_log", []), "mol_debug_log.csv")]:
        entries = list(entries or [])
        if not entries:
            continue
        fieldnames = sorted({k for e in entries for k in e.keys()})
        with open(case_dir / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row = {k: float(e[k]) if isinstance(e.get(k), np.floating) else
                          int(e[k]) if isinstance(e.get(k), np.integer) else e.get(k, "")
                       for k in fieldnames}
                writer.writerow(row)

    return {"profiles_dir": str(case_dir)}


def eig_split(A):
    """Return (R, Rin, lam, Aplus, Aminus) for real diagonalizable A.
    Aplus = R diag(max(lam,0)) Rin; Aminus = R diag(min(lam,0)) Rin.
    """
    lam, R = np.linalg.eig(A)
    Rin = np.linalg.inv(R)
    lam = lam.real  # assume hyperbolic / real spectrum
    Aplus  = R @ np.diag(np.maximum(lam, 0.0)) @ Rin
    Aminus = R @ np.diag(np.minimum(lam, 0.0)) @ Rin
    return R, Rin, lam, Aplus.real, Aminus.real

def mesh_widths(Z, L, h_floor=1e-14):
    """
    Local widths h_k on a nonuniform mesh.
    - At interior nodes: min(left spacing, right spacing).
    - At left boundary: use spacing to the right.
    - At right boundary: use spacing to the left.
    Small floor applied to avoid zero widths.
    """
    Z = np.asarray(Z)
    K = len(Z)
    h = np.empty(K)
    dZ = np.diff(Z)
    h[0] = dZ[0]
    h[-1] = dZ[-1]
    if K > 2:
        h[1:-1] = np.minimum(dZ[:-1], dZ[1:])
    np.maximum(h, h_floor, out=h)
    return h

def interp_ref_to(z_ref, U_ref_T, Z_target):
    """
    Interpolate the reference snapshot U_ref_T (shape: n x K_ref) defined on z_ref (K_ref,)
    onto the target grid Z_target (K_tgt,).
    """
    z_ref = np.asarray(z_ref, dtype=float)
    Z_target = np.asarray(Z_target, dtype=float)
    n, Kref = U_ref_T.shape
    U_on = np.empty((n, Z_target.size), dtype=float)
    for i in range(n):
        # Use endpoint values for any tiny extrapolation at the edges, if needed
        U_on[i, :] = np.interp(Z_target, z_ref, U_ref_T[i, :],
                               left=U_ref_T[i, 0], right=U_ref_T[i, -1])
    return U_on

def l2_error_total(u_T, Z, Uref_on_Z, L):
    """
    L2 error over [0,L] combining all components: ||u - u_ref||_{L2}.
    u_T, Uref_on_Z: shape (n, K); Z: (K,), L: float
    """
    w = quad_weights_on_grid(Z, L)  # (K,)
    diff2 = (u_T - Uref_on_Z) ** 2  # (n, K)
    node_sq = diff2.sum(axis=0)  # (K,)
    return float(np.sqrt(np.dot(w, node_sq)))

def quad_weights_on_grid(Z, L):
    """
    Quadrature weights on a 1D grid Z for integrals over [0, L].

    Cell widths:
      w_0     = Z_1 - Z_0
      w_{K-1} = Z_{K-1} - Z_{K-2}
      w_k     = 0.5 (Z_{k+1} - Z_{k-1})

    Any nodes outside [0, L) get weight 0.
    """
    Z = np.asarray(Z, dtype=float).reshape(-1)
    K = int(Z.size)
    if K == 0:
        return np.empty(0, dtype=float)
    if K == 1:
        w = np.array([float(L)], dtype=float)
        if not (0.0 <= Z[0] < L):
            w[0] = 0.0
        return w

    w = np.empty(K, dtype=float)
    w[0] = Z[1] - Z[0]
    w[-1] = Z[-1] - Z[-2]
    w[1:-1] = 0.5 * (Z[2:] - Z[:-2])
    w[(Z < 0.0) | (Z >= L)] = 0.0
    return w


def analytic_solution_constA(A, Z, T, u0_fun, L):
    """Exact solution of u_t + A u_x = 0 (constant A, g=0) via characteristics."""
    R, Rin, lam, _, _ = eig_split(A)
    n = len(lam)
    W = np.zeros((n, Z.size), dtype=float)
    for i in range(n):
        zi0 = Z - lam[i] * T                          # characteristic footpoints
        mask = (zi0 >= 0.0) & (zi0 < L)
        if np.any(mask):
            U0_foot = np.stack([u0_fun(float(z)) for z in zi0[mask]], axis=1)
            W[i, mask] = Rin[i, :] @ U0_foot          # project onto i-th characteristic variable
    return (R @ W).real

def lte_one_step_exact_constA(
    A, u0_fun, dt, *, L, K, selector_mode="minimax", scheme_name="SSPRK33",
    u_in_left=None, u_in_right=None,
):
    """LTE defect τ = ||Φ_dt(u0) - u(dt)||_L2 / dt for constant A, g=0."""
    dt = float(dt)
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    Z0 = np.linspace(L / (2 * K), L - L / (2 * K), K, dtype=float)
    u0 = np.stack([np.asarray(u0_fun(float(z)), dtype=float) for z in Z0], axis=1)

    def A_fun(_u, _z, _t): return A
    def g_fun(_u, _z, _t): return np.zeros(n, dtype=float)

    scheme = RK_SCHEMES[scheme_name]
    u_mol, _, _ = step_mol(u0, Z0, 0.0, dt, L, A_fun, g_fun, scheme=scheme,
                           u_in_left=u_in_left, u_in_right=u_in_right)
    u_hyb, _, Z1, _, _ = step_hybrid(u0, Z0, 0.0, dt, L, A_fun, g_fun, scheme=scheme,
                                     selector_mode=selector_mode,
                                     u_in_left=u_in_left, u_in_right=u_in_right)

    U_exact_Z0 = analytic_solution_constA(A, Z0, dt, u0_fun, L)
    U_exact_Z1 = analytic_solution_constA(A, Z1, dt, u0_fun, L)

    return {
        "tau_mol_L2": l2_error_total(u_mol, Z0, U_exact_Z0, L) / dt,
        "tau_hyb_L2": l2_error_total(u_hyb, Z1, U_exact_Z1, L) / dt,
    }

def V_selector_from_eigs(lam, mode="minimax"):
    """Design speed from precomputed eigenvalues: 'expected' → mean, 'minimax' → midrange."""
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if mode == "expected":
        return float(np.mean(lam))
    return 0.5 * (float(np.min(lam)) + float(np.max(lam)))

def precompute_A_eigs(u, Z, t, A_fun):
    """Eigendecomposition of A_fun at each node.

    Returns a list of (Ak, R, Rin, lam) tuples supporting ``eig_data[k]`` indexing.
    """
    u = np.asarray(u, float)
    Z = np.asarray(Z, float)
    n, K = u.shape
    I = np.eye(n)
    out = []
    for k in range(K):
        Ak = A_fun(u[:, k], float(Z[k]), float(t))
        if np.allclose(Ak, np.diag(np.diag(Ak))):
            lam = np.diag(Ak).astype(float)
            R = Rin = I
        else:
            R, Rin, lam, *_ = eig_split(Ak)
        out.append((Ak, R, Rin, lam))
    return out

def assemble_residual(u, Z, t, L, v_nodes, A_fun, g_fun,
                      u_in_left=None, u_in_right=None, eig_data=None):
    """
    Residual r_k = -[(A - v I) u_z]_k + g_k on [0,L] with no-inflow BCs determined by sign of A's eigenvalues.
    Note: g_fun uses the sign convention g_fun = -g_paper (so +g_fun here equals -g_paper, consistent with u_t + A u_z + g_paper = 0).
    Uses upwind flux splitting: r_k = -(A+ Dm + A- Dp) + g_fun, where A± = R diag(max/min(λ-v,0)) R⁻¹.
    """
    def flux_split(R, Rin, lam_rel):
        return (R @ np.diag(np.maximum(lam_rel, 0.0)) @ Rin,   # A+
                R @ np.diag(np.minimum(lam_rel, 0.0)) @ Rin)   # A-

    u = np.asarray(u, float)
    Z = np.asarray(Z, float)
    v_nodes = np.asarray(v_nodes, float)
    n, K = u.shape
    r = np.zeros_like(u)
    u_in_left  = evaluate_boundary(u_in_left,  t, n)
    u_in_right = evaluate_boundary(u_in_right, t, n)

    h_nom = float(np.median(np.diff(Z)))
    h_floor = max(1e-8 * max(h_nom, 1.0), 1e-15)

    # ---- k = 0 (left boundary) ----
    z0, v0 = float(Z[0]), float(v_nodes[0])
    if eig_data is not None:
        A0, R0, Rin0, lam0 = eig_data[0]
    else:
        A0 = A_fun(u[:, 0], z0, t)
        R0, Rin0, lam0 = eig_split(A0)[:3]
    lam_rel0 = lam0 - v0

    # Apply inflow BC in characteristic variables (physical inflow: lam(A) > 0 on left)
    w = Rin0 @ u[:, 0]
    wbL = Rin0 @ u_in_left
    w[lam0 > 0] = wbL[lam0 > 0]
    uL = R0 @ w

    Dm = (u[:, 0] - uL) / max(Z[1] - Z[0], h_floor)           # ghost cell at zL = Z[0]-(Z[1]-Z[0])
    Dp = (u[:, 1] - u[:, 0]) / max(Z[1] - Z[0], h_floor)

    # For relative-positive modes with no physical inflow, fill Dm from Dp (no upstream data)
    Dm_e, Dp_e = Rin0 @ Dm, Rin0 @ Dp
    maskL = (lam_rel0 > 0) & (lam0 <= 0)
    if np.any(maskL):
        Dm_e[maskL] = Dp_e[maskL]
    Dm, Dp = (R0 @ Dm_e).real, (R0 @ Dp_e).real

    Aplus, Aminus = flux_split(R0, Rin0, lam_rel0)
    r[:, 0] = -(Aplus @ Dm + Aminus @ Dp) + g_fun(u[:, 0], z0, t)

    # ---- interior nodes k = 1, ..., K-2 ----
    for k in range(1, K - 1):
        zk, vk = float(Z[k]), float(v_nodes[k])
        if eig_data is not None:
            Ak, Rk, Rink, lamk = eig_data[k]
        else:
            Ak = A_fun(u[:, k], zk, t); Rk, Rink, lamk = eig_split(Ak)[:3]
        Dm = (u[:, k] - u[:, k-1]) / max(Z[k] - Z[k-1], h_floor)
        Dp = (u[:, k+1] - u[:, k]) / max(Z[k+1] - Z[k], h_floor)
        Aplus, Aminus = flux_split(Rk, Rink, lamk - vk)
        r[:, k] = -(Aplus @ Dm + Aminus @ Dp) + g_fun(u[:, k], zk, t)

    # ---- k = K-1 (right boundary) ----
    zR, vR = float(Z[-1]), float(v_nodes[-1])
    if eig_data is not None:
        AR, RR, RinR, lamR = eig_data[-1]
    else:
        AR = A_fun(u[:, -1], zR, t); RR, RinR, lamR = eig_split(AR)[:3]
    lam_relR = lamR - vR

    # Apply inflow BC in characteristic variables (physical inflow: lam(A) < 0 on right)
    w = RinR @ u[:, -1]
    wbR = RinR @ u_in_right
    w[lamR < 0] = wbR[lamR < 0]
    uR = RR @ w

    Dm = (u[:, -1] - u[:, -2]) / max(Z[-1] - Z[-2], h_floor)
    Dp = (uR - u[:, -1]) / max(Z[-1] - Z[-2], h_floor)        # ghost cell at zR = Z[-1]+(Z[-1]-Z[-2])

    # For relative-negative modes with no physical inflow, fill Dp from Dm (no upstream data)
    Dm_e, Dp_e = RinR @ Dm, RinR @ Dp
    maskR = (lam_relR < 0) & (lamR >= 0)
    if np.any(maskR):
        Dp_e[maskR] = Dm_e[maskR]
    Dm, Dp = (RR @ Dm_e).real, (RR @ Dp_e).real

    Aplus, Aminus = flux_split(RR, RinR, lam_relR)
    r[:, -1] = -(Aplus @ Dm + Aminus @ Dp) + g_fun(u[:, -1], zR, t)

    return r

def _recycle_lines(u, Z, zL, zR, L, t, A_fun, g_fun,
                   u_in_left=None, u_in_right=None, eig_data=None):
    """
    Remove nodes that exited [0,L] and insert replacements at the opposite boundary.
    Incoming physical characteristic components are set from boundary data;
    outgoing components are copied from the adjacent interior node.
    zL, zR: anchor positions for newly inserted left/right nodes (e.g. 1e-6 and L-1e-6).
    """
    u = np.asarray(u, float)
    Z = np.asarray(Z, float)
    n = u.shape[0]
    eig_out = list(eig_data) if eig_data is not None else None
    I = np.eye(n)

    def _eig_tuple(u_vec, z_val):
        Ak = A_fun(np.asarray(u_vec, float).reshape(-1), float(z_val), float(t))
        if np.allclose(Ak, np.diag(np.diag(Ak))):  # R = I
            lam = np.diag(Ak).astype(float); R = Rin = I
        else:
            R, Rin, lam, *_ = eig_split(Ak)
        return (Ak, R, Rin, lam)

    u_in_left_t  = evaluate_boundary(u_in_left,  t, n)
    u_in_right_t = evaluate_boundary(u_in_right, t, n)

    # Remove nodes outside the domain
    boundary_tol = zL / Z.size / 4
    n_left_exit  = int(np.sum(Z < boundary_tol))
    n_right_exit = int(np.sum(Z > L - boundary_tol))
    if n_left_exit + n_right_exit:
        keep = (Z >= boundary_tol) & (Z <= L - boundary_tol)
        u, Z = u[:, keep], Z[keep]
        if eig_out is not None:
            eig_out = [e for e, k in zip(eig_out, keep, strict=True) if bool(k)]

    # Insert on LEFT if a node exited on RIGHT (anchor at zL)
    if n_right_exit > 0:
        if n_right_exit > 1:
            raise ValueError("Two chars. exited")
        _, R, Rin, lam = eig_out[0] if eig_out is not None else precompute_A_eigs(u, Z, t, A_fun)[0]
        # Outgoing components copied from adjacent interior node; incoming overwritten by BC
        w = Rin @ u[:, 0]
        w[lam > 0] = (Rin @ u_in_left_t)[lam > 0]
        uL = (R @ w).real
        u = np.concatenate([uL.reshape(-1, 1), u], axis=1)
        Z = np.concatenate([[zL], Z])
        if eig_out is not None:
            eig_out = [_eig_tuple(uL, zL)] + eig_out

    # Insert on RIGHT if a node exited on LEFT (anchor at zR)
    if n_left_exit > 0:
        if n_left_exit > 1:
            raise ValueError("Two chars. exited")
        _, R, Rin, lam = eig_out[-1] if eig_out is not None else precompute_A_eigs(u, Z, t, A_fun)[-1]
        # Outgoing components copied from adjacent interior node; incoming overwritten by BC
        w = Rin @ u[:, -1]
        w[lam < 0] = (Rin @ u_in_right_t)[lam < 0]
        uR = (R @ w).real
        u = np.concatenate([u, uR.reshape(-1, 1)], axis=1)
        Z = np.concatenate([Z, [zR]])
        if eig_out is not None:
            eig_out = eig_out + [_eig_tuple(uR, zR)]

    return u, Z, eig_out

def cfl_dt(Z, v_nodes, L, A_fun, u, t, alpha_p=1.0, safety=1, eig_data=None):
    """CFL-limited step: dt = safety * alpha_p * min_k h_k / rho_k(A - v I)."""
    n, K = u.shape
    h = mesh_widths(Z, L)
    if eig_data is not None and len(eig_data) == K:
        rho = np.array([max(1e-14, np.max(np.abs(
                            np.asarray(eig_data[k][3], float) - float(v_nodes[k]))))
                        for k in range(K)])
    else:
        rho = np.array([max(1e-14, np.max(np.abs(
                            np.linalg.eigvals(A_fun(u[:, k], Z[k], t) - v_nodes[k] * np.eye(n)).real)))
                        for k in range(K)])
    return float(safety * alpha_p * np.min(h / rho))

def nf_dt_cap(Z, v_nodes, theta=1, eps=1e-14):
    """No-folding condition: dt <= theta * min_k h_k / |dv_k|."""
    dZ = np.diff(Z)
    dv = np.diff(v_nodes)
    return float(theta * np.min(dZ / np.maximum(np.abs(dv), eps)))

def _dt_controller(dt, err, err_order, safety=0.9, fac_min=0.2, fac_max=5.0):
    """I-controller: dt_new = dt * clip(safety * err^(-1/p), fac_min, fac_max)."""
    fac = 2.0 if (err == 0 or not np.isfinite(err)) else safety * err**(-1.0 / float(err_order))
    return dt * float(np.clip(fac, fac_min, fac_max))

def _error_rms(u_high, u_low, atol=1e-6, rtol=1e-3):
    """RMS of the mixed absolute/relative scaled difference: sqrt(mean(((u_high-u_low)/scale)^2))."""
    scale = atol + rtol * np.maximum(np.abs(u_high), np.abs(u_low))
    rel = (u_high - u_low) / scale
    return float(np.sqrt(np.mean(rel**2)))

def step_hybrid(u0, Z0, t0, dt, L, A_fun, g_fun, scheme=None, selector_mode="minimax",
                u_in_left=None, u_in_right=None):

    if scheme is None:
        scheme = RK_SCHEMES["SSPRK33"]

    A = scheme["A"]
    b = scheme["b"]
    c = scheme["c"]
    bE = scheme["b_embedded"]
    s = len(b)

    # Stage containers
    w_stage = [None] * (s + 1)
    w_stage[0] = u0
    Z_stage = [None] * (s + 1)
    Z_stage[0] = Z0
    r = [None] * (s + 1)
    v = [None] * (s + 1)

    for m in range(1, s + 1):
        incr_w = np.zeros_like(u0)
        incr_v = np.zeros_like(Z0)
        for ell in range(1, m):
            incr_w += A[m-1, ell-1] * r[ell]
            incr_v += A[m-1, ell-1] * v[ell]
        w_stage[m] = u0 + dt * incr_w
        Z_stage[m] = Z0 + dt * incr_v
        t_m = t0 + c[m-1] * dt

        eig_data_stage = precompute_A_eigs(w_stage[m], Z_stage[m], t_m, A_fun)
        v[m] = np.array([
            V_selector_from_eigs(eig_data_stage[k][3], mode=selector_mode)
            for k in range(Z0.size)
        ], dtype=float)
        r[m] = assemble_residual(
            w_stage[m], Z_stage[m], t_m, L, v[m], A_fun, g_fun, u_in_left, u_in_right,
            eig_data=eig_data_stage,
        )

    u_high = u0.copy()
    u_low  = u0.copy()
    Z_high = Z0.copy()
    for m in range(1, s + 1):
        u_high += dt * b[m-1]  * r[m]
        u_low  += dt * bE[m-1] * r[m]
        Z_high += dt * b[m-1]  * v[m]

    eig_final = precompute_A_eigs(u_high, Z_high, t0 + dt, A_fun)
    return u_high, u_low, Z_high, v[s], eig_final

def step_mol(u0, Z, t0, dt, L, A_fun, g_fun, scheme=None, u_in_left=None, u_in_right=None):
    if scheme is None:
        scheme = RK_SCHEMES["SSPRK33"]

    A = scheme["A"]
    b = scheme["b"]
    c = scheme["c"]
    bE = scheme["b_embedded"]
    s = len(b)
    n, K = u0.shape
    zeros_v = np.zeros(K)

    # Stage containers
    w = [None] * (s + 1)
    r = [None] * (s + 1)
    w[0] = u0.copy()

    for m in range(1, s + 1):
        incr = np.zeros_like(u0)
        for ell in range(1, m):
            incr += A[m-1, ell-1] * r[ell]
        w[m] = w[0] + dt * incr
        t_m = t0 + c[m-1] * dt
        eig_data_stage = precompute_A_eigs(w[m], Z, t_m, A_fun)
        r[m] = assemble_residual(w[m], Z, t_m, L, zeros_v, A_fun, g_fun, u_in_left, u_in_right,
                                 eig_data=eig_data_stage)
    u_high = u0.copy()
    u_low  = u0.copy()
    for m in range(1, s + 1):
        u_high += dt * b[m-1]  * r[m]
        u_low  += dt * bE[m-1] * r[m]

    eig_final = precompute_A_eigs(u_high, Z, t0 + dt, A_fun)
    return u_high, u_low, eig_final

def solve_mol(
    u0,
    Z,
    T,
    L,
    A_fun,
    g_fun,
    scheme_name="SSPRK33",
    atol=1e-6,
    rtol=1e-3,
    cfl_safety=1.0,
    record_every=1,
    dt_init=None,
    dt_min=1e-12,
    dt_max=np.inf,
    u_in_left=None,
    u_in_right=None,
    max_steps=200000,
    collect_debug: bool = True,
):
    scheme = RK_SCHEMES[scheme_name]
    order = scheme.get("order", 3)
    err_order = int(scheme.get("error_order", order))
    alpha_p = scheme.get("alpha_p", 1.0)

    debug_log = [] if collect_debug else None
    t = 0.0
    u = u0.copy()
    ts = [t]
    solutions = [u.copy()]
    dts = []
    rejects = 0
    accepts = 0

    # initial dt guess
    v0 = np.zeros(Z.size)
    dt_cfl = cfl_dt(Z, v0, L, A_fun, u, t, alpha_p=alpha_p, safety=cfl_safety)
    dt_try = float(min(dt_init or dt_cfl, dt_cfl, dt_max))

    eig_now = None

    while t < T and (accepts + rejects) < max_steps:
        dt_user = float(dt_try)
        dt_cfl = float(cfl_dt(Z, v0, L, A_fun, u, t, alpha_p=alpha_p, safety=cfl_safety, eig_data=eig_now))
        dt_final = float(T - t)
        dt_try = float(min(dt_user, dt_cfl, dt_final, dt_max))
        if dt_try < dt_min:
            raise RuntimeError("Δt underflow in adaptive MOL")

        if collect_debug:
            limiter = min({"CFL": dt_cfl, "final": dt_final, "adapter": dt_user}.items(),
                          key=lambda kv: kv[1])[0]
            entry = {"t": float(t), "dt": float(dt_try), "dt_cfl": float(dt_cfl), "limiter": limiter}
            debug_log.append(entry)

        # step
        u_high, u_low, eig_final = step_mol(
            u, Z, t, dt_try, L, A_fun, g_fun,
            scheme=scheme, u_in_left=u_in_left, u_in_right=u_in_right,
        )
        err = float(_error_rms(u_high, u_low, atol, rtol))
        if collect_debug:
            entry["err"] = err
            entry["accepted"] = int(err <= 1.0)

        if err <= 1.0:
            # accept
            u = u_high
            t += dt_try
            accepts += 1
            dts.append(dt_try)
            if (accepts % record_every) == 0 or t >= T:
                ts.append(t)
                solutions.append(u.copy())

            eig_now = eig_final
            dt_try = float(_dt_controller(dt_try, err if err > 0 else 1e-12, err_order))
        else:
            # reject and shrink
            rejects += 1
            dt_try = float(_dt_controller(dt_try, max(err, 1e-12), err_order))

    return (
        np.array(ts),
        solutions,
        np.array(dts),
        {"accepted": accepts, "rejected": rejects, "debug_log": (debug_log or [])},
    )

def solve_hybrid(
    u0,
    Z0,
    T,
    L,
    A_fun,
    g_fun,
    selector_mode="minimax",
    scheme_name="SSPRK33",
    atol=1e-6,
    rtol=1e-3,
    cfl_safety=1.0,
    nf_theta=0.9,
    record_every=1,
    dt_init=None,
    dt_min=1e-12,
    dt_max=np.inf,
    u_in_left=None,
    u_in_right=None,
    max_steps=200000,
    collect_debug: bool = True,
):
    scheme = RK_SCHEMES[scheme_name]
    order = scheme.get("order", 3)
    err_order = int(scheme.get("error_order", order))
    alpha_p = scheme.get("alpha_p", 1.0)

    zL = 1e-6
    zR = float(L) - 1e-6

    debug_log = [] if collect_debug else None
    v_hist = [] if collect_debug else None

    t = 0.0
    u = u0.copy()
    Z = Z0.copy()
    ts = [t]
    solutions = [u.copy()]
    meshes = [Z.copy()]
    dts = []
    rejects = 0
    accepts = 0

    # initial dt guess
    eig_now = precompute_A_eigs(u, Z, t, A_fun)
    v_pred = np.array(
        [V_selector_from_eigs(eig_now[k][3], mode=selector_mode) for k in range(Z.size)],
        dtype=float,
    )
    dt_cfl = float(cfl_dt(Z, v_pred, L, A_fun, u, t, alpha_p=alpha_p, safety=cfl_safety, eig_data=eig_now))
    dt_nf = float(nf_dt_cap(Z, v_pred, theta=nf_theta))
    dt_try = float(min(dt_init or dt_cfl, dt_cfl, dt_nf, dt_max))

    while t < T and (accepts + rejects) < max_steps:
        # calculate dt limits on the current (accepted) state
        dt_user = float(dt_try)
        dt_cfl = float(cfl_dt(Z, v_pred, L, A_fun, u, t, alpha_p=alpha_p, safety=cfl_safety, eig_data=eig_now))
        dt_nf = float(nf_dt_cap(Z, v_pred, theta=nf_theta))

        # optional characteristic/boundary guard (only for outward-moving boundary nodes)
        dt_char = np.inf
        if Z.size >= 2:
            eps_v = 1e-14
            cand = []
            if float(v_pred[1]) < -eps_v:
                cand.append((0.0 - float(Z[1])) / float(v_pred[1]))
            if float(v_pred[-2]) > eps_v:
                cand.append((float(L) - float(Z[-2])) / float(v_pred[-2]))
            if cand:
                dt_char = float(max(cand))

        dt_final = float(T - t)
        dt_try = float(min(dt_user, dt_cfl, dt_nf, dt_final, dt_max, 0.99 * dt_char))
        if dt_try < dt_min:
            raise RuntimeError("Δt underflow in adaptive hybrid")

        if collect_debug:
            limiter = min({"CFL": dt_cfl, "NF": dt_nf, "char": dt_char,
                           "final": dt_final, "adapter": dt_user}.items(),
                          key=lambda kv: kv[1])[0]
            entry = {"t": float(t), "dt": float(dt_try), "dt_cfl": float(dt_cfl), "limiter": limiter}
            debug_log.append(entry)

        # step
        u_high, u_low, Z_high, v_last, eig_final = step_hybrid(
            u, Z, t, dt_try, L, A_fun, g_fun,
            scheme=scheme, selector_mode=selector_mode,
            u_in_left=u_in_left, u_in_right=u_in_right,
        )
        err = float(_error_rms(u_high, u_low, atol, rtol))

        if collect_debug:
            entry["err"] = err
            entry["accepted"] = int(err <= 1.0)

        if err <= 1.0:
            # accept
            t += dt_try
            accepts += 1
            dts.append(dt_try)

            # recycle to keep the moving mesh within [0,L]
            u, Z, eig_now = _recycle_lines(
                u_high, Z_high, zL, zR, L, t, A_fun, g_fun,
                u_in_left, u_in_right, eig_data=eig_final,
            )
            if eig_now is None:
                eig_now = precompute_A_eigs(u, Z, t, A_fun)
            v_pred = np.array(
                [V_selector_from_eigs(eig_now[k][3], mode=selector_mode) for k in range(Z.size)],
                dtype=float,
            )
            if collect_debug:
                v_hist.append(np.asarray(v_last, dtype=float).copy())

            if (accepts % record_every) == 0 or t >= T:
                ts.append(float(t))
                solutions.append(u.copy())
                meshes.append(Z.copy())

            # propose next dt
            dt_try = float(_dt_controller(dt_try, err if err > 0 else 1e-12, err_order))
        else:
            # reject and shrink
            rejects += 1
            dt_try = float(_dt_controller(dt_try, max(err, 1e-12), err_order))

    return (
        np.array(ts),
        solutions,
        meshes,
        np.asarray(dts, dtype=float),
        {"accepted": accepts, "rejected": rejects, "v_last": (v_hist or []), "debug_log": (debug_log or [])},
    )
