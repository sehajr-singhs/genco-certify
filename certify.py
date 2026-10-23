"""Certify-or-reject layer for learned AC power-flow solutions (model agnostic).

Given a network (bus admittance matrix, bus types and specifications, branch admittances and limits) and a
model's predicted operating point (Vm, Va), the layer runs a full AC Newton-Raphson power flow warm-started
from the prediction and returns:

  CERTIFIED  if the solve converges from the prediction, the converged point is within tolerance of the
             prediction, and the converged point satisfies every voltage, branch-thermal and generator
             reactive limit;
  REJECTED   otherwise, with the reason (no convergence, prediction too far from the true solution, or a
             limit violated at the true solution). Rejected cases fall back to a full study.

Everything is plain numpy/scipy so it can sit behind any model, including GENCO in gridfm-graphkit.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Network:
    Y: sp.csr_matrix            # complex bus admittance matrix, per unit
    pq: np.ndarray              # bool masks over buses
    pv: np.ndarray
    ref: np.ndarray
    P_spec: np.ndarray          # net injection, per unit (gen - load), used at PV and PQ buses
    Q_spec: np.ndarray          # net injection, per unit, used at PQ buses
    Vm_set: np.ndarray          # voltage magnitude setpoints at PV and REF buses
    Va_ref: float               # reference angle, rad
    vmin: np.ndarray
    vmax: np.ndarray
    Qd: np.ndarray              # bus reactive load, per unit (to recover Qg at PV/REF)
    qg_min: np.ndarray          # bus-aggregated generator Q limits, per unit (+-inf where no gen)
    qg_max: np.ndarray
    br_f: np.ndarray            # in-service branches: from bus, to bus, 2x2 admittance blocks, rating (pu)
    br_t: np.ndarray
    Yff: np.ndarray
    Yft: np.ndarray
    Ytf: np.ndarray
    Ytt: np.ndarray
    rate: np.ndarray


@dataclass
class Settings:
    tol_mismatch: float = 1e-8      # pu, power mismatch for convergence
    max_iter: int = 15
    tol_vm: float = 0.01            # pu, how close the prediction must be to the true solution
    tol_va_deg: float = 1.0         # deg
    limit_slack_v: float = 1e-4     # pu, numerical slack on voltage limits
    limit_slack_s: float = 1e-3     # fraction of rating
    limit_slack_q: float = 1e-3     # pu


def _mismatch(net, V):
    S = V * np.conj(net.Y @ V)
    dP = S.real - net.P_spec
    dQ = S.imag - net.Q_spec
    return np.concatenate([dP[net.pv | net.pq], dQ[net.pq]]), S


def newton(net, Vm0, Va0, st=Settings()):
    """Polar Newton-Raphson. Returns (converged, iterations, Vm, Va, S, final mismatch)."""
    Vm = Vm0.astype(float).copy()
    Va = Va0.astype(float).copy()
    Vm[net.pv | net.ref] = net.Vm_set[net.pv | net.ref]
    Va[net.ref] = net.Va_ref
    pvpq = np.where(net.pv | net.pq)[0]
    pq = np.where(net.pq)[0]
    V = Vm * np.exp(1j * Va)
    F, S = _mismatch(net, V)
    it = 0
    while np.max(np.abs(F)) > st.tol_mismatch and it < st.max_iter:
        Ibus = net.Y @ V
        dV = sp.diags(V)
        dI = sp.diags(Ibus)
        dVn = sp.diags(V / np.abs(V))
        dS_dVa = 1j * dV @ np.conj(dI - net.Y @ dV)
        dS_dVm = dV @ np.conj(net.Y @ dVn) + np.conj(dI) @ dVn
        J11 = dS_dVa[pvpq][:, pvpq].real
        J12 = dS_dVm[pvpq][:, pq].real
        J21 = dS_dVa[pq][:, pvpq].imag
        J22 = dS_dVm[pq][:, pq].imag
        J = sp.bmat([[J11, J12], [J21, J22]], format="csc")
        try:
            dx = spla.spsolve(J, -F)
        except Exception:
            return False, it, Vm, Va, S, np.inf
        if not np.all(np.isfinite(dx)):
            return False, it, Vm, Va, S, np.inf
        Va[pvpq] += dx[: len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
        V = Vm * np.exp(1j * Va)
        F, S = _mismatch(net, V)
        it += 1
    m = float(np.max(np.abs(F)))
    return bool(m <= st.tol_mismatch), it, Vm, Va, S, m


def limits(net, Vm, Va, S, st=Settings()):
    """Limit check at an operating point. Returns dict of worst margins (negative = violated)."""
    V = Vm * np.exp(1j * Va)
    v_low = float(np.min(Vm - net.vmin))
    v_high = float(np.min(net.vmax - Vm))
    Vf, Vt = V[net.br_f], V[net.br_t]
    If = net.Yff * Vf + net.Yft * Vt
    It = net.Ytf * Vf + net.Ytt * Vt
    Sf = np.abs(Vf * np.conj(If))
    St = np.abs(Vt * np.conj(It))
    rated = net.rate > 0
    load = np.maximum(Sf, St)[rated] / net.rate[rated]
    s_margin = float(1.0 - np.max(load)) if rated.any() else 1.0
    gen = net.pv | net.ref
    Qg = S.imag + net.Qd
    q_margin = float(np.min(np.minimum(Qg - net.qg_min, net.qg_max - Qg)[gen])) if gen.any() else 1.0
    ok = (v_low >= -st.limit_slack_v and v_high >= -st.limit_slack_v and s_margin >= -st.limit_slack_s
          and q_margin >= -st.limit_slack_q)
    return {"ok": bool(ok), "v_low_margin": v_low, "v_high_margin": v_high, "thermal_margin": s_margin,
            "qg_margin": q_margin, "max_loading": float(np.max(load)) if rated.any() else 0.0}


def certify(net, Vm_pred, Va_pred, st=Settings()):
    """Certify or reject a predicted operating point (Vm in pu, Va in rad)."""
    conv, it, Vm, Va, S, mis = newton(net, Vm_pred, Va_pred, st)
    out = {"converged": conv, "nr_iterations": it, "final_mismatch": mis}
    if not conv:
        out.update(verdict="REJECTED", reason="no_convergence_from_prediction")
        return out, None
    dvm = float(np.max(np.abs(Vm - Vm_pred)))
    dva = float(np.degrees(np.max(np.abs(np.angle(np.exp(1j * (Va - Va_pred)))))))
    out.update(dist_vm=dvm, dist_va_deg=dva)
    lim = limits(net, Vm, Va, S, st)
    out["limits"] = lim
    if dvm > st.tol_vm or dva > st.tol_va_deg:
        out.update(verdict="REJECTED", reason="prediction_far_from_solution")
    elif not lim["ok"]:
        out.update(verdict="REJECTED", reason="limit_violated_at_solution")
    else:
        out.update(verdict="CERTIFIED", reason="")
    return out, (Vm, Va, S)
