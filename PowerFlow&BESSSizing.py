"""
Power Flow Sizing Tool — Streamlit App
Replicates the Excel VBA power flow model for solar farm POI sizing.

3-Bus topology:
  Bus 1 = Grid / POI (swing, V1 = v_target, theta1 = 0)
  Bus 2 = MPT secondary (PQ bus, Q from NLL + tap model)
  Bus 3 = Inverter / ISU secondary (PQ bus, P+jQ injected by inverters)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Power Flow Sizing Tool", layout="wide", page_icon="⚡")

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem;}
  h1 {color:#1a3a5c; margin-bottom:2px;}
  .stMetric label {font-size:.78rem;}
  div[data-testid="stSidebarContent"] {background:#f5f8fc;}
  .section {font-weight:700; color:#1a5276; border-bottom:2px solid #2874a6;
            padding-bottom:3px; margin-top:14px; margin-bottom:6px; font-size:.95rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  POWER FLOW ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def tap_value(has_oltc: bool, ntaps: int, tap_range_pct: float,
              fixed_tap_pu: float, tap_number: int = 0) -> float:
    """Return tap value in pu."""
    if not has_oltc:
        return fixed_tap_pu
    step = (tap_range_pct / 100.0) / ((ntaps - 1) / 2)
    return tap_number * step


def build_ybus(sbase: float, params: dict, tap_c: float) -> np.ndarray:
    """
    Build 3×3 admittance matrix.
    All ISUs aggregated into one equivalent transformer (S_total = n_inv × S_isu).
    Tap modelled on primary (MPT) or secondary (ISU DETC at 0 for simplicity).
    """
    # ── MPT ──────────────────────────────────────────────────────────────────
    s_mpt = params['s_mpt_mva'] * params['n_mpt']
    z_mpt = params['z_mpt']
    xr_mpt = params['xr_mpt']
    r_mpt = z_mpt * np.cos(np.arctan(xr_mpt))
    x_mpt = z_mpt * np.sin(np.arctan(xr_mpt))
    Ys_mpt = sbase / s_mpt / complex(r_mpt, x_mpt)

    g_m = (params['p0_mpt'] * params['n_mpt']) / 1e3 / s_mpt
    b_m = np.sqrt(max(params['i0_mpt']**2 - (g_m)**2, 0))
    Ym_mpt = complex(g_m, b_m) * (sbase / s_mpt)

    c = 1.0 - tap_c
    if abs(c) < 1e-9:
        c = 1.0
    Y10_mpt = (1.0 - c) / c * Ys_mpt + Ym_mpt / c**2
    Y12_mpt = c * Ys_mpt
    Y20_mpt = (c - 1.0) * Ys_mpt

    # ── Aggregated ISU ────────────────────────────────────────────────────────
    n_inv = params['n_inverters']
    s_isu_total = params['s_isu_mva'] * n_inv
    z_isu = params['z_isu']
    xr_isu = params['xr_isu']
    r_isu = z_isu * np.cos(np.arctan(xr_isu))
    x_isu = z_isu * np.sin(np.arctan(xr_isu))
    Ys_isu = sbase / s_isu_total / complex(r_isu, x_isu)

    g_isu = (params['p0_isu'] * n_inv) / 1e3 / s_isu_total
    b_isu = np.sqrt(max(params['i0_isu']**2 - g_isu**2, 0))
    Ym_isu = complex(g_isu, -b_isu) * (sbase / s_isu_total)

    # ── Capacitor bank at bus 2 ───────────────────────────────────────────────
    y_cap = complex(0.0, params['q_cap'] / sbase)

    # ── Assemble Ybus ─────────────────────────────────────────────────────────
    Y = np.zeros((3, 3), dtype=complex)
    # Bus 1–2 (MPT with tap)
    Y[0, 0] = Y10_mpt + Y12_mpt
    Y[0, 1] = -Y12_mpt
    Y[1, 0] = -Y12_mpt
    Y[1, 1] = Y12_mpt + Y20_mpt
    # Bus 2–3 (aggregated ISU)
    Y[1, 1] += Ys_isu + Ym_isu + y_cap
    Y[1, 2] = -Ys_isu
    Y[2, 1] = -Ys_isu
    Y[2, 2] = Ys_isu

    return Y


def nr_pf(Ybus: np.ndarray, P2: float, Q2: float, P3: float, Q3: float,
          V1: float = 1.0, V2_init: float = None, V3_init: float = None,
          max_iter: int = 50, tol: float = 1e-7):
    """
    Newton-Raphson power flow for 3-bus system.
    Bus 1 = swing (V1 fixed, theta1=0).
    Bus 2, 3 = PQ buses.
    Returns (V, theta, converged).
    """
    if V2_init is None:
        V2_init = V1 * 0.99
    if V3_init is None:
        V3_init = V1 * 0.98

    V = np.array([V1, V2_init, V3_init])
    th = np.zeros(3)
    Psp = np.array([0.0, P2, P3])
    Qsp = np.array([0.0, Q2, Q3])

    for _ in range(max_iter):
        Pc = np.zeros(3)
        Qc = np.zeros(3)
        for i in range(3):
            for j in range(3):
                G = Ybus[i, j].real
                B = Ybus[i, j].imag
                d = th[i] - th[j]
                Pc[i] += V[i] * V[j] * (G * np.cos(d) + B * np.sin(d))
                Qc[i] += V[i] * V[j] * (G * np.sin(d) - B * np.cos(d))

        dP2 = Psp[1] - Pc[1]
        dP3 = Psp[2] - Pc[2]
        dQ2 = Qsp[1] - Qc[1]
        dQ3 = Qsp[2] - Qc[2]
        mis = np.array([dP2, dP3, dQ2, dQ3])

        if np.max(np.abs(mis)) < tol:
            return V, th, True

        # Jacobian (4×4 for buses 2 and 3)
        J = np.zeros((4, 4))
        buses = [1, 2]

        def dP_dth_ii(i):
            s = 0.0
            for k in range(3):
                if k != i:
                    G = Ybus[i, k].real; B = Ybus[i, k].imag; d = th[i] - th[k]
                    s += V[k] * (-G * np.sin(d) + B * np.cos(d))
            return V[i] * s

        def dP_dth_ij(i, j):
            G = Ybus[i, j].real; B = Ybus[i, j].imag; d = th[i] - th[j]
            return V[i] * V[j] * (G * np.sin(d) - B * np.cos(d))

        def dP_dV_ii(i):
            s = sum(V[k] * (Ybus[i, k].real * np.cos(th[i]-th[k]) + Ybus[i, k].imag * np.sin(th[i]-th[k]))
                    for k in range(3))
            return s + Ybus[i, i].real * V[i]

        def dP_dV_ij(i, j):
            G = Ybus[i, j].real; B = Ybus[i, j].imag; d = th[i] - th[j]
            return V[i] * (G * np.cos(d) + B * np.sin(d))

        def dQ_dth_ii(i):
            s = sum(V[k] * (Ybus[i, k].real * np.cos(th[i]-th[k]) + Ybus[i, k].imag * np.sin(th[i]-th[k]))
                    for k in range(3) if k != i)
            return V[i] * s

        def dQ_dth_ij(i, j):
            G = Ybus[i, j].real; B = Ybus[i, j].imag; d = th[i] - th[j]
            return -V[i] * V[j] * (G * np.cos(d) + B * np.sin(d))

        def dQ_dV_ii(i):
            s = sum(V[k] * (Ybus[i, k].real * np.sin(th[i]-th[k]) - Ybus[i, k].imag * np.cos(th[i]-th[k]))
                    for k in range(3))
            return s - Ybus[i, i].imag * V[i]

        def dQ_dV_ij(i, j):
            G = Ybus[i, j].real; B = Ybus[i, j].imag; d = th[i] - th[j]
            return V[i] * (G * np.sin(d) - B * np.cos(d))

        for ri, bi in enumerate(buses):
            for ci, bj in enumerate(buses):
                if bi == bj:
                    J[ri, ci] = dP_dth_ii(bi)
                    J[ri, 2+ci] = dP_dV_ii(bi)
                    J[2+ri, ci] = dQ_dth_ii(bi)
                    J[2+ri, 2+ci] = dQ_dV_ii(bi)
                else:
                    J[ri, ci] = dP_dth_ij(bi, bj)
                    J[ri, 2+ci] = dP_dV_ij(bi, bj)
                    J[2+ri, ci] = dQ_dth_ij(bi, bj)
                    J[2+ri, 2+ci] = dQ_dV_ij(bi, bj)

        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return V, th, False

        th[1] += dx[0]; th[2] += dx[1]
        V[1] += dx[2]; V[3 - 1] += dx[3]

    return V, th, False


def optimal_tap(params: dict, sbase: float, V_target: float) -> tuple:
    """
    Sweep tap positions and find the one that gives V_POI closest to V_target
    at nominal active power export.
    """
    if not params['has_oltc']:
        return tap_value(False, 1, 0, params['fixed_tap']), None

    ntaps = params['ntaps']
    tap_range = params['tap_range']
    P_nom = params['n_inverters'] * params['s_inv_mva'] * params['pf_inv'] / sbase
    Q_nom = params['n_inverters'] * params['s_inv_mva'] * np.sqrt(1 - params['pf_inv']**2) / sbase

    best_tap = 0.0
    best_err = 1e9
    tap_results = []

    for k in range(-(ntaps // 2), ntaps // 2 + 1):
        tc = tap_value(True, ntaps, tap_range, 0.0, k)
        try:
            Y = build_ybus(sbase, params, tc)
            V, _, conv = nr_pf(Y, 0, 0, P_nom, Q_nom, V1=V_target)
            if conv:
                err = abs(V[1] - V_target)
                tap_results.append({'tap_pos': k, 'tap_pu': tc, 'V_POI': V[1], 'error': err})
                if err < best_err:
                    best_err = err
                    best_tap = tc
        except Exception:
            pass

    return best_tap, pd.DataFrame(tap_results) if tap_results else None


@st.cache_data(show_spinner=False)
def run_sweep(params_frozen: tuple) -> pd.DataFrame:
    """Run full P-Q sweep, returns DataFrame of results."""
    params = dict(params_frozen)
    sbase = params['contract_power']
    n_inv = params['n_inverters']
    s_inv = params['s_inv_mva']
    pf = params['pf_inv']
    V_target = params['v_poi_calc']

    S_max_pu = n_inv * s_inv / sbase
    P_max_pu = n_inv * s_inv * pf / sbase

    # Determine tap
    tc = tap_value(params['has_oltc'], params['ntaps'], params['tap_range'],
                   params['fixed_tap'], params.get('tap_number', 0))
    Y = build_ybus(sbase, params, tc)

    results = []
    n_p = 101
    p_vals = np.linspace(-P_max_pu, P_max_pu, n_p)

    V2_prev = V_target * 0.99
    V3_prev = V_target * 0.98
    th2_prev = 0.0
    th3_prev = 0.0

    for p3 in p_vals:
        # Reactive capability curve
        q_cap = np.sqrt(max(S_max_pu**2 - p3**2, 0))
        for sign in [1, -1]:
            q3 = sign * q_cap
            V, th, conv = nr_pf(Y, 0.0, 0.0, p3, q3, V1=V_target,
                                 V2_init=V2_prev, V3_init=V3_prev)
            if conv:
                V2_prev, V3_prev = V[1], V[2]
                # Grid-side power (computed from swing bus)
                P1 = sum(V[0] * V[j] * (Y[0, j].real * np.cos(th[0] - th[j]) +
                                          Y[0, j].imag * np.sin(th[0] - th[j]))
                         for j in range(3))
                Q1 = sum(V[0] * V[j] * (Y[0, j].real * np.sin(th[0] - th[j]) -
                                          Y[0, j].imag * np.cos(th[0] - th[j]))
                         for j in range(3))
                results.append({
                    'P_inv_MW': p3 * sbase,
                    'Q_inv_MVAR': q3 * sbase,
                    'V_POI': V[1],
                    'V_ISU': V[2],
                    'theta_POI_deg': np.degrees(th[1]),
                    'theta_ISU_deg': np.degrees(th[2]),
                    'P_grid_MW': P1 * sbase,
                    'Q_grid_MVAR': Q1 * sbase,
                    'tap_pu': tc,
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Input Parameters")

    st.markdown('<div class="section">Grid Requirement</div>', unsafe_allow_html=True)
    contract_power = st.number_input("Contract Power (MW)", value=200.0, min_value=1.0, max_value=5000.0, step=10.0)
    contract_pf = st.number_input("Contract Power Factor", value=0.95, min_value=0.5, max_value=1.0, step=0.01)

    st.markdown('<div class="section">POI Voltage</div>', unsafe_allow_html=True)
    v_max = st.number_input("Max (pu)", value=1.05, min_value=1.0, max_value=1.15, step=0.01)
    v_min = st.number_input("Min (pu)", value=0.95, min_value=0.8, max_value=1.0, step=0.01)
    v_calc = st.number_input("This calculation (pu)", value=1.00, min_value=0.8, max_value=1.15, step=0.01)

    st.markdown('<div class="section">Main Power Transformer (MPT)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n_mpt = c1.number_input("# MPTs", value=1, min_value=1, max_value=10)
    s_mpt = c2.number_input("MVA/MPT", value=240.0, min_value=10.0, max_value=2000.0, step=10.0)
    z_mpt = st.number_input("Impedance (pu)", value=0.08, min_value=0.01, max_value=0.25, step=0.005, format="%.3f")
    xr_mpt = st.number_input("X/R", value=40.0, min_value=1.0, max_value=100.0, step=1.0, key="xr_mpt")
    c3, c4 = st.columns(2)
    i0_mpt = c3.number_input("No-load I (pu)", value=0.001, min_value=0.0, max_value=0.05, step=0.0005, format="%.4f")
    p0_mpt = c4.number_input("No-load P (kW)", value=10.0, min_value=0.0, max_value=500.0, step=1.0)

    has_oltc = st.checkbox("Has OLTC tap changer?", value=True)
    if has_oltc:
        c5, c6 = st.columns(2)
        ntaps = c5.number_input("# Taps", value=31, min_value=3, max_value=99, step=2)
        tap_range = c6.number_input("Range (%)", value=10.0, min_value=1.0, max_value=30.0, step=1.0)
        tap_number = st.slider("Tap position", -(ntaps // 2), ntaps // 2, 0)
        fixed_tap = tap_value(True, ntaps, tap_range, 0.0, tap_number)
        st.caption(f"Tap value = **{fixed_tap:+.4f} pu**")
    else:
        ntaps = 31; tap_range = 10.0; tap_number = 0
        fixed_tap = st.number_input("Fixed tap value (pu)", value=0.025, min_value=-0.2, max_value=0.2, step=0.005, format="%.3f")

    st.markdown('<div class="section">Inverter Skid</div>', unsafe_allow_html=True)
    n_inverters = st.number_input("Number of Inverters", value=53, min_value=1, max_value=1000)

    st.markdown("**Inverter Step-Up (ISU)**")
    c7, c8 = st.columns(2)
    s_isu = c7.number_input("MVA/ISU", value=5.04, min_value=0.1, max_value=50.0, step=0.1)
    z_isu = c8.number_input("Imp. (pu)", value=0.08, min_value=0.01, max_value=0.2, step=0.005, format="%.3f")
    c9, c10 = st.columns(2)
    xr_isu = c9.number_input("X/R ISU", value=8.83, min_value=1.0, max_value=50.0, step=0.5)
    i0_isu = c10.number_input("NL I (pu)", value=0.005, min_value=0.0, max_value=0.05, step=0.001, format="%.4f")
    p0_isu = st.number_input("NL losses/ISU (kW)", value=8.0, min_value=0.0, max_value=100.0, step=1.0)

    st.markdown("**Inverter**")
    c11, c12 = st.columns(2)
    s_inv_mva = c11.number_input("MVA/inv", value=5.0, min_value=0.1, max_value=50.0, step=0.5)
    pf_inv = c12.number_input("Power factor", value=0.87, min_value=0.5, max_value=1.0, step=0.01)
    c13, c14 = st.columns(2)
    v_inv_max = c13.number_input("V max (pu)", value=1.10, min_value=1.0, max_value=1.2, step=0.01)
    v_inv_min = c14.number_input("V min (pu)", value=0.90, min_value=0.7, max_value=1.0, step=0.01)

    st.markdown('<div class="section">Capacitor Bank</div>', unsafe_allow_html=True)
    q_cap = st.number_input("Capacitor Bank (MVAR)", value=0.0, min_value=0.0, max_value=500.0, step=5.0)

    st.divider()
    run_btn = st.button("▶  Run Calculations", type="primary", use_container_width=True)
    auto_tap_btn = st.button("🔧 Find Optimal Tap", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD params dict (hashable tuple for caching)
# ══════════════════════════════════════════════════════════════════════════════

params = {
    'contract_power': float(contract_power),
    'contract_pf': float(contract_pf),
    'v_poi_calc': float(v_calc),
    'v_poi_min': float(v_min),
    'v_poi_max': float(v_max),
    'n_mpt': int(n_mpt),
    's_mpt_mva': float(s_mpt),
    'z_mpt': float(z_mpt),
    'xr_mpt': float(xr_mpt),
    'i0_mpt': float(i0_mpt),
    'p0_mpt': float(p0_mpt),
    'has_oltc': bool(has_oltc),
    'ntaps': int(ntaps),
    'tap_range': float(tap_range),
    'fixed_tap': float(fixed_tap),
    'tap_number': int(tap_number) if has_oltc else 0,
    'n_inverters': int(n_inverters),
    's_isu_mva': float(s_isu),
    'z_isu': float(z_isu),
    'xr_isu': float(xr_isu),
    'i0_isu': float(i0_isu),
    'p0_isu': float(p0_isu),
    's_inv_mva': float(s_inv_mva),
    'pf_inv': float(pf_inv),
    'v_inv_max': float(v_inv_max),
    'v_inv_min': float(v_inv_min),
    'q_cap': float(q_cap),
}

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# ⚡ Power Flow Sizing Tool")
st.caption("Solar farm POI sizing — 3-bus Newton-Raphson power flow")

# System overview metrics
total_s = n_inverters * s_inv_mva
total_p = total_s * pf_inv
mpt_total = n_mpt * s_mpt
q_contract = contract_power * np.tan(np.arccos(contract_pf))
ratio = total_s / mpt_total if mpt_total > 0 else 0

cols = st.columns(5)
cols[0].metric("Total Inverter S", f"{total_s:.1f} MVA")
cols[1].metric("Total Active P", f"{total_p:.1f} MW")
cols[2].metric("MPT Total Rating", f"{mpt_total:.0f} MVA")
cols[3].metric("Inv/MPT Ratio", f"{ratio:.2f}",
               delta=("OK" if 0.85 <= ratio <= 1.15 else "Check sizing"))
cols[4].metric("Contract Q demand", f"{q_contract:.1f} MVAR")

st.divider()

# ── Optimal tap search ────────────────────────────────────────────────────────
if auto_tap_btn and has_oltc:
    with st.spinner("Sweeping tap positions..."):
        best_tc, tap_df = optimal_tap(params, contract_power, v_calc)
    if tap_df is not None:
        st.success(f"✅ Optimal tap: **{best_tc:+.4f} pu** → V_POI closest to {v_calc} pu at rated power")
        with st.expander("Tap sweep results"):
            st.dataframe(tap_df.round(5), use_container_width=True)
    else:
        st.warning("Tap sweep did not converge. Check parameters.")

# ── Main power flow run ────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running power flow sweep (101 P-points × 2 Q-directions)..."):
        df = run_sweep(tuple(sorted(params.items())))

    if df.empty:
        st.error("❌ Power flow did not converge. Try adjusting parameters (V_target, impedances, tap).")
        st.stop()

    conv_pct = len(df) / (101 * 2) * 100

    # ── Summary ──────────────────────────────────────────────────────────────
    idx_nom = (df['P_inv_MW'] - contract_power).abs().idxmin() if len(df) > 0 else 0
    v_nom = df.loc[idx_nom, 'V_POI'] if len(df) > 0 else np.nan
    v_within = ((df['V_POI'] >= v_min) & (df['V_POI'] <= v_max)).mean() * 100

    st.subheader("📊 Results Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("V_POI @ Contract P", f"{v_nom:.4f} pu",
              delta="✓ In limits" if v_min <= v_nom <= v_max else "⚠ Violation")
    c2.metric("V_POI Min", f"{df['V_POI'].min():.4f} pu",
              delta="✓" if df['V_POI'].min() >= v_min else f"Below {v_min}")
    c3.metric("V_POI Max", f"{df['V_POI'].max():.4f} pu",
              delta="✓" if df['V_POI'].max() <= v_max else f"Above {v_max}")
    c4.metric("% Points in V-limits", f"{v_within:.1f}%")
    c5.metric("Convergence", f"{conv_pct:.0f}%")

    # ── Plots ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 PV Curve", "📉 QV Curve", "🗺 P-Q Capability", "📋 Data Table"])

    df_lead = df[df['Q_inv_MVAR'] >= 0].sort_values('P_inv_MW')
    df_lag  = df[df['Q_inv_MVAR'] < 0].sort_values('P_inv_MW')

    with tab1:
        fig = go.Figure()
        # Voltage band
        fig.add_hrect(y0=v_min, y1=v_max, fillcolor="lightgreen", opacity=0.12, line_width=0)
        fig.add_hline(y=v_min, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"V_min = {v_min} pu", annotation_position="bottom left")
        fig.add_hline(y=v_max, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"V_max = {v_max} pu", annotation_position="top left")
        fig.add_hline(y=v_calc, line_dash="dot", line_color="steelblue", line_width=1.5)
        fig.add_vline(x=contract_power, line_dash="dot", line_color="gray",
                      annotation_text=f"Contract = {contract_power:.0f} MW")

        if len(df_lead):
            fig.add_trace(go.Scatter(x=df_lead['P_inv_MW'], y=df_lead['V_POI'],
                                     mode='lines', name='Leading (gen Q)',
                                     line=dict(color='royalblue', width=2.5)))
        if len(df_lag):
            fig.add_trace(go.Scatter(x=df_lag['P_inv_MW'], y=df_lag['V_POI'],
                                     mode='lines', name='Lagging (abs Q)',
                                     line=dict(color='darkorange', width=2.5)))

        y_lo = max(0.80, df['V_POI'].min() - 0.03)
        y_hi = min(1.25, df['V_POI'].max() + 0.03)
        fig.update_layout(title="PV Curve — POI Voltage vs Active Power",
                          xaxis_title="Active Power Injection (MW)",
                          yaxis_title="V_POI (pu)",
                          yaxis=dict(range=[y_lo, y_hi]),
                          height=430, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_hrect(y0=v_min, y1=v_max, fillcolor="lightgreen", opacity=0.12, line_width=0)
        fig2.add_hline(y=v_min, line_dash="dash", line_color="red", line_width=1.5)
        fig2.add_hline(y=v_max, line_dash="dash", line_color="red", line_width=1.5)
        fig2.add_hline(y=v_calc, line_dash="dot", line_color="steelblue", line_width=1.5)

        df_q = df.sort_values('Q_inv_MVAR')
        fig2.add_trace(go.Scatter(
            x=df_q['Q_inv_MVAR'], y=df_q['V_POI'],
            mode='markers', name='Operating points',
            marker=dict(size=5, color=df_q['P_inv_MW'],
                        colorscale='Plasma', showscale=True,
                        colorbar=dict(title="P_inv (MW)"))
        ))
        fig2.update_layout(title="QV Curve — POI Voltage vs Reactive Power",
                           xaxis_title="Reactive Power Q (MVAR)",
                           yaxis_title="V_POI (pu)", height=430)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        # Inverter S_max circle
        ang = np.linspace(0, 2 * np.pi, 300)
        fig3.add_trace(go.Scatter(x=total_s * np.cos(ang), y=total_s * np.sin(ang),
                                  mode='lines', name=f'S_max = {total_s:.0f} MVA',
                                  line=dict(color='lightgray', dash='dash')))
        # Contract point
        fig3.add_trace(go.Scatter(x=[contract_power, contract_power],
                                   y=[q_contract, -q_contract],
                                   mode='markers+text',
                                   text=['Contract (lead)', 'Contract (lag)'],
                                   textposition='top right',
                                   name='Contract points',
                                   marker=dict(size=14, color='red', symbol='star')))
        # Feasible envelope (coloured by V_POI)
        v_ok = (df['V_POI'] >= v_min) & (df['V_POI'] <= v_max)
        for mask, label, color in [(v_ok, 'V in limits', None), (~v_ok, 'V violation', 'red')]:
            sub = df[mask]
            if len(sub):
                marker_kw = dict(size=4, color=sub['V_POI'], colorscale='RdYlGn',
                                 cmin=v_min, cmax=v_max, showscale=True,
                                 colorbar=dict(title="V_POI (pu)")) if color is None \
                    else dict(size=4, color='red', symbol='x')
                fig3.add_trace(go.Scatter(
                    x=sub['P_inv_MW'], y=sub['Q_inv_MVAR'],
                    mode='markers', name=label, marker=marker_kw
                ))

        fig3.add_vline(x=0, line_color='black', line_width=1)
        fig3.add_hline(y=0, line_color='black', line_width=1)
        fig3.update_layout(title="P-Q Capability Diagram",
                           xaxis_title="Active Power P (MW)",
                           yaxis_title="Reactive Power Q (MVAR)",
                           height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        disp = df[['P_inv_MW', 'Q_inv_MVAR', 'V_POI', 'V_ISU',
                   'theta_POI_deg', 'theta_ISU_deg', 'P_grid_MW', 'Q_grid_MVAR']].copy()
        disp.columns = ['P_inv (MW)', 'Q_inv (MVAR)', 'V_POI (pu)', 'V_ISU (pu)',
                        'θ_POI (°)', 'θ_ISU (°)', 'P_grid (MW)', 'Q_grid (MVAR)']
        st.dataframe(disp.round(5).style.background_gradient(subset=['V_POI (pu)'],
                     cmap='RdYlGn', vmin=v_min, vmax=v_max), use_container_width=True, height=400)
        csv = disp.round(6).to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "power_flow_results.csv", "text/csv")

    # ── Violations ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Constraint Check")
    viols = []
    if df['V_POI'].min() < v_min:
        viols.append(f"⚠️ **Under-voltage**: {(df['V_POI'] < v_min).sum()} points below V_min = {v_min} pu "
                     f"(worst = {df['V_POI'].min():.4f} pu)")
    if df['V_POI'].max() > v_max:
        viols.append(f"⚠️ **Over-voltage**: {(df['V_POI'] > v_max).sum()} points above V_max = {v_max} pu "
                     f"(worst = {df['V_POI'].max():.4f} pu)")
    if df['V_ISU'].min() < v_inv_min:
        viols.append(f"⚠️ **ISU under-voltage**: V_ISU drops to {df['V_ISU'].min():.4f} pu "
                     f"(limit {v_inv_min} pu)")
    if df['V_ISU'].max() > v_inv_max:
        viols.append(f"⚠️ **ISU over-voltage**: V_ISU rises to {df['V_ISU'].max():.4f} pu "
                     f"(limit {v_inv_max} pu)")
    if total_s > mpt_total * 1.05:
        viols.append(f"⚠️ **MPT overloaded**: {total_s:.1f} MVA inverters > {mpt_total:.1f} MVA MPT rating")
    if viols:
        for v in viols:
            st.warning(v)
    else:
        st.success("✅ All constraints satisfied across the full P-Q operating envelope.")

else:
    st.info("👈 Set parameters in the sidebar, then press **▶ Run Calculations**.")
    with st.expander("ℹ️ Model description"):
        st.markdown("""
**3-Bus Power Flow Model:**

| Bus | Description | Type |
|-----|-------------|------|
| 1   | Grid / POI  | Swing (V₁ = V_target, θ₁ = 0) |
| 2   | MPT secondary | PQ (no load losses only) |
| 3   | ISU secondary / Inverter terminals | PQ (P+jQ injected) |

**Transformer modelling:**
- MPT uses a π-model with tap changer on the primary side
- ISUs aggregated into one equivalent transformer (S_total = N × S_isu)
- No-load losses modelled as magnetising branch (G+jB)

**Sweep:**  
Active power swept from −P_max to +P_max at both leading and lagging reactive limits,
tracing the full P-Q capability envelope.

**Outputs:** PV curve, QV curve, P-Q capability diagram, voltage violations.
        """)