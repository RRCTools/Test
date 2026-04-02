"""
Power Flow Sizing Tool — Streamlit App
Solar farm POI sizing — 3-bus Newton-Raphson | ISU S-circle sweep

Key model facts:
  - Bus 1 = Grid swing (V=v_target, θ=0)
  - Bus 2 = MPT secondary = POI (PQ, no injection)
  - Bus 3 = ISU secondary = inverter bus (PQ injection)
  - Y = S_equip / (sbase * Z_pu)  [correct per-unit conversion]
  - Sweep traces ISU S-circle: P²+Q² = (n_isu * S_isu)²
  - n_isu = n_inverters / inv_per_isu
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Power Flow Sizing Tool", layout="wide", page_icon="⚡")

st.markdown("""
<style>
  .block-container{padding-top:1rem;}
  h1{color:#1a3a5c;margin-bottom:2px;}
  .stMetric label{font-size:.78rem;}
  div[data-testid="stSidebarContent"]{background:#f5f8fc;}
  .sec{font-weight:700;color:#1a5276;border-bottom:2px solid #2874a6;
       padding-bottom:3px;margin-top:12px;margin-bottom:4px;font-size:.92rem;}
  .suggest{background:#e8f4fd;border-left:3px solid #2980b9;padding:6px 10px;
           border-radius:3px;font-size:.85rem;margin:4px 0;}
</style>
""", unsafe_allow_html=True)


# ── ENGINE ─────────────────────────────────────────────────────────────────────

def build_ybus(sbase, p, tap_c):
    """3×3 Ybus. Y_equip = S_equip / (sbase * Z_pu). tap_c on MPT primary side."""
    # MPT
    s_m = p['s_mpt_mva'] * p['n_mpt']
    rm  = p['z_mpt'] * np.cos(np.arctan(p['xr_mpt']))
    xm  = p['z_mpt'] * np.sin(np.arctan(p['xr_mpt']))
    Ys  = (s_m / sbase) / complex(rm, xm)
    gm  = (p['p0_mpt'] * p['n_mpt'] / 1e3) / s_m
    bm  = np.sqrt(max(p['i0_mpt']**2 - gm**2, 0.0))
    Ym  = complex(gm, bm) * (s_m / sbase)
    c   = 1.0 - tap_c
    if abs(c) < 1e-9: c = 1.0
    Y10 = (1-c)/c * Ys + Ym/c**2
    Y12 = c * Ys
    Y20 = (c-1) * Ys

    # Aggregate ISU  (n_isu units, each s_isu_mva)
    n_isu = p['n_isu']
    si    = p['s_isu_mva'] * n_isu
    ri    = p['z_isu'] * np.cos(np.arctan(p['xr_isu']))
    xi    = p['z_isu'] * np.sin(np.arctan(p['xr_isu']))
    Yi    = (si / sbase) / complex(ri, xi)
    gi    = (p['p0_isu'] * n_isu / 1e3) / si
    bi2   = np.sqrt(max(p['i0_isu']**2 - gi**2, 0.0))
    Ymi   = complex(gi, -bi2) * (si / sbase)

    yc = complex(0.0, p['q_cap'] / sbase)

    Y = np.zeros((3, 3), dtype=complex)
    Y[0,0] = Y10 + Y12;  Y[0,1] = -Y12;   Y[1,0] = -Y12
    Y[1,1] = Y12 + Y20 + Yi + Ymi + yc
    Y[1,2] = -Yi;        Y[2,1] = -Yi;     Y[2,2] = Yi
    return Y


def nr_pf(Y, P3, Q3, V1=1.0, V2i=None, V3i=None, tol=1e-9, maxiter=100):
    """NR power flow: bus1=swing, bus2+3=PQ."""
    if V2i is None: V2i = V1
    if V3i is None: V3i = V1
    V  = np.array([V1, V2i, V3i])
    th = np.zeros(3)
    for _ in range(maxiter):
        Pc = np.zeros(3); Qc = np.zeros(3)
        for i in range(3):
            for j in range(3):
                G=Y[i,j].real; B=Y[i,j].imag; d=th[i]-th[j]
                Pc[i] += V[i]*V[j]*(G*np.cos(d)+B*np.sin(d))
                Qc[i] += V[i]*V[j]*(G*np.sin(d)-B*np.cos(d))
        mis = np.array([0-Pc[1], P3-Pc[2], 0-Qc[1], Q3-Qc[2]])
        if np.max(np.abs(mis)) < tol:
            return V, th, True
        J = np.zeros((4,4))
        for ri, bi in enumerate([1,2]):
            for ci, bj in enumerate([1,2]):
                if bi == bj:
                    J[ri,   ci]   = V[bi]*sum(V[k]*(-Y[bi,k].real*np.sin(th[bi]-th[k])+Y[bi,k].imag*np.cos(th[bi]-th[k])) for k in range(3) if k!=bi)
                    J[ri,   2+ci] = sum(V[k]*(Y[bi,k].real*np.cos(th[bi]-th[k])+Y[bi,k].imag*np.sin(th[bi]-th[k])) for k in range(3))+Y[bi,bi].real*V[bi]
                    J[2+ri, ci]   = V[bi]*sum(V[k]*(Y[bi,k].real*np.cos(th[bi]-th[k])+Y[bi,k].imag*np.sin(th[bi]-th[k])) for k in range(3) if k!=bi)
                    J[2+ri, 2+ci] = sum(V[k]*(Y[bi,k].real*np.sin(th[bi]-th[k])-Y[bi,k].imag*np.cos(th[bi]-th[k])) for k in range(3))-Y[bi,bi].imag*V[bi]
                else:
                    G=Y[bi,bj].real; B=Y[bi,bj].imag; d=th[bi]-th[bj]
                    J[ri,   ci]   =  V[bi]*V[bj]*(G*np.sin(d)-B*np.cos(d))
                    J[ri,   2+ci] =  V[bi]*(G*np.cos(d)+B*np.sin(d))
                    J[2+ri, ci]   = -V[bi]*V[bj]*(G*np.cos(d)+B*np.sin(d))
                    J[2+ri, 2+ci] =  V[bi]*(G*np.sin(d)-B*np.cos(d))
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return V, th, False
        th[1]+=dx[0]; th[2]+=dx[1]; V[1]+=dx[2]; V[2]+=dx[3]
    return V, th, False


def get_tap(has_oltc, ntaps, tap_range_pct, fixed_tap_pu, tap_number=0):
    if not has_oltc:
        return fixed_tap_pu
    step = (tap_range_pct / 100.0) / ((ntaps - 1) / 2.0)
    return tap_number * step


@st.cache_data(show_spinner=False)
def run_sweep(pkey):
    p     = dict(pkey)
    sbase = p['contract_power']
    n_isu = p['n_isu']
    S_isu = n_isu * p['s_isu_mva'] / sbase        # total ISU apparent power (pu)
    P_max = p['n_inverters'] * p['s_inv_mva'] * p['pf_inv'] / sbase  # inverter P limit (pu)
    Vt    = p['v_poi_calc']

    tap_c = get_tap(p['has_oltc'], p['ntaps'], p['tap_range'], p['fixed_tap'], p['tap_number'])
    Y     = build_ybus(sbase, p, tap_c)

    results = []
    V2p, V3p = Vt, Vt
    for theta in np.linspace(0, 2*np.pi, 121, endpoint=False):
        P3 = np.clip(S_isu * np.cos(theta), -P_max, P_max)
        Q3 = S_isu * np.sin(theta)
        V, th, conv = nr_pf(Y, P3, Q3, V1=Vt, V2i=V2p, V3i=V3p)
        if conv:
            V2p, V3p = V[1], V[2]
            P1 = sum(V[0]*V[j]*(Y[0,j].real*np.cos(th[0]-th[j])+Y[0,j].imag*np.sin(th[0]-th[j])) for j in range(3))
            Q1 = sum(V[0]*V[j]*(Y[0,j].real*np.sin(th[0]-th[j])-Y[0,j].imag*np.cos(th[0]-th[j])) for j in range(3))
            results.append({
                'P_inv_MW':      P3 * sbase,
                'Q_inv_MVAR':    Q3 * sbase,
                'V_POI':         V[1],
                'V_ISU':         V[2],
                'theta_POI_deg': np.degrees(th[1]),
                'P_grid_MW':     P1 * sbase,
                'Q_grid_MVAR':   Q1 * sbase,
                'tap_pu':        tap_c,
            })
    return pd.DataFrame(results)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Input Parameters")

    # ── Grid ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Grid Requirement</div>', unsafe_allow_html=True)
    contract_power = st.number_input("Contract Power (MW)", value=200.0, min_value=1.0, max_value=5000.0, step=10.0)
    contract_pf    = st.number_input("Contract Power Factor", value=0.95, min_value=0.5, max_value=1.0, step=0.01)

    # ── POI Voltage ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">POI Voltage</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    v_max  = ca.number_input("Max (pu)", value=1.05, min_value=1.0, max_value=1.15, step=0.01)
    v_min  = cb.number_input("Min (pu)", value=0.95, min_value=0.80, max_value=1.0,  step=0.01)
    v_calc = st.number_input("This calculation (pu)", value=1.00, min_value=0.8, max_value=1.15, step=0.01)

    # ── MPT ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Main Power Transformer (MPT)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n_mpt = c1.number_input("# MPTs",    value=1,     min_value=1, max_value=10)
    s_mpt = c2.number_input("MVA / MPT", value=240.0, min_value=10.0, max_value=2000.0, step=10.0)
    c3, c4 = st.columns(2)
    z_mpt  = c3.number_input("Z (pu)",  value=0.08, min_value=0.01, max_value=0.25, step=0.005, format="%.3f")
    xr_mpt = c4.number_input("X/R",     value=40.0, min_value=1.0,  max_value=100.0, step=1.0)
    c5, c6 = st.columns(2)
    i0_mpt = c5.number_input("No-load I (pu)", value=0.001,  min_value=0.0, max_value=0.05, step=0.0005, format="%.4f")
    p0_mpt = c6.number_input("No-load P (kW)", value=10.0,   min_value=0.0, max_value=500.0, step=1.0)

    has_oltc = st.checkbox("Has OLTC tap changer?", value=True)
    if has_oltc:
        c7, c8 = st.columns(2)
        ntaps      = c7.number_input("# Taps",    value=31,   min_value=3,  max_value=99, step=2)
        tap_range  = c8.number_input("Range (%)", value=10.0, min_value=1.0, max_value=30.0, step=1.0)
        tap_number = st.slider("Tap position", int(-(ntaps//2)), int(ntaps//2), 0)
        fixed_tap  = get_tap(True, ntaps, tap_range, 0.0, tap_number)
        st.caption(f"Tap = **{fixed_tap:+.4f} pu**  |  c = {1-fixed_tap:.3f}")
    else:
        ntaps = 31; tap_range = 10.0; tap_number = 0
        fixed_tap = st.number_input("Fixed tap (pu)", value=0.0, min_value=-0.2, max_value=0.2, step=0.005, format="%.3f")

    # ── Inverter Skid ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Inverter Skid</div>', unsafe_allow_html=True)
    n_inverters = st.number_input("Number of Inverters", value=53, min_value=1, max_value=5000)

    st.markdown("**Inverter Step-Up (ISU)**")
    c9, c10 = st.columns(2)
    s_isu  = c9.number_input("MVA / ISU",  value=5.04, min_value=0.1, max_value=100.0, step=0.1)
    z_isu  = c10.number_input("Z (pu)",    value=0.08, min_value=0.01, max_value=0.2, step=0.005, format="%.3f")
    c11, c12 = st.columns(2)
    xr_isu = c11.number_input("X/R ISU",        value=8.83,  min_value=1.0, max_value=50.0, step=0.5)
    i0_isu = c12.number_input("NL current (pu)", value=0.005, min_value=0.0, max_value=0.05, step=0.001, format="%.4f")
    p0_isu = st.number_input("NL losses / ISU (kW)", value=8.0, min_value=0.0, max_value=100.0, step=1.0)

    st.markdown("**Inverter**")
    c13, c14 = st.columns(2)
    s_inv_mva = c13.number_input("MVA / inv",    value=5.0,  min_value=0.01, max_value=50.0, step=0.1)
    pf_inv    = c14.number_input("Power factor", value=0.87, min_value=0.5,  max_value=1.0,  step=0.01)
    c15, c16 = st.columns(2)
    v_inv_max = c15.number_input("V max (pu)", value=1.10, min_value=1.0, max_value=1.2, step=0.01)
    v_inv_min = c16.number_input("V min (pu)", value=0.90, min_value=0.7, max_value=1.0, step=0.01)

    # ── ISU grouping suggestion ────────────────────────────────────────────────
    suggested_ipu = None
    if s_isu > 0 and s_inv_mva > 0:
        raw_ratio = s_isu / s_inv_mva
        candidates = [r for r in [round(raw_ratio), int(raw_ratio), int(raw_ratio)+1] if r >= 1]
        for cand in candidates:
            if abs(cand * s_inv_mva - s_isu) / s_isu < 0.05:   # within 5%
                suggested_ipu = cand
                break

    if suggested_ipu:
        st.markdown(
            f'<div class="suggest">💡 {s_isu:.2g} MVA ISU ÷ {s_inv_mva:.2g} MVA inv = '
            f'<b>{suggested_ipu} inverters per ISU</b></div>',
            unsafe_allow_html=True)
        inv_per_isu = st.number_input(
            "Inverters per ISU", value=suggested_ipu, min_value=1, max_value=50)
    else:
        inv_per_isu = st.number_input(
            "Inverters per ISU", value=1, min_value=1, max_value=50,
            help="How many inverters share one ISU transformer")

    n_isu_calc = n_inverters / inv_per_isu
    s_isu_total = n_isu_calc * s_isu
    st.caption(f"→ **{n_isu_calc:.1f} ISUs** × {s_isu} MVA = **{s_isu_total:.1f} MVA** total")

    # ── Cap bank ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Capacitor Bank</div>', unsafe_allow_html=True)
    q_cap = st.number_input("Capacitor Bank (MVAR)", value=0.0, min_value=0.0, max_value=500.0, step=5.0)

    st.divider()
    run_btn = st.button("▶  Run Calculations", type="primary", use_container_width=True)


# ── PARAMS DICT ────────────────────────────────────────────────────────────────

params = {
    'contract_power': float(contract_power),
    'contract_pf':    float(contract_pf),
    'v_poi_calc':     float(v_calc),
    'v_poi_min':      float(v_min),
    'v_poi_max':      float(v_max),
    'n_mpt':          int(n_mpt),
    's_mpt_mva':      float(s_mpt),
    'z_mpt':          float(z_mpt),
    'xr_mpt':         float(xr_mpt),
    'i0_mpt':         float(i0_mpt),
    'p0_mpt':         float(p0_mpt),
    'has_oltc':       bool(has_oltc),
    'ntaps':          int(ntaps),
    'tap_range':      float(tap_range),
    'fixed_tap':      float(fixed_tap),
    'tap_number':     int(tap_number) if has_oltc else 0,
    'n_inverters':    int(n_inverters),
    'inv_per_isu':    int(inv_per_isu),
    'n_isu':          float(n_isu_calc),
    's_isu_mva':      float(s_isu),
    'z_isu':          float(z_isu),
    'xr_isu':         float(xr_isu),
    'i0_isu':         float(i0_isu),
    'p0_isu':         float(p0_isu),
    's_inv_mva':      float(s_inv_mva),
    'pf_inv':         float(pf_inv),
    'v_inv_max':      float(v_inv_max),
    'v_inv_min':      float(v_inv_min),
    'q_cap':          float(q_cap),
}

# ── HEADER METRICS ─────────────────────────────────────────────────────────────

st.markdown("# ⚡ Power Flow Sizing Tool")
st.caption("Solar farm POI sizing — 3-bus Newton-Raphson | ISU S-circle sweep")

total_inv_s = n_inverters * s_inv_mva
total_inv_p = total_inv_s * pf_inv
s_isu_tot   = s_isu_total                   # n_isu_calc * s_isu
mpt_tot     = n_mpt * s_mpt
q_ctr       = contract_power * np.tan(np.arccos(contract_pf))
ratio       = s_isu_tot / mpt_tot if mpt_tot > 0 else 0.0

mc = st.columns(5)
mc[0].metric("Inverter S total", f"{total_inv_s:.1f} MVA")
mc[1].metric("Inverter P max",   f"{total_inv_p:.1f} MW")
mc[2].metric("ISU S total",      f"{s_isu_tot:.1f} MVA")
mc[3].metric("MPT rating",       f"{mpt_tot:.0f} MVA")
mc[4].metric("ISU/MPT ratio",    f"{ratio:.2f}",
             delta="OK" if 0.85 <= ratio <= 1.15 else "Check")

st.divider()

# ── RUN & DISPLAY ──────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running power flow sweep (121 points on ISU S-circle)…"):
        df = run_sweep(tuple(sorted(params.items())))

    if df.empty:
        st.error("❌ Power flow did not converge for any point. "
                 "Check impedances, tap position, and voltage target.")
        st.stop()

    v_nom     = df.loc[(df['P_inv_MW'] - contract_power).abs().idxmin(), 'V_POI']
    v_in_band = ((df['V_POI'] >= v_min) & (df['V_POI'] <= v_max)).mean() * 100

    st.subheader("📊 Results Summary")
    sc = st.columns(5)
    sc[0].metric("V_POI @ Contract P", f"{v_nom:.4f} pu",
                 delta="✓ In limits" if v_min <= v_nom <= v_max else "⚠ Violation")
    sc[1].metric("V_POI min", f"{df['V_POI'].min():.4f} pu",
                 delta="✓" if df['V_POI'].min() >= v_min else f"Below {v_min}")
    sc[2].metric("V_POI max", f"{df['V_POI'].max():.4f} pu",
                 delta="✓" if df['V_POI'].max() <= v_max else f"Above {v_max}")
    sc[3].metric("% in V-limits", f"{v_in_band:.1f}%")
    sc[4].metric("Converged pts", f"{len(df)}/121")

    df_lead = df[df['Q_inv_MVAR'] >= 0].sort_values('P_inv_MW')
    df_lag  = df[df['Q_inv_MVAR'] <  0].sort_values('P_inv_MW')

    t1, t2, t3, t4 = st.tabs(["📈 PV Curve", "📉 QV Curve", "🗺 P-Q Capability", "📋 Data"])

    with t1:
        fig = go.Figure()
        fig.add_hrect(y0=v_min, y1=v_max, fillcolor="lightgreen", opacity=0.12, line_width=0)
        fig.add_hline(y=v_min,  line_dash="dash", line_color="red",       line_width=1.5,
                      annotation_text=f"V_min={v_min}", annotation_position="bottom left")
        fig.add_hline(y=v_max,  line_dash="dash", line_color="red",       line_width=1.5,
                      annotation_text=f"V_max={v_max}", annotation_position="top left")
        fig.add_hline(y=v_calc, line_dash="dot",  line_color="steelblue", line_width=1.5)
        fig.add_vline(x=contract_power, line_dash="dot", line_color="gray",
                      annotation_text=f"Contract={contract_power:.0f}MW")
        if len(df_lead):
            fig.add_trace(go.Scatter(x=df_lead['P_inv_MW'], y=df_lead['V_POI'], mode='lines',
                                     name='Leading / gen Q',
                                     line=dict(color='royalblue', width=2.5)))
        if len(df_lag):
            fig.add_trace(go.Scatter(x=df_lag['P_inv_MW'],  y=df_lag['V_POI'],  mode='lines',
                                     name='Lagging / abs Q',
                                     line=dict(color='darkorange', width=2.5)))
        y_lo = max(0.75, df['V_POI'].min() - 0.03)
        y_hi = min(1.30, df['V_POI'].max() + 0.03)
        fig.update_layout(title="PV Curve — POI Voltage vs Active Power",
                          xaxis_title="P (MW)", yaxis_title="V_POI (pu)",
                          yaxis=dict(range=[y_lo, y_hi]),
                          height=440, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2 = go.Figure()
        fig2.add_hrect(y0=v_min, y1=v_max, fillcolor="lightgreen", opacity=0.12, line_width=0)
        fig2.add_hline(y=v_min, line_dash="dash", line_color="red", line_width=1.5)
        fig2.add_hline(y=v_max, line_dash="dash", line_color="red", line_width=1.5)
        dq = df.sort_values('Q_inv_MVAR')
        fig2.add_trace(go.Scatter(
            x=dq['Q_inv_MVAR'], y=dq['V_POI'], mode='markers',
            marker=dict(size=5, color=dq['P_inv_MW'], colorscale='Plasma',
                        showscale=True, colorbar=dict(title="P_inv (MW)"))))
        fig2.update_layout(title="QV Curve — POI Voltage vs Reactive Power",
                           xaxis_title="Q (MVAR)", yaxis_title="V_POI (pu)", height=440)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        fig3 = go.Figure()
        ang = np.linspace(0, 2*np.pi, 300)
        fig3.add_trace(go.Scatter(
            x=s_isu_tot * np.cos(ang), y=s_isu_tot * np.sin(ang),
            mode='lines', name=f'ISU S_max={s_isu_tot:.0f}MVA',
            line=dict(color='lightgray', dash='dash')))
        fig3.add_trace(go.Scatter(
            x=total_inv_s * np.cos(ang), y=total_inv_s * np.sin(ang),
            mode='lines', name=f'Inv S_max={total_inv_s:.0f}MVA',
            line=dict(color='lightblue', dash='dot')))
        fig3.add_trace(go.Scatter(
            x=[contract_power, contract_power], y=[q_ctr, -q_ctr],
            mode='markers+text', text=['Contract (lead)', 'Contract (lag)'],
            textposition='top right',
            marker=dict(size=14, color='red', symbol='star'), name='Contract'))
        v_ok   = (df['V_POI'] >= v_min) & (df['V_POI'] <= v_max)
        sub_ok = df[v_ok]; sub_viol = df[~v_ok]
        if len(sub_ok):
            fig3.add_trace(go.Scatter(
                x=sub_ok['P_inv_MW'], y=sub_ok['Q_inv_MVAR'], mode='markers',
                name='V in limits',
                marker=dict(size=5, color=sub_ok['V_POI'],
                            colorscale='RdYlGn', cmin=v_min, cmax=v_max,
                            showscale=True, colorbar=dict(title="V_POI (pu)"))))
        if len(sub_viol):
            fig3.add_trace(go.Scatter(
                x=sub_viol['P_inv_MW'], y=sub_viol['Q_inv_MVAR'], mode='markers',
                name='V violation', marker=dict(size=5, color='red', symbol='x')))
        fig3.add_vline(x=0, line_color='black', line_width=1)
        fig3.add_hline(y=0, line_color='black', line_width=1)
        fig3.update_layout(title="P-Q Capability Diagram",
                           xaxis_title="P (MW)", yaxis_title="Q (MVAR)", height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        disp = df[['P_inv_MW','Q_inv_MVAR','V_POI','V_ISU',
                   'theta_POI_deg','P_grid_MW','Q_grid_MVAR']].round(5).copy()
        disp.columns = ['P_inv (MW)','Q_inv (MVAR)','V_POI (pu)','V_ISU (pu)',
                        'θ_POI (°)','P_grid (MW)','Q_grid (MVAR)']
        st.dataframe(disp, use_container_width=True, height=400)
        st.download_button("⬇ Download CSV", disp.to_csv(index=False),
                           "power_flow_results.csv", "text/csv")

    st.divider()
    st.subheader("🔍 Constraint Check")
    viols = []
    if df['V_POI'].min() < v_min:
        viols.append(f"⚠️ **Under-voltage**: {(df['V_POI']<v_min).sum()} pts "
                     f"below {v_min} pu (worst {df['V_POI'].min():.4f})")
    if df['V_POI'].max() > v_max:
        viols.append(f"⚠️ **Over-voltage**: {(df['V_POI']>v_max).sum()} pts "
                     f"above {v_max} pu (worst {df['V_POI'].max():.4f})")
    if df['V_ISU'].min() < v_inv_min:
        viols.append(f"⚠️ **ISU under-voltage**: {df['V_ISU'].min():.4f} pu "
                     f"(inverter limit {v_inv_min})")
    if df['V_ISU'].max() > v_inv_max:
        viols.append(f"⚠️ **ISU over-voltage**: {df['V_ISU'].max():.4f} pu "
                     f"(inverter limit {v_inv_max})")
    if s_isu_tot > mpt_tot * 1.05:
        viols.append(f"⚠️ **MPT overloaded**: {s_isu_tot:.1f} > {mpt_tot:.1f} MVA")
    if viols:
        for v in viols:
            st.warning(v)
    else:
        st.success("✅ No constraint violations across the full operating envelope.")

else:
    st.info("👈 Configure parameters in the sidebar, then press **▶ Run Calculations**.")
    with st.expander("ℹ️ Model description"):
        st.markdown("""
**3-bus NR power flow — same structure as the Excel VBA:**

| Bus | Node | Type |
|-----|------|------|
| 1 | Grid (swing) | V₁ = V_target, θ₁ = 0 |
| 2 | MPT secondary = **POI** | PQ (losses only) |
| 3 | ISU secondary = inverter bus | PQ (P+jQ injected) |

**Sweep:** 121 points on the ISU apparent power circle  
P² + Q² = (n_ISU × S_ISU)², clamped at inverter active P limit.

**ISU grouping:** `n_ISU = n_inverters / inverters_per_ISU`  
The app suggests the grouping automatically when S_ISU ≈ k × S_inverter.

**Transformer admittance:** Y = S_equip / (S_base × Z_pu)
        """)