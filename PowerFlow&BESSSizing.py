"""
BESS Sizing Tool — Streamlit App
Replicates the Excel MAIN DC BESS sizing engine with 3-bus NR power flow backing.

Sizing logic (backwards from POI → DC):
  POI target → gen-tie → MPT → MV bus (subtract aux) → MVT → PCS → DC cable → battery

Power Flow: 3-bus Newton-Raphson
  Bus 1 = Grid (swing),  Bus 2 = MPT secondary = POI,  Bus 3 = ISU secondary = PCS bus
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import ceil

st.set_page_config(page_title="BESS Sizing Tool", layout="wide", page_icon="🔋")

st.markdown("""
<style>
  .block-container{padding-top:.5rem;}
  h1{color:#1a3a5c;margin-bottom:0;}
  .stMetric label{font-size:.75rem;}
  div[data-testid="stSidebarContent"]{background:#f0f4f8;}
  .sec{font-weight:700;color:#1a5276;border-bottom:2px solid #2874a6;
       padding-bottom:2px;margin-top:10px;margin-bottom:3px;font-size:.85rem;}
  .ok{color:#1e8449;font-weight:700;}
  .fail{color:#c0392b;font-weight:700;}
  .warn{color:#d68910;font-weight:700;}
  .kpi-box{background:#eaf4fb;border-radius:6px;padding:10px 14px;margin:4px 0;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT SOH DEGRADATION CURVE (from Excel MAIN DC Engine)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_SOH = [1.0000, 0.9342, 0.9115, 0.8933, 0.8775, 0.8633, 0.8502,
               0.8381, 0.8266, 0.8158, 0.8054, 0.7955, 0.7859, 0.7767,
               0.7677, 0.7590, 0.7506, 0.7424, 0.7343, 0.7265, 0.7188]


# ══════════════════════════════════════════════════════════════════════════════
#  3-BUS POWER FLOW ENGINE  (identical to test1final.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_ybus(sbase, mpt_s, mpt_z, mpt_xr, mpt_i0, mpt_p0kw,
               isu_s_total, isu_z, isu_xr, isu_i0, isu_p0kw_total,
               tap_c, q_cap_mvar=0.0):
    """Y = S_equip / (sbase * Z_pu). Tap_c on MPT primary."""
    rm = mpt_z * np.cos(np.arctan(mpt_xr))
    xm = mpt_z * np.sin(np.arctan(mpt_xr))
    Ys = (mpt_s / sbase) / complex(rm, xm)
    gm = (mpt_p0kw / 1e3) / mpt_s
    bm = np.sqrt(max(mpt_i0**2 - gm**2, 0.0))
    Ym = complex(gm, bm) * (mpt_s / sbase)
    c  = 1.0 - tap_c
    if abs(c) < 1e-9: c = 1.0
    Y10 = (1-c)/c * Ys + Ym/c**2
    Y12 = c * Ys
    Y20 = (c-1) * Ys

    ri  = isu_z * np.cos(np.arctan(isu_xr))
    xi  = isu_z * np.sin(np.arctan(isu_xr))
    Yi  = (isu_s_total / sbase) / complex(ri, xi)
    gi  = (isu_p0kw_total / 1e3) / isu_s_total
    bi2 = np.sqrt(max(isu_i0**2 - gi**2, 0.0))
    Ymi = complex(gi, -bi2) * (isu_s_total / sbase)

    yc = complex(0.0, q_cap_mvar / sbase)
    Y  = np.zeros((3, 3), dtype=complex)
    Y[0,0] = Y10+Y12; Y[0,1] = -Y12;  Y[1,0] = -Y12
    Y[1,1] = Y12+Y20+Yi+Ymi+yc
    Y[1,2] = -Yi;     Y[2,1] = -Yi;   Y[2,2] = Yi
    return Y


def nr_pf(Y, P3, Q3, V1=1.0, V2i=None, V3i=None, tol=1e-9, maxiter=100):
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
        if np.max(np.abs(mis)) < tol: return V, th, True
        J = np.zeros((4,4))
        for ri, bi in enumerate([1,2]):
            for ci, bj in enumerate([1,2]):
                if bi==bj:
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
        try: dx = np.linalg.solve(J, mis)
        except: return V, th, False
        th[1]+=dx[0]; th[2]+=dx[1]; V[1]+=dx[2]; V[2]+=dx[3]
    return V, th, False


def get_tap(has_oltc, ntaps, tap_range_pct, fixed_tap, tap_number=0):
    if not has_oltc: return fixed_tap
    step = (tap_range_pct/100.0) / ((ntaps-1)/2.0)
    return tap_number * step


# ══════════════════════════════════════════════════════════════════════════════
#  BESS SIZING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def bess_sizing(inputs):
    """
    Core sizing engine. Returns dict of all computed quantities.
    Works backwards: POI → DC battery through the loss chain.
    """
    i = inputs

    # ── Reactive requirement ────────────────────────────────────────────────
    if i['pf_or_mvar'] == 'Target PF':
        pf_poi   = i['target_pf']
        q_poi    = i['poi_mw'] * np.tan(np.arccos(pf_poi))
    else:
        q_poi    = i['target_mvar']
        pf_poi   = i['poi_mw'] / np.sqrt(i['poi_mw']**2 + q_poi**2)
    s_poi = np.sqrt(i['poi_mw']**2 + q_poi**2)

    # ── Loss factors ────────────────────────────────────────────────────────
    lf = (i['eta_transmission'] * i['eta_mpt'] * i['eta_mv_cable'] *
          i['eta_mvt'] * i['eta_pcs'] * i['eta_dc_cable'] *
          i['eta_charge_disch'])
    # Power loss: POI → inverter terminals (excl. DC chain)
    lf_ac = i['eta_transmission'] * i['eta_mpt'] * i['eta_mv_cable'] * i['eta_mvt'] * i['eta_pcs']
    # Energy loss: full chain
    lf_energy = lf * i['eta_auxiliary']

    # ── Power chain (back-calculate to inverter) ─────────────────────────────
    p_gentie   = i['poi_mw']   / i['eta_transmission']
    p_mpt_lv   = p_gentie      / i['eta_mpt']
    p_aux_total = i['aux_kw_per_block'] / 1000.0 * i['n_blocks_guess']
    p_mv_bus   = p_mpt_lv - p_aux_total        # after subtracting aux
    p_mvt_out  = p_mv_bus  / i['eta_mv_cable']
    p_inv_total = p_mvt_out / i['eta_mvt']

    # Reactive at POI → MV Bus (approximate, losses add reactance)
    q_mv_bus   = q_poi + i['poi_mw'] * 0.20   # rule of thumb from engine

    # ── Per-block / per-PCS sizing ──────────────────────────────────────────
    inv_mva    = i['inv_mva']                  # de-rated inverter MVA
    pcs_pf     = i['pcs_max_pf']               # operating PF at PCS
    inv_mw     = inv_mva * pcs_pf
    inv_mvar   = inv_mva * np.sqrt(max(1 - pcs_pf**2, 0))
    block_mwh_dc = i['batt_mwh_per_unit'] * i['units_per_block']

    # ── Required blocks for POWER ───────────────────────────────────────────
    p_needed_inv = p_inv_total / i['eta_pcs']   # at inverter DC side
    n_blocks_power = ceil(p_needed_inv / inv_mw)

    # ── Required blocks for ENERGY (BOL, accounting for degradation) ────────
    soh = i['soh_curve']
    soh_eol = soh[min(i['project_years'], len(soh)-1)]
    # If augmenting, EOL SOH resets at augmentation
    if i['aug_year'] > 0 and i['aug_year'] < i['project_years']:
        soh_eol = soh[min(i['project_years'] - i['aug_year'], len(soh)-1)]

    # Energy needed at DC level to deliver poi_mwh at POI
    energy_dc_needed = i['poi_mwh'] / lf_energy
    energy_dc_bol    = energy_dc_needed / soh_eol  # overbuild for degradation
    n_blocks_energy  = ceil(energy_dc_bol / block_mwh_dc)

    # ── Final block count (max of power and energy) ──────────────────────────
    n_blocks = max(n_blocks_power, n_blocks_energy, 1)

    # ── Actual quantities ────────────────────────────────────────────────────
    n_batt_units  = n_blocks * i['units_per_block']
    n_pcs         = n_blocks
    n_mvt         = n_blocks

    # Actual energy cascade
    energy_dc_actual    = n_blocks * block_mwh_dc
    energy_inv_actual   = energy_dc_actual * i['eta_charge_disch'] * i['eta_dc_cable'] * i['eta_pcs']
    energy_mvbus_actual = energy_inv_actual * i['eta_mvt'] * i['eta_mv_cable'] * i['eta_auxiliary']
    energy_poi_actual   = energy_mvbus_actual * i['eta_mpt'] * i['eta_transmission']

    # Actual power cascade
    p_inv_actual   = n_blocks * inv_mw
    p_mvt_actual   = p_inv_actual * i['eta_mvt']
    p_mvbus_actual = p_mvt_actual * i['eta_mv_cable']
    p_aux_actual   = i['aux_kw_per_block'] / 1000.0 * n_blocks
    p_mpt_in_actual = (p_mvbus_actual + p_aux_actual) * i['eta_mpt']  # rough
    p_poi_actual    = p_mpt_in_actual * i['eta_transmission']

    s_inv_total   = n_blocks * inv_mva
    # MVA at MV bus (for MPT sizing)
    q_inv_total   = n_blocks * inv_mvar
    s_mvbus_actual = np.sqrt(p_mvbus_actual**2 + q_mv_bus**2)
    min_mpt_mva   = s_mvbus_actual

    # ── Reactive power checks ────────────────────────────────────────────────
    q_available_poi = n_blocks * inv_mvar * i['eta_mvt'] * i['eta_mv_cable'] * i['eta_mpt'] * i['eta_transmission']
    q_meets = q_available_poi >= (q_poi - 0.5)

    # ── Degradation schedule ─────────────────────────────────────────────────
    deg_rows = []
    aug_added = 0
    for yr in range(i['project_years'] + 1):
        soh_yr  = soh[min(yr, len(soh)-1)]
        e_yr    = energy_poi_actual * soh_yr
        aug_mwh = 0.0
        if i['aug_year'] > 0 and yr == i['aug_year']:
            # add blocks to recover to BOL
            needed    = i['poi_mwh'] - e_yr
            aug_mwh   = max(needed, 0.0)
            aug_added = aug_mwh
        deg_rows.append({'Year': yr, 'SOH': soh_yr,
                         'Energy @ POI (MWh)': round(e_yr, 2),
                         'Augmentation (MWh)': round(aug_mwh, 2)})
    deg_df = pd.DataFrame(deg_rows)

    return {
        # Quantities
        'n_blocks':       n_blocks,
        'n_batt_units':   n_batt_units,
        'n_pcs':          n_pcs,
        'n_mvt':          n_mvt,
        # Power
        'p_poi_actual':   p_poi_actual,
        'p_inv_actual':   p_inv_actual,
        'p_mvbus_actual': p_mvbus_actual,
        's_inv_total':    s_inv_total,
        'min_mpt_mva':    min_mpt_mva,
        # Energy
        'energy_poi_actual':  energy_poi_actual,
        'energy_dc_actual':   energy_dc_actual,
        # Reactive
        'q_poi_needed':   q_poi,
        'q_available_poi': q_available_poi,
        'q_meets':        q_meets,
        's_poi':          s_poi,
        'pf_poi':         pf_poi,
        # Checks
        'power_meets':    p_poi_actual >= i['poi_mw'] * 0.999,
        'energy_meets':   energy_poi_actual * soh_eol >= i['poi_mwh'] * 0.999,
        'soh_eol':        soh_eol,
        # Degradation
        'deg_df':         deg_df,
        # Losses
        'lf_total':       lf_energy,
        'p_loss_pct':     (1 - lf_ac) * 100,
    }


@st.cache_data(show_spinner=False)
def run_pf_sweep(poi_mw, poi_mwh, n_blocks, inv_mva, pcs_pf,
                 mpt_s, mpt_z, mpt_xr, mpt_i0, mpt_p0kw,
                 isu_s_total, isu_z, isu_xr, isu_i0, isu_p0kw_total,
                 tap_c, q_cap, v_target, poi_mw_limit):
    """Run 121-point ISU S-circle PF sweep for the sized system."""
    sbase  = poi_mw_limit if poi_mw_limit > 0 else poi_mw
    S_isu  = isu_s_total / sbase
    P_max  = n_blocks * inv_mva * pcs_pf / sbase
    Y = build_ybus(sbase, mpt_s, mpt_z, mpt_xr, mpt_i0, mpt_p0kw,
                   isu_s_total, isu_z, isu_xr, isu_i0, isu_p0kw_total,
                   tap_c, q_cap)
    results = []
    V2p, V3p = v_target, v_target
    for theta in np.linspace(0, 2*np.pi, 121, endpoint=False):
        P3 = np.clip(S_isu * np.cos(theta), -P_max, P_max)
        Q3 = S_isu * np.sin(theta)
        V, th, conv = nr_pf(Y, P3, Q3, V1=v_target, V2i=V2p, V3i=V3p)
        if conv:
            V2p, V3p = V[1], V[2]
            P1 = sum(V[0]*V[j]*(Y[0,j].real*np.cos(th[0]-th[j])+Y[0,j].imag*np.sin(th[0]-th[j])) for j in range(3))
            Q1 = sum(V[0]*V[j]*(Y[0,j].real*np.sin(th[0]-th[j])-Y[0,j].imag*np.cos(th[0]-th[j])) for j in range(3))
            results.append({'P_inv_MW': P3*sbase, 'Q_inv_MVAR': Q3*sbase,
                            'V_POI': V[1], 'V_ISU': V[2],
                            'P_grid_MW': P1*sbase, 'Q_grid_MVAR': Q1*sbase})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  ─  INPUTS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔋 BESS Project Inputs")

    # ── Project ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Project</div>', unsafe_allow_html=True)
    project_name = st.text_input("Project Name", value="Demo BESS")
    iso = st.selectbox("ISO / Market", ["WECC", "CAISO", "ERCOT", "PJM", "MISO", "NYISO", "ISO-NE", "Other"])

    # ── POI Requirements ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">POI Requirements</div>', unsafe_allow_html=True)
    poi_mw       = st.number_input("BESS Nameplate Power @ POI (MW)",  value=100.0, min_value=1.0, step=5.0)
    poi_mwh      = st.number_input("BESS Nameplate Energy @ POI (MWh)", value=200.0, min_value=1.0, step=10.0)
    poi_limit_mw = st.number_input("POI Export Limit (MW)",            value=200.0, min_value=1.0, step=10.0)
    project_years= st.number_input("Project Term (years)",             value=20,    min_value=1,  max_value=40)
    aug_year     = st.number_input("Augmentation Year (0 = none)",     value=0,     min_value=0,  max_value=40,
                                   help="Year at which battery capacity is augmented to meet energy target")

    # ── Reactive Power ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Reactive Power</div>', unsafe_allow_html=True)
    pf_or_mvar = st.radio("Target type", ["Target PF", "Target MVAR"], horizontal=True)
    if pf_or_mvar == "Target PF":
        target_pf   = st.number_input("Target PF @ POI", value=0.95, min_value=0.5, max_value=1.0, step=0.01)
        target_mvar = poi_mw * np.tan(np.arccos(target_pf))
        st.caption(f"→ Q = **{target_mvar:.1f} MVAR**")
    else:
        target_mvar = st.number_input("Target MVAR @ POI", value=32.9, min_value=0.0, step=1.0)
        target_pf   = poi_mw / np.sqrt(poi_mw**2 + target_mvar**2) if target_mvar else 1.0
        st.caption(f"→ PF = **{target_pf:.3f}**")
    cap_bank = st.number_input("Capacitor Bank (MVAR)", value=0.0, min_value=0.0, step=5.0)

    # ── Grid / Interconnection ────────────────────────────────────────────────
    st.markdown('<div class="sec">Grid / Interconnection</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    poi_kv    = c1.number_input("POI Voltage (kV)", value=138.0, min_value=1.0)
    mv_kv     = c2.number_input("MV Bus (kV)",      value=34.5,  min_value=1.0)
    gentie_mi = st.number_input("Gen-tie Length (miles)", value=1.0, min_value=0.0)
    ca, cb, cc = st.columns(3)
    v_max  = ca.number_input("V max (pu)", value=1.05, min_value=1.0, max_value=1.15, step=0.01)
    v_min  = cb.number_input("V min (pu)", value=0.95, min_value=0.80, max_value=1.0, step=0.01)
    v_calc = cc.number_input("V calc (pu)", value=1.00, min_value=0.8, max_value=1.15, step=0.01)

    # ── MPT ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Main Power Transformer (MPT)</div>', unsafe_allow_html=True)
    cx, cy = st.columns(2)
    n_mpt  = cx.number_input("# MPTs",     value=1,     min_value=1)
    s_mpt  = cy.number_input("MVA / MPT",  value=120.0, min_value=10.0, step=5.0)
    cz, cw = st.columns(2)
    z_mpt  = cz.number_input("Z (pu)",  value=0.10, min_value=0.01, max_value=0.25, step=0.005, format="%.3f")
    xr_mpt = cw.number_input("X/R",     value=40.0, min_value=1.0)
    eta_mpt = st.number_input("MPT Efficiency", value=0.995, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")

    has_oltc = st.checkbox("OLTC tap changer?", value=True)
    if has_oltc:
        cd, ce = st.columns(2)
        ntaps     = cd.number_input("# Taps",    value=31,   min_value=3, step=2)
        tap_range = ce.number_input("Range (%)", value=10.0, min_value=1.0)
        tap_number = st.slider("Tap position", int(-(ntaps//2)), int(ntaps//2), 0)
        step = (tap_range/100.0) / ((ntaps-1)/2.0)
        fixed_tap = tap_number * step
        st.caption(f"tap_c = **{fixed_tap:+.4f}**  |  c = {1-fixed_tap:.3f}")
    else:
        ntaps=31; tap_range=10.0; tap_number=0
        fixed_tap = st.number_input("Fixed tap (pu)", value=0.0, step=0.005, format="%.3f")

    # ── PCS / Inverter Block ──────────────────────────────────────────────────
    st.markdown('<div class="sec">PCS / Inverter Block</div>', unsafe_allow_html=True)
    pcs_model = st.text_input("PCS Model", value="EPC POWER M10")
    cf, cg = st.columns(2)
    inv_mva   = cf.number_input("Inverter MVA (de-rated)", value=4.6, min_value=0.1, step=0.1)
    mvt_mva   = cg.number_input("MVT MVA (de-rated)",      value=4.6, min_value=0.1, step=0.1)
    pcs_max_pf = st.number_input("PCS Max Operating PF", value=0.90, min_value=0.5, max_value=1.0, step=0.01)

    # ── Battery ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Battery Unit</div>', unsafe_allow_html=True)
    batt_model   = st.text_input("Battery Model", value="Samsung SBB 1.7")
    ch, ci_ = st.columns(2)
    batt_mwh     = ch.number_input("MWh / unit (DC)", value=2.982, min_value=0.1, step=0.1, format="%.3f")
    units_per_blk = ci_.number_input("Units / block",  value=3,     min_value=1)
    aux_kw       = st.number_input("Aux load / block (kW)", value=75.6, min_value=0.0, step=1.0)

    # ── ISU (for power flow) ──────────────────────────────────────────────────
    st.markdown('<div class="sec">Inverter Step-Up (ISU / MVT)</div>', unsafe_allow_html=True)
    cj, ck = st.columns(2)
    z_isu  = cj.number_input("Z (pu)",  value=0.08, min_value=0.01, max_value=0.2, step=0.005, format="%.3f")
    xr_isu = ck.number_input("X/R",     value=8.83, min_value=1.0)

    # ── Loss Assumptions ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Loss Assumptions</div>', unsafe_allow_html=True)
    with st.expander("Edit loss factors"):
        eta_pcs        = st.number_input("PCS",              value=0.985, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_mvt        = st.number_input("MVT",              value=0.990, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_mv_cable   = st.number_input("MV Cable",         value=0.995, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_transmission = st.number_input("Transmission",   value=0.990, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_dc_cable   = st.number_input("DC Cable",         value=0.999, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_charge     = st.number_input("Charge/Discharge", value=0.955, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")
        eta_aux        = st.number_input("Auxiliary",        value=0.998, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")

    st.divider()
    run_btn = st.button("▶  Run Sizing", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"# 🔋 BESS Sizing Tool  <span style='font-size:.55em;color:#666;font-weight:400'>— {project_name}</span>", unsafe_allow_html=True)
st.caption(f"{iso}  |  {poi_mw:.0f} MW / {poi_mwh:.0f} MWh @ POI  |  POI limit {poi_limit_mw:.0f} MW  |  {poi_kv:.0f} kV")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:

    # ── First pass: guess n_blocks ────────────────────────────────────────────
    inp = {
        'poi_mw': poi_mw, 'poi_mwh': poi_mwh,
        'pf_or_mvar': pf_or_mvar, 'target_pf': target_pf, 'target_mvar': target_mvar,
        'eta_transmission': eta_transmission, 'eta_mpt': eta_mpt,
        'eta_mv_cable': eta_mv_cable, 'eta_mvt': eta_mvt, 'eta_pcs': eta_pcs,
        'eta_dc_cable': eta_dc_cable, 'eta_charge_disch': eta_charge,
        'eta_auxiliary': eta_aux,
        'inv_mva': inv_mva, 'pcs_max_pf': pcs_max_pf,
        'batt_mwh_per_unit': batt_mwh, 'units_per_block': units_per_blk,
        'aux_kw_per_block': aux_kw,
        'n_blocks_guess': 1,   # dummy, iterated below
        'project_years': int(project_years), 'aug_year': int(aug_year),
        'soh_curve': DEFAULT_SOH,
    }
    # Iterate to converge n_blocks (aux load depends on qty)
    for _ in range(5):
        res = bess_sizing(inp)
        inp['n_blocks_guess'] = res['n_blocks']

    # ── Derived ISU parameters for PF ────────────────────────────────────────
    n_isu       = res['n_pcs']          # one MVT per PCS block
    isu_s_total = n_isu * mvt_mva       # total ISU / MVT apparent power
    isu_p0kw    = 8.0 * n_isu           # ~8 kW NL per ISU (typical)

    tap_c = get_tap(has_oltc, ntaps, tap_range, fixed_tap, tap_number if has_oltc else 0)

    # ── Run power flow sweep ──────────────────────────────────────────────────
    with st.spinner("Running power flow sweep…"):
        df_pf = run_pf_sweep(
            poi_mw, poi_mwh, res['n_blocks'], inv_mva, pcs_max_pf,
            float(n_mpt * s_mpt), z_mpt, xr_mpt, 0.001, 10.0 * n_mpt,
            isu_s_total, z_isu, xr_isu, 0.005, isu_p0kw,
            tap_c, float(cap_bank), float(v_calc), float(poi_limit_mw)
        )

    pf_ok = len(df_pf) > 0
    v_nom = df_pf.loc[(df_pf['P_inv_MW'] - poi_mw).abs().idxmin(), 'V_POI'] if pf_ok else None

    # ══════════════════════════════════════════════════════════════════════════
    #  RESULTS LAYOUT
    # ══════════════════════════════════════════════════════════════════════════

    st.subheader("📦 Equipment Quantities")

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Power Blocks (PCS)", f"{res['n_blocks']}")
    q2.metric("Battery Enclosures", f"{res['n_batt_units']}")
    q3.metric("PCS Units", f"{res['n_pcs']}")
    q4.metric("MVT Units", f"{res['n_mvt']}")
    q5.metric("Min MPT Size", f"{res['min_mpt_mva']:.1f} MVA",
              delta="✓ OK" if (n_mpt * s_mpt) >= res['min_mpt_mva'] else "⚠ Undersize")

    st.divider()

    # ── Check badges ──────────────────────────────────────────────────────────
    st.subheader("✅ Sizing Checks")
    bc = st.columns(4)
    def badge(col, label, ok, val=""):
        css = "ok" if ok else "fail"
        sym = "✓" if ok else "✗"
        col.markdown(f'<div class="kpi-box"><span class="{css}">{sym} {label}</span><br>{val}</div>',
                     unsafe_allow_html=True)

    badge(bc[0], "Energy met",        res['energy_meets'],
          f"{res['energy_poi_actual']*res['soh_eol']:.1f} MWh ≥ {poi_mwh:.0f} MWh")
    badge(bc[1], "Active power met",  res['power_meets'],
          f"{res['p_poi_actual']:.1f} MW ≥ {poi_mw:.0f} MW")
    badge(bc[2], "Reactive power met", res['q_meets'],
          f"{res['q_available_poi']:.1f} MVAR available, {res['q_poi_needed']:.1f} needed")
    pf_v_ok = pf_ok and v_nom is not None and v_min <= v_nom <= v_max
    badge(bc[3], "POI Voltage",       pf_v_ok,
          f"V={v_nom:.4f} pu" if v_nom else "No PF result")

    st.divider()

    # ── Power cascade ─────────────────────────────────────────────────────────
    st.subheader("⚡ Power Flow Cascade")
    cas = st.columns(5)
    cas[0].metric("@ Inverter",  f"{res['s_inv_total']:.1f} MVA")
    cas[1].metric("@ MVT out",   f"{res['p_mvbus_actual']:.1f} MW")
    cas[2].metric("@ MV Bus",    f"{res['p_mvbus_actual']:.1f} MW")
    cas[3].metric("@ POI",       f"{res['p_poi_actual']:.1f} MW")
    cas[4].metric("Total losses", f"{res['p_loss_pct']:.1f}%")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 PV Curve (Power Flow)", "📉 Degradation", "📋 Equipment List", "🔢 Full Data"])

    with tab1:
        if not pf_ok:
            st.warning("Power flow did not converge. Check MPT size and tap position.")
        else:
            df_lead = df_pf[df_pf['Q_inv_MVAR'] >= 0].sort_values('P_inv_MW')
            df_lag  = df_pf[df_pf['Q_inv_MVAR'] <  0].sort_values('P_inv_MW')
            fig = go.Figure()
            fig.add_hrect(y0=v_min, y1=v_max, fillcolor="lightgreen", opacity=0.12, line_width=0)
            fig.add_hline(y=v_min, line_dash="dash", line_color="red", line_width=1.5,
                          annotation_text=f"V_min={v_min}", annotation_position="bottom left")
            fig.add_hline(y=v_max, line_dash="dash", line_color="red", line_width=1.5,
                          annotation_text=f"V_max={v_max}", annotation_position="top left")
            fig.add_hline(y=v_calc, line_dash="dot", line_color="steelblue", line_width=1.5)
            fig.add_vline(x=poi_mw, line_dash="dot", line_color="gray",
                          annotation_text=f"{poi_mw:.0f} MW")
            if len(df_lead):
                fig.add_trace(go.Scatter(x=df_lead['P_inv_MW'], y=df_lead['V_POI'],
                                         mode='lines', name='Leading / gen Q',
                                         line=dict(color='royalblue', width=2.5)))
            if len(df_lag):
                fig.add_trace(go.Scatter(x=df_lag['P_inv_MW'], y=df_lag['V_POI'],
                                         mode='lines', name='Lagging / abs Q',
                                         line=dict(color='darkorange', width=2.5)))
            y_lo = max(0.80, df_pf['V_POI'].min()-0.03)
            y_hi = min(1.25, df_pf['V_POI'].max()+0.03)
            fig.update_layout(title="PV Curve — POI Voltage vs Active Power",
                              xaxis_title="P_inv (MW)", yaxis_title="V_POI (pu)",
                              yaxis=dict(range=[y_lo, y_hi]), height=420,
                              legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)

            # Quick PF summary
            v_band_pct = ((df_pf['V_POI']>=v_min)&(df_pf['V_POI']<=v_max)).mean()*100
            pf1, pf2, pf3 = st.columns(3)
            pf1.metric("V_POI @ rated P", f"{v_nom:.4f} pu")
            pf2.metric("% pts in V-band",  f"{v_band_pct:.1f}%")
            pf3.metric("Converged pts",    f"{len(df_pf)}/121")

    with tab2:
        fig2 = go.Figure()
        fig2.add_hline(y=poi_mwh, line_dash="dash", line_color="red",
                       annotation_text=f"Target {poi_mwh:.0f} MWh")
        fig2.add_trace(go.Scatter(
            x=res['deg_df']['Year'],
            y=res['deg_df']['Energy @ POI (MWh)'],
            mode='lines+markers', name='Energy @ POI',
            line=dict(color='royalblue', width=2.5),
            marker=dict(size=6)))
        aug_rows = res['deg_df'][res['deg_df']['Augmentation (MWh)'] > 0]
        if len(aug_rows):
            fig2.add_trace(go.Bar(x=aug_rows['Year'], y=aug_rows['Augmentation (MWh)'],
                                  name='Augmentation', marker_color='green', opacity=0.6))
        fig2.update_layout(title="Energy Capacity over Project Life",
                           xaxis_title="Year", yaxis_title="MWh @ POI",
                           height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(res['deg_df'].set_index('Year'), use_container_width=True)

    with tab3:
        equip = pd.DataFrame([
            {"Equipment":     "Power Block (PCS + MVT + DC Enclosures)",
             "Qty":           res['n_blocks'],
             "Unit":          "blocks",
             "Notes":         f"{units_per_blk} battery units each, {inv_mva} MVA inverter"},
            {"Equipment":     f"Battery Enclosure ({batt_model})",
             "Qty":           res['n_batt_units'],
             "Unit":          "units",
             "Notes":         f"{batt_mwh} MWh/unit → {res['energy_dc_actual']:.1f} MWh DC total"},
            {"Equipment":     f"PCS ({pcs_model})",
             "Qty":           res['n_pcs'],
             "Unit":          "units",
             "Notes":         f"{inv_mva} MVA each, PF={pcs_max_pf:.2f}"},
            {"Equipment":     "MVT (Medium Voltage Transformer)",
             "Qty":           res['n_mvt'],
             "Unit":          "units",
             "Notes":         f"{mvt_mva} MVA each"},
            {"Equipment":     "MPT (Main Power Transformer)",
             "Qty":           n_mpt,
             "Unit":          "units",
             "Notes":         f"{s_mpt:.0f} MVA each, min required {res['min_mpt_mva']:.1f} MVA"},
        ])
        st.dataframe(equip.set_index("Equipment"), use_container_width=True)

        # Download
        st.download_button("⬇ Download Equipment List",
                           equip.to_csv(index=False), "bess_equipment.csv", "text/csv")

        # Loss chain table
        st.markdown("**Loss Chain Summary**")
        loss_tbl = pd.DataFrame([
            {"Stage": "Transmission Line", "Efficiency": eta_transmission},
            {"Stage": "MPT",               "Efficiency": eta_mpt},
            {"Stage": "MV Cable",          "Efficiency": eta_mv_cable},
            {"Stage": "MVT",               "Efficiency": eta_mvt},
            {"Stage": "PCS",               "Efficiency": eta_pcs},
            {"Stage": "DC Cable",          "Efficiency": eta_dc_cable},
            {"Stage": "Charge/Discharge",  "Efficiency": eta_charge},
            {"Stage": "Auxiliary",         "Efficiency": eta_aux},
            {"Stage": "→ TOTAL",           "Efficiency": res['lf_total']},
        ])
        st.dataframe(loss_tbl.set_index("Stage"), use_container_width=True)

    with tab4:
        if pf_ok:
            st.dataframe(df_pf.round(4), use_container_width=True, height=350)
            st.download_button("⬇ Download PF Data",
                               df_pf.to_csv(index=False), "pf_results.csv", "text/csv")

else:
    st.info("👈 Fill in project parameters in the sidebar, then press **▶ Run Sizing**.")

    with st.expander("ℹ️ How this tool works"):
        st.markdown("""
**Sizing logic (backwards from POI → DC battery):**

```
POI target (MW, MWh)
  ÷ η_transmission   →  gen-tie MW
  ÷ η_MPT            →  MV bus MW
  − P_aux            →  net MV bus after auxiliary
  ÷ η_MV_cable       →  MVT output
  ÷ η_MVT            →  inverter terminals
  ÷ η_PCS            →  DC output
  ÷ η_charge/disch   →  battery nameplate needed
  ÷ SOH_EOL          →  BOL DC capacity needed (overbuild)
```

**Equipment quantity:**
- `Power Blocks = max(blocks_for_power, blocks_for_energy)`
- `Battery Enclosures = blocks × units_per_block`

**Power flow validation:**
- 3-bus Newton-Raphson (same engine as solar sizing tool)
- Confirms V_POI is within limits at the rated operating point
- ISU S-circle sweep shows full PV curve
        """)