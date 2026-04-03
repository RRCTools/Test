"""
BESS Sizing Tool v4 — Auto-Sizing Architecture

User inputs:
  - Project requirements (MW, MWh, PF, voltage)
  - Equipment specs (Inverter MVA, MVT MVA, battery MWh/unit)
  - Overbuild % (controls initial installed energy)
  - Augmentation year

Tool outputs:
  - Required PCS qty and BESS enclosure qty
  - Operating PF at PCS output
  - PV curve from power flow
  - Energy degradation with overbuild and augmentation
  - Multiple config options (different BESS/PCS ratios)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import ceil

st.set_page_config(page_title="BESS Sizing Tool", layout="wide", page_icon="🔋")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.block-container{padding-top:.5rem;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;}
div[data-testid="stSidebarContent"]{background:#0d1117;}
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p,
div[data-testid="stSidebarContent"] span{color:#c9d1d9 !important;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:.70rem;font-weight:600;
     letter-spacing:.12em;text-transform:uppercase;color:#58a6ff;
     border-bottom:1px solid #21262d;padding-bottom:2px;
     margin-top:12px;margin-bottom:4px;}
.card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin:4px 0;}
.cfg-head{font-family:'IBM Plex Mono',monospace;font-size:.8rem;font-weight:600;color:#1e40af;margin-bottom:4px;}
.ok{color:#16a34a;font-weight:700;} .fail{color:#dc2626;font-weight:700;}
.warn{color:#d97706;font-weight:700;}
</style>
""", unsafe_allow_html=True)

DEFAULT_SOH = [1.0,0.9342,0.9115,0.8933,0.8775,0.8633,0.8502,0.8381,
               0.8266,0.8158,0.8054,0.7955,0.7859,0.7767,0.7677,0.759,
               0.7506,0.7424,0.7343,0.7265,0.7188]

# ── 3-BUS POWER FLOW ─────────────────────────────────────────────────────────
def build_ybus(sbase,mpt_s,mpt_z,mpt_xr,mpt_p0kw,isu_s,isu_z,isu_xr,isu_p0kw,tap_c,q_cap=0.):
    rm=mpt_z*np.cos(np.arctan(mpt_xr)); xm=mpt_z*np.sin(np.arctan(mpt_xr))
    Ys=(mpt_s/sbase)/complex(rm,xm)
    gm=(mpt_p0kw/1e3)/mpt_s; bm=np.sqrt(max(.001**2-gm**2,0))
    Ym=complex(gm,bm)*(mpt_s/sbase)
    c=1.-tap_c
    if abs(c)<1e-9: c=1.
    Y10=(1-c)/c*Ys+Ym/c**2; Y12=c*Ys; Y20=(c-1)*Ys
    ri=isu_z*np.cos(np.arctan(isu_xr)); xi=isu_z*np.sin(np.arctan(isu_xr))
    Yi=(isu_s/sbase)/complex(ri,xi)
    gi=(isu_p0kw/1e3)/isu_s; bi=np.sqrt(max(.005**2-gi**2,0))
    Ymi=complex(gi,-bi)*(isu_s/sbase)
    yc=complex(0,q_cap/sbase)
    Y=np.zeros((3,3),dtype=complex)
    Y[0,0]=Y10+Y12; Y[0,1]=-Y12; Y[1,0]=-Y12
    Y[1,1]=Y12+Y20+Yi+Ymi+yc; Y[1,2]=-Yi; Y[2,1]=-Yi; Y[2,2]=Yi
    return Y

def nr_pf(Y,P3,Q3,V1=1.,V2i=None,V3i=None,tol=1e-9,mi=100):
    if V2i is None: V2i=V1
    if V3i is None: V3i=V1
    V=np.array([V1,V2i,V3i]); th=np.zeros(3)
    for _ in range(mi):
        Pc=np.zeros(3); Qc=np.zeros(3)
        for i in range(3):
            for j in range(3):
                G=Y[i,j].real; B=Y[i,j].imag; d=th[i]-th[j]
                Pc[i]+=V[i]*V[j]*(G*np.cos(d)+B*np.sin(d))
                Qc[i]+=V[i]*V[j]*(G*np.sin(d)-B*np.cos(d))
        mis=np.array([0-Pc[1],P3-Pc[2],0-Qc[1],Q3-Qc[2]])
        if np.max(np.abs(mis))<tol: return V,th,True
        J=np.zeros((4,4))
        for ri,bi in enumerate([1,2]):
            for ci,bj in enumerate([1,2]):
                if bi==bj:
                    J[ri,ci]  =V[bi]*sum(V[k]*(-Y[bi,k].real*np.sin(th[bi]-th[k])+Y[bi,k].imag*np.cos(th[bi]-th[k])) for k in range(3) if k!=bi)
                    J[ri,2+ci]=sum(V[k]*(Y[bi,k].real*np.cos(th[bi]-th[k])+Y[bi,k].imag*np.sin(th[bi]-th[k])) for k in range(3))+Y[bi,bi].real*V[bi]
                    J[2+ri,ci]=V[bi]*sum(V[k]*(Y[bi,k].real*np.cos(th[bi]-th[k])+Y[bi,k].imag*np.sin(th[bi]-th[k])) for k in range(3) if k!=bi)
                    J[2+ri,2+ci]=sum(V[k]*(Y[bi,k].real*np.sin(th[bi]-th[k])-Y[bi,k].imag*np.cos(th[bi]-th[k])) for k in range(3))-Y[bi,bi].imag*V[bi]
                else:
                    G=Y[bi,bj].real; B=Y[bi,bj].imag; d=th[bi]-th[bj]
                    J[ri,ci]  = V[bi]*V[bj]*(G*np.sin(d)-B*np.cos(d))
                    J[ri,2+ci]= V[bi]*(G*np.cos(d)+B*np.sin(d))
                    J[2+ri,ci]=-V[bi]*V[bj]*(G*np.cos(d)+B*np.sin(d))
                    J[2+ri,2+ci]=V[bi]*(G*np.sin(d)-B*np.cos(d))
        try: dx=np.linalg.solve(J,mis)
        except: return V,th,False
        th[1]+=dx[0]; th[2]+=dx[1]; V[1]+=dx[2]; V[2]+=dx[3]
    return V,th,False

@st.cache_data(show_spinner=False)
def run_pf_sweep(sbase,mpt_s,mpt_z,mpt_xr,mpt_p0kw,isu_s,isu_z,isu_xr,isu_p0kw,P_max_pu,tap_c,q_cap,vt):
    Y=build_ybus(sbase,mpt_s,mpt_z,mpt_xr,mpt_p0kw,isu_s,isu_z,isu_xr,isu_p0kw,tap_c,q_cap)
    S=isu_s/sbase; rows=[]; V2p,V3p=vt,vt
    for ang in np.linspace(0,2*np.pi,121,endpoint=False):
        P3=np.clip(S*np.cos(ang),-P_max_pu,P_max_pu); Q3=S*np.sin(ang)
        V,th,conv=nr_pf(Y,P3,Q3,V1=vt,V2i=V2p,V3i=V3p)
        if conv:
            V2p,V3p=V[1],V[2]
            P1=sum(V[0]*V[j]*(Y[0,j].real*np.cos(th[0]-th[j])+Y[0,j].imag*np.sin(th[0]-th[j])) for j in range(3))
            Q1=sum(V[0]*V[j]*(Y[0,j].real*np.sin(th[0]-th[j])-Y[0,j].imag*np.cos(th[0]-th[j])) for j in range(3))
            rows.append({'P_inv_MW':P3*sbase,'Q_inv_MVAR':Q3*sbase,'V_POI':V[1],'V_ISU':V[2],
                         'P_grid_MW':P1*sbase,'Q_grid_MVAR':Q1*sbase})
    return pd.DataFrame(rows)

# ── SIZING ENGINE ─────────────────────────────────────────────────────────────
def compute_eta(eta_pcs,eta_mvt,eta_mv,eta_mpt,eta_tx,eta_dc,eta_chg,eta_aux):
    return eta_pcs*eta_mvt*eta_mv*eta_mpt*eta_tx*eta_dc*eta_chg*eta_aux

def size_config(poi_mw, poi_mwh, target_pf,
                inv_mva, batt_per_pcs,
                batt_mwh_dc, aux_kw_per_pcs,
                eta_pcs,eta_mvt,eta_mv,eta_mpt,eta_tx,eta_dc,eta_chg,eta_aux,
                aug_year, overbuild_pct, project_years, soh_curve, cap_mvar=0.):
    """
    Auto-size PCS qty given a fixed batt_per_pcs ratio.
    PCS operating PF is derived from the P/Q requirement (not a fixed input).
    overbuild_pct: how much extra energy to install at BOL (e.g. 15% means
                   E_installed = poi_mwh * (1 + overbuild_pct/100))
    """
    eta_all = compute_eta(eta_pcs,eta_mvt,eta_mv,eta_mpt,eta_tx,eta_dc,eta_chg,eta_aux)
    soh = soh_curve + [soh_curve[-1]]*(max(project_years+2-len(soh_curve),0))

    # ── Q requirement at POI ──────────────────────────────────────────────────
    q_poi_target = poi_mw * np.tan(np.arccos(target_pf))

    # ── POWER: find minimum n_pcs where P AND Q are both met ─────────────────
    # At each n, operating PF = minimum PF to meet P (maximises Q headroom)
    eta_q = eta_mvt * eta_mv * eta_mpt * eta_tx  # Q loss chain (no chg/dc)
    n_pcs = 1
    for _ in range(30):
        aux_mw   = n_pcs * aux_kw_per_pcs / 1000.
        p_mpt_lv = poi_mw / (eta_mpt * eta_tx)
        p_mv     = p_mpt_lv + aux_mw
        p_mvt    = p_mv / eta_mv
        p_inv_nd = p_mvt / eta_mvt
        pf_op    = p_inv_nd / (n_pcs * inv_mva)   # derived operating PF
        if pf_op > 1.0:
            n_pcs += 1
            continue
        # Check Q headroom at this n and pf
        q_avail_poi = n_pcs * inv_mva * np.sqrt(max(1-pf_op**2,0)) * eta_q + cap_mvar * eta_mpt * eta_tx
        p_meets = True   # by construction
        q_meets = q_avail_poi >= q_poi_target - 0.01
        if p_meets and q_meets:
            break
        n_pcs += 1
    n_pcs_power = n_pcs

    # ── ENERGY: size for overbuild target ────────────────────────────────────
    # E_target_bol = poi_mwh * (1 + overbuild_pct/100)
    # This ensures at aug_year, energy is still >= poi_mwh
    e_target_bol = poi_mwh * (1.0 + overbuild_pct / 100.0)
    blk_mwh_poi  = batt_per_pcs * batt_mwh_dc * eta_all
    n_pcs_energy = ceil(e_target_bol / blk_mwh_poi)

    n_pcs_base = max(n_pcs_power, n_pcs_energy)
    n_batt_base = n_pcs_base * batt_per_pcs

    # Derived operating PF at final n_pcs_base
    aux_mw_final = n_pcs_base * aux_kw_per_pcs / 1000.
    p_mpt_lv_f   = poi_mw / (eta_mpt * eta_tx)
    p_mv_f       = p_mpt_lv_f + aux_mw_final
    p_inv_nd_f   = (p_mv_f / eta_mv) / eta_mvt
    pf_operating = min(p_inv_nd_f / (n_pcs_base * inv_mva), 1.0)

    # ── AUGMENTATION ─────────────────────────────────────────────────────────
    e_bol_base   = n_pcs_base * blk_mwh_poi
    if aug_year > 0:
        soh_at_aug   = soh[min(aug_year, len(soh)-1)]
        e_at_aug     = e_bol_base * soh_at_aug
        aug_gap      = max(poi_mwh - e_at_aug, 0.)
        n_pcs_aug    = ceil(aug_gap / blk_mwh_poi) if aug_gap > 0 else 0
    else:
        soh_at_aug = soh[min(project_years, len(soh)-1)]
        n_pcs_aug  = 0
    n_batt_aug = n_pcs_aug * batt_per_pcs

    # ── Actuals ───────────────────────────────────────────────────────────────
    n_pcs_total  = n_pcs_base  # aug added later
    e_dc_bol     = n_pcs_total * batt_per_pcs * batt_mwh_dc
    e_poi_bol    = e_dc_bol * eta_all
    overbuild_actual = (e_poi_bol - poi_mwh) / poi_mwh * 100

    # Power cascade
    aux_mw       = n_pcs_total * aux_kw_per_pcs / 1000.
    p_inv_act    = n_pcs_total * inv_mva * pf_operating
    q_inv_act    = n_pcs_total * inv_mva * np.sqrt(max(1-pf_operating**2,0))
    s_inv_act    = n_pcs_total * inv_mva
    p_mvt_out    = p_inv_act * eta_mvt; q_mvt_out = q_inv_act * eta_mvt
    p_mv_bus     = p_mvt_out * eta_mv;  q_mv_bus  = q_mvt_out * eta_mv
    p_mpt_in     = p_mv_bus - aux_mw;   q_mpt_in  = q_mv_bus + cap_mvar
    p_poi        = p_mpt_in * eta_mpt * eta_tx
    q_poi        = q_mpt_in * eta_mpt * eta_tx
    s_poi        = np.sqrt(p_poi**2+q_poi**2)
    pf_poi       = p_poi/s_poi if s_poi>0 else 1.
    min_mpt_mva  = np.sqrt(p_mpt_in**2+q_mpt_in**2)

    # Q checks
    q_avail_mvbus = n_pcs_total*inv_mva*np.sqrt(max(1-pf_operating**2,0))*eta_mvt*eta_mv + cap_mvar
    q_needed_mvbus= q_poi_target*(q_mpt_in/q_poi) if q_poi>0.01 else q_poi_target/(eta_mpt*eta_tx)

    # ── Degradation table ─────────────────────────────────────────────────────
    e_base_bol = e_poi_bol
    e_aug_bol  = n_pcs_aug * blk_mwh_poi
    deg = []
    for yr in range(project_years+1):
        s_yr   = soh[min(yr, len(soh)-1)]
        e_base = e_base_bol * s_yr
        if aug_year > 0 and yr >= aug_year:
            s_rel  = soh[min(yr-aug_year, len(soh)-1)]
            e_aug  = e_aug_bol * s_rel
        else:
            e_aug  = 0.
        e_tot  = e_base + e_aug
        aug_ev = round(e_aug_bol,1) if (aug_year>0 and yr==aug_year) else 0.
        deg.append({'Year':yr,'SOH (%)':round(s_yr*100,2),
                    'Base E@POI (MWh)':round(e_base,1),
                    'Aug E added (MWh)':aug_ev,
                    'Total E@POI (MWh)':round(e_tot,1)})
    deg_df = pd.DataFrame(deg,
        columns=['Year','SOH (%)','Base E@POI (MWh)','Aug E added (MWh)','Total E@POI (MWh)'])

    # First year below target
    yr_below = next((r['Year'] for r in deg if r['Total E@POI (MWh)'] < poi_mwh), None)

    return {
        'n_pcs_base':n_pcs_base,'n_batt_base':n_batt_base,
        'n_pcs_aug':n_pcs_aug,  'n_batt_aug':n_batt_aug,
        'batt_per_pcs':batt_per_pcs,
        'pf_operating':pf_operating, 'pf_poi':pf_poi,
        'p_poi':p_poi,'q_poi':q_poi,'s_poi':s_poi,
        'p_inv_nd':p_inv_nd_f,
        'n_pcs_power':n_pcs_power,'n_pcs_energy':n_pcs_energy,
        'driver':'POWER' if n_pcs_power>=n_pcs_energy else 'ENERGY',
        'e_poi_bol':e_poi_bol,'e_dc_bol':e_dc_bol,
        'overbuild_actual':overbuild_actual,
        'min_mpt_mva':min_mpt_mva,
        'p_mpt_in':p_mpt_in,'q_mpt_in':q_mpt_in,
        'aux_mw':aux_mw,'eta_all':eta_all,
        'blk_mwh_poi':blk_mwh_poi,
        'soh_at_aug':soh_at_aug,
        # Checks
        'check1_energy': e_poi_bol >= poi_mwh*0.999,
        'check2_power':  p_poi >= poi_mw*0.999,
        'check3_reactive': q_avail_mvbus >= q_needed_mvbus*0.999,
        'q_poi_target':q_poi_target,'q_avail_mvbus':q_avail_mvbus,'q_needed_mvbus':q_needed_mvbus,
        'yr_below':yr_below,
        'deg_df':deg_df,
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 BESS Sizing")

    st.markdown('<div class="sec">Project</div>', unsafe_allow_html=True)
    proj_name = st.text_input("", value="Demo BESS", label_visibility="collapsed")
    iso = st.selectbox("ISO",["WECC","CAISO","ERCOT","PJM","MISO","NYISO","ISO-NE","Other"])

    st.markdown('<div class="sec">POI Requirements</div>', unsafe_allow_html=True)
    poi_mw   = st.number_input("Power @ POI (MW)",   value=200., min_value=1., step=10.)
    poi_mwh  = st.number_input("Energy @ POI (MWh)", value=800., min_value=1., step=50.)
    poi_lim  = st.number_input("Export Limit (MW)",  value=200., min_value=1., step=10.)
    proj_yrs = st.number_input("Project Term (yr)",  value=20, min_value=1, max_value=40)
    pf_mode  = st.radio("", ["Target PF","Target MVAR"], horizontal=True, label_visibility="collapsed")
    if pf_mode=="Target PF":
        target_pf = st.number_input("Target PF @ POI", value=0.95, min_value=0.5, max_value=1., step=.01)
        st.caption(f"Q = {poi_mw*np.tan(np.arccos(target_pf)):.1f} MVAR")
    else:
        tmvar = st.number_input("Target MVAR @ POI", value=65.7, min_value=0., step=1.)
        target_pf = poi_mw/np.sqrt(poi_mw**2+tmvar**2) if tmvar else 1.
        st.caption(f"PF = {target_pf:.3f}")
    cap_bank = st.number_input("Capacitor Bank (MVAR)", value=0., min_value=0., step=5.)

    st.markdown('<div class="sec">Energy Strategy</div>', unsafe_allow_html=True)
    overbuild_pct = st.number_input(
        "Overbuild % at BOL", value=15.0, min_value=0., max_value=100., step=1.,
        help="Extra energy installed above nameplate so degradation reaches target at aug_year. "
             "E_installed = poi_mwh × (1 + overbuild%/100)")
    aug_year = st.number_input("Augmentation Year (0=none)", value=10, min_value=0, max_value=40,
        help="Year when aug PCS+batteries are added. Sized to restore to poi_mwh target.")
    st.caption(f"E_BOL target = {poi_mwh*(1+overbuild_pct/100):.0f} MWh  "
               f"({'reaches target at aug yr' if aug_year>0 else 'must survive full term'})")

    st.markdown('<div class="sec">Grid / MPT</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    n_mpt=c1.number_input("#MPTs",   value=1,    min_value=1)
    s_mpt=c2.number_input("MVA/MPT", value=240., min_value=10., step=10.)
    c3,c4=st.columns(2)
    z_mpt =c3.number_input("Z(pu)",  value=0.10, min_value=.01, step=.005, format="%.3f")
    xr_mpt=c4.number_input("X/R",    value=40.,  min_value=1.)
    eta_mpt=st.number_input("MPT η",  value=0.995, min_value=.9, max_value=1., step=.001, format="%.3f")
    has_oltc=st.checkbox("OLTC?", value=True)
    if has_oltc:
        c5,c6=st.columns(2)
        ntaps  =c5.number_input("#Taps",  value=31, min_value=3, step=2)
        tap_rng=c6.number_input("Range%", value=10., min_value=1.)
        tap_num=st.slider("Tap", int(-(ntaps//2)), int(ntaps//2), 0)
        tap_c  =tap_num*(tap_rng/100.)/((ntaps-1)/2.)
        st.caption(f"tap_c={tap_c:+.4f}  c={1-tap_c:.3f}")
    else:
        tap_c=st.number_input("Fixed tap(pu)", value=0., step=.005, format="%.3f")

    st.markdown('<div class="sec">PCS / MVT Specs</div>', unsafe_allow_html=True)
    pcs_model=st.text_input("PCS Model", value="EPC POWER M10")
    c7,c8=st.columns(2)
    inv_mva=c7.number_input("Inv MVA",  value=5.3, min_value=.1, step=.1)
    mvt_mva=c8.number_input("MVT MVA",  value=5.3, min_value=.1, step=.1)
    eta_pcs=st.number_input("PCS η", value=0.985, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">Battery Unit</div>', unsafe_allow_html=True)
    batt_model=st.text_input("Battery Model", value="BESS Unit")
    batt_mwh  =st.number_input("MWh/unit (DC)", value=6.138, min_value=.1, step=.1, format="%.3f")
    aux_kw    =st.number_input("Aux load/PCS (kW)", value=75.6, min_value=0., step=1.)

    st.markdown('<div class="sec">MVT / ISU</div>', unsafe_allow_html=True)
    c9,c10=st.columns(2)
    z_isu =c9.number_input("Z(pu)",   value=0.08, min_value=.01, step=.005, format="%.3f")
    xr_isu=c10.number_input("X/R ISU",value=8.83, min_value=1.)
    eta_mvt=st.number_input("MVT η",  value=0.990, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">Loss Factors</div>', unsafe_allow_html=True)
    with st.expander("AC losses"):
        eta_mv  =st.number_input("MV Cable",    value=.995, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_tx  =st.number_input("Transmission",value=.990, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_aux =st.number_input("Auxiliary",   value=.998, min_value=.9, max_value=1., step=.001, format="%.3f")
    with st.expander("DC losses"):
        eta_dc  =st.number_input("DC Cable",        value=.999, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_chg =st.number_input("Charge/Discharge", value=.955, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">BESS/PCS Ratios to Evaluate</div>', unsafe_allow_html=True)
    st.caption("Tool will size PCS qty for each ratio and show all options")
    ratios_str = st.text_input("BESS units per PCS (comma-separated)", value="2, 3, 4, 5, 6")
    try:
        bess_ratios = [int(x.strip()) for x in ratios_str.split(',') if x.strip().isdigit() and int(x.strip())>0]
    except:
        bess_ratios = [2,3,4,5,6]

    st.markdown('<div class="sec">SOH Degradation</div>', unsafe_allow_html=True)
    with st.expander("Edit SOH curve"):
        soh_vals=[]
        for yr in range(min(int(proj_yrs)+1,21)):
            dv=DEFAULT_SOH[yr] if yr<len(DEFAULT_SOH) else DEFAULT_SOH[-1]
            r1,r2=st.columns([1,2])
            r1.markdown(f"<div style='padding-top:8px;font-size:.8rem'>Yr {yr}</div>",unsafe_allow_html=True)
            v=r2.number_input(f"s{yr}",value=float(round(dv*100,2)),min_value=0.,max_value=100.,
                              step=.1,label_visibility="collapsed",key=f"soh{yr}")
            soh_vals.append(v/100.)
    soh_curve=soh_vals

    st.markdown('<div class="sec">POI Voltage Limits</div>', unsafe_allow_html=True)
    ca,cb,cc=st.columns(3)
    v_min=ca.number_input("Min",  value=.95, min_value=.8, max_value=1.,  step=.01)
    v_max=cb.number_input("Max",  value=1.05,min_value=1., max_value=1.15,step=.01)
    v_calc=cc.number_input("Calc",value=1.00,min_value=.8, max_value=1.15,step=.01)

    st.divider()
    run_btn=st.button("▶  Size All Configurations", type="primary", use_container_width=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🔋 BESS Sizing Tool")
st.caption(f"**{proj_name}** · {iso} · {poi_mw:.0f} MW / {poi_mwh:.0f} MWh @ POI · "
           f"PF {target_pf:.2f} · {overbuild_pct:.0f}% overbuild · aug yr {int(aug_year)}")
st.divider()

# ── RUN ───────────────────────────────────────────────────────────────────────
if run_btn:
    eta_all_ref = compute_eta(eta_pcs,eta_mvt,eta_mv,eta_mpt,eta_tx,eta_dc,eta_chg,eta_aux)

    with st.spinner(f"Sizing {len(bess_ratios)} configurations…"):
        results = {}
        for ratio in bess_ratios:
            results[ratio] = size_config(
                poi_mw, poi_mwh, target_pf,
                inv_mva, ratio, batt_mwh, aux_kw,
                eta_pcs,eta_mvt,eta_mv,eta_mpt,eta_tx,eta_dc,eta_chg,eta_aux,
                int(aug_year), float(overbuild_pct), int(proj_yrs), soh_curve, float(cap_bank))

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    st.subheader("📊 Configuration Comparison")

    cmp_rows = []
    for ratio, r in results.items():
        c1ok = "✅" if r['check1_energy'] else "❌"
        c2ok = "✅" if r['check2_power']  else "❌"
        c3ok = "✅" if r['check3_reactive'] else "❌"
        all_ok = r['check1_energy'] and r['check2_power'] and r['check3_reactive']
        mpt_ok = n_mpt*s_mpt >= r['min_mpt_mva']
        cmp_rows.append({
            'BESS/PCS': ratio,
            'PCS (base)': r['n_pcs_base'],
            'BESS (base)': r['n_batt_base'],
            'PCS (aug)': r['n_pcs_aug'],
            'BESS (aug)': r['n_batt_aug'],
            'PCS op. PF': round(r['pf_operating'],3),
            'PF @ POI': round(r['pf_poi'],3),
            'P @ POI (MW)': round(r['p_poi'],1),
            'Q @ POI (MVAR)': round(r['q_poi'],1),
            'E BOL (MWh)': round(r['e_poi_bol'],1),
            'Overbuild %': round(r['overbuild_actual'],1),
            'Yr < target': r['yr_below'] if r['yr_below'] else '—',
            'Min MPT (MVA)': round(r['min_mpt_mva'],1),
            'MPT OK': '✅' if mpt_ok else '❌',
            'P✅ Q✅ E✅': '✅ ALL PASS' if all_ok else f"{c2ok}P {c1ok}E {c3ok}Q",
            'Driver': r['driver'],
        })
    cmp_df = pd.DataFrame(cmp_rows).set_index('BESS/PCS')
    st.dataframe(cmp_df, use_container_width=True)

    # Highlight best (all checks pass, fewest total BESS units)
    passing = [(r, results[r]['n_batt_base']+results[r]['n_batt_aug'])
               for r in bess_ratios
               if results[r]['check1_energy'] and results[r]['check2_power'] and results[r]['check3_reactive']]
    if passing:
        best_ratio = min(passing, key=lambda x: x[1])[0]
        st.success(f"✅ Most efficient passing config: **{best_ratio} BESS/PCS** — "
                   f"{results[best_ratio]['n_pcs_base']} PCS × {best_ratio} batteries = "
                   f"{results[best_ratio]['n_batt_base']} units | "
                   f"PCS PF = {results[best_ratio]['pf_operating']:.3f}")
    else:
        st.warning("⚠️ No configuration passes all 3 checks. Adjust inputs or increase PCS/BESS counts.")

    st.divider()

    # ── DETAIL VIEW ───────────────────────────────────────────────────────────
    st.subheader("🔍 Detail View")
    sel_ratio = st.selectbox(
        "Select configuration to inspect",
        options=bess_ratios,
        format_func=lambda r: f"{r} BESS/PCS → {results[r]['n_pcs_base']} PCS × {r} = {results[r]['n_batt_base']} batteries",
        index=bess_ratios.index(best_ratio) if passing else 0)
    res = results[sel_ratio]

    # Checks
    def check_card(col,num,title,ok,actual,required,note=""):
        bg="#0d3321" if ok else "#3d1a1a"; brd="#238636" if ok else "#b62324"
        tc="#3fb950" if ok else "#f85149"; sym="✓" if ok else "✗"
        nh=f'<div style="font-size:.70rem;color:#8b949e;margin-top:3px">{note}</div>' if note else ""
        col.markdown(
            f'<div style="background:{bg};border:1px solid {brd};border-radius:6px;padding:10px 12px">'
            f'<div style="font-family:monospace;font-size:.65rem;color:#8b949e">CHECK {num}</div>'
            f'<div style="font-size:.9rem;font-weight:700;color:{tc}">{sym} {title}</div>'
            f'<div style="font-size:.80rem;color:#c9d1d9">Actual: <b>{actual}</b></div>'
            f'<div style="font-size:.80rem;color:#c9d1d9">Required: <b>{required}</b></div>'
            f'{nh}</div>', unsafe_allow_html=True)

    bc=st.columns(4)
    check_card(bc[0],"2","Active Power",res['check2_power'],
               f"{res['p_poi']:.2f} MW",f"{poi_mw:.1f} MW",
               f"PCS PF = {res['pf_operating']:.3f} (derived)")
    check_card(bc[1],"1","Energy / Overbuild",res['check1_energy'],
               f"{res['e_poi_bol']:.1f} MWh BOL",f"{poi_mwh:.0f} MWh",
               f"Overbuild: {res['overbuild_actual']:+.1f}%")
    check_card(bc[2],"3","Reactive FERC 827",res['check3_reactive'],
               f"{res['q_avail_mvbus']:.2f} MVAR @ MV bus",
               f"{res['q_needed_mvbus']:.2f} MVAR @ MV bus",
               f"POI target: {res['q_poi_target']:.2f} MVAR")
    mpt_ok=n_mpt*s_mpt>=res['min_mpt_mva']
    check_card(bc[3],"—","MPT Adequate",mpt_ok,
               f"{n_mpt*s_mpt:.0f} MVA specified",
               f"{res['min_mpt_mva']:.1f} MVA min",
               f"Margin: {n_mpt*s_mpt-res['min_mpt_mva']:+.1f} MVA")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs(["📈 PV Curve","📉 Degradation","📋 Equipment","🔢 All Configs"])

    with t1:
        with st.spinner("Running power flow sweep…"):
            isu_s = res['n_pcs_base'] * mvt_mva
            df_pf = run_pf_sweep(
                float(poi_lim), float(n_mpt*s_mpt), z_mpt, xr_mpt, 10.*n_mpt,
                isu_s, z_isu, xr_isu, 8.*res['n_pcs_base'],
                res['n_pcs_base']*inv_mva*res['pf_operating']/poi_lim,
                tap_c, float(cap_bank), float(v_calc))

        if len(df_pf):
            dl=df_pf[df_pf['Q_inv_MVAR']>=0].sort_values('P_inv_MW')
            dg=df_pf[df_pf['Q_inv_MVAR']< 0].sort_values('P_inv_MW')
            fig=go.Figure()
            fig.add_hrect(y0=v_min,y1=v_max,fillcolor='lightgreen',opacity=.10,line_width=0)
            fig.add_hline(y=v_min,line_dash='dash',line_color='red',line_width=1.5,
                          annotation_text=f"V_min={v_min}")
            fig.add_hline(y=v_max,line_dash='dash',line_color='red',line_width=1.5,
                          annotation_text=f"V_max={v_max}", annotation_position="top left")
            fig.add_vline(x=poi_mw,line_dash='dot',line_color='gray',
                          annotation_text=f"{poi_mw:.0f} MW")
            if len(dl): fig.add_trace(go.Scatter(x=dl['P_inv_MW'],y=dl['V_POI'],mode='lines',
                name='Leading Q',line=dict(color='royalblue',width=2.5)))
            if len(dg): fig.add_trace(go.Scatter(x=dg['P_inv_MW'],y=dg['V_POI'],mode='lines',
                name='Lagging Q',line=dict(color='darkorange',width=2.5)))
            y_lo=max(.80,df_pf['V_POI'].min()-.03); y_hi=min(1.25,df_pf['V_POI'].max()+.03)
            fig.update_layout(title=f"PV Curve — {res['n_pcs_base']} PCS × {sel_ratio} BESS",
                xaxis_title="P_inv (MW)",yaxis_title="V_POI (pu)",
                yaxis=dict(range=[y_lo,y_hi]),height=420,legend=dict(x=.01,y=.99))
            st.plotly_chart(fig,use_container_width=True)
            v_nom=df_pf.loc[(df_pf['P_inv_MW']-poi_mw).abs().idxmin(),'V_POI']
            p1,p2,p3,p4=st.columns(4)
            p1.metric("V_POI @ rated P",f"{v_nom:.4f} pu",
                      delta="✓ In limits" if v_min<=v_nom<=v_max else "⚠ Violation")
            p2.metric("% pts in V-band",f"{((df_pf['V_POI']>=v_min)&(df_pf['V_POI']<=v_max)).mean()*100:.1f}%")
            p3.metric("Converged",f"{len(df_pf)}/121")
            p4.metric("PCS op. PF",f"{res['pf_operating']:.3f}")
        else:
            st.warning("Power flow did not converge — check tap position and MPT size.")

    with t2:
        ddf = res['deg_df']
        y_lo_deg = max(0, ddf['Total E@POI (MWh)'].min() * 0.85)
        y_hi_deg = ddf['Total E@POI (MWh)'].max() * 1.08

        fig2 = go.Figure()
        # Shaded "below target" band
        fig2.add_hrect(y0=y_lo_deg, y1=poi_mwh,
                       fillcolor='#fee2e2', opacity=0.15, line_width=0)
        fig2.add_hline(y=poi_mwh, line_dash='dash', line_color='red', line_width=2,
                       annotation_text=f"Target {poi_mwh:.0f} MWh",
                       annotation_position="top left",
                       annotation_font=dict(color='red', size=11))

        # Total energy line
        fig2.add_trace(go.Scatter(
            x=ddf['Year'], y=ddf['Total E@POI (MWh)'],
            mode='lines+markers', name='Total E @ POI',
            line=dict(color='royalblue', width=3),
            marker=dict(size=7, color='royalblue'),
            fill='tozeroy', fillcolor='rgba(30,64,175,0.06)'))

        # Augmentation bar — on secondary axis so scale doesn't compress main line
        aug_rows = ddf[ddf['Aug E added (MWh)'] > 0]
        if len(aug_rows):
            fig2.add_trace(go.Bar(
                x=aug_rows['Year'], y=aug_rows['Aug E added (MWh)'],
                name=f'Augmentation (+{aug_rows["Aug E added (MWh)"].sum():.0f} MWh)',
                marker_color='#16a34a', opacity=0.9,
                text=[f"+{v:.0f}" for v in aug_rows['Aug E added (MWh)']],
                textposition='outside', yaxis='y2'))

        fig2.update_layout(
            title=f"Energy @ POI — {res['n_pcs_base']} PCS × {sel_ratio} BESS  |  "
                  f"BOL {res['e_poi_bol']:.0f} MWh  |  Overbuild {res['overbuild_actual']:+.1f}%",
            xaxis_title="Year",
            yaxis=dict(title="MWh @ POI", range=[y_lo_deg, y_hi_deg]),
            yaxis2=dict(title="Aug MWh added", overlaying='y', side='right',
                        range=[0, aug_rows['Aug E added (MWh)'].max()*6 if len(aug_rows) else 100],
                        showgrid=False),
            height=420, legend=dict(x=0.01, y=0.99),
            barmode='overlay')
        st.plotly_chart(fig2, use_container_width=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("E @ BOL", f"{res['e_poi_bol']:.1f} MWh")
        m2.metric("Overbuild", f"{res['overbuild_actual']:+.1f}%")
        m3.metric("Target E", f"{poi_mwh:.0f} MWh")
        yr_drop = res['yr_below']
        m4.metric("First yr < target", str(yr_drop) if yr_drop is not None else "Never ✅")
        m5.metric("Aug adds", f"{res['n_pcs_aug']} PCS / {res['n_batt_aug']} BESS")

        st.dataframe(ddf.set_index('Year'), use_container_width=True)
        st.download_button("⬇ CSV", ddf.to_csv(index=False), "degradation.csv","text/csv")

    with t3:
        eq_rows = [
            {"Equipment": f"PCS ({pcs_model})", "Qty (base)": res['n_pcs_base'],
             "Qty (aug)": res['n_pcs_aug'], "Unit": f"{inv_mva} MVA"},
            {"Equipment": f"Battery ({batt_model})", "Qty (base)": res['n_batt_base'],
             "Qty (aug)": res['n_batt_aug'], "Unit": f"{batt_mwh} MWh DC"},
            {"Equipment": "MVT", "Qty (base)": res['n_pcs_base'],
             "Qty (aug)": res['n_pcs_aug'], "Unit": f"{mvt_mva} MVA"},
            {"Equipment": "MPT", "Qty (base)": n_mpt,
             "Qty (aug)": 0, "Unit": f"{s_mpt:.0f} MVA (min {res['min_mpt_mva']:.1f})"},
        ]
        eq_df = pd.DataFrame(eq_rows).set_index("Equipment")
        eq_df['Total (after aug)'] = eq_df['Qty (base)'] + eq_df['Qty (aug)']
        st.dataframe(eq_df, use_container_width=True)
        st.download_button("⬇ Equipment CSV", eq_df.to_csv(), "equipment.csv","text/csv")

        # Power cascade
        st.markdown("**Power Cascade**")
        cdf = pd.DataFrame([
            {'Stage':'Inverter Output','P (MW)':round(res['p_poi']/eta_mpt/eta_tx/eta_mv/eta_mvt*eta_pcs/(eta_pcs),1),
             'Note':f"PCS PF = {res['pf_operating']:.3f} (derived from P/Q requirement)"},
            {'Stage':'MV Bus (−Aux)',  'P (MW)':round(res['p_mpt_in'],2), 'Note':f"Aux = {res['aux_mw']:.2f} MW"},
            {'Stage':'POI',            'P (MW)':round(res['p_poi'],2),    'Note':f"PF = {res['pf_poi']:.3f}"},
            {'Stage':'▶ POI Target',   'P (MW)':poi_mw,                   'Note':f"PF = {target_pf:.2f}"},
        ])
        st.dataframe(cdf.set_index('Stage'), use_container_width=True)

    with t4:
        # Multi-config degradation overlay
        st.markdown("#### Degradation overlay — all configs")
        fig4 = go.Figure()
        fig4.add_hline(y=poi_mwh, line_dash='dash', line_color='red', line_width=1.5,
                       annotation_text=f"Target {poi_mwh:.0f} MWh")
        colors = ['royalblue','darkorange','green','purple','crimson','teal']
        all_mins = []
        for i,ratio in enumerate(bess_ratios):
            r = results[ratio]
            ddf2 = r['deg_df']
            all_mins.append(ddf2['Total E@POI (MWh)'].min())
            fig4.add_trace(go.Scatter(
                x=ddf2['Year'], y=ddf2['Total E@POI (MWh)'],
                mode='lines+markers', name=f"{ratio} BESS/PCS ({r['n_pcs_base']} PCS)",
                line=dict(color=colors[i%len(colors)], width=2),
                marker=dict(size=4)))
        fig4.update_layout(
            title="Energy @ POI — all BESS/PCS ratios",
            xaxis_title="Year", yaxis_title="MWh @ POI",
            yaxis=dict(range=[max(0,min(all_mins)*0.9), max(r['e_poi_bol'] for r in results.values())*1.05]),
            height=400, legend=dict(x=0.01,y=0.99))
        st.plotly_chart(fig4, use_container_width=True)

        # Full table
        st.markdown("#### Full sizing table")
        st.dataframe(cmp_df, use_container_width=True)
        st.download_button("⬇ Full comparison CSV", cmp_df.to_csv(), "sizing_comparison.csv","text/csv")

else:
    st.info("👈 Set parameters in the sidebar, then press **▶ Size All Configurations**")
    with st.expander("ℹ️ How this tool works"):
        st.markdown(f"""
**Auto-sizing flow:**

1. **For each BESS/PCS ratio** (e.g. 2, 3, 4, 5, 6 batteries per PCS):
   - Find minimum PCS qty where both **P and Q are met simultaneously**
   - PCS operating PF is **derived** (not a user input) — minimum PF needed to deliver MW target, maximising Q headroom
   - Energy PCS qty = ceil(poi_mwh × (1+overbuild%) / (batt/pcs × batt_mwh × η_total))
   - n_pcs_base = max(power_qty, energy_qty)

2. **Overbuild %** controls initial installed energy:
   `E_BOL = poi_mwh × (1 + overbuild%/100)`
   This ensures energy degrades to ≥ poi_mwh at the augmentation year

3. **3 Checks** (matching Excel R29–R31):
   - ✅ Energy: E_BOL ≥ poi_mwh
   - ✅ Power: P@POI ≥ poi_mw  
   - ✅ Reactive (FERC 827): Q@MV_bus ≥ Q_needed

4. **Power flow** validates V_POI across full P-Q operating range
        """)