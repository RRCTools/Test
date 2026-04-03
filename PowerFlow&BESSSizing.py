"""
BESS Sizing Tool v2 — corrected engine matching Excel MAIN DC BESS
Confirmed sizing logic:
  n_power  = ceil(P_inv_needed / (inv_mva * pcs_pf))  [iterative for aux]
  n_energy = ceil(poi_mwh / (units_per_blk * batt_mwh * ALL_eta))
  n_final  = max(n_power, n_energy)
  batteries = n_final * units_per_block
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil

st.set_page_config(page_title="BESS Sizing Tool", layout="wide", page_icon="🔋")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.block-container{padding-top:.6rem;padding-bottom:1rem;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;}
div[data-testid="stSidebarContent"]{background:#0d1117;color:#e6edf3;}
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p,
div[data-testid="stSidebarContent"] span{color:#c9d1d9 !important;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:600;
     letter-spacing:.12em;text-transform:uppercase;color:#58a6ff;
     border-bottom:1px solid #21262d;padding-bottom:3px;
     margin-top:14px;margin-bottom:5px;}
.badge-ok  {background:#0d3321;color:#3fb950;border:1px solid #238636;
            padding:4px 10px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.8rem;}
.badge-fail{background:#3d1a1a;color:#f85149;border:1px solid #b62324;
            padding:4px 10px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.8rem;}
.cascade-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
              padding:10px 14px;text-align:center;font-size:.82rem;}
.cascade-label{color:#64748b;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;}
.cascade-p{color:#1e40af;font-weight:700;font-size:1.05rem;}
.cascade-q{color:#7c3aed;font-weight:600;}
.cascade-s{color:#0f766e;font-weight:600;}
.cascade-pf{color:#b45309;font-size:.78rem;}
.arrow{color:#94a3b8;font-size:1.4rem;display:flex;align-items:center;justify-content:center;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT SOH CURVE  (from Excel MAIN DC Engine rows 111-131)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_SOH = [1.0000,0.9342,0.9115,0.8933,0.8775,0.8633,0.8502,
               0.8381,0.8266,0.8158,0.8054,0.7955,0.7859,0.7767,
               0.7677,0.7590,0.7506,0.7424,0.7343,0.7265,0.7188]

# ══════════════════════════════════════════════════════════════════════════════
#  3-BUS NR POWER FLOW  (same engine as test1final.py)
# ══════════════════════════════════════════════════════════════════════════════
def build_ybus(sbase, mpt_s, mpt_z, mpt_xr, mpt_i0, mpt_p0kw,
               isu_s, isu_z, isu_xr, isu_i0, isu_p0kw, tap_c, q_cap=0.0):
    rm = mpt_z*np.cos(np.arctan(mpt_xr)); xm = mpt_z*np.sin(np.arctan(mpt_xr))
    Ys = (mpt_s/sbase)/complex(rm,xm)
    gm = (mpt_p0kw/1e3)/mpt_s; bm = np.sqrt(max(mpt_i0**2-gm**2,0))
    Ym = complex(gm,bm)*(mpt_s/sbase)
    c  = 1.0-tap_c
    if abs(c)<1e-9: c=1.0
    Y10=(1-c)/c*Ys+Ym/c**2; Y12=c*Ys; Y20=(c-1)*Ys
    ri=isu_z*np.cos(np.arctan(isu_xr)); xi=isu_z*np.sin(np.arctan(isu_xr))
    Yi=(isu_s/sbase)/complex(ri,xi)
    gi=(isu_p0kw/1e3)/isu_s; bi=np.sqrt(max(isu_i0**2-gi**2,0))
    Ymi=complex(gi,-bi)*(isu_s/sbase)
    yc=complex(0,q_cap/sbase)
    Y=np.zeros((3,3),dtype=complex)
    Y[0,0]=Y10+Y12; Y[0,1]=-Y12; Y[1,0]=-Y12
    Y[1,1]=Y12+Y20+Yi+Ymi+yc; Y[1,2]=-Yi; Y[2,1]=-Yi; Y[2,2]=Yi
    return Y

def nr_pf(Y,P3,Q3,V1=1.0,V2i=None,V3i=None,tol=1e-9,maxiter=100):
    if V2i is None: V2i=V1
    if V3i is None: V3i=V1
    V=np.array([V1,V2i,V3i]); th=np.zeros(3)
    for _ in range(maxiter):
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
def run_pf_sweep(sbase,mpt_s,mpt_z,mpt_xr,mpt_p0kw,
                 isu_s,isu_z,isu_xr,isu_p0kw,
                 P_max_pu,tap_c,q_cap,v_tgt):
    Y=build_ybus(sbase,mpt_s,mpt_z,mpt_xr,0.001,mpt_p0kw,
                 isu_s,isu_z,isu_xr,0.005,isu_p0kw,tap_c,q_cap)
    S_isu=isu_s/sbase
    rows=[]; V2p,V3p=v_tgt,v_tgt
    for th in np.linspace(0,2*np.pi,121,endpoint=False):
        P3=np.clip(S_isu*np.cos(th),-P_max_pu,P_max_pu)
        Q3=S_isu*np.sin(th)
        V,ang,conv=nr_pf(Y,P3,Q3,V1=v_tgt,V2i=V2p,V3i=V3p)
        if conv:
            V2p,V3p=V[1],V[2]
            P1=sum(V[0]*V[j]*(Y[0,j].real*np.cos(ang[0]-ang[j])+Y[0,j].imag*np.sin(ang[0]-ang[j])) for j in range(3))
            Q1=sum(V[0]*V[j]*(Y[0,j].real*np.sin(ang[0]-ang[j])-Y[0,j].imag*np.cos(ang[0]-ang[j])) for j in range(3))
            rows.append({'P_inv_MW':P3*sbase,'Q_inv_MVAR':Q3*sbase,
                         'V_POI':V[1],'V_ISU':V[2],
                         'P_grid_MW':P1*sbase,'Q_grid_MVAR':Q1*sbase})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
#  BESS SIZING ENGINE  (matches Excel MAIN DC Engine exactly)
# ══════════════════════════════════════════════════════════════════════════════
def size_bess(poi_mw, poi_mwh, target_pf,
              inv_mva, pcs_pf, units_per_blk, batt_mwh_dc, aux_kw_per_blk,
              eta_pcs, eta_mvt, eta_mv_cable, eta_mpt, eta_transmission,
              eta_dc_cable, eta_charge, eta_auxiliary,
              project_years, aug_year, soh_curve, cap_bank_mvar=0.0):

    # ── Loss chain (full, for energy) ──────────────────────────────────────
    eta_all = eta_pcs*eta_mvt*eta_mv_cable*eta_mpt*eta_transmission*eta_dc_cable*eta_charge*eta_auxiliary

    # ── POWER constraint: iterative (aux depends on n) ──────────────────────
    n = 1
    for _ in range(20):
        aux_mw = n * aux_kw_per_blk / 1000.0
        # Back-calculate required inverter power
        p_mpt_lv    = poi_mw / (eta_mpt * eta_transmission)
        p_mv_needed = p_mpt_lv + aux_mw
        p_mvt_out   = p_mv_needed / eta_mv_cable
        p_inv_needed= p_mvt_out  / eta_mvt
        n_new = ceil(p_inv_needed / (inv_mva * pcs_pf))
        if n_new == n: break
        n = n_new
    n_power = n

    # ── ENERGY constraint: size for BOL nameplate target ───────────────────
    block_mwh_poi = units_per_blk * batt_mwh_dc * eta_all
    n_energy = ceil(poi_mwh / block_mwh_poi)

    # ── Final count ──────────────────────────────────────────────────────────
    n_blk = max(n_power, n_energy)

    # ── Actuals (forward cascade) ────────────────────────────────────────────
    aux_mw_act   = n_blk * aux_kw_per_blk / 1000.0
    p_inv_act    = n_blk * inv_mva * pcs_pf
    q_inv_act    = n_blk * inv_mva * np.sqrt(max(1-pcs_pf**2,0))
    s_inv_act    = n_blk * inv_mva
    p_mvt_out    = p_inv_act    * eta_mvt
    q_mvt_out    = q_inv_act    * eta_mvt
    p_mv_bus     = p_mvt_out   * eta_mv_cable
    q_mv_bus     = q_mvt_out   * eta_mv_cable
    p_mpt_in     = p_mv_bus    - aux_mw_act      # aux subtracted at MV bus
    q_mpt_in     = q_mv_bus    + cap_bank_mvar    # cap bank helps reactive
    p_poi_act    = p_mpt_in    * eta_mpt * eta_transmission
    q_poi_act    = q_mpt_in    * eta_mpt * eta_transmission   # approx
    s_poi_act    = np.sqrt(p_poi_act**2 + q_poi_act**2)
    pf_poi_act   = p_poi_act / s_poi_act if s_poi_act > 0 else 1.0

    # Reactive requirement
    q_poi_needed = poi_mw * np.tan(np.arccos(target_pf))
    s_poi_needed = poi_mw / target_pf
    q_meets      = q_poi_act >= (q_poi_needed - 0.5)

    # Energy
    e_dc_act     = n_blk * units_per_blk * batt_mwh_dc
    e_poi_bol    = e_dc_act * eta_all

    # Min MPT MVA (from MV bus apparent power)
    s_mvbus      = np.sqrt(p_mpt_in**2 + q_mpt_in**2)
    min_mpt_mva  = s_mvbus

    # ── Degradation schedule ─────────────────────────────────────────────────
    soh = soh_curve + [soh_curve[-1]]*(max(project_years+1-len(soh_curve),0))
    deg = []
    for yr in range(project_years+1):
        soh_yr = soh[min(yr, len(soh)-1)]
        e_yr   = e_poi_bol * soh_yr
        aug    = max(poi_mwh - e_yr, 0) if aug_year > 0 and yr == aug_year else 0.0
        deg.append({'Year':yr,'SOH (%)':round(soh_yr*100,2),
                    'Energy @ POI (MWh)':round(e_yr,1),
                    'Augmentation (MWh)':round(aug,1)})
    deg_df = pd.DataFrame(deg)

    # ── Power cascade table ──────────────────────────────────────────────────
    def pf_(p,q): return p/np.sqrt(p**2+q**2) if np.sqrt(p**2+q**2)>0 else 1.0
    def s_(p,q):  return np.sqrt(p**2+q**2)

    cascade = {
        'Inverter Output': {'P':p_inv_act,'Q':q_inv_act,'S':s_inv_act,'PF':pcs_pf},
        'MVT Output':      {'P':p_mvt_out,'Q':q_mvt_out,'S':s_(p_mvt_out,q_mvt_out),'PF':pf_(p_mvt_out,q_mvt_out)},
        'MV Bus\n(−Aux)':  {'P':p_mpt_in, 'Q':q_mpt_in, 'S':s_(p_mpt_in,q_mpt_in),'PF':pf_(p_mpt_in,q_mpt_in)},
        'POI':             {'P':p_poi_act,'Q':q_poi_act,'S':s_poi_act,'PF':pf_poi_act},
    }

    return {
        'n_blk': n_blk, 'n_batt': n_blk*units_per_blk,
        'n_pcs': n_blk, 'n_mvt': n_blk,
        'n_power': n_power, 'n_energy': n_energy,
        'p_poi_act': p_poi_act, 'q_poi_act': q_poi_act,
        's_inv_act': s_inv_act,
        'e_poi_bol': e_poi_bol, 'e_dc_act': e_dc_act,
        'q_poi_needed': q_poi_needed,
        'q_meets': q_meets, 'p_meets': p_poi_act >= poi_mw*0.999,
        'min_mpt_mva': min_mpt_mva,
        'p_loss_pct': (1-eta_all)*100,
        'eta_all': eta_all,
        'block_mwh_poi': block_mwh_poi,
        'cascade': cascade,
        'deg_df': deg_df,
        'p_inv_needed': p_inv_needed,
        'aux_mw': aux_mw_act,
        'p_mv_bus': p_mv_bus, 'q_mv_bus': q_mv_bus,
        'p_mpt_in': p_mpt_in, 'q_mpt_in': q_mpt_in,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔋 BESS Sizing")

    st.markdown('<div class="sec">Project</div>', unsafe_allow_html=True)
    project_name = st.text_input("Project Name", value="Demo BESS", label_visibility="collapsed")
    iso = st.selectbox("ISO", ["WECC","CAISO","ERCOT","PJM","MISO","NYISO","ISO-NE","Other"])

    st.markdown('<div class="sec">POI Requirements</div>', unsafe_allow_html=True)
    poi_mw    = st.number_input("Nameplate Power @ POI (MW)",  value=200.0, min_value=1.0, step=10.0)
    poi_mwh   = st.number_input("Nameplate Energy @ POI (MWh)", value=800.0, min_value=1.0, step=50.0)
    poi_limit = st.number_input("POI Export Limit (MW)",         value=200.0, min_value=1.0, step=10.0)
    proj_yrs  = st.number_input("Project Term (years)", value=20, min_value=1, max_value=40)
    aug_year  = st.number_input("Augmentation Year (0=none)",   value=0,     min_value=0)

    st.markdown('<div class="sec">Reactive Power</div>', unsafe_allow_html=True)
    pf_or_q = st.radio("", ["Target PF","Target MVAR"], horizontal=True, label_visibility="collapsed")
    if pf_or_q == "Target PF":
        target_pf   = st.number_input("Target PF @ POI", value=0.95, min_value=0.5, max_value=1.0, step=0.01)
        st.caption(f"Q = {poi_mw*np.tan(np.arccos(target_pf)):.1f} MVAR")
    else:
        tmvar = st.number_input("Target MVAR @ POI", value=65.7, min_value=0.0, step=1.0)
        target_pf = poi_mw/np.sqrt(poi_mw**2+tmvar**2) if tmvar else 1.0
        st.caption(f"PF = {target_pf:.3f}")
    cap_bank = st.number_input("Capacitor Bank (MVAR)", value=0.0, min_value=0.0, step=5.0)

    st.markdown('<div class="sec">Grid / MPT</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    n_mpt = ca.number_input("#MPTs", value=1, min_value=1)
    s_mpt = cb.number_input("MVA/MPT", value=240.0, min_value=10.0, step=10.0)
    cc,cd = st.columns(2)
    z_mpt  = cc.number_input("Z (pu)",  value=0.10, min_value=0.01, step=0.005, format="%.3f")
    xr_mpt = cd.number_input("X/R MPT", value=40.0, min_value=1.0)
    eta_mpt = st.number_input("MPT Efficiency", value=0.995, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")

    has_oltc = st.checkbox("OLTC?", value=True)
    if has_oltc:
        ce,cf = st.columns(2)
        ntaps   = ce.number_input("#Taps", value=31, min_value=3, step=2)
        tap_rng = cf.number_input("Range%", value=10.0, min_value=1.0)
        tap_num = st.slider("Tap position", int(-(ntaps//2)), int(ntaps//2), 0)
        tap_c   = tap_num * (tap_rng/100.0) / ((ntaps-1)/2.0)
        st.caption(f"tap_c={tap_c:+.4f}  c={1-tap_c:.3f}")
        fixed_tap = tap_c
    else:
        fixed_tap = st.number_input("Fixed tap (pu)", value=0.0, step=0.005, format="%.3f")
        tap_c = fixed_tap

    st.markdown('<div class="sec">PCS / Inverter Block</div>', unsafe_allow_html=True)
    pcs_model = st.text_input("PCS Model", value="EPC POWER M10")
    cg,ch = st.columns(2)
    inv_mva = cg.number_input("Inv MVA", value=5.3, min_value=0.1, step=0.1)
    mvt_mva = ch.number_input("MVT MVA", value=5.3, min_value=0.1, step=0.1)
    pcs_pf  = st.number_input("PCS Operating PF", value=0.90, min_value=0.5, max_value=1.0, step=0.01,
                               help="Operating PF of PCS. Determines P vs Q split per block.")
    eta_pcs = st.number_input("PCS Efficiency", value=0.985, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")

    st.markdown('<div class="sec">Battery Unit</div>', unsafe_allow_html=True)
    batt_model = st.text_input("Battery Model", value="BESS Unit")
    ci,cj = st.columns(2)
    batt_mwh    = ci.number_input("MWh/unit (DC)", value=6.138, min_value=0.1, step=0.1, format="%.3f")
    units_per_blk = cj.number_input("Units/block", value=4, min_value=1)
    aux_kw = st.number_input("Aux load/block (kW)", value=75.6, min_value=0.0, step=1.0)

    st.markdown('<div class="sec">MVT / ISU</div>', unsafe_allow_html=True)
    ck,cl = st.columns(2)
    z_isu  = ck.number_input("Z (pu)",  value=0.08, min_value=0.01, step=0.005, format="%.3f")
    xr_isu = cl.number_input("X/R ISU", value=8.83, min_value=1.0)
    eta_mvt = st.number_input("MVT Efficiency", value=0.990, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")

    st.markdown('<div class="sec">Loss Factors</div>', unsafe_allow_html=True)
    with st.expander("Edit AC losses"):
        eta_mv_cable  = st.number_input("MV Cable",      value=0.995, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")
        eta_trans     = st.number_input("Transmission",  value=0.990, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")
        eta_auxiliary = st.number_input("Auxiliary",     value=0.998, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")
    with st.expander("Edit DC losses"):
        eta_dc_cable  = st.number_input("DC Cable",        value=0.999, min_value=0.9, max_value=1.0, step=0.001, format="%.3f")
        eta_charge    = st.number_input("Charge/Discharge",value=0.955, min_value=0.8, max_value=1.0, step=0.001, format="%.3f")

    # ── SOH Degradation Curve (editable) ─────────────────────────────────────
    st.markdown('<div class="sec">Battery Degradation Curve</div>', unsafe_allow_html=True)
    with st.expander("Edit SOH curve (Year → SOH%)"):
        st.caption("Default: Samsung SBB / standard Li-ion. Edit any year below.")
        soh_rows = []
        cols_header = st.columns([1,2])
        cols_header[0].markdown("**Year**"); cols_header[1].markdown("**SOH %**")
        for yr in range(min(int(proj_yrs)+1, 21)):
            default_soh = DEFAULT_SOH[yr] if yr < len(DEFAULT_SOH) else DEFAULT_SOH[-1]
            r1, r2 = st.columns([1,2])
            r1.markdown(f"<div style='padding-top:8px'>{yr}</div>", unsafe_allow_html=True)
            val = r2.number_input(f"soh_{yr}", value=float(round(default_soh*100,2)),
                                  min_value=0.0, max_value=100.0, step=0.1,
                                  label_visibility="collapsed", key=f"soh_yr_{yr}")
            soh_rows.append(val/100.0)
        soh_curve = soh_rows

    st.divider()
    run_btn = st.button("▶  Run Sizing", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"# 🔋 BESS Sizing Tool")
st.caption(f"**{project_name}** · {iso} · {poi_mw:.0f} MW / {poi_mwh:.0f} MWh @ POI · {poi_limit:.0f} MW export limit")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Sizing BESS system…"):
        res = size_bess(
            poi_mw=poi_mw, poi_mwh=poi_mwh, target_pf=target_pf,
            inv_mva=inv_mva, pcs_pf=pcs_pf,
            units_per_blk=int(units_per_blk), batt_mwh_dc=batt_mwh,
            aux_kw_per_blk=aux_kw,
            eta_pcs=eta_pcs, eta_mvt=eta_mvt, eta_mv_cable=eta_mv_cable,
            eta_mpt=eta_mpt, eta_transmission=eta_trans,
            eta_dc_cable=eta_dc_cable, eta_charge=eta_charge,
            eta_auxiliary=eta_auxiliary,
            project_years=int(proj_yrs), aug_year=int(aug_year),
            soh_curve=soh_curve, cap_bank_mvar=float(cap_bank)
        )

    # ── Equipment quantities ─────────────────────────────────────────────────
    st.subheader("📦 Equipment Quantities")
    q1,q2,q3,q4,q5,q6 = st.columns(6)
    q1.metric("Power Blocks (PCS)", str(res['n_blk']),
              help=f"Power: {res['n_power']} blocks, Energy: {res['n_energy']} blocks → max")
    q2.metric("Battery Units", str(res['n_batt']),
              help=f"{res['n_blk']} blocks × {units_per_blk} units")
    q3.metric("PCS Units", str(res['n_pcs']))
    q4.metric("MVT Units", str(res['n_mvt']))
    q5.metric("Min MPT MVA", f"{res['min_mpt_mva']:.1f} MVA",
              delta=f"Specified {n_mpt*s_mpt:.0f} MVA {'✓' if n_mpt*s_mpt>=res['min_mpt_mva'] else '⚠'}")
    q6.metric("Total DC Energy", f"{res['e_dc_act']:.0f} MWh")

    st.divider()

    # ── Sizing checks ────────────────────────────────────────────────────────
    st.subheader("✅ Sizing Checks")
    bc = st.columns(4)
    def badge(col, ok, label, detail):
        cls = "badge-ok" if ok else "badge-fail"
        sym = "✓" if ok else "✗"
        col.markdown(f'<div class="{cls}">{sym} {label}</div><div style="font-size:.78rem;color:#64748b;margin-top:3px">{detail}</div>', unsafe_allow_html=True)

    badge(bc[0], res['p_meets'],  "Active Power",
          f"{res['p_poi_act']:.1f} MW ≥ {poi_mw:.0f} MW")
    badge(bc[1], res['e_poi_bol'] >= poi_mwh*0.999, "Energy @ BOL",
          f"{res['e_poi_bol']:.1f} MWh ≥ {poi_mwh:.0f} MWh")
    badge(bc[2], res['q_meets'],  "Reactive Power",
          f"{res['q_poi_act']:.1f} MVAR avail, {res['q_poi_needed']:.1f} needed")
    mpt_ok = (n_mpt*s_mpt) >= res['min_mpt_mva']
    badge(bc[3], mpt_ok, "MPT Adequate",
          f"Required {res['min_mpt_mva']:.1f} MVA, have {n_mpt*s_mpt:.0f} MVA")

    st.divider()

    # ── SINGLE LINE DIAGRAM ──────────────────────────────────────────────────
    st.subheader("📐 Single Line Diagram — Power Flow Cascade")

    cas = res['cascade']
    stages = list(cas.keys())

    # Build SVG-style diagram using plotly
    fig_sld = go.Figure()
    fig_sld.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        height=260, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(visible=False, range=[-0.5,4.5]),
        yaxis=dict(visible=False, range=[-0.3,1.5]),
        showlegend=False
    )

    # Bus bar line
    for i, (stage, vals) in enumerate(cas.items()):
        x = i
        pf_val = vals['PF']
        pf_color = '#16a34a' if pf_val >= target_pf else '#dc2626'
        # Bus rectangle
        fig_sld.add_shape(type='rect', x0=x-0.35, x1=x+0.35, y0=0.55, y1=0.65,
                          fillcolor='#1e40af', line_color='#1e40af')
        # Label above
        label = stage.replace('\n',' ')
        fig_sld.add_annotation(x=x, y=1.30, text=f"<b>{label}</b>",
                               showarrow=False, font=dict(size=10, color='#1e40af'))
        # P
        fig_sld.add_annotation(x=x, y=1.10, text=f"P = {vals['P']:.1f} MW",
                               showarrow=False, font=dict(size=10, color='#1e40af'))
        # Q
        fig_sld.add_annotation(x=x, y=0.90, text=f"Q = {vals['Q']:.1f} MVAR",
                               showarrow=False, font=dict(size=10, color='#7c3aed'))
        # S
        fig_sld.add_annotation(x=x, y=0.72, text=f"S = {vals['S']:.1f} MVA",
                               showarrow=False, font=dict(size=10, color='#0f766e'))
        # PF
        fig_sld.add_annotation(x=x, y=0.30, text=f"PF = {pf_val:.3f}",
                               showarrow=False, font=dict(size=10, color=pf_color))
        # Arrow between stages
        if i < len(stages)-1:
            fig_sld.add_annotation(x=x+0.55, y=0.60, text="→",
                                   showarrow=False, font=dict(size=18, color='#94a3b8'))

    # Add aux label at MV bus
    fig_sld.add_annotation(x=2, y=0.10,
        text=f"⬆ Aux = {res['aux_mw']:.2f} MW",
        showarrow=False, font=dict(size=9, color='#b45309'))

    # Add POI target
    poi_P = cas['POI']['P']; poi_Q = cas['POI']['Q']; poi_S = cas['POI']['S']
    poi_pf_col = '#16a34a' if cas['POI']['PF'] >= target_pf else '#dc2626'

    st.plotly_chart(fig_sld, use_container_width=True)

    # Numeric cascade table below diagram
    cascade_data = []
    for stage, vals in cas.items():
        cascade_data.append({
            'Stage': stage.replace('\n',' '),
            'P (MW)': round(vals['P'],2),
            'Q (MVAR)': round(vals['Q'],2),
            'S (MVA)': round(vals['S'],2),
            'PF': round(vals['PF'],4),
        })
    # Add POI target row
    q_need = res['q_poi_needed']
    cascade_data.append({
        'Stage': '▶ POI Target',
        'P (MW)': poi_mw,
        'Q (MVAR)': round(q_need,2),
        'S (MVA)': round(poi_mw/target_pf,2),
        'PF': target_pf,
    })
    st.dataframe(pd.DataFrame(cascade_data).set_index('Stage'), use_container_width=True)

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["📈 PV Curve","📉 Degradation","📋 Equipment List","🔢 Loss Chain"])

    with t1:
        with st.spinner("Running power flow…"):
            isu_s_total = res['n_mvt'] * mvt_mva
            isu_p0kw    = 8.0 * res['n_mvt']
            P_max_pu    = res['n_blk'] * inv_mva * pcs_pf / poi_limit
            df_pf = run_pf_sweep(
                poi_limit, float(n_mpt*s_mpt), z_mpt, xr_mpt, 10.0*n_mpt,
                isu_s_total, z_isu, xr_isu, isu_p0kw,
                P_max_pu, tap_c, float(cap_bank), float(
                    st.session_state.get('v_calc', 1.0))
            )

        v_calc_pf = 1.0
        v_min_pf = 0.95; v_max_pf = 1.05

        if len(df_pf) > 0:
            df_lead = df_pf[df_pf['Q_inv_MVAR']>=0].sort_values('P_inv_MW')
            df_lag  = df_pf[df_pf['Q_inv_MVAR']< 0].sort_values('P_inv_MW')
            fig_pv = go.Figure()
            fig_pv.add_hrect(y0=v_min_pf, y1=v_max_pf, fillcolor='lightgreen', opacity=0.10, line_width=0)
            fig_pv.add_hline(y=v_min_pf, line_dash='dash', line_color='red', line_width=1.5)
            fig_pv.add_hline(y=v_max_pf, line_dash='dash', line_color='red', line_width=1.5)
            fig_pv.add_vline(x=poi_mw, line_dash='dot', line_color='gray',
                             annotation_text=f"{poi_mw:.0f} MW")
            if len(df_lead): fig_pv.add_trace(go.Scatter(x=df_lead['P_inv_MW'],y=df_lead['V_POI'],
                mode='lines',name='Leading Q',line=dict(color='royalblue',width=2.5)))
            if len(df_lag):  fig_pv.add_trace(go.Scatter(x=df_lag['P_inv_MW'], y=df_lag['V_POI'],
                mode='lines',name='Lagging Q',line=dict(color='darkorange',width=2.5)))
            y_lo = max(0.80, df_pf['V_POI'].min()-0.03)
            y_hi = min(1.25, df_pf['V_POI'].max()+0.03)
            fig_pv.update_layout(title="PV Curve — POI Voltage vs Active Power",
                xaxis_title="P_inv (MW)",yaxis_title="V_POI (pu)",
                yaxis=dict(range=[y_lo,y_hi]),height=420,legend=dict(x=0.01,y=0.99))
            st.plotly_chart(fig_pv, use_container_width=True)
            v_nom = df_pf.loc[(df_pf['P_inv_MW']-poi_mw).abs().idxmin(),'V_POI']
            c1,c2,c3 = st.columns(3)
            c1.metric("V_POI @ rated P", f"{v_nom:.4f} pu")
            c2.metric("% pts in band", f"{((df_pf['V_POI']>=v_min_pf)&(df_pf['V_POI']<=v_max_pf)).mean()*100:.1f}%")
            c3.metric("Converged pts", f"{len(df_pf)}/121")
        else:
            st.warning("Power flow did not converge. Check tap position and MPT size.")

    with t2:
        fig_deg = go.Figure()
        fig_deg.add_hline(y=poi_mwh, line_dash='dash', line_color='red',
                          annotation_text=f"Target {poi_mwh:.0f} MWh")
        fig_deg.add_trace(go.Scatter(x=res['deg_df']['Year'],
            y=res['deg_df']['Energy @ POI (MWh)'],
            mode='lines+markers', name='Energy @ POI',
            line=dict(color='royalblue',width=2.5), marker=dict(size=6)))
        aug_rows = res['deg_df'][res['deg_df']['Augmentation (MWh)']>0]
        if len(aug_rows):
            fig_deg.add_trace(go.Bar(x=aug_rows['Year'],y=aug_rows['Augmentation (MWh)'],
                name='Augmentation',marker_color='green',opacity=0.6))
        fig_deg.update_layout(title="Energy @ POI over Project Life",
            xaxis_title="Year",yaxis_title="MWh",height=380)
        st.plotly_chart(fig_deg, use_container_width=True)
        st.dataframe(res['deg_df'].set_index('Year'), use_container_width=True)
        st.download_button("⬇ Download Degradation Table",
            res['deg_df'].to_csv(index=False),"degradation.csv","text/csv")

    with t3:
        equip = pd.DataFrame([
            {"Equipment": f"PCS Block ({pcs_model})","Qty": res['n_pcs'],
             "Unit Size": f"{inv_mva} MVA","Total":f"{res['s_inv_act']:.1f} MVA"},
            {"Equipment": f"Battery Unit ({batt_model})","Qty": res['n_batt'],
             "Unit Size": f"{batt_mwh} MWh DC","Total":f"{res['e_dc_act']:.1f} MWh DC"},
            {"Equipment": "MVT (Medium Voltage Transformer)","Qty": res['n_mvt'],
             "Unit Size": f"{mvt_mva} MVA","Total":f"{res['n_mvt']*mvt_mva:.1f} MVA"},
            {"Equipment": "MPT (Main Power Transformer)","Qty": n_mpt,
             "Unit Size": f"{s_mpt:.0f} MVA","Total":f"{n_mpt*s_mpt:.0f} MVA"},
        ])
        st.dataframe(equip.set_index("Equipment"), use_container_width=True)
        st.download_button("⬇ Download Equipment List",
            equip.to_csv(index=False),"equipment.csv","text/csv")

        st.markdown("**Sizing Drivers**")
        driver = "POWER" if res['n_power'] >= res['n_energy'] else "ENERGY"
        st.info(f"Sizing driven by **{driver}** constraint\n"
                f"- Power blocks needed: **{res['n_power']}** "
                f"(p_inv={res['p_inv_needed']:.1f} MW ÷ {inv_mva*pcs_pf:.2f} MW/block)\n"
                f"- Energy blocks needed: **{res['n_energy']}** "
                f"({poi_mwh:.0f} MWh ÷ {res['block_mwh_poi']:.2f} MWh/block @ POI)")

    with t4:
        loss_df = pd.DataFrame([
            {"Stage":"Transmission", "η":eta_trans,  "Loss %":f"{(1-eta_trans)*100:.1f}%"},
            {"Stage":"MPT",          "η":eta_mpt,    "Loss %":f"{(1-eta_mpt)*100:.1f}%"},
            {"Stage":"MV Cable",     "η":eta_mv_cable,"Loss %":f"{(1-eta_mv_cable)*100:.1f}%"},
            {"Stage":"MVT",          "η":eta_mvt,    "Loss %":f"{(1-eta_mvt)*100:.1f}%"},
            {"Stage":"PCS",          "η":eta_pcs,    "Loss %":f"{(1-eta_pcs)*100:.1f}%"},
            {"Stage":"DC Cable",     "η":eta_dc_cable,"Loss %":f"{(1-eta_dc_cable)*100:.1f}%"},
            {"Stage":"Charge/Disch", "η":eta_charge, "Loss %":f"{(1-eta_charge)*100:.1f}%"},
            {"Stage":"Auxiliary",    "η":eta_auxiliary,"Loss %":f"{(1-eta_auxiliary)*100:.1f}%"},
            {"Stage":"→ TOTAL",      "η":round(res['eta_all'],5),"Loss %":f"{res['p_loss_pct']:.2f}%"},
        ])
        st.dataframe(loss_df.set_index("Stage"), use_container_width=True)

else:
    st.info("👈 Fill project parameters, then press **▶ Run Sizing**")
    with st.expander("ℹ️ Sizing methodology"):
        st.markdown("""
**Two constraints, max wins:**

| Constraint | Formula |
|-----------|---------|
| Power | `ceil(P_inv_needed / (inv_MVA × PCS_PF))`  |
| Energy | `ceil(POI_MWh / (units × batt_MWh × η_total))` |
| **Final** | **`max(power_blocks, energy_blocks)`** |

**Loss chain (POI ← losses ← DC battery):**
```
DC battery → η_charge → η_dc_cable → η_pcs → η_mvt
          → η_mv_cable → −Aux → η_mpt → η_transmission → POI
```
**Power flow:** 3-bus NR (Bus1=Grid swing, Bus2=POI, Bus3=PCS bus).
Sweep traces ISU/MVT S-circle to generate PV curve.
        """)