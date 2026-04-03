"""
BESS Sizing Tool v3 — correct augmentation logic
Augmentation strategy:
  - n_base = max(n_power, ceil(poi_mwh / (block_mwh * SOH[aug_year])))
  - All n_base+n_aug PCS installed at BOL for power/reactive compliance
  - Base blocks fully loaded; aug blocks partially loaded (aug_units_bol batteries)
  - At aug_year: fill aug blocks to full battery complement
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
.ok  {color:#3fb950;font-weight:700;}
.fail{color:#f85149;font-weight:700;}
.aug-box{background:#0f2c1a;border:1px solid #238636;border-radius:6px;
         padding:10px 14px;margin:6px 0;font-size:.85rem;}
.base-box{background:#0c1929;border:1px solid #1f6feb;border-radius:6px;
          padding:10px 14px;margin:6px 0;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

DEFAULT_SOH = [1.0000,0.9342,0.9115,0.8933,0.8775,0.8633,0.8502,
               0.8381,0.8266,0.8158,0.8054,0.7955,0.7859,0.7767,
               0.7677,0.7590,0.7506,0.7424,0.7343,0.7265,0.7188]

# ── 3-BUS NR POWER FLOW ──────────────────────────────────────────────────────
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
            rows.append({'P_inv_MW':P3*sbase,'Q_inv_MVAR':Q3*sbase,'V_POI':V[1],'V_ISU':V[2],'P_grid_MW':P1*sbase,'Q_grid_MVAR':Q1*sbase})
    return pd.DataFrame(rows)

# ── BESS SIZING ENGINE ────────────────────────────────────────────────────────
def size_bess(poi_mw, poi_mwh, target_pf,
              inv_mva, pcs_pf, groups, batt_mwh_dc, aux_kw_per_blk,
              eta_pcs, eta_mvt, eta_mv, eta_mpt, eta_tx,
              eta_dc, eta_chg, eta_aux,
              project_years, aug_year,
              soh_curve, cap_mvar=0.):
    """
    groups = list of (n_pcs, units_per_pcs) tuples.
    E.g. [(39,4),(5,2)] = 39 PCS×4 BESS + 5 PCS×2 BESS.
    The engine uses these exact counts for energy/power.
    Auto-sizing suggestion is also computed for comparison.
    """
    eta_all = eta_pcs*eta_mvt*eta_mv*eta_mpt*eta_tx*eta_dc*eta_chg*eta_aux
    soh = soh_curve + [soh_curve[-1]]*(max(project_years+2-len(soh_curve),0))

    # Base group config (group 1) drives power sizing
    base_units   = groups[0][1] if groups else 4
    blk_mwh_base = base_units * batt_mwh_dc * eta_all

    # ── AUTO-SIZING (suggestion) ──────────────────────────────────────────────
    n = 1
    for _ in range(25):
        aux_mw   = n * aux_kw_per_blk / 1000.
        p_mpt_lv = poi_mw / (eta_mpt * eta_tx)
        p_mv     = p_mpt_lv + aux_mw
        p_mvt    = p_mv / eta_mv
        p_inv_nd = p_mvt / eta_mvt
        n_new    = ceil(p_inv_nd / (inv_mva * pcs_pf))
        if n_new == n: break
        n = n_new
    n_power = n
    soh_aug  = soh[min(aug_year, len(soh)-1)] if aug_year > 0 else soh[min(project_years, len(soh)-1)]
    n_energy = ceil(poi_mwh / (blk_mwh_base * soh_aug))
    n_base_auto = max(n_power, n_energy)
    if aug_year > 0:
        e_at_aug  = n_base_auto * blk_mwh_base * soh_aug
        n_aug_auto= ceil(max(poi_mwh - e_at_aug, 0.) / blk_mwh_base)
    else:
        n_aug_auto = 0

    # ── ACTUAL CONFIG from user groups ────────────────────────────────────────
    groups_act  = [(n,u) for n,u in groups if n > 0] or [(1, base_units)]
    total_pcs   = sum(n for n,_ in groups_act)
    total_batt  = sum(n*u for n,u in groups_act)
    e_dc_bol    = total_batt * batt_mwh_dc
    e_poi_bol   = e_dc_bol * eta_all

    # Group 1 = base; rest = aug/secondary
    n_base   = groups_act[0][0]
    n_aug    = sum(n for n,_ in groups_act[1:])
    n_batt_g = [(n*u) for n,u in groups_act]   # batteries per group

    # ── Power cascade (total PCS) ─────────────────────────────────────────────
    n_pcs_total = total_pcs
    aux_mw_act  = n_pcs_total * aux_kw_per_blk / 1000.
    p_inv_act   = n_pcs_total * inv_mva * pcs_pf
    q_inv_act   = n_pcs_total * inv_mva * np.sqrt(max(1-pcs_pf**2,0))
    s_inv_act   = n_pcs_total * inv_mva
    p_mvt_out   = p_inv_act * eta_mvt
    q_mvt_out   = q_inv_act * eta_mvt
    p_mv_bus    = p_mvt_out * eta_mv
    q_mv_bus    = q_mvt_out * eta_mv
    p_mpt_in    = p_mv_bus - aux_mw_act
    q_mpt_in    = q_mv_bus + cap_mvar
    p_poi       = p_mpt_in * eta_mpt * eta_tx
    q_poi       = q_mpt_in * eta_mpt * eta_tx
    s_poi       = np.sqrt(p_poi**2+q_poi**2)
    pf_poi      = p_poi/s_poi if s_poi>0 else 1.
    q_poi_need  = poi_mw * np.tan(np.arccos(target_pf))
    s_mv_bus    = np.sqrt(p_mpt_in**2+q_mpt_in**2)
    min_mpt_mva = s_mv_bus

    # ── Degradation: each group degrades at same SOH curve ───────────────────
    # Group 1 (base) installed at BOL; group 2+ treated as aug (installed at aug_year)
    e_group_bol = [n*u*batt_mwh_dc*eta_all for n,u in groups_act]

    deg = []
    for yr in range(project_years+1):
        s_yr  = soh[min(yr, len(soh)-1)]
        e_tot = 0.
        aug_event = 0.
        for gi, eg in enumerate(e_group_bol):
            if gi == 0:
                # Base group: installed at BOL, degrades from yr 0
                e_tot += eg * s_yr
            else:
                # Secondary groups: installed at aug_year, degrade from aug_year
                if aug_year > 0 and yr >= aug_year:
                    s_rel  = soh[min(yr - aug_year, len(soh)-1)]
                    e_tot += eg * s_rel
                    if yr == aug_year:
                        aug_event += eg
        deg.append({'Year':yr, 'SOH (%)':round(s_yr*100,2),
                    'G1 E@POI (MWh)':round(e_group_bol[0]*s_yr,1),
                    'Aug E added (MWh)':round(aug_event,1),
                    'Total E@POI (MWh)':round(e_tot,1)})
    deg_df = pd.DataFrame(deg)
    # Rename base column
    deg_df = deg_df.rename(columns={'G1 E@POI (MWh)':'Base E@POI (MWh)'})

    # ── 3 Checks ─────────────────────────────────────────────────────────────
    q_poi_target   = poi_mw * np.tan(np.arccos(target_pf))
    q_actual_mvbus = n_pcs_total * inv_mva * np.sqrt(max(1-pcs_pf**2,0)) * eta_mvt * eta_mv + cap_mvar
    q_needed_mvbus = q_poi_target * (q_mpt_in/q_poi) if q_poi > 0.01 else q_poi_target/max(eta_mpt*eta_tx,0.01)

    overbuild_years = 0
    e_base_bol = e_group_bol[0]
    for _yr in range(project_years+1):
        if e_base_bol * soh[min(_yr, len(soh)-1)] >= poi_mwh:
            overbuild_years = _yr
    overbuild_pct = (e_poi_bol - poi_mwh) / poi_mwh * 100

    # ── Cascade dict for SLD ─────────────────────────────────────────────────
    def pf_(p,q): return p/np.sqrt(p**2+q**2) if np.sqrt(p**2+q**2)>0 else 1.
    cascade = {
        'Inverter\nOutput': {'P':p_inv_act,'Q':q_inv_act,'S':s_inv_act,'PF':pcs_pf},
        'MVT\nOutput':      {'P':p_mvt_out,'Q':q_mvt_out,'S':np.sqrt(p_mvt_out**2+q_mvt_out**2),'PF':pf_(p_mvt_out,q_mvt_out)},
        'MV Bus\n(−Aux)':   {'P':p_mpt_in, 'Q':q_mpt_in, 'S':s_mv_bus,'PF':pf_(p_mpt_in,q_mpt_in)},
        'POI':              {'P':p_poi,     'Q':q_poi,     'S':s_poi,   'PF':pf_poi},
    }

    return {
        # User groups
        'groups_act': groups_act,
        'n_batt_g':   n_batt_g,
        # Summary quantities
        'n_base':n_base,'n_aug':n_aug,'n_pcs':n_pcs_total,
        'n_batt_bol':total_batt, 'n_batt_aug': sum(n*u for n,u in groups_act[1:]),
        'base_units':base_units,
        # Auto-sizing suggestion
        'n_base_auto':n_base_auto,'n_aug_auto':n_aug_auto,
        'n_power':n_power,'n_energy':n_energy,
        # Power
        'p_poi':p_poi,'q_poi':q_poi,'s_inv':s_inv_act,
        'min_mpt_mva':min_mpt_mva,'aux_mw':aux_mw_act,
        'p_mv_bus':p_mv_bus,'q_mv_bus':q_mv_bus,
        'p_mpt_in':p_mpt_in,'q_mpt_in':q_mpt_in,
        'p_inv_nd':p_inv_nd, 'p_loss_pct':(1-eta_all)*100,
        # Energy
        'e_poi_bol':e_poi_bol,'e_dc_bol':e_dc_bol,
        # 3 Checks (Excel R29-R31)
        'check1_energy':   e_poi_bol >= poi_mwh * 0.999,
        'check2_power':    p_poi >= poi_mw * 0.999,
        'check3_reactive': q_actual_mvbus >= q_needed_mvbus * 0.999,
        'q_actual_mvbus':  q_actual_mvbus,
        'q_needed_mvbus':  q_needed_mvbus,
        'q_poi_target':    q_poi_target,
        'overbuild_pct':   overbuild_pct,
        'overbuild_years': overbuild_years,
        # Legacy
        'p_meets':p_poi>=poi_mw*.999,'e_meets':e_poi_bol>=poi_mwh*.999,
        'q_meets':q_actual_mvbus>=q_needed_mvbus*.999,'q_poi_need':q_poi_target,
        # Drivers
        'blk_mwh_base':blk_mwh_base,'eta_all':eta_all,'soh_aug':soh_aug,
        # Tables
        'cascade':cascade,'deg_df':deg_df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔋 BESS Sizing")

    st.markdown('<div class="sec">Project</div>', unsafe_allow_html=True)
    proj_name = st.text_input("", value="Demo BESS", label_visibility="collapsed")
    iso = st.selectbox("ISO", ["WECC","CAISO","ERCOT","PJM","MISO","NYISO","ISO-NE","Other"])

    st.markdown('<div class="sec">POI Requirements</div>', unsafe_allow_html=True)
    poi_mw   = st.number_input("Power @ POI (MW)",  value=200., min_value=1., step=10.)
    poi_mwh  = st.number_input("Energy @ POI (MWh)", value=800., min_value=1., step=50.)
    poi_lim  = st.number_input("Export Limit (MW)",  value=200., min_value=1., step=10.)
    proj_yrs = st.number_input("Project Term (yr)",  value=20,  min_value=1, max_value=40)

    st.markdown('<div class="sec">Augmentation</div>', unsafe_allow_html=True)
    aug_year = st.number_input("Augmentation Year (0=none)", value=5, min_value=0, max_value=40,
        help="Year when aug batteries are added. Base blocks sized to sustain until this year.")
    # aug blocks are fully new PCS+batteries added at aug_year (no partial pre-install)

    st.markdown('<div class="sec">Reactive Power</div>', unsafe_allow_html=True)
    pf_mode = st.radio("", ["Target PF","Target MVAR"], horizontal=True, label_visibility="collapsed")
    if pf_mode=="Target PF":
        target_pf = st.number_input("Target PF @ POI", value=0.95, min_value=0.5, max_value=1., step=.01)
        st.caption(f"Q = {poi_mw*np.tan(np.arccos(target_pf)):.1f} MVAR")
    else:
        tmvar = st.number_input("Target MVAR @ POI", value=65.7, min_value=0., step=1.)
        target_pf = poi_mw/np.sqrt(poi_mw**2+tmvar**2) if tmvar else 1.
        st.caption(f"PF = {target_pf:.3f}")
    cap_bank = st.number_input("Capacitor Bank (MVAR)", value=0., min_value=0., step=5.)

    st.markdown('<div class="sec">Grid / MPT</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    n_mpt=c1.number_input("#MPTs",   value=1,     min_value=1)
    s_mpt=c2.number_input("MVA/MPT", value=240.,  min_value=10., step=10.)
    c3,c4 = st.columns(2)
    z_mpt =c3.number_input("Z(pu)",  value=0.10, min_value=.01, step=.005, format="%.3f")
    xr_mpt=c4.number_input("X/R",    value=40.,  min_value=1.)
    eta_mpt=st.number_input("MPT η", value=0.995, min_value=.9, max_value=1., step=.001, format="%.3f")

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

    st.markdown('<div class="sec">PCS / Inverter</div>', unsafe_allow_html=True)
    pcs_model=st.text_input("PCS Model", value="EPC POWER M10")
    c7,c8=st.columns(2)
    inv_mva=c7.number_input("Inv MVA",  value=5.3, min_value=.1, step=.1)
    mvt_mva=c8.number_input("MVT MVA",  value=5.3, min_value=.1, step=.1)
    pcs_pf =st.number_input("PCS Operating PF", value=0.90, min_value=.5, max_value=1., step=.01,
        help="Sets the P/Q split per block. Drives the power constraint.")
    eta_pcs=st.number_input("PCS η", value=0.985, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">Battery Unit</div>', unsafe_allow_html=True)
    batt_model = st.text_input("Battery Model", value="BESS Unit")
    batt_mwh   = st.number_input("MWh / unit (DC)", value=6.138, min_value=.1, step=.1, format="%.3f")
    aux_kw     = st.number_input("Aux load / block (kW)", value=75.6, min_value=0., step=1.)

    st.markdown('<div class="sec">PCS → Battery Groups</div>', unsafe_allow_html=True)
    st.caption("Define 1–3 groups. Each group: N PCS blocks, each with K batteries.")
    st.markdown('<div style="font-size:.72rem;color:#8b949e;margin-bottom:6px">Example: 39×4 + 5×2</div>', unsafe_allow_html=True)

    n_groups = int(st.number_input("Number of groups", value=1, min_value=1, max_value=3, step=1))
    group_colors = ["#1f6feb","#238636","#9e6a03"]
    group_labels = ["Base","Aug/Partial","Extra"]
    group_defaults_n = [44, 0, 0]
    group_defaults_u = [4, 2, 4]

    groups = []
    for g in range(n_groups):
        st.markdown(
            f'<div style="border-left:3px solid {group_colors[g]};padding:2px 0 2px 8px;'
            f'margin:4px 0;font-size:.78rem;color:#c9d1d9"><b>Group {g+1} — {group_labels[g]}</b></div>',
            unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        gn = gc1.number_input(f"# PCS", value=group_defaults_n[g], min_value=0, key=f"gn{g}",
                               label_visibility="visible")
        gu = gc2.number_input(f"BESS/PCS", value=group_defaults_u[g], min_value=0, max_value=20, key=f"gu{g}",
                               label_visibility="visible")
        groups.append((int(gn), int(gu)))

    # Always ensure at least one group with PCS > 0
    groups_active = [(n,u) for n,u in groups if n > 0]
    if not groups_active:
        groups_active = [(1, 4)]

    total_pcs  = sum(n for n,_ in groups_active)
    total_batt = sum(n*u for n,u in groups_active)
    total_dc   = total_batt * batt_mwh
    st.caption(f"PCS: **{total_pcs}** | Batteries: **{total_batt}** | DC: **{total_dc:.1f} MWh**")

    # base_units = batteries per PCS in group 1 (used by power sizing engine)
    base_units = groups_active[0][1]

    st.markdown('<div class="sec">MVT/ISU</div>', unsafe_allow_html=True)
    c11,c12=st.columns(2)
    z_isu =c11.number_input("Z(pu)",  value=0.08, min_value=.01, step=.005, format="%.3f")
    xr_isu=c12.number_input("X/R",    value=8.83, min_value=1.)
    eta_mvt=st.number_input("MVT η",  value=0.990, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">Loss Factors</div>', unsafe_allow_html=True)
    with st.expander("AC losses"):
        eta_mv  =st.number_input("MV Cable",    value=.995, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_tx  =st.number_input("Transmission",value=.990, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_aux =st.number_input("Auxiliary",   value=.998, min_value=.9, max_value=1., step=.001, format="%.3f")
    with st.expander("DC losses"):
        eta_dc  =st.number_input("DC Cable",       value=.999, min_value=.9, max_value=1., step=.001, format="%.3f")
        eta_chg =st.number_input("Charge/Disch",   value=.955, min_value=.8, max_value=1., step=.001, format="%.3f")

    st.markdown('<div class="sec">SOH Degradation Curve</div>', unsafe_allow_html=True)
    with st.expander("Edit SOH by year"):
        soh_vals=[]
        for yr in range(min(int(proj_yrs)+1,21)):
            dv=DEFAULT_SOH[yr] if yr<len(DEFAULT_SOH) else DEFAULT_SOH[-1]
            r1,r2=st.columns([1,2])
            r1.markdown(f"<div style='padding-top:8px;font-size:.82rem'>Yr {yr}</div>",unsafe_allow_html=True)
            v=r2.number_input(f"s{yr}",value=float(round(dv*100,2)),min_value=0.,max_value=100.,
                              step=.1,label_visibility="collapsed",key=f"soh{yr}")
            soh_vals.append(v/100.)
    soh_curve=soh_vals

    st.divider()
    run_btn=st.button("▶  Run Sizing", type="primary", use_container_width=True)
    v_min_pf=st.number_input("V min (pu)",value=.95,min_value=.8,max_value=1.,step=.01)
    v_max_pf=st.number_input("V max (pu)",value=1.05,min_value=1.,max_value=1.15,step=.01)
    v_calc  =st.number_input("V calc(pu)",value=1.00,min_value=.8,max_value=1.15,step=.01)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"# 🔋 BESS Sizing Tool")
st.caption(f"**{proj_name}** · {iso} · {poi_mw:.0f} MW / {poi_mwh:.0f} MWh @ POI")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Sizing…"):
        res = size_bess(
            poi_mw, poi_mwh, target_pf,
            inv_mva, pcs_pf, groups_active, batt_mwh, aux_kw,
            eta_pcs, eta_mvt, eta_mv, eta_mpt, eta_tx,
            eta_dc, eta_chg, eta_aux,
            int(proj_yrs), int(aug_year),
            soh_curve, float(cap_bank)
        )

    # ── Equipment quantities ─────────────────────────────────────────────────
    st.subheader("📦 Equipment Quantities")

    # Group breakdown boxes
    group_colors = ["#1f6feb","#238636","#9e6a03"]
    group_labels = ["Base Build","Aug / Partial","Extra"]
    for gi, (gn, gu) in enumerate(res['groups_act']):
        gbatts = res['n_batt_g'][gi]
        ge_dc  = gbatts * batt_mwh
        ge_poi = ge_dc * res['eta_all']
        tag    = group_labels[gi] if gi < len(group_labels) else f"Group {gi+1}"
        install_note = "Day 1" if gi == 0 else (f"Year {int(aug_year)}" if int(aug_year) > 0 else "Day 1")
        bg    = ["#0c1929","#0f2c1a","#2c2000"][gi % 3]
        brd   = group_colors[gi % len(group_colors)]
        html  = (
            f'<div style="background:{bg};border:1px solid {brd};border-radius:6px;'
            f'padding:10px 14px;margin:4px 0;font-size:.85rem">'
            f'<b>{'🏗' if gi==0 else '🔄'} Group {gi+1} — {tag}</b> &nbsp;'
            f'<span style="font-size:.72rem;color:#8b949e">({install_note})</span><br>'
            f'<b>{gn}</b> PCS × <b>{gu}</b> batteries = <b>{gbatts}</b> units | '
            f'DC: <b>{ge_dc:.1f}</b> MWh | E@POI: <b>{ge_poi:.1f}</b> MWh'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    # Auto-sizing suggestion
    if res['n_base_auto'] != res['n_base'] or res['n_aug_auto'] != res['n_aug']:
        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #553c9a;border-radius:6px;'
            f'padding:8px 12px;margin:4px 0;font-size:.80rem;color:#c9d1d9">'
            f'💡 <b>Auto-sizing suggests:</b> {res["n_base_auto"]} base PCS × {base_units} batteries'
            f'{f" + {res["n_aug_auto"]} aug PCS" if res["n_aug_auto"]>0 else ""}'
            f'</div>',
            unsafe_allow_html=True
        )

    q1,q2,q3,q4,q5 = st.columns(5)
    q1.metric("Total PCS (BOL)", str(res['n_pcs']),
              help=f"n_base={res['n_base']} + n_aug={res['n_aug']}")
    q2.metric("Batteries @ BOL", str(res['n_batt_bol']))
    q3.metric("Batteries @ Aug", str(res['n_batt_aug']),
              delta=f"Year {aug_year}" if aug_year>0 else "None")
    q4.metric("Min MPT MVA", f"{res['min_mpt_mva']:.1f}",
              delta=f"{'OK' if n_mpt*s_mpt>=res['min_mpt_mva'] else 'UNDERSIZE'}")
    q5.metric("BOL overbuild", f"{res['overbuild_pct']:+.1f}%")

    st.divider()

    # ── Checks ──────────────────────────────────────────────────────────────
    st.subheader("✅ Sizing Checks")

    def check_card(col, num, title, ok, actual, required, note=""):
        bg  = "#0d3321" if ok else "#3d1a1a"
        brd = "#238636" if ok else "#b62324"
        tc  = "#3fb950" if ok else "#f85149"
        sym = "✓" if ok else "✗"
        nh  = f'<div style="font-size:.70rem;color:#8b949e;margin-top:3px">{note}</div>' if note else ""
        html = (
            f'<div style="background:{bg};border:1px solid {brd};border-radius:6px;padding:10px 12px;margin-bottom:4px">'
            f'<div style="font-family:monospace;font-size:.65rem;color:#8b949e;letter-spacing:.1em">CHECK {num}</div>'
            f'<div style="font-size:.92rem;font-weight:700;color:{tc}">{sym} {title}</div>'
            f'<div style="font-size:.80rem;color:#c9d1d9">Actual: <b>{actual}</b></div>'
            f'<div style="font-size:.80rem;color:#c9d1d9">Required: <b>{required}</b></div>'
            f'{nh}</div>'
        )
        col.markdown(html, unsafe_allow_html=True)

    bc = st.columns(4)
    check_card(bc[0], "2", "Active Power (R30)",
        ok=res['check2_power'],
        actual=f"{res['p_poi']:.2f} MW @ POI",
        required=f"{poi_mw:.1f} MW",
        note=f"Loss chain: {res['p_loss_pct']:.1f}%")

    check_card(bc[1], "1", "Energy / Overbuild (R29)",
        ok=res['check1_energy'],
        actual=f"{res['e_poi_bol']:.1f} MWh @ BOL",
        required=f"{poi_mwh:.0f} MWh",
        note=f"Overbuild: {res['overbuild_pct']:+.1f}%  |  Yrs above target: {res['overbuild_years']}")

    check_card(bc[2], "3", "Reactive FERC 827 (R31)",
        ok=res['check3_reactive'],
        actual=f"{res['q_actual_mvbus']:.2f} MVAR @ MV bus",
        required=f"{res['q_needed_mvbus']:.2f} MVAR @ MV bus",
        note=f"POI target: {res['q_poi_target']:.2f} MVAR  |  PF {target_pf:.2f}")

    mpt_ok = n_mpt*s_mpt >= res['min_mpt_mva']
    check_card(bc[3], "—", "MPT Adequate",
        ok=mpt_ok,
        actual=f"{n_mpt*s_mpt:.0f} MVA specified",
        required=f"{res['min_mpt_mva']:.1f} MVA min",
        note=f"Margin: {n_mpt*s_mpt - res['min_mpt_mva']:+.1f} MVA")

    st.divider()

    # ── Single Line Diagram ──────────────────────────────────────────────────
    st.subheader("📐 Power Flow Cascade")

    cas = res['cascade']
    labels = [k.replace('\n',' ') for k in cas]
    fig_sld = go.Figure()
    fig_sld.update_layout(
        height=300, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=5,r=5,t=30,b=5),
        xaxis=dict(visible=False,range=[-0.6,4.6]),
        yaxis=dict(visible=False,range=[-0.15,1.55]),
        showlegend=False, title="Inverter → MVT → MV Bus → POI"
    )
    for i,(stage,v) in enumerate(cas.items()):
        x=i; pf_col='#16a34a' if v['PF']>=target_pf else '#dc2626'
        fig_sld.add_shape(type='rect',x0=x-.38,x1=x+.38,y0=.52,y1=.62,
                          fillcolor='#1e40af',line_color='#1e40af')
        lbl=stage.replace('\n',' ')
        fig_sld.add_annotation(x=x,y=1.48,text=f"<b>{lbl}</b>",showarrow=False,font=dict(size=11,color='#1e293b'))
        fig_sld.add_annotation(x=x,y=1.28,text=f"P = {v['P']:.1f} MW",showarrow=False,font=dict(size=10,color='#1e40af'))
        fig_sld.add_annotation(x=x,y=1.10,text=f"Q = {v['Q']:.1f} MVAR",showarrow=False,font=dict(size=10,color='#6d28d9'))
        fig_sld.add_annotation(x=x,y=0.92,text=f"S = {v['S']:.1f} MVA",showarrow=False,font=dict(size=10,color='#0f766e'))
        fig_sld.add_annotation(x=x,y=0.32,text=f"PF = {v['PF']:.3f}",showarrow=False,font=dict(size=10,color=pf_col))
        if i<3:
            fig_sld.add_annotation(x=x+.55,y=.57,text="→",showarrow=False,font=dict(size=20,color='#94a3b8'))
    # Aux annotation
    fig_sld.add_annotation(x=2,y=0.12,text=f"↑ Aux {res['aux_mw']:.2f} MW subtracted",
                           showarrow=False,font=dict(size=9,color='#b45309'))
    # POI target
    fig_sld.add_annotation(x=3.5,y=0.12,
        text=f"Target: {poi_mw:.0f} MW / {res['q_poi_need']:.1f} MVAR / PF {target_pf:.2f}",
        showarrow=False,font=dict(size=9,color='#374151'))
    st.plotly_chart(fig_sld,use_container_width=True)

    # Cascade table
    cdf = pd.DataFrame([
        {'Stage':k.replace('\n',' '),'P (MW)':round(v['P'],2),
         'Q (MVAR)':round(v['Q'],2),'S (MVA)':round(v['S'],2),'PF':round(v['PF'],4)}
        for k,v in cas.items()
    ])
    cdf.loc[len(cdf)] = {'Stage':'▶ POI Target','P (MW)':poi_mw,
        'Q (MVAR)':round(res['q_poi_need'],2),'S (MVA)':round(poi_mw/target_pf,2),'PF':target_pf}
    st.dataframe(cdf.set_index('Stage'),use_container_width=True)

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs(["📈 PV Curve","📉 Degradation","📋 Equipment","🔢 Sizing Logic"])

    with t1:
        with st.spinner("Running power flow sweep…"):
            isu_s = res['n_pcs'] * mvt_mva
            df_pf = run_pf_sweep(
                float(poi_lim), float(n_mpt*s_mpt), z_mpt, xr_mpt, 10.*n_mpt,
                isu_s, z_isu, xr_isu, 8.*res['n_pcs'],
                res['n_pcs']*inv_mva*pcs_pf/poi_lim,
                tap_c, float(cap_bank), float(v_calc)
            )
        if len(df_pf):
            dl=df_pf[df_pf['Q_inv_MVAR']>=0].sort_values('P_inv_MW')
            dg=df_pf[df_pf['Q_inv_MVAR']< 0].sort_values('P_inv_MW')
            fig=go.Figure()
            fig.add_hrect(y0=v_min_pf,y1=v_max_pf,fillcolor='lightgreen',opacity=.10,line_width=0)
            fig.add_hline(y=v_min_pf,line_dash='dash',line_color='red',line_width=1.5)
            fig.add_hline(y=v_max_pf,line_dash='dash',line_color='red',line_width=1.5)
            fig.add_vline(x=poi_mw,line_dash='dot',line_color='gray',annotation_text=f"{poi_mw:.0f} MW")
            if len(dl): fig.add_trace(go.Scatter(x=dl['P_inv_MW'],y=dl['V_POI'],mode='lines',name='Leading Q',line=dict(color='royalblue',width=2.5)))
            if len(dg): fig.add_trace(go.Scatter(x=dg['P_inv_MW'],y=dg['V_POI'],mode='lines',name='Lagging Q', line=dict(color='darkorange',width=2.5)))
            y_lo=max(.80,df_pf['V_POI'].min()-.03); y_hi=min(1.25,df_pf['V_POI'].max()+.03)
            fig.update_layout(title="PV Curve — POI Voltage vs Active Power",
                xaxis_title="P_inv (MW)",yaxis_title="V_POI (pu)",
                yaxis=dict(range=[y_lo,y_hi]),height=420,legend=dict(x=.01,y=.99))
            st.plotly_chart(fig,use_container_width=True)
            v_nom=df_pf.loc[(df_pf['P_inv_MW']-poi_mw).abs().idxmin(),'V_POI']
            p1,p2,p3=st.columns(3)
            p1.metric("V@rated P",f"{v_nom:.4f} pu")
            p2.metric("% in V-band",f"{((df_pf['V_POI']>=v_min_pf)&(df_pf['V_POI']<=v_max_pf)).mean()*100:.1f}%")
            p3.metric("Converged",f"{len(df_pf)}/121")
        else:
            st.warning("Power flow did not converge. Check tap position.")

    with t2:
        fig2=go.Figure()
        fig2.add_hline(y=poi_mwh,line_dash='dash',line_color='red',annotation_text=f"Target {poi_mwh:.0f} MWh")
        fig2.add_trace(go.Scatter(x=res['deg_df']['Year'],y=res['deg_df']['Total E@POI (MWh)'],
            mode='lines+markers',name='Total E@POI',line=dict(color='royalblue',width=2.5),marker=dict(size=6)))
        fig2.add_trace(go.Scatter(x=res['deg_df']['Year'],y=res['deg_df']['Base E@POI (MWh)'],
            mode='lines',name='Base only',line=dict(color='lightblue',width=1.5,dash='dot')))
        aug_rows=res['deg_df'][res['deg_df']['Aug E added (MWh)']>0]
        if len(aug_rows):
            fig2.add_trace(go.Bar(x=aug_rows['Year'],y=aug_rows['Aug E added (MWh)'],
                name='Augmentation added',marker_color='#16a34a',opacity=.7))
        fig2.update_layout(title="Energy @ POI over Project Life",
            xaxis_title="Year",yaxis_title="MWh",height=380)
        st.plotly_chart(fig2,use_container_width=True)
        st.dataframe(res['deg_df'].set_index('Year'),use_container_width=True)
        st.download_button("⬇ Degradation CSV",res['deg_df'].to_csv(index=False),"degradation.csv","text/csv")

    with t3:
        glabels = ["Base","Aug/Partial","Extra"]
        rows = []
        for gi, (gn, gu) in enumerate(res['groups_act']):
            lbl = glabels[gi] if gi < len(glabels) else f"Group {gi+1}"
            install = "Day 1" if gi==0 else (f"Year {int(aug_year)}" if int(aug_year)>0 else "Day 1")
            rows.append({
                "Group": f"G{gi+1} — {lbl} ({install})",
                "PCS qty": gn,
                "BESS/PCS": gu,
                "Total BESS": gn*gu,
                "DC MWh": round(gn*gu*batt_mwh, 1),
                "E@POI MWh": round(gn*gu*batt_mwh*res['eta_all'], 1),
            })
        rows += [
            {"Group":"MVT","PCS qty":res['n_pcs'],"BESS/PCS":"—","Total BESS":"—",
             "DC MWh":"—","E@POI MWh":f"{res['n_pcs']*mvt_mva:.0f} MVA total"},
            {"Group":"MPT","PCS qty":n_mpt,"BESS/PCS":"—","Total BESS":"—",
             "DC MWh":"—","E@POI MWh":f"{n_mpt*s_mpt:.0f} MVA (min {res['min_mpt_mva']:.1f})"},
        ]
        df_eq = pd.DataFrame(rows)
        st.dataframe(df_eq.set_index("Group"), use_container_width=True)
        st.download_button("⬇ Equipment CSV", df_eq.to_csv(index=False), "equipment.csv","text/csv")

    with t4:
        driver = "POWER" if res['n_power']>=res['n_energy'] else "ENERGY"
        soh_lbl = f"SOH[{int(aug_year)}]={res['soh_aug']:.4f}" if int(aug_year)>0 else f"SOH[{int(proj_yrs)}]={res['soh_aug']:.4f}"
        st.markdown("#### Auto-Sizing Suggestion")
        st.info(
            f"Driver: **{driver}** | "
            f"n_power={res['n_power']}, n_energy={res['n_energy']} ({soh_lbl}) → "
            f"**n_base_auto={res['n_base_auto']}** + n_aug_auto={res['n_aug_auto']}\n\n"
            f"Your config: {' + '.join(f'{n}×{u}batt' for n,u in res['groups_act'])} "
            f"= **{res['n_pcs']} PCS, {res['n_batt_bol']} batteries**"
        )
        st.markdown("#### User Groups vs Auto")
        gdf_rows = []
        for gi,(gn,gu) in enumerate(res['groups_act']):
            gdf_rows.append({'Group':f'G{gi+1}','PCS':gn,'BESS/PCS':gu,
                'DC MWh':round(gn*gu*batt_mwh,1),'E@POI MWh':round(gn*gu*batt_mwh*res['eta_all'],1)})
        gdf_rows.append({'Group':'TOTAL','PCS':res['n_pcs'],'BESS/PCS':'—',
            'DC MWh':round(res['e_dc_bol'],1),'E@POI MWh':round(res['e_poi_bol'],1)})
        st.dataframe(pd.DataFrame(gdf_rows).set_index('Group'),use_container_width=True)

        loss_df=pd.DataFrame([
            {"Stage":"Transmission","η":eta_tx},{"Stage":"MPT","η":eta_mpt},
            {"Stage":"MV Cable","η":eta_mv},{"Stage":"MVT","η":eta_mvt},
            {"Stage":"PCS","η":eta_pcs},{"Stage":"DC Cable","η":eta_dc},
            {"Stage":"Charge/Disch","η":eta_chg},{"Stage":"Auxiliary","η":eta_aux},
            {"Stage":"→ TOTAL","η":round(res['eta_all'],5)},
        ])
        st.dataframe(loss_df.set_index("Stage"),use_container_width=True)

else:
    st.info("👈 Set parameters in the sidebar and press **▶ Run Sizing**")
    with st.expander("ℹ️ Augmentation logic"):
        st.markdown("""
**With augmentation (aug_year > 0):**

1. **n_power** = ceil(P_inv_needed / (inv_MVA × PCS_PF))  — from power delivery requirement
2. **n_energy** = ceil(POI_MWh / (block_MWh × SOH[aug_year]))  — base must last *until* aug_year
3. **n_base** = max(n_power, n_energy)
4. **n_aug** = ceil(gap_at_aug_year / (aug_units × batt_MWh × η))
5. All n_base+n_aug **PCS installed at BOL** (for power/reactive compliance)
6. Aug blocks carry `aug_units_bol` batteries at BOL, filled to full at aug_year

**Example: 200MW / 800MWh, aug_year=5, PCS_PF=0.90, 5.3 MVA, 6.138 MWh/unit, 4 units/block:**
- n_power = 44, n_energy = ceil(800/22.34/0.8633) = 42 → n_base = 44
- E@aug_yr5 = 44×22.34×0.8633 = 849 MWh > 800 → n_aug = 0
- Result: 44 PCS × 4 batteries = 176 units

**For 39+5 split:** use aug_year=5 with 2 batteries in aug blocks, fewer base blocks
        """)