import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="RRC Loss Report Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }
  .stApp { background-color: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  .rrc-header {
      background: linear-gradient(90deg, #161b22 0%, #1c2128 100%);
      border-bottom: 2px solid #e85d04;
      padding: 18px 32px; margin: -1rem -1rem 2rem -1rem;
      display: flex; align-items: center; gap: 16px;
  }
  .rrc-header-subtitle { font-size: 0.85rem; color: #8b949e; margin-left: auto; text-transform: uppercase; letter-spacing: 1px; }
  .rrc-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 24px; margin-bottom: 20px; }
  .rrc-card-header { font-size: 0.7rem; font-weight: 600; color: #e85d04; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
  .rrc-card-title { font-size: 1.15rem; font-weight: 600; color: #ffffff; margin-bottom: 16px; }
  .metric-row { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .metric-card { flex: 1; min-width: 120px; background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-card.highlight { border-color: #e85d04; }
  .metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
  .metric-unit { font-size: 0.8rem; color: #8b949e; margin-left: 3px; }
  .metric-card.orange .metric-value { color: #e85d04; }
  .metric-card.green .metric-value  { color: #3fb950; }
  .metric-card.blue .metric-value   { color: #58a6ff; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
  .pill-waiting { background: #21262d; color: #8b949e; }
  .pill-ready   { background: #1a3a2a; color: #3fb950; }
  [data-testid="stFileUploader"] { background: #0d1117 !important; border: 2px dashed #30363d !important; border-radius: 8px !important; }
  .stButton > button { background: #e85d04 !important; color: white !important; border: none !important; border-radius: 6px !important; font-weight: 600 !important; }
  .stButton > button:hover { background: #c94d00 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #21262d; }
  .stTabs [data-baseweb="tab"] { color: #8b949e !important; }
  .stTabs [aria-selected="true"] { color: #e85d04 !important; border-bottom: 2px solid #e85d04 !important; }
  .progress-bar-wrap { background: #21262d; border-radius: 4px; height: 6px; margin-top: 8px; }
  .progress-bar-fill { background: #e85d04; border-radius: 4px; height: 6px; }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in [("etap_uploaded", False), ("sld_uploaded", False), ("page", "Upload")]:
    if k not in st.session_state:
        st.session_state[k] = v

REQUIRED_SHEETS = ["General", "Bus", "Branch", "Load", "Source"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 24px 0; text-align:center;'>
        <div style='display:flex; align-items:center; justify-content:center; gap:4px;'>
            <span style='font-size:1.5rem; font-weight:900; color:#5a6272; letter-spacing:1px;'>RRC</span>
            <svg width="32" height="12" viewBox="0 0 38 14" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:-2px;">
              <path d="M0 10 Q10 2 20 7 Q30 12 38 4" stroke="#4a90c4" stroke-width="3.5" fill="none" stroke-linecap="round"/>
              <path d="M0 13 Q10 6 20 10 Q30 14 38 8" stroke="#e85d04" stroke-width="3.5" fill="none" stroke-linecap="round"/>
            </svg>
        </div>
        <div style='font-size:0.7rem; color:#8b949e; letter-spacing:1.5px; margin-top:4px;'>LOSS REPORT CALCULATOR</div>
    </div>
    <hr style='border-color:#21262d; margin-bottom:20px;'>
    """, unsafe_allow_html=True)

    pages = [
        ("📁", "Upload",     "Upload Files"),
        ("🔌", "LV Losses",  "LV Loss Breakdown"),
        ("⚡", "MV Losses",  "MV Loss Breakdown"),
        ("🔧", "Aux Losses", "Auxiliary Losses"),
        ("📊", "Summary",    "Total Loss Summary"),
    ]
    for icon, key, label in pages:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key

    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>File Status</div>", unsafe_allow_html=True)
    etap_pill = "<span class='pill pill-ready'>&#10003; Loaded</span>" if st.session_state.etap_uploaded else "<span class='pill pill-waiting'>Waiting</span>"
    sld_pill  = "<span class='pill pill-ready'>&#10003; Loaded</span>" if st.session_state.sld_uploaded  else "<span class='pill pill-waiting'>Waiting</span>"
    st.markdown(f"""
    <div style='font-size:0.82rem; margin-bottom:8px; display:flex; justify-content:space-between;'><span>ETAP File</span>{etap_pill}</div>
    <div style='font-size:0.82rem; display:flex; justify-content:space-between;'><span>SLD File</span>{sld_pill}</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; color:#8b949e; text-align:center;'>© 2026 RRC Power &amp; Energy</div>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='rrc-header'>
    <div style='display:flex; align-items:center; gap:4px;'>
        <span style='font-size:1.8rem; font-weight:900; color:#5a6272; letter-spacing:1px;'>RRC</span>
        <svg width="38" height="14" viewBox="0 0 38 14" xmlns="http://www.w3.org/2000/svg" style="margin-left:2px; margin-bottom:-2px;">
          <path d="M0 10 Q10 2 20 7 Q30 12 38 4" stroke="#4a90c4" stroke-width="3.5" fill="none" stroke-linecap="round"/>
          <path d="M0 13 Q10 6 20 10 Q30 14 38 8" stroke="#e85d04" stroke-width="3.5" fill="none" stroke-linecap="round"/>
        </svg>
    </div>
    <div style='font-size:1rem; color:#c9d1d9; font-weight:500; margin-left:8px;'>Loss Report Calculator</div>
    <div class='rrc-header-subtitle'>ETAP Loss Analyzer</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Upload":

    st.markdown("### Step 1 — Upload Project Files")
    st.markdown("<p style='color:#8b949e; margin-bottom:24px;'>Upload your ETAP losses file and Single Line Diagram to begin the loss analysis.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class='rrc-card'>
            <div class='rrc-card-header'>Step 01</div>
            <div class='rrc-card-title'>&#9881; ETAP Losses File</div>
        </div>
        """, unsafe_allow_html=True)

        etap_file = st.file_uploader(
            "Upload ETAP Excel file (.xlsx / .xls)",
            type=["xlsx", "xls"],
            key="etap_uploader",
            help="Excel file must contain sheets: General, Bus, Branch, Load, Source"
        )

        if etap_file:
            try:
                xl = pd.ExcelFile(etap_file)
                found = xl.sheet_names
                missing = [s for s in REQUIRED_SHEETS if s not in found]

                if missing:
                    st.session_state.etap_uploaded = False
                    st.error(f"Missing sheet(s): **{', '.join(missing)}** — please re-upload the correct file.")
                    st.markdown(
                        "<div style='background:#1a0a0a; border:1px solid #f85149; border-radius:6px; padding:12px; margin-top:8px; font-size:0.82rem; color:#8b949e;'>"
                        f"&#128196; File: <span style='color:#e6edf3;'>{etap_file.name}</span><br>"
                        f"&#128230; Size: <span style='color:#e6edf3;'>{etap_file.size / 1024:.1f} KB</span><br>"
                        f"&#128203; Sheets found: <span style='color:#c9d1d9;'>{' | '.join(found) if found else 'None'}</span><br>"
                        f"&#9888; Missing: <span style='color:#f85149; font-weight:600;'>{' | '.join(missing)}</span><br><br>"
                        "<span style='color:#c9d1d9;'>Required sheets: </span>"
                        "<span style='color:#3fb950;'>General &nbsp;·&nbsp; Bus &nbsp;·&nbsp; Branch &nbsp;·&nbsp; Load &nbsp;·&nbsp; Source</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.etap_uploaded = True
                    st.success(f"All required sheets found in **{etap_file.name}**")
                    st.markdown(
                        "<div style='background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:12px; margin-top:8px; font-size:0.82rem; color:#8b949e;'>"
                        f"&#128196; File: <span style='color:#e6edf3;'>{etap_file.name}</span><br>"
                        f"&#128230; Size: <span style='color:#e6edf3;'>{etap_file.size / 1024:.1f} KB</span><br>"
                        "&#128203; Sheets validated: <span style='color:#3fb950;'>General &nbsp;·&nbsp; Bus &nbsp;·&nbsp; Branch &nbsp;·&nbsp; Load &nbsp;·&nbsp; Source</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.session_state.etap_uploaded = False
                st.error(f"Could not read file: {e}. Please upload a valid Excel (.xlsx) file.")
        else:
            st.markdown(
                "<div style='font-size:0.82rem; color:#8b949e; margin-top:8px; padding:12px; background:#0d1117; border-radius:6px; border:1px solid #21262d;'>"
                "&#128203; <b style='color:#c9d1d9;'>Required sheets:</b><br>"
                "&nbsp;&nbsp;· General<br>&nbsp;&nbsp;· Bus<br>&nbsp;&nbsp;· Branch<br>&nbsp;&nbsp;· Load<br>&nbsp;&nbsp;· Source"
                "</div>",
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("""
        <div class='rrc-card'>
            <div class='rrc-card-header'>Step 02</div>
            <div class='rrc-card-title'>&#128208; Single Line Diagram (SLD)</div>
        </div>
        """, unsafe_allow_html=True)

        sld_file = st.file_uploader(
            "Upload SLD file (.xlsx or .pdf)",
            type=["xlsx", "xls", "pdf"],
            key="sld_uploader",
            help="SLD with standardized naming for automatic component identification"
        )
        if sld_file:
            st.session_state.sld_uploaded = True
            st.success(f"**{sld_file.name}** uploaded successfully!")
            st.markdown(
                "<div style='background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:12px; margin-top:8px; font-size:0.82rem; color:#8b949e;'>"
                f"&#128196; File: <span style='color:#e6edf3;'>{sld_file.name}</span><br>"
                f"&#128230; Size: <span style='color:#e6edf3;'>{sld_file.size / 1024:.1f} KB</span><br>"
                "&#128269; Detects: <span style='color:#58a6ff;'>Inverter &nbsp;·&nbsp; Transformer &nbsp;·&nbsp; MV Cable &nbsp;·&nbsp; MPT &nbsp;·&nbsp; Transmission Line</span>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:0.82rem; color:#8b949e; margin-top:8px; padding:12px; background:#0d1117; border-radius:6px; border:1px solid #21262d;'>"
                "&#128269; <b style='color:#c9d1d9;'>Auto-detected via naming convention:</b><br>"
                "&nbsp;&nbsp;· Inverters — INV_xxx<br>"
                "&nbsp;&nbsp;· Transformers — TX_xxx<br>"
                "&nbsp;&nbsp;· Main Power TX — MPT_xxx<br>"
                "&nbsp;&nbsp;· MV Cables — MVCAB_xxx<br>"
                "&nbsp;&nbsp;· Transmission Lines — TL_xxx<br>"
                "&nbsp;&nbsp;· Auxiliary Loads — AUX_xxx"
                "</div>",
                unsafe_allow_html=True
            )

    # Progress
    st.markdown("<hr style='border-color:#21262d; margin:28px 0 20px 0;'>", unsafe_allow_html=True)
    files_done = int(st.session_state.etap_uploaded) + int(st.session_state.sld_uploaded)
    pct = files_done * 50
    st.markdown(
        f"<div style='margin-bottom:8px; display:flex; justify-content:space-between; font-size:0.82rem;'>"
        f"<span style='color:#8b949e;'>Upload Progress</span>"
        f"<span style='color:#e85d04; font-weight:600;'>{files_done}/2 files</span></div>"
        f"<div class='progress-bar-wrap'><div class='progress-bar-fill' style='width:{pct}%;'></div></div>",
        unsafe_allow_html=True
    )

    if files_done == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Proceed to Loss Analysis"):
            st.session_state.page = "LV Losses"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SHARED LOSS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def loss_page(step, title, subtitle, sheet_num, components):
    st.markdown(f"### Step {step} — {title}")
    st.markdown(f"<p style='color:#8b949e; margin-bottom:24px;'>{subtitle}</p>", unsafe_allow_html=True)
    if not st.session_state.etap_uploaded:
        st.warning("Please upload and validate the ETAP file first (Step 1 — Upload).")
        return
    st.markdown(f"<div class='rrc-card'><div class='rrc-card-header'>Components Detected</div><div class='rrc-card-title'>{title}</div></div>", unsafe_allow_html=True)
    st.info(f"Data will be parsed automatically from the '{['','General','Bus','Branch','Load','Source'][sheet_num] if sheet_num <= 5 else sheet_num}' sheet of your ETAP file.")
    st.markdown("""
    <div class='metric-row'>
        <div class='metric-card orange'><div class='metric-label'>Load Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
        <div class='metric-card blue'><div class='metric-label'>No-Load Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
        <div class='metric-card highlight'><div class='metric-label'>Total Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><b>Expected components from SLD:</b>", unsafe_allow_html=True)
    cols = st.columns(len(components))
    for i, (icon, label) in enumerate(components):
        with cols[i]:
            st.markdown(f"{icon} {label}")

if st.session_state.page == "LV Losses":
    loss_page(2, "LV Loss Breakdown", "Low voltage losses from inverters, LV transformers, and LV cables.", 2,
              [("🔋", "Inverters (INV_xxx)"), ("🔲", "LV Transformers (TX_LV_xxx)"), ("〰️", "LV Cables (LVCAB_xxx)")])

elif st.session_state.page == "MV Losses":
    loss_page(3, "MV Loss Breakdown", "Medium voltage losses from MV cables, transformers, and feeder circuits.", 3,
              [("⚡", "MV Cables (MVCAB_xxx)"), ("🔲", "Main Power TX (MPT_xxx)"), ("〰️", "Transmission Lines (TL_xxx)")])

elif st.session_state.page == "Aux Losses":
    loss_page(4, "Auxiliary Losses", "Auxiliary power consumption and parasitic losses.", 4,
              [("🔧", "Aux Loads (AUX_xxx)"), ("🏭", "Station Service"), ("📡", "Controls & SCADA")])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Summary":
    st.markdown("### Step 5 — Total Loss Summary")
    st.markdown("<p style='color:#8b949e; margin-bottom:24px;'>Combined loss report across all voltage levels and components.</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-row'>
        <div class='metric-card orange'><div class='metric-label'>LV Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
        <div class='metric-card blue'><div class='metric-label'>MV Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
        <div class='metric-card green'><div class='metric-label'>Aux Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
        <div class='metric-card highlight'><div class='metric-label'>&#9889; Total Losses</div><div class='metric-value'>—<span class='metric-unit'>kW</span></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#21262d; margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='rrc-card'><div class='rrc-card-header'>Loss Breakdown</div><div class='rrc-card-title'>Component-Level Summary Table</div></div>", unsafe_allow_html=True)
    st.info("Full breakdown table and export options will appear here after files are processed.")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Export to Excel")
    with col2:
        st.button("Export PDF Report")
   