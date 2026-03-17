import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="RRC EE Loss Report Calculator",
    page_icon="⚡⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer { visibility: hidden; } header { visibility: visible; }
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
  .stButton > button { background: #e85d04 !important; color: white !important; border: none !important; border-radius: 6px !important; font-weight: 600 !important; }
  .stButton > button:hover { background: #c94d00 !important; }
  .unload-btn > button { background: #21262d !important; color: #f85149 !important; border: 1px solid #f85149 !important; font-size: 0.8rem !important; padding: 4px 12px !important; }
  .progress-bar-wrap { background: #21262d; border-radius: 4px; height: 6px; margin-top: 8px; }
  .progress-bar-fill { background: #e85d04; border-radius: 4px; height: 6px; }
  .conv-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 20px; }
  .conv-table th { background: #21262d; color: #e85d04; text-align: left; padding: 10px 14px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
  .conv-table td { padding: 9px 14px; border-bottom: 1px solid #21262d; color: #c9d1d9; }
  .conv-table tr:hover td { background: #1c2128; }
  .conv-code { background: #21262d; color: #e85d04; padding: 2px 7px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; font-weight: 600; }
  .section-tag { display: inline-block; background: #1c2840; color: #58a6ff; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-left: 8px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in [("etap_uploaded", False), ("sld_uploaded", False), ("page", "Upload"),
             ("etap_filename", ""), ("etap_filesize", 0), ("sld_filename", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

REQUIRED_SHEETS = ["General", "Bus", "Branch", "Load", "Source"]

import re, io

def parse_sld(sld_file):
    """Extract component IDs from SLD PDF or Excel using naming convention."""
    fname = sld_file.name.lower()
    text = ""
    if fname.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(sld_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + " "
        except Exception as e:
            return {"error": str(e)}
    elif fname.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if fname.endswith(".xlsx") else "xlrd"
        xl = pd.ExcelFile(sld_file, engine=engine)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            for col in df.columns:
                text += " ".join(df[col].dropna().astype(str).tolist()) + " "

    tokens = re.findall(r'[A-Za-z0-9_.\-]+', text)
    result = {
        "INV":   sorted(set(t for t in tokens if re.match(r'INV\.', t, re.I))),
        "MVT":   sorted(set(t for t in tokens if re.match(r'MVT\.', t, re.I))),
        "CABLE": sorted(set(t for t in tokens if re.match(r'CABLE\.', t, re.I))),
        "MV_FEEDER": sorted(set(t for t in tokens if re.match(r'MV[-_]', t, re.I) and len(t) > 5)),
        "NLL":   sorted(set(t for t in tokens if re.match(r'NLL\.', t, re.I))),
        "CB":    sorted(set(t for t in tokens if re.match(r'CB\.', t, re.I))),
        "BATT":  sorted(set(t for t in tokens if re.match(r'BATT\.', t, re.I))),
        "MPT":   sorted(set(t for t in tokens if re.match(r'MPT', t, re.I) and 'NLL' not in t.upper())),
        "GSU":   sorted(set(t for t in tokens if re.match(r'GSU', t, re.I))),
        "UAT":   sorted(set(t for t in tokens if re.match(r'UAT', t, re.I))),
        "LV":    sorted(set(t for t in tokens if re.match(r'LV[._]', t, re.I))),
        "ISU":   sorted(set(t for t in tokens if re.match(r'ISU\.', t, re.I))),
        "PV":    sorted(set(t for t in tokens if re.match(r'PV\.', t, re.I))),
        "SWGR":  sorted(set(t for t in tokens if re.match(r'SWGR\.', t, re.I))),
    }
    return result

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

    # Naming convention first
    if st.button("📋  ETAP RRC Naming Convention", key="nav_Naming", use_container_width=True):
        st.session_state.page = "Naming"

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    pages = [
        ("📁", "Upload",     "Upload Files"),
        ("🔍", "SLD Check",  "SLD vs Excel Check"),
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
# PAGE: NAMING CONVENTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Naming":

    st.markdown("## 📋 ETAP RRC Naming Convention")
    st.markdown("<p style='color:#8b949e; margin-bottom:28px;'>Standard component naming for RRC's ETAP studies. All uploaded ETAP files must follow these conventions.</p>", unsafe_allow_html=True)

    # Download button for original docx
    import os
    docx_path = "ETAP_Component_Names.docx"
    if os.path.exists(docx_path):
        with open(docx_path, "rb") as f:
            st.download_button(
                label="⬇  Download Original Document (.docx)",
                data=f,
                file_name="ETAP_Component_Names.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    else:
        st.info("Place ETAP_Component_Names.docx in your repo root to enable download.")

    st.markdown("<hr style='border-color:#21262d; margin:24px 0;'>", unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    <div class='rrc-card' style='max-height:260px; overflow-y:auto;'>
        <div class='rrc-card-header'>Introduction</div>
        <div class='rrc-card-title'>Component Naming Convention for Solar Projects</div>
        <p style='color:#c9d1d9; font-size:0.9rem; line-height:1.7;'>
        The purpose of this section is to standardize the equipment naming across RRC's ETAP studies.
        This added consistency will reduce mistakes and contribute to higher client satisfaction.
        The below names should be used as the default naming conventions, with exceptions being made where necessary.
        </p>
        <p style='color:#c9d1d9; font-size:0.9rem; line-height:1.7; margin-top:10px;'>
        The first block should be built (block 01) according to the below naming conventions, while entering as much
        data as possible that is known at the time. Even if the protective devices are not known, it is much easier
        to enter that information later, than it is to add a component to every block.
        </p>
        <p style='color:#c9d1d9; font-size:0.9rem; line-height:1.7; margin-top:10px;'>
        When a component is copied/pasted, the component number 1 higher will be in the dumpster. You must right click
        and select "Remove from Dumpster" to bring it into the oneline. When there is a leading zero (ex: GSU 01),
        the component "GSU 1" will be in the dumpster, and "GSU 2" will appear on the oneline. In this case, it is
        best to leave the components in the dumpster, paste all the blocks that are needed, and add the leading zeros
        once all blocks are pasted.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # AC Model table
    st.markdown("### ⚡ AC Model Components")

    st.markdown("""
    <table class='conv-table'>
      <tr><th>Component</th><th>Standard Name</th><th>Notes</th></tr>
      <tr><td>Point of Interconnection</td><td><span class='conv-code'>POI</span></td><td></td></tr>
      <tr><td>Main Power Transformer</td><td><span class='conv-code'>MPT 1</span></td><td></td></tr>
      <tr><td>MPT Primary Bus</td><td><span class='conv-code'>MPT 1 HV</span></td><td></td></tr>
      <tr><td>MPT Secondary Bus</td><td><span class='conv-code'>MPT 1 LV</span></td><td></td></tr>
      <tr><td>MV Substation Bus</td><td><span class='conv-code'>34.5 kV MPT 1</span></td><td></td></tr>
      <tr><td>MPT No Load Losses (Static Load)</td><td><span class='conv-code'>MPT 1 NLL</span></td><td></td></tr>
      <tr><td>Generator Step Up Transformer</td><td><span class='conv-code'>GSU 01</span></td><td>Use 001 if 100+ power blocks</td></tr>
      <tr><td>GSU Primary Bus</td><td><span class='conv-code'>GSU HV 01</span></td><td>If no accessible MV switchgear; else <span class='conv-code'>MV SWGR 01</span></td></tr>
      <tr><td>GSU Secondary Bus</td><td><span class='conv-code'>GSU LV 01</span></td><td></td></tr>
      <tr><td>Inverter Output</td><td><span class='conv-code'>INV AC 01</span></td><td></td></tr>
      <tr><td>GSU No Load Losses (Static Load)</td><td><span class='conv-code'>GSU NLL 01</span></td><td></td></tr>
      <tr><td>Expulsion Fuse</td><td><span class='conv-code'>EXP 01</span></td><td></td></tr>
      <tr><td>Current Limiting Fuse</td><td><span class='conv-code'>CLF 01</span></td><td></td></tr>
      <tr><td>MV Breaker</td><td><span class='conv-code'>CB 01</span></td><td></td></tr>
      <tr><td>Current Transformer</td><td><span class='conv-code'>CT 01</span></td><td></td></tr>
      <tr><td>MV Relay</td><td><span class='conv-code'>RELAY 01</span></td><td>Or name per drawings + 01</td></tr>
      <tr><td>Auxiliary Transformer</td><td><span class='conv-code'>AUX XFMR 01</span></td><td></td></tr>
      <tr><td>Aux Transformer Primary Bus</td><td><span class='conv-code'>AUX XFMR HV 01</span></td><td>Check drawings for alternate names</td></tr>
      <tr><td>Aux Transformer Secondary Bus</td><td><span class='conv-code'>AUX XFMR LV 01</span></td><td>Check drawings for alternate names</td></tr>
      <tr><td>Auxiliary Transformer Fuse</td><td><span class='conv-code'>AUX FUSE 01</span></td><td></td></tr>
      <tr><td>Auxiliary Transformer Circuit Breaker</td><td><span class='conv-code'>AUX CB 01</span></td><td></td></tr>
      <tr><td>Inverter</td><td><span class='conv-code'>INV 01</span></td><td></td></tr>
      <tr><td>DC Bus</td><td><span class='conv-code'>DC 01</span></td><td></td></tr>
      <tr><td>PV Array</td><td><span class='conv-code'>PV 01</span></td><td></td></tr>
      <tr><td>MV Cables</td><td><span class='conv-code'>See IO Sheet</span></td><td>Circuit ID column in MV Circuits tab</td></tr>
      <tr><td>Junction Box</td><td><span class='conv-code'>JB 01</span></td><td>Or per Single Line Drawing</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("### 🏗 Substation Devices")
    st.markdown("""
    <table class='conv-table'>
      <tr><th>Component</th><th>Standard Name</th><th>Notes</th></tr>
      <tr><td>HV &amp; MV Breakers</td><td><span class='conv-code'>CB 1</span></td><td>Name after feeder number or per substation oneline drawing</td></tr>
      <tr><td>MV Relay</td><td><span class='conv-code'>RELAY 1</span></td><td>Name after feeder number or per oneline / .RDB file</td></tr>
      <tr><td>Current Transformer</td><td><span class='conv-code'>CT 1</span></td><td></td></tr>
      <tr><td>Capacitor Bank</td><td><span class='conv-code'>CAP 1</span></td><td></td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1c2840; border:1px solid #58a6ff; border-radius:8px; padding:14px 18px; margin-bottom:24px; font-size:0.87rem; color:#c9d1d9;'>
        <b style='color:#58a6ff;'>BESS Projects:</b> Either add <span class='conv-code'>BESS</span> to the beginning of each component name, or name according to the project drawings.
    </div>
    """, unsafe_allow_html=True)

    # Images
    import os
    st.markdown("### 📸 Reference Examples")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MPT Example — Big Star Solar**")
        if os.path.exists("images/image1.png"):
            st.image("images/image1.png", caption="MPT Example from Big Star Solar", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**HV & MV Breaker Example — Maplewood**")
        if os.path.exists("images/image3.png"):
            st.image("images/image3.png", caption="HV & MV Breaker Example from Maplewood", use_container_width=True)

    with col2:
        st.markdown("**Power Block Example — Big Star Solar**")
        if os.path.exists("images/image2.png"):
            st.image("images/image2.png", caption="Power Block Example from Big Star Solar", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**DC Model Example — Shoals BLA (Slate)**")
        img4_path = "images/image4.png"
        if os.path.exists(img4_path):
            st.image(img4_path, caption="DC Model Example - Shoals BLA", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Upload":

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

        if st.session_state.etap_uploaded:
            st.success(f"All required sheets validated in **{st.session_state.etap_filename}**")
            st.markdown(
                "<div style='background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:12px; margin-top:8px; font-size:0.82rem; color:#8b949e;'>"
                f"&#128196; File: <span style='color:#e6edf3;'>{st.session_state.etap_filename}</span><br>"
                f"&#128230; Size: <span style='color:#e6edf3;'>{st.session_state.etap_filesize:.1f} KB</span><br>"
                "&#128203; Sheets validated: <span style='color:#3fb950;'>General &nbsp;·&nbsp; Bus &nbsp;·&nbsp; Branch &nbsp;·&nbsp; Load &nbsp;·&nbsp; Source</span>"
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("<div class='unload-btn'>", unsafe_allow_html=True)
            if st.button("✕  Remove ETAP File", key="unload_etap"):
                st.session_state.etap_uploaded = False
                st.session_state.etap_filename = ""
                st.session_state.etap_filesize = 0
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            etap_file = st.file_uploader(
                "Upload ETAP Excel file (.xlsx / .xls)",
                type=["xlsx", "xls"],
                key="etap_uploader",
                help="Excel file must contain sheets: General, Bus, Branch, Load, Source"
            )
            if etap_file:
                try:
                    engine = "openpyxl" if etap_file.name.endswith(".xlsx") else "xlrd"
                    xl = pd.ExcelFile(etap_file, engine=engine)
                    found = xl.sheet_names
                    missing = [s for s in REQUIRED_SHEETS if s not in found]
                    if missing:
                        st.error(f"Missing sheet(s): **{', '.join(missing)}** — please re-upload the correct file.")
                        st.markdown(
                            "<div style='background:#1a0a0a; border:1px solid #f85149; border-radius:6px; padding:12px; margin-top:8px; font-size:0.82rem; color:#8b949e;'>"
                            f"&#128196; File: <span style='color:#e6edf3;'>{etap_file.name}</span><br>"
                            f"&#128230; Size: <span style='color:#e6edf3;'>{etap_file.size / 1024:.1f} KB</span><br>"
                            f"&#128203; Sheets found: <span style='color:#c9d1d9;'>{' | '.join(found) if found else 'None'}</span><br>"
                            f"&#9888; Missing: <span style='color:#f85149; font-weight:600;'>{' | '.join(missing)}</span><br><br>"
                            "<span style='color:#c9d1d9;'>Required: </span>"
                            "<span style='color:#3fb950;'>General &nbsp;·&nbsp; Bus &nbsp;·&nbsp; Branch &nbsp;·&nbsp; Load &nbsp;·&nbsp; Source</span>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.session_state.etap_uploaded = True
                        st.session_state.etap_filename = etap_file.name
                        st.session_state.etap_filesize = etap_file.size / 1024
                        # Parse and cache all sheets
                        st.session_state.etap_data = {
                            "branch": pd.read_excel(etap_file, sheet_name="Branch", engine=engine),
                            "load":   pd.read_excel(etap_file, sheet_name="Load",   engine=engine),
                            "bus":    pd.read_excel(etap_file, sheet_name="Bus",     engine=engine),
                        }
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")
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

        if st.session_state.sld_uploaded:
            st.success(f"**{st.session_state.sld_filename}** uploaded successfully!")
            st.markdown("<div class='unload-btn'>", unsafe_allow_html=True)
            if st.button("✕  Remove SLD File", key="unload_sld"):
                st.session_state.sld_uploaded = False
                st.session_state.sld_filename = ""
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            sld_file = st.file_uploader(
                "Upload SLD file (.xlsx or .pdf)",
                type=["xlsx", "xls", "pdf"],
                key="sld_uploader",
            )
            if sld_file:
                sld_components = parse_sld(sld_file)
                st.session_state.sld_uploaded = True
                st.session_state.sld_filename = sld_file.name
                st.session_state.sld_components = sld_components
                st.rerun()
            else:
                st.markdown(
                    "<div style='font-size:0.82rem; color:#8b949e; margin-top:8px; padding:12px; background:#0d1117; border-radius:6px; border:1px solid #21262d;'>"
                    "&#128269; <b style='color:#c9d1d9;'>Auto-detected via naming convention:</b><br>"
                    "&nbsp;&nbsp;· Inverters — <span style='color:#e85d04;'>INV 01</span><br>"
                    "&nbsp;&nbsp;· GSU Transformers — <span style='color:#e85d04;'>GSU 01</span><br>"
                    "&nbsp;&nbsp;· Main Power TX — <span style='color:#e85d04;'>MPT 1</span><br>"
                    "&nbsp;&nbsp;· MV Cables — <span style='color:#e85d04;'>See IO Sheet</span><br>"
                    "&nbsp;&nbsp;· Auxiliary Transformer — <span style='color:#e85d04;'>AUX XFMR 01</span>"
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
def loss_page(step, title, subtitle, components):
    st.markdown(f"### Step {step} — {title}")
    st.markdown(f"<p style='color:#8b949e; margin-bottom:24px;'>{subtitle}</p>", unsafe_allow_html=True)
    if not st.session_state.etap_uploaded:
        st.warning("Please upload and validate the ETAP file first (Step 1 — Upload).")
        return
    st.markdown(f"<div class='rrc-card'><div class='rrc-card-header'>Components Detected</div><div class='rrc-card-title'>{title}</div></div>", unsafe_allow_html=True)
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

if st.session_state.page == "SLD Check":
    st.markdown("### 🔍 SLD vs Excel Component Check")
    st.markdown("<p style='color:#8b949e; margin-bottom:24px;'>Preliminary check comparing component counts between the uploaded SLD and ETAP Excel file.</p>", unsafe_allow_html=True)

    if not st.session_state.etap_uploaded:
        st.warning("Please upload and validate the ETAP file first.")
    elif not st.session_state.sld_uploaded:
        st.warning("Please upload the SLD file first.")
    elif "sld_components" not in st.session_state:
        st.warning("SLD was uploaded before parsing was available. Please remove and re-upload the SLD file.")
    elif "error" in st.session_state.sld_components:
        st.error(f"Could not parse SLD: {st.session_state.sld_components['error']}")
    else:
        sld = st.session_state.sld_components
        branch_df = st.session_state.etap_data["branch"].copy()
        load_df   = st.session_state.etap_data["load"].copy()
        branch_df["ID"] = branch_df["ID"].astype(str)
        load_df["ID"]   = load_df["ID"].astype(str)

        # Build Excel component counts
        excel_counts = {
            "INV":        len(branch_df[branch_df["ID"].str.upper().str.startswith("INV.")]),
            "MVT":        len(branch_df[branch_df["ID"].str.upper().str.startswith("MVT.") & branch_df["ID"].str.upper().str.endswith("-P")]),
            "CABLE":      len(branch_df[branch_df["ID"].str.upper().str.startswith("CABLE.")]),
            "MV_FEEDER":  len(branch_df[branch_df["ID"].str.upper().str.startswith("MV-")]),
            "NLL":        len(load_df[load_df["ID"].str.upper().str.startswith("NLL.")]),
            "CB":         len(branch_df[branch_df["ID"].str.upper().str.startswith("CB.")]),
            "MPT":        len(branch_df[branch_df["ID"].str.upper().str.startswith("MPT") & branch_df["ID"].str.upper().str.endswith("-P")]),
            "GSU":        len(branch_df[branch_df["ID"].str.upper().str.startswith("GSU") & branch_df["ID"].str.upper().str.endswith("-P")]),
            "UAT":        len(branch_df[branch_df["ID"].str.upper().str.startswith("UAT") & branch_df["ID"].str.upper().str.endswith("-P")]),
            "LV":         len(branch_df[branch_df["ID"].str.upper().str.startswith("LV")]),
            "BATT":       0,
        }

        labels = {
            "INV": "Inverters (INV)",
            "MVT": "MV Transformers (MVT)",
            "CABLE": "MV Cables (CABLE)",
            "MV_FEEDER": "MV Feeder Segments (MV-)",
            "NLL": "No-Load Loss Loads (NLL)",
            "CB": "Circuit Breakers (CB)",
            "MPT": "Main Power Transformer (MPT)",
            "GSU": "GSU Transformers (GSU)",
            "UAT": "Aux Transformers (UAT)",
            "LV": "LV Components (LV)",
            "BATT": "Battery Units (BATT)",
        }

        rows = []
        for key, label in labels.items():
            sld_count = len(sld.get(key, []))
            exc_count = excel_counts.get(key, 0)
            if sld_count == 0 and exc_count == 0:
                continue
            match = sld_count == exc_count
            rows.append({
                "Component": label,
                "SLD Count": sld_count,
                "Excel Count": exc_count,
                "Status": "✅ Match" if match else f"⚠️ Diff: {abs(sld_count - exc_count)}"
            })

        matches   = sum(1 for r in rows if "Match" in r["Status"])
        mismatches = sum(1 for r in rows if "Diff" in r["Status"])

        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card green'>
                <div class='metric-label'>Matching</div>
                <div class='metric-value'>{matches}</div>
            </div>
            <div class='metric-card orange'>
                <div class='metric-label'>Mismatches</div>
                <div class='metric-value'>{mismatches}</div>
            </div>
            <div class='metric-card highlight'>
                <div class='metric-label'>Total Checked</div>
                <div class='metric-value'>{len(rows)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

        for row in rows:
            is_match = "Match" in row["Status"]
            border = "#3fb950" if is_match else "#e85d04"
            status_color = "#3fb950" if is_match else "#f0a500"
            st.markdown(
                f"<div style='background:#161b22; border:1px solid {border}; border-radius:6px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='color:#c9d1d9; font-size:0.9rem;'>{row['Component']}</span>"
                f"<span style='color:#8b949e; font-size:0.85rem;'>SLD: <b style='color:#58a6ff;'>{row['SLD Count']}</b> &nbsp;|&nbsp; Excel: <b style='color:#e85d04;'>{row['Excel Count']}</b></span>"
                f"<span style='color:{status_color}; font-weight:600; font-size:0.85rem;'>{row['Status']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        if mismatches > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning("⚠️ Some component counts differ between SLD and Excel. Review mismatches before proceeding with loss calculations.")
        else:
            st.success("✅ All component counts match between SLD and Excel file!")

elif st.session_state.page == "LV Losses":
    st.markdown("### Step 2 — LV Loss Breakdown")
    st.markdown("<p style='color:#8b949e; margin-bottom:24px;'>Low voltage load losses (Branch sheet) and no-load losses (Load sheet) for LV cables and auxiliary transformers.</p>", unsafe_allow_html=True)

    if not st.session_state.etap_uploaded:
        st.warning("Please upload and validate the ETAP file first (Step 1 — Upload).")
    elif "etap_data" not in st.session_state:
        st.warning("ETAP data not parsed yet. Please re-upload the file.")
    else:
        branch_df = st.session_state.etap_data["branch"]
        load_df   = st.session_state.etap_data["load"]

        # ── Load Losses from Branch sheet ──────────────────────────────────
        # Drop rows where ID is NaN before any string operations
        branch_df = branch_df.dropna(subset=["ID"]).copy()
        branch_df["ID"] = branch_df["ID"].astype(str)
        load_df = load_df.dropna(subset=["ID"]).copy()
        load_df["ID"] = load_df["ID"].astype(str)

        lv_cables = branch_df[branch_df["ID"].str.upper().str.startswith("LV")].copy()
        lv_cables["Category"] = "LV Cable"

        uat_tx = branch_df[
            branch_df["ID"].str.upper().str.startswith("UAT") &
            branch_df["ID"].str.upper().str.endswith("-P") &
            branch_df["kW Losses"].notna()
        ].copy()
        uat_tx["Category"] = "Aux Transformer (UAT)"

        parts = [df for df in [lv_cables, uat_tx] if not df.empty]
        if parts:
            load_losses_df = pd.concat(parts)[["Category", "ID", "Type", "kW Losses"]] \
                .dropna(subset=["kW Losses"]).reset_index(drop=True)
            load_losses_df.columns = ["Category", "Component ID", "Type", "kW Losses"]
        else:
            load_losses_df = pd.DataFrame(columns=["Category", "Component ID", "Type", "kW Losses"])

        # ── No-Load Losses from Load sheet ─────────────────────────────────
        uat_loads = load_df[load_df["ID"].str.upper().str.contains("UAT")].copy()
        if not uat_loads.empty:
            uat_loads["Category"] = "Aux Transformer No-Load (UAT)"
            no_load_df = uat_loads[["Category", "ID", "kW"]].reset_index(drop=True)
            no_load_df.columns = ["Category", "Component ID", "kW No-Load"]
        else:
            no_load_df = pd.DataFrame(columns=["Category", "Component ID", "kW No-Load"])

        # ── Totals ──────────────────────────────────────────────────────────
        total_load   = load_losses_df["kW Losses"].sum() if not load_losses_df.empty else 0.0
        total_noload = no_load_df["kW No-Load"].sum()   if not no_load_df.empty   else 0.0
        total_lv     = total_load + total_noload

        # ── Metric cards ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card orange'>
                <div class='metric-label'>Load Losses (Branch)</div>
                <div class='metric-value'>{total_load:,.2f}<span class='metric-unit'>kW</span></div>
            </div>
            <div class='metric-card blue'>
                <div class='metric-label'>No-Load Losses (Load)</div>
                <div class='metric-value'>{total_noload:,.2f}<span class='metric-unit'>kW</span></div>
            </div>
            <div class='metric-card highlight'>
                <div class='metric-label'>Total LV Losses</div>
                <div class='metric-value'>{total_lv:,.2f}<span class='metric-unit'>kW</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

        # ── Load Losses Table ────────────────────────────────────────────────
        st.markdown("#### ⚡ Load Losses — Branch Sheet")
        if load_losses_df.empty:
            st.info("ℹ️ No LV or UAT load loss components identified in the Branch sheet for this project.")
        else:
            st.markdown(f"<p style='color:#8b949e; font-size:0.85rem;'>{len(load_losses_df)} components &nbsp;·&nbsp; Total: <b style='color:#e85d04;'>{total_load:,.2f} kW</b></p>", unsafe_allow_html=True)
            st.dataframe(load_losses_df, use_container_width=True, hide_index=True)
            cat_totals = load_losses_df.groupby("Category")["kW Losses"].sum().reset_index()
            col1, _ = st.columns(2)
            with col1:
                for _, row in cat_totals.iterrows():
                    st.markdown(f"<div style='background:#161b22; border:1px solid #21262d; border-radius:6px; padding:10px 14px; margin-bottom:8px; font-size:0.85rem;'><span style='color:#8b949e;'>{row['Category']}</span><span style='float:right; color:#e85d04; font-weight:600;'>{row['kW Losses']:,.4f} kW</span></div>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

        # ── No-Load Losses Table ─────────────────────────────────────────────
        st.markdown("#### 🔲 No-Load Losses — Load Sheet")
        if no_load_df.empty:
            st.info("ℹ️ No LV or UAT no-load loss components identified in the Load sheet for this project.")
        else:
            st.markdown(f"<p style='color:#8b949e; font-size:0.85rem;'>{len(no_load_df)} components &nbsp;·&nbsp; Total: <b style='color:#58a6ff;'>{total_noload:,.2f} kW</b></p>", unsafe_allow_html=True)
            st.dataframe(no_load_df, use_container_width=True, hide_index=True)

elif st.session_state.page == "MV Losses":
    loss_page(3, "MV Loss Breakdown", "Medium voltage losses from MV cables, GSU HV buses, and Main Power Transformer.",
              [("⚡", "MV Cables (IO Sheet)"), ("🔲", "MPT 1"), ("〰️", "GSU HV 01")])

elif st.session_state.page == "Aux Losses":
    loss_page(4, "Auxiliary Losses", "Auxiliary transformer and station service losses.",
              [("🔧", "AUX XFMR 01"), ("🏭", "AUX XFMR LV 01"), ("📡", "AUX CB 01")])

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
