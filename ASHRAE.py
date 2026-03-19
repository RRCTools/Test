"""
ASHRAE Design Temperature Lookup
─────────────────────────────────────────────────────────────────────────────
Standalone Streamlit page.  Run: streamlit run ashrae_page.py
Or integrate: call render_ashrae_page() inside your main app.py nav block.
─────────────────────────────────────────────────────────────────────────────
API:  http://ashrae-meteo.info
  Step 1 — POST request_places.php          → nearest WMO station from lat/lon
  Step 2 — POST request_meteo_parametres.php → full climate data for that WMO

Key fields used (ASHRAE 2021, SI units):
  cooling_DB_MCWB_04_DB                              = Yearly cooling 0.4% DB
  cooling_DB_MCWB_2_DB                               = Yearly cooling 2.0% DB
  monthly_design_day_cooling_DB_range_<Mon>_DB_04    = Monthly 0.4% (x12)
  monthly_design_day_cooling_DB_range_<Mon>_DB_2     = Monthly 2.0% (x12)
  n-year_return_period_values_of_extreme_DB_50_max   = Extreme annual max N=50
  n-year_return_period_values_of_extreme_DB_50_min   = Extreme annual min N=50
"""

import streamlit as st
import requests
import json
import pandas as pd

# ── API constants ─────────────────────────────────────────────────────────────

STATION_URL = "http://ashrae-meteo.info/request_places.php"
METEO_URL   = "http://ashrae-meteo.info/request_meteo_parametres.php"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MONTHLY_04 = [f"monthly_design_day_cooling_DB_range_{m}_DB_04" for m in MONTHS]
MONTHLY_2  = [f"monthly_design_day_cooling_DB_range_{m}_DB_2"  for m in MONTHS]

# Rows shown in the results table
# (display label, api_field_or_computed_key, unit)
DISPLAY_ROWS = [
    ("Highest Monthly 0.4%",    "_monthly_04_max",                                           "°C"),
    ("Highest Monthly 2.0%",    "_monthly_2_max",                                            "°C"),
    ("Yearly 0.4%",             "cooling_DB_MCWB_04_DB",                                    "°C"),
    ("Yearly 2.0%",             "cooling_DB_MCWB_2_DB",                                     "°C"),
    ("Extreme Mean Annual Max", "n-year_return_period_values_of_extreme_DB_50_max",          "°C"),
    ("Extreme Mean Annual Min", "n-year_return_period_values_of_extreme_DB_50_min",          "°C"),
    ("Min Recorded Temp (N=50)","n-year_return_period_values_of_extreme_DB_50_min",          "°C"),
    ("Max Recorded Temp (N=50)","n-year_return_period_values_of_extreme_DB_50_max",          "°C"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _post(url: str, params: dict) -> dict:
    try:
        resp = requests.post(url, data=params, timeout=15)
        raw  = resp.content.decode("utf-8-sig").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


def find_station(lat: float, lon: float, version: str = "2021") -> dict:
    data     = _post(STATION_URL, {"lat": lat, "long": lon, "number": "1", "ashrae_version": version})
    if "_error" in data:
        return data
    stations = data.get("meteo_stations", [])
    return stations[0] if stations else {"_error": "No station found near these coordinates"}


def fetch_climate(wmo: str, version: str = "2021") -> dict:
    data     = _post(METEO_URL, {"wmo": wmo, "ashrae_version": version, "si_ip": "SI"})
    if "_error" in data:
        return data
    stations = data.get("meteo_stations", [])
    return stations[0] if stations else {"_error": f"No climate data for WMO {wmo}"}


def to_float(val) -> float | None:
    try:
        return float(str(val).strip())
    except Exception:
        return None


def monthly_max(climate: dict, fields: list) -> float | None:
    vals = [to_float(climate.get(f)) for f in fields]
    vals = [v for v in vals if v is not None]
    return round(max(vals), 1) if vals else None


# ── Page ──────────────────────────────────────────────────────────────────────

def render_ashrae_page():
    st.markdown("### 🌡️ ASHRAE Design Temperature Lookup")
    st.markdown(
        "<p style='color:#8b949e; margin-bottom:24px;'>"
        "Enter coordinates for up to 5 project locations. Fetches ASHRAE 2021 climatic design "
        "conditions and recommends the worst-case inverter design temperature.</p>",
        unsafe_allow_html=True,
    )

    # ── Location inputs ──────────────────────────────────────────────────────
    st.markdown("#### 📍 Project Locations")

    if "ashrae_locs" not in st.session_state:
        st.session_state.ashrae_locs = [{"name": "Location A", "lat": "", "lon": ""}]

    locs      = st.session_state.ashrae_locs
    to_delete = None

    # Header row
    h1, h2, h3, _ = st.columns([2, 2, 2, 0.5])
    h1.markdown("<span style='color:#8b949e; font-size:0.78rem;'>LOCATION NAME</span>", unsafe_allow_html=True)
    h2.markdown("<span style='color:#8b949e; font-size:0.78rem;'>LATITUDE (°N)</span>",  unsafe_allow_html=True)
    h3.markdown("<span style='color:#8b949e; font-size:0.78rem;'>LONGITUDE (°E / °W negative)</span>", unsafe_allow_html=True)

    for i, loc in enumerate(locs):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 0.5])
        locs[i]["name"] = c1.text_input("n", value=loc["name"], key=f"n_{i}",  label_visibility="collapsed")
        locs[i]["lat"]  = c2.text_input("la", value=loc["lat"],  key=f"la_{i}", label_visibility="collapsed", placeholder="e.g. 45.60")
        locs[i]["lon"]  = c3.text_input("lo", value=loc["lon"],  key=f"lo_{i}", label_visibility="collapsed", placeholder="e.g. -121.50")
        if len(locs) > 1:
            if c4.button("✕", key=f"del_{i}"):
                to_delete = i

    if to_delete is not None:
        st.session_state.ashrae_locs.pop(to_delete)
        st.rerun()

    ca, cb = st.columns([1, 3])
    with ca:
        if len(locs) < 5 and st.button("＋ Add Location"):
            st.session_state.ashrae_locs.append(
                {"name": f"Location {chr(65 + len(locs))}", "lat": "", "lon": ""}
            )
            st.rerun()
    with cb:
        run = st.button("▶  Fetch ASHRAE Data", type="primary", use_container_width=True)

    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

    if not run:
        st.markdown(
            "<div style='background:#161b22; border:1px solid #21262d; border-radius:8px; "
            "padding:16px 20px; color:#8b949e; font-size:0.85rem; line-height:1.8;'>"
            "Enter project coordinates and click <b style='color:#e85d04;'>▶ Fetch ASHRAE Data</b>.<br>"
            "Data source: "
            "<a href='https://ashrae-meteo.info/v3.0/' target='_blank' style='color:#58a6ff;'>"
            "ashrae-meteo.info</a> · ASHRAE 2021 Handbook of Fundamentals"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Validate ─────────────────────────────────────────────────────────────
    valid = []
    for loc in locs:
        try:
            valid.append({
                "name": loc["name"].strip() or "Unnamed",
                "lat":  float(loc["lat"]),
                "lon":  float(loc["lon"]),
            })
        except Exception:
            if loc["lat"] or loc["lon"]:
                st.warning(f"⚠️ '{loc['name']}' skipped — enter valid decimal coordinates.")

    if not valid:
        st.error("No valid locations. Enter lat/lon as decimal degrees (e.g. 45.6, -121.5).")
        return

    # ── Fetch ─────────────────────────────────────────────────────────────────
    results = []
    prog    = st.progress(0, text="Connecting to ASHRAE database…")

    for i, loc in enumerate(valid):
        prog.progress(i / len(valid), text=f"Fetching {loc['name']}…")

        station = find_station(loc["lat"], loc["lon"])
        if "_error" in station:
            results.append({"name": loc["name"], "error": station["_error"]})
            continue

        wmo    = station.get("wmo", "")
        s_name = station.get("place_name", "")
        s_ctry = station.get("country", "")
        s_dist = station.get("distance", "?")

        climate = fetch_climate(wmo)
        if "_error" in climate:
            results.append({"name": loc["name"], "wmo": wmo, "station": s_name,
                             "error": climate["_error"]})
            continue

        # Inject computed fields
        climate["_monthly_04_max"] = monthly_max(climate, MONTHLY_04)
        climate["_monthly_2_max"]  = monthly_max(climate, MONTHLY_2)

        row = {
            "name":    loc["name"],
            "station": f"{s_name}, {s_ctry}",
            "wmo":     wmo,
            "dist_km": s_dist,
            "error":   None,
            "_raw":    climate,
        }
        for label, field, _ in DISPLAY_ROWS:
            row[label] = to_float(climate.get(field))

        results.append(row)

    prog.progress(1.0, text="Done!")
    prog.empty()

    # ── Render ────────────────────────────────────────────────────────────────
    ok  = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]

    for r in bad:
        st.error(f"❌ **{r['name']}**: {r['error']}")

    if not ok:
        return

    # Station cards
    st.markdown("#### 📡 Nearest ASHRAE 2021 Stations")
    cols = st.columns(len(ok))
    for col, r in zip(cols, ok):
        col.markdown(
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px;'>"
            f"<div style='color:#e85d04; font-weight:600; font-size:0.95rem;'>{r['name']}</div>"
            f"<div style='color:#c9d1d9; font-size:0.8rem; margin-top:4px;'>{r['station']}</div>"
            f"<div style='color:#8b949e; font-size:0.75rem; margin-top:2px;'>WMO {r['wmo']} · {r['dist_km']} km away</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    st.markdown("#### 📊 Design Temperatures (°C) — ASHRAE 2021")

    table = {"Parameter": [lbl for lbl, _, _ in DISPLAY_ROWS],
             "Unit":      [u   for _, _, u  in DISPLAY_ROWS]}
    for r in ok:
        table[r["name"]] = [
            f"{r.get(lbl):.1f}" if r.get(lbl) is not None else "N/A"
            for lbl, _, _ in DISPLAY_ROWS
        ]

    df = pd.DataFrame(table)

    def highlight_hm04(row):
        if row["Parameter"] == "Highest Monthly 0.4%":
            return [f"background-color:rgba(232,93,4,0.12); font-weight:600"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight_hm04, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

    # Recommendation
    st.markdown("#### ✅ Inverter Design Temperature Recommendation")

    hm04 = {r["name"]: r.get("Highest Monthly 0.4%")
             for r in ok if r.get("Highest Monthly 0.4%") is not None}

    if not hm04:
        st.warning(
            "⚠️ Monthly 0.4% temperatures could not be computed — "
            "the monthly fields may use a different naming convention in ASHRAE 2021. "
            "Expand the debug panel below to inspect available field names."
        )
    else:
        worst_name = max(hm04, key=hm04.get)
        worst_val  = hm04[worst_name]

        st.markdown(
            f"<div style='background:#161b22; border:2px solid #e85d04; border-radius:10px; padding:20px 24px;'>"
            f"<div style='color:#8b949e; font-size:0.75rem; text-transform:uppercase; "
            f"letter-spacing:0.06em; margin-bottom:10px;'>"
            f"Recommended inverter design temperature · ASHRAE 2021 · Highest Monthly 0.4%</div>"
            f"<div style='display:flex; align-items:baseline; gap:24px; flex-wrap:wrap;'>"
            f"<div><span style='color:#8b949e; font-size:0.85rem;'>Worst-case location: </span>"
            f"<span style='color:#e85d04; font-size:1.1rem; font-weight:600;'>{worst_name}</span></div>"
            f"<div><span style='color:#8b949e; font-size:0.85rem;'>Design temp: </span>"
            f"<span style='color:#c9d1d9; font-size:1.8rem; font-weight:600;'>{worst_val:.1f} °C</span></div>"
            f"</div>"
            f"<div style='color:#8b949e; font-size:0.78rem; margin-top:12px; line-height:1.7;'>"
            f"The <b style='color:#c9d1d9;'>Highest Monthly 0.4%</b> dry-bulb temperature is exceeded "
            f"only 0.4% of hours in the hottest month (~3 hrs/month). "
            f"This is the RRC standard basis for inverter thermal derating analysis per ASHRAE guidelines."
            f"</div></div>",
            unsafe_allow_html=True,
        )

        if len(hm04) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            for name, val in sorted(hm04.items(), key=lambda x: -x[1]):
                is_worst = name == worst_name
                color    = "#e85d04" if is_worst else "#4a90c4"
                pct      = int(val / (worst_val + 8) * 100)
                tag      = "  ← worst case" if is_worst else ""
                st.markdown(
                    f"<div style='margin-bottom:10px;'>"
                    f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
                    f"<span style='color:#c9d1d9; font-size:0.85rem;'>{name}{tag}</span>"
                    f"<span style='color:{color}; font-weight:600;'>{val:.1f} °C</span></div>"
                    f"<div style='background:#21262d; border-radius:4px; height:8px;'>"
                    f"<div style='background:{color}; width:{pct}%; height:8px; border-radius:4px;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # Debug expander
    with st.expander("🔍 Raw API response (debug / field name verification)"):
        st.markdown(
            "<p style='color:#8b949e; font-size:0.8rem;'>"
            "All temperature-related fields returned by the API. "
            "Use this to verify field names if any values show as N/A.</p>",
            unsafe_allow_html=True,
        )
        for r in ok:
            st.markdown(f"**{r['name']}** — {r['station']} (WMO {r['wmo']})")
            raw = r.get("_raw", {})
            temp_fields = {
                k: v for k, v in raw.items()
                if any(x in k.lower() for x in ["db", "cool", "heat", "extreme", "monthly", "temp"])
                   and not k.startswith("_")
            }
            st.json(temp_fields)


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="ASHRAE Lookup · RRC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"] { background: #161b22; }
    .stButton > button { border: 1px solid #e85d04; color: #e85d04; }
    .stButton > button[kind="primary"] { background: #e85d04; color: white; border: none; }
    </style>""", unsafe_allow_html=True)
    render_ashrae_page()