"""
PDF & Image Merger
──────────────────
Standalone Streamlit page.
Run: streamlit run pdf_merger.py

Accepts: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp
Converts images to PDF pages then merges everything in order.
"""

import streamlit as st
import io
from PIL import Image
from pypdf import PdfWriter, PdfReader


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_pdf_bytes(img_bytes: bytes, filename: str) -> bytes:
    """Convert an image file to a single-page PDF in memory."""
    img = Image.open(io.BytesIO(img_bytes))
    # Convert to RGB (removes alpha channel which PDF doesn't support)
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    pdf_buf = io.BytesIO()
    img.save(pdf_buf, format="PDF", resolution=150)
    return pdf_buf.getvalue()


def merge_files(file_list: list) -> bytes:
    """
    Merge a list of (name, bytes, type) tuples into one PDF.
    Returns the merged PDF as bytes.
    """
    writer = PdfWriter()

    for name, data, ftype in file_list:
        if ftype == "pdf":
            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages:
                writer.add_page(page)
        else:  # image
            pdf_bytes = image_to_pdf_bytes(data, name)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            writer.add_page(reader.pages[0])

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return "pdf" if ext == "pdf" else "image"


def page_count(data: bytes, ftype: str) -> int:
    if ftype == "pdf":
        return len(PdfReader(io.BytesIO(data)).pages)
    return 1


# ── Page ──────────────────────────────────────────────────────────────────────

def render_merger_page():
    st.markdown("### 📎 PDF & Image Merger")
    st.markdown(
        "<p style='color:#8b949e; margin-bottom:24px;'>"
        "Upload PDFs and images in any order. Drag to reorder, then download the merged PDF.</p>",
        unsafe_allow_html=True,
    )

    # ── Init state ────────────────────────────────────────────────────────────
    if "merger_files" not in st.session_state:
        st.session_state.merger_files = []   # list of {name, data, ftype, pages}

    files = st.session_state.merger_files

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown("#### ＋ Add Files")

    uploaded = st.file_uploader(
        "Upload PDFs or images",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Add as many files as you like — each click adds to the list",
    )

    if uploaded:
        existing_names = {f["name"] for f in files}
        added = 0
        for uf in uploaded:
            if uf.name not in existing_names:
                data  = uf.read()
                ftype = file_type(uf.name)
                try:
                    pages = page_count(data, ftype)
                except Exception:
                    pages = 1
                files.append({"name": uf.name, "data": data, "ftype": ftype, "pages": pages})
                existing_names.add(uf.name)
                added += 1
        if added:
            st.session_state.merger_files = files
            st.rerun()

    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

    # ── File list ─────────────────────────────────────────────────────────────
    if not files:
        st.markdown(
            "<div style='background:#161b22; border:1px dashed #30363d; border-radius:8px; "
            "padding:40px; text-align:center; color:#8b949e;'>"
            "No files added yet — use the uploader above to add PDFs or images."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    total_pages = sum(f["pages"] for f in files)
    st.markdown(
        f"#### 📋 Files to Merge "
        f"<span style='color:#8b949e; font-size:0.85rem; font-weight:400;'>"
        f"— {len(files)} file{'s' if len(files)!=1 else ''} · {total_pages} page{'s' if total_pages!=1 else ''} total"
        f"</span>",
        unsafe_allow_html=True,
    )

    to_delete = None
    move_up   = None
    move_down = None

    for i, f in enumerate(files):
        is_pdf = f["ftype"] == "pdf"
        icon   = "📄" if is_pdf else "🖼️"
        pg_txt = f"{f['pages']} page{'s' if f['pages']!=1 else ''}" if is_pdf else "1 page (image)"

        col_num, col_icon, col_name, col_pages, col_up, col_dn, col_del = st.columns(
            [0.4, 0.4, 5, 1.5, 0.5, 0.5, 0.5]
        )

        col_num.markdown(
            f"<div style='color:#8b949e; font-size:0.85rem; padding-top:8px; text-align:center;'>{i+1}</div>",
            unsafe_allow_html=True,
        )
        col_icon.markdown(
            f"<div style='font-size:1.2rem; padding-top:4px; text-align:center;'>{icon}</div>",
            unsafe_allow_html=True,
        )
        col_name.markdown(
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:6px; "
            f"padding:8px 12px; color:#c9d1d9; font-size:0.85rem; "
            f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
            f"{f['name']}</div>",
            unsafe_allow_html=True,
        )
        col_pages.markdown(
            f"<div style='color:#8b949e; font-size:0.78rem; padding-top:10px;'>{pg_txt}</div>",
            unsafe_allow_html=True,
        )

        if i > 0 and col_up.button("↑", key=f"up_{i}", help="Move up"):
            move_up = i
        if i < len(files) - 1 and col_dn.button("↓", key=f"dn_{i}", help="Move down"):
            move_down = i
        if col_del.button("✕", key=f"del_{i}", help="Remove"):
            to_delete = i

    # Handle reorder / delete
    if move_up is not None:
        files[move_up - 1], files[move_up] = files[move_up], files[move_up - 1]
        st.session_state.merger_files = files
        st.rerun()
    if move_down is not None:
        files[move_down], files[move_down + 1] = files[move_down + 1], files[move_down]
        st.session_state.merger_files = files
        st.rerun()
    if to_delete is not None:
        files.pop(to_delete)
        st.session_state.merger_files = files
        st.rerun()

    st.markdown("<hr style='border-color:#21262d; margin:20px 0;'>", unsafe_allow_html=True)

    # ── Clear all ─────────────────────────────────────────────────────────────
    col_clear, col_merge = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.merger_files = []
            st.rerun()

    # ── Merge & download ──────────────────────────────────────────────────────
    with col_merge:
        if st.button("⬇️  Merge & Download PDF", type="primary", use_container_width=True):
            with st.spinner("Merging files…"):
                try:
                    merged = merge_files(
                        [(f["name"], f["data"], f["ftype"]) for f in files]
                    )
                    st.success(f"✅ Merged {len(files)} files into {total_pages} pages.")
                    st.download_button(
                        label="📥 Download merged.pdf",
                        data=merged,
                        file_name="merged.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"❌ Merge failed: {e}")


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="PDF Merger · RRC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"] { background: #161b22; }
    .stButton > button { border: 1px solid #30363d; color: #c9d1d9; }
    .stButton > button[kind="primary"] { background: #e85d04; color: white; border: none; }
    .stDownloadButton > button { background: #1f6feb; color: white; border: none; }
    </style>""", unsafe_allow_html=True)
    render_merger_page()