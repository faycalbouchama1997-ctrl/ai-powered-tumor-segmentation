import streamlit as st
import httpx
import base64
import io
import time
import uuid
import datetime
from pathlib import Path
from PIL import Image

XLSX_PATH = Path(__file__).parent.parent / "results_tumor.xlsx"

MODELL_KEYS = {
    1: "Gen1_SegResNet",
    3: "Gen3_SwinUNETR",
    4: "Gen4_MedSAM",
}

GEN_CONFIGS = [
    (1, "SegResNet",   "CNN (SegResNet)",            "CNN / Residual-Bloecke / MONAI"),
    (3, "Swin-UNETR",  "Transformer (Swin-UNETR)",   "Transformer / Swin-ViT / MONAI"),
    (4, "MedSAM",      "Foundation Model (MedSAM)",   "Foundation Model / 1.5M Bilder"),
]

API_URL = "http://127.0.0.1:8000"

T = {
    "DE": {
        "page_title":        "TumorSeg",
        "connecting":        "Verbindung zum Backend wird hergestellt...",
        "backend_error":     "Backend nicht erreichbar -- bitte run.bat starten",
        "patient_data":      "Patientendaten",
        "required_note":     "* Pflichtfeld",
        "lastname":          "Nachname *",
        "firstname":         "Vorname *",
        "dob":               "Geburtsdatum",
        "patient_id":        "Patient-ID *",
        "date":              "Datum",
        "time":              "Uhrzeit",
        "upload_section":    "Datei hochladen und Analyse",
        "upload_label":      "Datei hochladen (.nii.gz)",
        "analyze_btn":       "Alle 3 Modelle gleichzeitig analysieren",
        "required_missing":  "Pflichtfelder ausfüllen (* erforderlich):",
        "analyzing":         "wird analysiert...",
        "analysis_start":    "Analyse wird gestartet...",
        "analysis_done":     "Analyse abgeschlossen",
        "not_analyzed":      "Noch nicht analysiert",
        "unknown_error":     "Unbekannter Fehler",
        "axial":             "AXIAL",
        "sagittal":          "SAGITTAL",
        "coronal":           "KORONAL",
        "this_case":         "Dieses Bild",
        "avg_over":          "Ø über",
        "cases":             "Fälle",
        "radiologist_ref":   "Radiologen Inter-Rater Dice: 0.74–0.85 (Menze et al. 2015)",
        "pdf_btn":           "PDF-Befundbericht herunterladen",
        "pdf_unavailable":   "PDF nicht verfügbar -- pip install reportlab",
        "pdf_report_title":  "Befundbericht Tumorsegmentierung",
        "tumor_finding":     "Tumorbefund",
        "tumor_yes":         "Ja",
        "tumor_no":          "Nein",
        "lastname_lbl":      "Nachname",
        "firstname_lbl":     "Vorname",
        "dob_lbl":           "Geburtsdatum",
        "pid_lbl":           "Patient-ID",
        "date_lbl":          "Datum",
        "time_lbl":          "Uhrzeit",
        "filename_lbl":      "Dateiname",
        "placeholder_last":  "Mustermann",
        "placeholder_first": "Max",
        "placeholder_dob":   "TT.MM.JJJJ",
        "lang_label":        "Sprache / Language",
    },
    "EN": {
        "page_title":        "TumorSeg",
        "connecting":        "Connecting to backend...",
        "backend_error":     "Backend not reachable -- please start run.bat",
        "patient_data":      "Patient Data",
        "required_note":     "* Required field",
        "lastname":          "Last Name *",
        "firstname":         "First Name *",
        "dob":               "Date of Birth",
        "patient_id":        "Patient ID *",
        "date":              "Date",
        "time":              "Time",
        "upload_section":    "Upload File & Analysis",
        "upload_label":      "Upload file (.nii.gz)",
        "analyze_btn":       "Analyze with all 3 models simultaneously",
        "required_missing":  "Please fill required fields (*required):",
        "analyzing":         "analyzing...",
        "analysis_start":    "Starting analysis...",
        "analysis_done":     "Analysis complete",
        "not_analyzed":      "Not yet analyzed",
        "unknown_error":     "Unknown error",
        "axial":             "AXIAL",
        "sagittal":          "SAGITTAL",
        "coronal":           "CORONAL",
        "this_case":         "This image",
        "avg_over":          "Avg over",
        "cases":             "cases",
        "radiologist_ref":   "Radiologist Inter-Rater Dice: 0.74–0.85 (Menze et al. 2015)",
        "pdf_btn":           "Download PDF Report",
        "pdf_unavailable":   "PDF not available -- pip install reportlab",
        "pdf_report_title":  "Tumor Segmentation Report",
        "tumor_finding":     "Tumor Finding",
        "tumor_yes":         "Yes",
        "tumor_no":          "No",
        "lastname_lbl":      "Last Name",
        "firstname_lbl":     "First Name",
        "dob_lbl":           "Date of Birth",
        "pid_lbl":           "Patient ID",
        "date_lbl":          "Date",
        "time_lbl":          "Time",
        "filename_lbl":      "Filename",
        "placeholder_last":  "Doe",
        "placeholder_first": "John",
        "placeholder_dob":   "DD.MM.YYYY",
        "lang_label":        "Sprache / Language",
    },
}

def t(key):
    lang = st.session_state.get("lang", "DE")
    return T[lang].get(key, key)

@st.cache_data(ttl=60, show_spinner=False)
def load_metrics_from_xlsx() -> dict:
    if not XLSX_PATH.exists():
        return {"avg": {}, "individual": {}}
    try:
        import pandas as pd
        df = pd.read_excel(XLSX_PATH)
        models_dice = {}
        models_sen  = {}
        individual  = {}
        for _, row in df.iterrows():
            m    = str(row.get("Modell", ""))
            fall = str(row.get("Fall", ""))
            try:
                d = float(row["Dice"])
                if pd.notna(d):
                    models_dice.setdefault(m, []).append(d)
                    individual.setdefault((fall, m), {})["dice"] = d
            except (ValueError, TypeError):
                pass
            try:
                s = float(row["Sensitivitaet"])
                if pd.notna(s):
                    models_sen.setdefault(m, []).append(s)
                    individual.setdefault((fall, m), {})["sen"] = s
            except (ValueError, TypeError):
                pass
        avg = {
            m: {
                "avg_dice": sum(models_dice.get(m, [])) / len(models_dice[m]) if models_dice.get(m) else None,
                "avg_sen":  sum(models_sen.get(m, []))  / len(models_sen[m])  if models_sen.get(m)  else None,
                "n": len(models_dice.get(m, [])),
            }
            for m in set(list(models_dice.keys()) + list(models_sen.keys()))
        }
        return {"avg": avg, "individual": individual}
    except Exception:
        return {"avg": {}, "individual": {}}

def check_api() -> bool:
    for _ in range(6):
        try:
            return httpx.get(f"{API_URL}/models/status", timeout=15).status_code == 200
        except httpx.ReadTimeout:
            continue
        except Exception:
            time.sleep(2)
    return False

@st.cache_data(ttl=8, show_spinner=False)
def get_status() -> dict:
    try:
        return httpx.get(f"{API_URL}/models/status", timeout=60).json()
    except Exception:
        return {"gen1": False, "gen3": False, "gen4": False}

def call_analyze(gen: int, data: bytes, name: str) -> dict:
    try:
        r = httpx.post(
            f"{API_URL}/analyze/{gen}",
            files={"file": (name, data, "application/octet-stream")},
            timeout=300,
        )
        return r.json() if r.status_code == 200 else {"success": False, "error": r.json().get("detail", "?")}
    except Exception as e:
        return {"success": False, "error": str(e)}

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def generate_pdf(patient: dict, results: dict, fname: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, Image as RLImage, HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        lang = st.session_state.get("lang", "DE")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        styles = getSampleStyleSheet()

        def ps(n, **kw):
            return ParagraphStyle(n, parent=styles["Normal"], **kw)

        title_s = ps("T", fontSize=14, fontName="Helvetica-Bold",
                     spaceAfter=4, textColor=colors.HexColor("#0f3460"))
        label_s = ps("L", fontSize=8, fontName="Helvetica",
                     textColor=colors.HexColor("#6b7280"))
        value_s = ps("V", fontSize=10, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#1a1a2e"))
        sub_s   = ps("S", fontSize=9, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#0f3460"), spaceBefore=8)
        note_s  = ps("N", fontSize=7.5, fontName="Helvetica-Oblique",
                     textColor=colors.HexColor("#9ca3af"))

        story = []
        story.append(Paragraph(T[lang]["pdf_report_title"], title_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
        story.append(Spacer(1, 0.4*cm))

        def cell(lbl, val):
            return [Paragraph(lbl, label_s), Paragraph(str(val) if val else "--", value_s)]

        left_rows = [
            cell(T[lang]["lastname_lbl"],  patient.get("name", "")),
            cell(T[lang]["firstname_lbl"], patient.get("vorname", "")),
            cell(T[lang]["dob_lbl"],       patient.get("geburtsdatum", "")),
            cell(T[lang]["pid_lbl"],       patient.get("patient_id", "")),
        ]
        right_rows = [
            cell(T[lang]["date_lbl"],     patient.get("datum", "")),
            cell(T[lang]["time_lbl"],     patient.get("uhrzeit", "")),
            cell(T[lang]["filename_lbl"], fname),
        ]

        def info_tbl(rows):
            tbl = Table([[r[0], r[1]] for r in rows], colWidths=[3*cm, 5.5*cm])
            tbl.setStyle(TableStyle([
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ]))
            return tbl

        combo = Table([[info_tbl(left_rows), info_tbl(right_rows)]], colWidths=[8.5*cm, 8.5*cm])
        combo.setStyle(TableStyle([
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(combo)
        story.append(Spacer(1, 0.25*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
        story.append(Spacer(1, 0.2*cm))

        for gen, title, arch, _ in GEN_CONFIGS:
            res = results.get(str(gen))
            if not res or not res.get("success"):
                continue
            story.append(Paragraph(arch, sub_s))

            views     = res.get("views", {})
            img_cells = []
            axis_labels = [
                ("axial",    T[lang]["axial"].capitalize()),
                ("sagittal", T[lang]["sagittal"].capitalize()),
                ("coronal",  T[lang]["coronal"].capitalize()),
            ]
            for axis, label in axis_labels:
                b64 = views.get(axis, {}).get("image_base64")
                if b64:
                    pil = b64_to_pil(b64)
                    ib  = io.BytesIO()
                    pil.save(ib, format="PNG")
                    ib.seek(0)
                    img_cells.append([RLImage(ib, width=3.8*cm, height=3.8*cm), Paragraph(label, note_s)])

            if img_cells:
                while len(img_cells) < 3:
                    img_cells.append([Paragraph("", note_s), Paragraph("", note_s)])
                it = Table(
                    [[c[0] for c in img_cells], [c[1] for c in img_cells]],
                    colWidths=[4.5*cm, 4.5*cm, 4.5*cm],
                )
                it.setStyle(TableStyle([
                    ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(it)
            story.append(Spacer(1, 0.15*cm))

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return b""

st.set_page_config(page_title="TumorSeg", layout="wide", initial_sidebar_state="collapsed")

CSS = (
    '<style>'
    '[data-testid="stFileUploaderDropzone"] ~ div > button[kind="minimal"] { display: none !important; }'
    '[data-testid="fileUploaderDeleteBtn"] {'
    '    display: flex !important; position: absolute !important;'
    '    top: -6px !important; right: -6px !important;'
    '    width: 18px !important; height: 18px !important; min-width: unset !important;'
    '    padding: 0 !important; border-radius: 50% !important;'
    '    background: #e5e7eb !important; border: none !important;'
    '    font-size: 10px !important; line-height: 18px !important;'
    '    justify-content: center !important; align-items: center !important; cursor: pointer !important;'
    '}'
    '[data-testid="stFileUploaderFile"] { position: relative !important; }'
    '[data-testid="stFileUploaderDropzone"] small { display: none !important; }'
    '#MainMenu { visibility: hidden; }'
    'header[data-testid="stHeader"] { display: none; }'
    'footer { display: none; }'
    '.stDeployButton { display: none; }'
    '[data-testid="stToolbar"] { display: none; }'
    '[data-testid="stDecoration"] { display: none; }'
    'body, .stApp { background-color: #f4f6f9; color: #1a1a2e; }'
    '.page-title { font-size: 1.4rem; font-weight: 700; color: #0f3460; margin-bottom: 0.2rem; }'
    '.section-title {'
    '    font-size: 0.85rem; font-weight: 700; color: #0f3460;'
    '    margin: 0.8rem 0 0.4rem 0; text-transform: uppercase;'
    '    letter-spacing: 0.06em; border-bottom: 2px solid #0f3460; padding-bottom: 4px;'
    '}'
    '.gen-header { font-size: 1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }'
    '.report-box {'
    '    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;'
    '    padding: 0.75rem; margin-top: 0.5rem;'
    '    font-size: 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);'
    '}'
    '.report-row { display: flex; justify-content: space-between; margin: 4px 0; }'
    '.rl { color: #6b7280; }'
    '.rv { color: #1a1a2e; font-weight: 500; }'
    'div[data-testid="stImage"] img {'
    '    border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'
    '}'
    '.stButton > button {'
    '    background: #0f3460; color: #ffffff; border: none; border-radius: 8px;'
    '    padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.2s; width: 100%;'
    '}'
    '.stButton > button:hover { background: #16213e; color: #ffffff; }'
    '.view-label {'
    '    text-align: center; font-size: 0.72rem; color: #6b7280;'
    '    margin-bottom: 4px; font-weight: 600; letter-spacing: 0.05em;'
    '}'
    '.stFileUploader { background: #ffffff; border-radius: 8px; }'
    'div[data-testid="stFileUploader"] { background: #ffffff; }'
    '</style>'
)
st.markdown(CSS, unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "DE"

lang_col, _ = st.columns([1, 9])
with lang_col:
    chosen = st.radio(
        t("lang_label"),
        options=["DE", "EN"],
        index=0 if st.session_state["lang"] == "DE" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="lang_radio",
    )
    if chosen != st.session_state["lang"]:
        st.session_state["lang"] = chosen
        st.rerun()

with st.spinner(t("connecting")):
    api_ok = check_api()

if not api_ok:
    st.error(t("backend_error"))
    st.stop()

st.divider()

st.markdown(f'<div class="section-title">{t("patient_data")}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.6rem">{t("required_note")}</div>', unsafe_allow_html=True)

now         = datetime.datetime.now()
default_pid = f"PAT-{now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

col1, col2, col3 = st.columns(3)
with col1:
    pat_name = st.text_input(t("lastname"),    key="pat_name",    placeholder=t("placeholder_last"))
    pat_geb  = st.text_input(t("dob"),         key="pat_geb",     placeholder=t("placeholder_dob"))
with col2:
    pat_vorname = st.text_input(t("firstname"), key="pat_vorname", placeholder=t("placeholder_first"))
    pat_id      = st.text_input(t("patient_id"),key="pat_id",      value=default_pid)
with col3:
    pat_datum   = st.text_input(t("date"),      key="pat_datum",   value=now.strftime("%d.%m.%Y"))
    pat_uhrzeit = st.text_input(t("time"),      key="pat_uhrzeit", value=now.strftime("%H:%M"))

patient_info = {
    "name":         pat_name,
    "vorname":      pat_vorname,
    "geburtsdatum": pat_geb,
    "patient_id":   pat_id,
    "datum":        pat_datum,
    "uhrzeit":      pat_uhrzeit,
}

st.divider()

st.markdown(f'<div class="section-title">{t("upload_section")}</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(t("upload_label"), key="tumor_upload")

if uploaded:
    file_bytes = uploaded.read()
    fname      = uploaded.name
    st.divider()

    for gen, *_ in GEN_CONFIGS:
        if f"res_{gen}" not in st.session_state:
            st.session_state[f"res_{gen}"] = None

    required_missing = [f for f, v in [
        (t("lastname")[:-2],    pat_name),
        (t("firstname")[:-2],   pat_vorname),
        (t("patient_id")[:-2],  pat_id),
    ] if not v.strip()]

    if required_missing:
        st.warning(f"{t('required_missing')} {', '.join(required_missing)}")

    if st.button(t("analyze_btn"), type="primary",
                 use_container_width=True, disabled=bool(required_missing)):
        bar       = st.progress(0, text=t("analysis_start"))
        collected = {}
        for i, (gen, title, arch, _) in enumerate(GEN_CONFIGS):
            bar.progress(i / 3, text=f"{title} {t('analyzing')}")
            res = call_analyze(gen, file_bytes, fname)
            st.session_state[f"res_{gen}"] = res
            collected[str(gen)] = res
        bar.progress(1.0, text=t("analysis_done"))
        time.sleep(0.5)
        st.rerun()

    st.divider()

    xlsx_data  = load_metrics_from_xlsx()
    avg_map    = xlsx_data.get("avg", {})
    indiv_map  = xlsx_data.get("individual", {})
    _base      = Path(fname).name.replace(".nii.gz", "").replace(".nii", "")
    case_stems = [_base, _base + ".nii"]

    cols = st.columns(3, gap="medium")
    for col, (gen, title, arch, desc) in zip(cols, GEN_CONFIGS):
        with col:
            st.markdown(f'<div class="gen-header">{arch}</div>', unsafe_allow_html=True)
            key   = MODELL_KEYS.get(gen, "")
            info  = avg_map.get(key, {})
            indiv = next(
                (indiv_map.get((s, key)) for s in case_stems if indiv_map.get((s, key)) is not None),
                None,
            )

            if indiv:
                d_val = indiv.get("dice")
                s_val = indiv.get("sen")
                parts = []
                if d_val is not None:
                    parts.append(f'Dice: <b style="color:#0f3460">{d_val:.3f}</b>')
                if s_val is not None:
                    sens_lbl = "Sensitivität" if st.session_state.get("lang", "DE") == "DE" else "Sensitivity"
                    parts.append(f'{sens_lbl}: <b style="color:#0f3460">{s_val:.3f}</b>')
                if parts:
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#6b7280;margin-bottom:2px;">'
                        f'{t("this_case")} (<b>{_base}</b>): {" / ".join(parts)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if info:
                avg_d = info.get("avg_dice")
                avg_s = info.get("avg_sen")
                n     = info.get("n", 0)
                row1  = []
                if avg_d is not None:
                    row1.append(f'Dice: <b style="color:#0f3460">{avg_d:.3f}</b>')
                if avg_s is not None:
                    sens_lbl = "Sensitivität" if st.session_state.get("lang", "DE") == "DE" else "Sensitivity"
                    row1.append(f'{sens_lbl}: <b style="color:#0f3460">{avg_s:.3f}</b>')
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#6b7280;margin-bottom:2px;">'
                    f'{t("avg_over")} <b>{n} {t("cases")}</b> (Task01): {" / ".join(row1)}'
                    f'</div>'
                    f'<div style="font-size:0.65rem;color:#9ca3af;margin-bottom:6px;">'
                    f'{t("radiologist_ref")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            res = st.session_state.get(f"res_{gen}")
            if res is None:
                st.markdown(
                    f'<div style="color:#9ca3af;text-align:center;padding:2rem 0;font-size:0.85rem">'
                    f'{t("not_analyzed")}</div>',
                    unsafe_allow_html=True,
                )
            elif not res.get("success"):
                st.error(res.get("error", t("unknown_error")))
            else:
                views = res.get("views")
                if views:
                    v_ax  = views.get("axial", {})
                    v_sag = views.get("sagittal", {})
                    v_cor = views.get("coronal", {})
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="view-label">{t("axial")}</div>', unsafe_allow_html=True)
                        if v_ax.get("image_base64"):
                            st.image(b64_to_pil(v_ax["image_base64"]), use_container_width=True)
                    with c2:
                        st.markdown(f'<div class="view-label">{t("sagittal")}</div>', unsafe_allow_html=True)
                        if v_sag.get("image_base64"):
                            st.image(b64_to_pil(v_sag["image_base64"]), use_container_width=True)
                    with c3:
                        st.markdown(f'<div class="view-label">{t("coronal")}</div>', unsafe_allow_html=True)
                        if v_cor.get("image_base64"):
                            st.image(b64_to_pil(v_cor["image_base64"]), use_container_width=True)
                else:
                    img = b64_to_pil(res["image_base64"])
                    st.image(img, use_container_width=True,
                             caption=f"{t('axial')} #{res.get('slice_index', '?')}")

    any_result = any(st.session_state.get(f"res_{gen}") for gen, *_ in GEN_CONFIGS)
    if any_result:
        st.divider()
        all_results = {str(gen): st.session_state.get(f"res_{gen}") for gen, *_ in GEN_CONFIGS}
        pdf_bytes   = generate_pdf(patient_info, all_results, fname)
        if pdf_bytes:
            lang       = st.session_state.get("lang", "DE")
            def _safe(s): return (s or "").strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
            if lang == "DE":
                pdf_name = f"Befundbericht_{_safe(pat_id)}_{_safe(pat_name)}_{_safe(pat_vorname)}_{_safe(pat_geb)}.pdf"
            else:
                pdf_name = f"Report_{_safe(pat_id)}_{_safe(pat_name)}_{_safe(pat_vorname)}_{_safe(pat_geb)}.pdf"
            st.download_button(
                label=t("pdf_btn"),
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(t("pdf_unavailable"))
