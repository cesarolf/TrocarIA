import json
import re

import easyocr
import numpy as np
import streamlit as st
from PIL import Image
from spellchecker import SpellChecker

from analysis import ERROR_CATEGORIES, ERROR_CATEGORY_DESCRIPTIONS, analyze_words, build_features


st.set_page_config(page_title="Traço | Escrita", page_icon="T", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #e8eee8; --muted: #b2beb4; --paper: #151a17; --surface: #202722; --surface-raised: #28322b; --line: #3c4940; --green: #9bc9a7; --coral: #ffb398; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p, label { color: var(--ink); }
    [data-testid="stCaptionContainer"], [data-testid="stFileUploader"] small { color: var(--muted); }
    a { color: var(--green); }
    .block-container { max-width: 1180px; padding-top: 2.4rem; }
    h1 { font-size: 2.1rem !important; letter-spacing: 0 !important; font-weight: 800 !important; }
    h2, h3 { letter-spacing: 0 !important; }
    .eyebrow { font: 500 0.76rem 'DM Mono', monospace; color: var(--green); text-transform: uppercase; }
    .subtle { color: var(--muted); font-size: 0.93rem; }
    .section-label { border-bottom: 1px solid var(--line); padding-bottom: .6rem; margin: .5rem 0 1rem; font-weight: 700; }
    .notice { border-left: 3px solid var(--coral); padding: .7rem 1rem; background: #30251f; color: #f4d9cd; font-size: .88rem; }
    .word-chip { display: inline-block; margin: .18rem; padding: .28rem .55rem; border: 1px solid var(--line); background: var(--surface-raised); color: var(--ink); font-family: 'DM Mono', monospace; font-size: .85rem; }
    div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: .85rem 1rem; border-radius: 4px; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { background: var(--surface); color: var(--ink); border-color: var(--line); }
    [data-testid="stFileUploader"] section { background: var(--surface); border-color: #607568; }
    [data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line); }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--ink); }
    div.stButton > button[kind="primary"] { background: #31583e; border-color: #6e9c79; color: #f4f7f2; }
    div.stButton > button[kind="secondary"] { background: var(--surface-raised); border-color: var(--line); color: var(--ink); }
    div.stButton > button:hover { border-color: var(--green); color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparando o leitor de escrita...")
def get_reader():
    return easyocr.Reader(["pt"], gpu=False)


@st.cache_resource
def get_spellchecker():
    return SpellChecker(language="pt")


def recognize_image(image):
    reader = get_reader()
    results = reader.readtext(np.array(image), detail=1, paragraph=False)
    lines = [re.sub(r"\s+", " ", text).strip() for _, text, _ in results if text.strip()]
    confidence = [float(score) for _, _, score in results]
    return " ".join(lines), confidence


def spelling_suggestions(words):
    checker = get_spellchecker()
    flagged = []
    for word in words:
        clean = re.sub(r"[^\wÀ-ÿ'-]", "", word, flags=re.UNICODE)
        if clean and clean.lower() in checker.unknown([clean.lower()]):
            flagged.append({"detected": clean, "type": "Possível erro ortográfico", "suggestions": list(checker.candidates(clean.lower()) or [])[:3]})
    return flagged


def find_differences(transcription, reference):
    findings = analyze_words(transcription, reference)
    if not reference.strip():
        findings = spelling_suggestions(transcription.split())
    return findings


def render_findings(findings):
    if not findings:
        st.success("Nenhum possível erro foi sinalizado nesta análise.")
        return
    for item in findings:
        detected = item.get("detected", "")
        expected = item.get("expected")
        label = item["type"]
        detail = f"{detected} → {expected}" if expected else detected
        suggestions = item.get("suggestions", [])
        if suggestions:
            detail += f"  ·  Sugestões: {', '.join(suggestions)}"
        st.markdown(f"- **{label}:** {detail}")
        if item.get("requires_visual_review"):
            st.caption("Confirme o traçado diretamente na imagem; a transcrição OCR não preserva a forma visual das letras.")


st.markdown('<div class="eyebrow">Fonoaudiologia · apoio à análise</div>', unsafe_allow_html=True)
st.title("Leitura assistida da escrita")
st.markdown('<p class="subtle">Transcrição de palavras manuscritas e sinalização de possíveis diferenças ortográficas.</p>', unsafe_allow_html=True)
st.markdown('<div class="notice"><strong>Sugestões para revisão profissional, não diagnóstico.</strong> O OCR não foi ajustado especificamente para escrita clínica manuscrita. Categorias que dependem do traçado precisam ser confirmadas na imagem.</div>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([0.92, 1.08], gap="large")
with left:
    st.markdown('<div class="section-label">01 · Amostra</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Imagem da escrita", type=["png", "jpg", "jpeg", "webp"], help="Prefira uma foto nítida, bem iluminada e sem inclinação.")
    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
    reference = st.text_area(
        "Texto esperado (opcional)",
        placeholder="Digite as palavras que foram solicitadas ao paciente para comparar a escrita.",
        help="A comparação com uma referência conhecida é mais confiável do que a revisão baseada apenas em dicionário.",
        height=100,
        key="reference_text",
    )
    run_analysis = st.button("Analisar imagem", type="primary", disabled=image is None, use_container_width=True)

with right:
    st.markdown('<div class="section-label">02 · Resultado</div>', unsafe_allow_html=True)
    if run_analysis and image:
        try:
            with st.spinner("Lendo a amostra..."):
                transcription, confidence = recognize_image(image)
                findings = find_differences(transcription, reference)
                st.session_state["transcription_review"] = transcription
                st.session_state["analysis_result"] = {
                    "transcription": transcription,
                    "confidence": confidence,
                    "findings": findings,
                    "features": build_features(transcription.split(), findings),
                    "reference": reference,
                }
        except Exception as error:
            st.error(f"Não foi possível concluir a leitura: {error}")

    result = st.session_state.get("analysis_result")
    if result:
        st.markdown("**Transcrição (revise antes da comparação)**")
        if "transcription_review" not in st.session_state:
            st.session_state["transcription_review"] = result["transcription"]
        st.text_area("Transcrição revisável", height=95, label_visibility="collapsed", key="transcription_review")
        if st.button("Recalcular diferenças"):
            result["transcription"] = st.session_state["transcription_review"]
            result["reference"] = st.session_state.get("reference_text", "")
            result["findings"] = find_differences(result["transcription"], result["reference"])
            result["features"] = build_features(result["transcription"].split(), result["findings"])
            st.session_state["analysis_result"] = result
        metrics = st.columns(3)
        metrics[0].metric("Palavras", len(result["transcription"].split()))
        metrics[1].metric("Apontamentos", len(result["findings"]))
        average = sum(result["confidence"]) / len(result["confidence"]) if result["confidence"] else 0
        metrics[2].metric("Confiança OCR", f"{average:.0%}" if result["confidence"] else "—")
        st.markdown("**Possíveis diferenças**")
        render_findings(result["findings"])
        if not result["reference"]:
            st.caption("Sem texto esperado: os apontamentos usam um dicionário e podem sinalizar nomes próprios ou palavras válidas fora do vocabulário.")
        st.download_button("Baixar análise em JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name="analise_escrita.json", mime="application/json")
    else:
        st.info("Envie uma imagem para iniciar a leitura.")

st.divider()
st.markdown('<div class="section-label">03 · Categorias e rede neural</div>', unsafe_allow_html=True)
st.markdown("As regras textuais sugerem categorias dos slides quando há texto esperado. A rede neural está definida para as dez classes, mas ainda não foi treinada e não gera previsões.")
with st.expander("Ver as dez categorias dos slides"):
    for category, description in zip(ERROR_CATEGORIES, ERROR_CATEGORY_DESCRIPTIONS):
        st.markdown(f"- **{category}:** {description}")
with st.expander("Hiperparâmetros iniciais sugeridos"):
    st.markdown(
        """
        - **Entrada:** 10 contagens, uma por categoria, normalizadas antes do treino.
        - **Camadas ocultas:** 64 e 32 neurônios, com **ReLU** e dropout de 0,2.
        - **Saída:** 10 neurônios, um por categoria, com **softmax** para probabilidades.
        - **Treino inicial:** Adam, taxa de aprendizado 0,001, batch size 32, até 50 épocas, early stopping (paciência 5) e validação de 20%.
        - **Ajuste obrigatório:** treinar com dados rotulados representativos, separados por paciente, e acompanhar métricas por classe.
        """
    )
    st.caption("Esses valores são um ponto de partida experimental. A rede ainda não foi treinada e não deve ser usada para decisões clínicas.")