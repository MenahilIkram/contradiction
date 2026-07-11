import streamlit as st
from utils import (
    load_model, 
    load_similarity_model, 
    analyze_articles, 
    LABEL_EMOJI, 
    LABEL_COLOR
)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContradictAI",
    page_icon="🔍",
    layout="wide"
)

# ─── Custom CSS — Dark Glassmorphism ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}

/* Result cards */
.contradiction-card {
    background: rgba(255, 75, 75, 0.08);
    border: 1px solid rgba(255, 75, 75, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
}
.entailment-card {
    background: rgba(0, 200, 83, 0.08);
    border: 1px solid rgba(0, 200, 83, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
}
.neutral-card {
    background: rgba(150,150,150,0.06);
    border: 1px solid rgba(150,150,150,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
}

/* Sentence text */
.sent-text {
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
    margin: 4px 0;
}
.source-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9b7fe8;
    margin-bottom: 4px;
}

/* Stats bar */
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-number {
    font-size: 32px;
    font-weight: 700;
    margin: 0;
}
.stat-label {
    font-size: 12px;
    color: #888;
    margin: 0;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}

/* Section header */
.pair-header {
    font-size: 16px;
    font-weight: 600;
    color: #c9b8f0;
    padding: 8px 0 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 12px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(135deg, #9b7fe8, #6ee7f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.subtitle {
    color: #888;
    font-size: 16px;
    margin-top: 4px;
}

/* Input labels */
.stTextArea label, .stTextInput label {
    color: #c0c0c0 !important;
    font-size: 13px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #9b7fe8, #6ee7f7);
    color: #000;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 10px 32px;
    font-size: 15px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 32px 0 24px'>
    <p class='main-title'>🔍 ContradictAI</p>
    <p class='subtitle'>Paste articles on the same topic — AI will find where they contradict each other</p>
</div>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading models (first load takes ~30s)..."):
    model     = load_model()
    sim_model = load_similarity_model()
st.success("✅ Models loaded!", icon="🤖")


# ─── Article Input Section ─────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Articles")
st.caption("Add 2–5 articles on the **same topic** (news, research, reviews — any domain)")

num_articles = st.slider("Number of articles", min_value=2, max_value=5, value=2)

articles = []
cols_per_row = 2
rows = (num_articles + cols_per_row - 1) // cols_per_row

art_idx = 0
for row in range(rows):
    cols = st.columns(cols_per_row)
    for col in cols:
        if art_idx >= num_articles:
            break
        with col:
            st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
            source = st.text_input(
                f"Source name (e.g. Dawn, BBC, Review 1)",
                value=f"Source {art_idx + 1}",
                key=f"source_{art_idx}"
            )
            text = st.text_area(
                f"Article {art_idx + 1} text",
                height=200,
                placeholder="Paste article content here...",
                key=f"text_{art_idx}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            articles.append({'source': source, 'text': text})
        art_idx += 1


# ─── Analyze Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button("🔍 Detect Contradictions")


# ─── Results ──────────────────────────────────────────────────────────────────
if analyze_clicked:

    # Validate inputs
    filled = [a for a in articles if a['text'].strip()]
    if len(filled) < 2:
        st.error("Please fill in at least 2 articles before analyzing.")
        st.stop()

    with st.spinner("🧠 Analyzing articles... this may take a moment"):
        findings = analyze_articles(model, sim_model, filled)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Results")

    # ── Summary Stats ──
    total_contradictions = sum(
        sum(1 for r in f['results'] if r['label'] == 'contradiction')
        for f in findings
    )
    total_agreements = sum(
        sum(1 for r in f['results'] if r['label'] == 'entailment')
        for f in findings
    )
    total_neutral = sum(
        sum(1 for r in f['results'] if r['label'] == 'neutral')
        for f in findings
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class='stat-box'>
            <p class='stat-number' style='color:#9b7fe8'>{len(filled)}</p>
            <p class='stat-label'>Articles Analyzed</p>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class='stat-box'>
            <p class='stat-number' style='color:#ff4b4b'>{total_contradictions}</p>
            <p class='stat-label'>Contradictions Found</p>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class='stat-box'>
            <p class='stat-number' style='color:#00c853'>{total_agreements}</p>
            <p class='stat-label'>Agreements Found</p>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class='stat-box'>
            <p class='stat-number' style='color:#888'>{total_neutral}</p>
            <p class='stat-label'>Neutral Pairs</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per Pair Results ──
    for finding in findings:
        src1 = finding['source_1']
        src2 = finding['source_2']
        results = finding['results']

        contradictions = [r for r in results if r['label'] == 'contradiction']
        agreements     = [r for r in results if r['label'] == 'entailment']
        neutrals       = [r for r in results if r['label'] == 'neutral']

        st.markdown(f"""
        <div class='pair-header'>
            🗞️ {src1} &nbsp;vs&nbsp; {src2}
            &nbsp;—&nbsp;
            <span style='color:#ff4b4b'>{len(contradictions)} contradiction(s)</span> &nbsp;
            <span style='color:#00c853'>{len(agreements)} agreement(s)</span>
        </div>""", unsafe_allow_html=True)

        # Show contradictions first (most important)
        if contradictions:
            st.markdown("**❌ Contradictions:**")
            for r in contradictions:
                conf_pct = int(r['confidence'] * 100)
                st.markdown(f"""
                <div class='contradiction-card'>
                    <p class='source-label'>{src1}</p>
                    <p class='sent-text'>"{r['sentence_1']}"</p>
                    <p class='source-label' style='margin-top:10px'>{src2}</p>
                    <p class='sent-text'>"{r['sentence_2']}"</p>
                    <div class='conf-bar-bg'>
                        <div style='width:{conf_pct}%; height:6px;
                             background: linear-gradient(90deg,#ff4b4b,#ff8a80);
                             border-radius:99px'></div>
                    </div>
                    <p style='font-size:11px; color:#ff4b4b; margin:4px 0 0'>
                        Confidence: {conf_pct}%
                    </p>
                </div>""", unsafe_allow_html=True)

        # Show agreements
        if agreements:
            with st.expander(f"✅ Agreements ({len(agreements)})"):
                for r in agreements:
                    conf_pct = int(r['confidence'] * 100)
                    st.markdown(f"""
                    <div class='entailment-card'>
                        <p class='source-label'>{src1}</p>
                        <p class='sent-text'>"{r['sentence_1']}"</p>
                        <p class='source-label' style='margin-top:10px'>{src2}</p>
                        <p class='sent-text'>"{r['sentence_2']}"</p>
                        <p style='font-size:11px; color:#00c853; margin:8px 0 0'>
                            Confidence: {conf_pct}%
                        </p>
                    </div>""", unsafe_allow_html=True)

        # Show neutral (collapsed by default)
        if neutrals:
            with st.expander(f"🤷 Neutral pairs ({len(neutrals)})"):
                for r in neutrals:
                    st.markdown(f"""
                    <div class='neutral-card'>
                        <p class='sent-text'><b>{src1}:</b> "{r['sentence_1']}"</p>
                        <p class='sent-text'><b>{src2}:</b> "{r['sentence_2']}"</p>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Overall Verdict ──
    if total_contradictions > 0:
        st.error(
            f"⚠️ **{total_contradictions} contradiction(s) detected** across the uploaded articles. "
            f"These sources have conflicting claims on the same topic."
        )
    else:
        st.success(
            "✅ **No significant contradictions found.** "
            "The articles appear to be largely consistent with each other."
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#444; font-size:12px; padding:40px 0 20px'>
    ContradictAI — NLP Semester Project &nbsp;|&nbsp; 
    Powered by sentence-transformers &amp; Streamlit
</div>
""", unsafe_allow_html=True)
