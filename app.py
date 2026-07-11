import streamlit as st
from utils import load_model, load_similarity_model, analyze_articles

# ─── Page Styling & Configuration ─────────────────────────────────────────────
st.set_page_config(page_title="AI Contradiction Detector", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #f5f7f8; }
    .main-title { font-size: 40px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 30px; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 5px solid #ff4b4b; }
    .source-tag { font-weight: bold; color: #1E3A8A; font-size: 14px; margin-bottom: 5px; }
    .sentence-text { font-style: italic; color: #333333; font-size: 16px; background-color: #fdf2f2; padding: 8px; border-radius: 5px; margin: 5px 0; }
    </style>
""", unsafe_allowed_html=True)

st.markdown("<div class='main-title'>🔍 AI Contradiction Detector Dashboard</div>", unsafe_allowed_html=True)

# ─── Model Lazy Loading ───────────────────────────────────────────────────────
with st.spinner("🔄 Loading High-Accuracy AI Models... Please wait (Takes a few seconds first time)"):
    nli_model = load_model()
    sim_model = load_similarity_model()

# ─── Helper Functions for File Reading ────────────────────────────────────────
def read_txt(file) -> str:
    return file.read().decode("utf-8")

def read_pdf(file) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        st.error("⚠️ Please add 'pypdf' inside your requirements.txt file to enable PDF processing!")
        return ""

def read_docx(file) -> str:
    try:
        import docx
        doc = docx.Document(file)
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        return "\n".join(text)
    except ImportError:
        st.error("⚠️ Please add 'python-docx' inside your requirements.txt file to enable DOCX processing!")
        return ""

# ─── Sidebar / File Upload Section ───────────────────────────────────────────
st.sidebar.header("📁 Upload Articles")
st.sidebar.write("Upload exactly 2 documents (PDF, DOCX, or TXT) to compare.")

# `type` parameter mein 'docx' bhi add kar diya hai
file1 = st.sidebar.file_uploader("Upload Article 1", type=["txt", "pdf", "docx"], key="art1")
file2 = st.sidebar.file_uploader("Upload Article 2", type=["txt", "pdf", "docx"], key="art2")

source1_name = file1.name if file1 else "Article 1"
source2_name = file2.name if file2 else "Article 2"

# ─── Main Execution Trigger ───────────────────────────────────────────────────
if st.sidebar.button("🚀 Analyze Documents", use_container_width=True):
    if not file1 or not file2:
        st.warning("⚠️ Please upload both Article 1 and Article 2 before processing!")
    else:
        # File 1 parsing logic based on extension
        if file1.name.endswith('.pdf'):
            text1 = read_pdf(file1)
        elif file1.name.endswith('.docx'):
            text1 = read_docx(file1)
        else:
            text1 = read_txt(file1)

        # File 2 parsing logic based on extension
        if file2.name.endswith('.pdf'):
            text2 = read_pdf(file2)
        elif file2.name.endswith('.docx'):
            text2 = read_docx(file2)
        else:
            text2 = read_txt(file2)

        if text1.strip() and text2.strip():
            articles_payload = [
                {'source': source1_name, 'text': text1},
                {'source': source2_name, 'text': text2}
            ]
            
            with st.spinner("🧠 AI is processing cross-matrix logic..."):
                findings = analyze_articles(nli_model, sim_model, articles_payload)
            
            st.success("✅ Document Analysis Complete!")
            
            for finding in findings:
                st.subheader(f"🗞️ {finding['source_1']}  vs  {finding['source_2']}  —  {len(finding['results'])} contradiction(s)")
                
                if not finding['results']:
                    st.info("🎉 Exceptional Consistency: No contradictions detected across these documents!")
                else:
                    for res in finding['results']:
                        st.markdown(f"""
                            <div class="card">
                                <div class="source-tag">📄 {finding['source_1']}</div>
                                <div class="sentence-text">"{res['sentence_1']}"</div>
                                <div style="font-weight: bold; color: #ff4b4b; text-align: center; margin: 10px 0;">❌ DIRECT CONTRADICTION (Confidence: {res['confidence']*100:.1f}%) ❌</div>
                                <div class="source-tag">📄 {finding['source_2']}</div>
                                <div class="sentence-text">"{res['sentence_2']}"</div>
                            </div>
                        """, unsafe_allowed_html=True)
        else:
            st.error("❌ Could not extract enough clear text from the uploaded files. Check file encoding.")
else:
    st.info("💡 Pro-Tip: Go to the left sidebar, upload your PDF, DOCX, or TXT files, and click 'Analyze Documents' to scan contradictions dynamically.")
