import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """NLI Model - For detecting Contradiction / Entailment"""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """Similarity Model - To find matching sentences before NLI"""
    # Agar possible ho toh 'all-mpnet-base-v2' use karna, wo zyada accurate hai
    return SentenceTransformer('all-MiniLM-L6-v2') 


# ─── Config & Labels ──────────────────────────────────────────────────────────

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {"contradiction": "❌", "entailment": "✅", "neutral": "🤷"}
LABEL_COLOR = {"contradiction": "#ff4b4b", "entailment": "#00c853", "neutral": "#888888"}


# ─── Text Processing ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences(text: str, max_sentences: int = 15) -> list:
    """Paragraphs ko sentences mein break karta hai aur chote/faltu sentences remove karta hai."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 5]
    
    unique_sentences = []
    for s in sentences:
        if s not in unique_sentences:
            unique_sentences.append(s)
            
    return unique_sentences[:max_sentences]


# ─── Main Analysis Engine (The Best Approach) ─────────────────────────────────

def analyze_articles(nli_model: CrossEncoder, sim_model: SentenceTransformer, articles: list) -> list:
    # Step 1: Har article ke sentences aur unke embeddings (vectors) ek hi baar nikal lo
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        # Embeddings calculate kar ke store kar rahe hain taake fast ho
        embeddings = sim_model.encode(sents, convert_to_tensor=True) if sents else None
        parsed.append({
            'source': art['source'], 
            'sentences': sents,
            'embeddings': embeddings
        })

    findings = []

    # Step 2: Articles ko pairs mein compare karo (A vs B, B vs C, etc.)
    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]
        pair_results = []

        if art_i['embeddings'] is None or art_j['embeddings'] is None:
            continue

        # ✨ THE MAGIC: Pura Matrix ek sath compare ho raha hai (Super Fast)
        cosine_scores = util.cos_sim(art_i['embeddings'], art_j['embeddings'])

        for idx_i, s1 in enumerate(art_i['sentences']):
            for idx_j, s2 in enumerate(art_j['sentences']):
                
                sim_score = float(cosine_scores[idx_i][idx_j])

                # 🛡️ STRICT SEMANTIC FILTER:
                # Agar sentences exactly ek hi topic (e.g., dono deaths, ya dono budget) 
                # ki baat nahi kar rahe, toh NLI ko bhej kar confuse nahi karna.
                if sim_score < 0.55:
                    continue

                # Agar similarity 55% se zyada hai, tabhi NLI se check karwao
                raw_scores = nli_model.predict([(s1, s2)])[0]
                probs = F.softmax(torch.tensor(raw_scores), dim=0)
                pred_idx = int(probs.argmax())
                conf = float(probs[pred_idx])
                label = LABEL_MAP[pred_idx]

                # Sirf solid contradictions ko hi list mein daalo
                if label == 'contradiction' and conf > 0.65:
                    pair_results.append({
                        'sentence_1': s1,
                        'sentence_2': s2,
                        'label': label,
                        'confidence': conf,
                        'similarity': sim_score
                    })

        # Jo sentences aapas mein sabse zyada similar the (par contradict kar gaye), unko top pe rakho
        pair_results.sort(key=lambda x: -x['similarity'])

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results': pair_results[:5]  # Top 5 most relevant contradictions
        })

    return findings
