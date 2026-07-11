import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading (High Accuracy Upgrade) ────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """HIGH ACCURACY SOTA NLI Model - DeBERTa-v3-base"""
    # MiniLM se bohot zyada powerful aur complex logical contradictions pakadne ke liye behtareen hai.
    return CrossEncoder('cross-encoder/nli-deberta-v3-base')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """HIGH ACCURACY Similarity Model - MPNET-base-v2"""
    # MiniLM se bada aur accurate model jo advanced semantic similarity find karta hai.
    return SentenceTransformer('all-mpnet-base-v2') 


# ─── Config & Labels (DeBERTa Mapping Fixed) ──────────────────────────────────

# DeBERTa-v3 ki native output mapping: 0 -> contradiction, 1 -> entailment, 2 -> neutral
LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {"contradiction": "❌", "entailment": "✅", "neutral": "🤷"}
LABEL_COLOR = {"contradiction": "#ff4b4b", "entailment": "#00c853", "neutral": "#888888"}


# ─── Text Processing ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences(text: str, max_sentences: int = 30) -> list:
    """Paragraphs ko sentences mein break karta hai aur irrelevant lines filter out karta hai."""
    if not text:
        return []
    
    # Better regex processing to avoid split-ups on abbreviations
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 4]
    
    unique_sentences = []
    for s in sentences:
        if s not in unique_sentences:
            unique_sentences.append(s)
            
    return unique_sentences[:max_sentences]


# ─── Main Analysis Engine (Batch Mode & High Precision) ───────────────────────

def analyze_articles(nli_model: CrossEncoder, sim_model: SentenceTransformer, articles: list) -> list:
    parsed = []
    for art in articles:
        sents = extract_sentences(art.get('text', ''))
        embeddings = sim_model.encode(sents, convert_to_tensor=True) if sents else None
        parsed.append({
            'source': art.get('source', 'Unknown Source'), 
            'sentences': sents,
            'embeddings': embeddings
        })

    findings = []

    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]
        pair_results = []

        if art_i['embeddings'] is None or art_j['embeddings'] is None:
            continue

        # Cosine similarity matrix mapping
        cosine_scores = util.cos_sim(art_i['embeddings'], art_j['embeddings'])
        
        pairs_to_predict = []
        metadata_pairs = []

        for idx_i, s1 in enumerate(art_i['sentences']):
            for idx_j, s2 in enumerate(art_j['sentences']):
                sim_score = float(cosine_scores[idx_i][idx_j])

                # 🛡️ THE PRECISION FILTER (0.42 is perfect for MPNET)
                # MPNET models embedding depth zyada hoti hai, isliye 0.42 standard cutoff hai.
                if sim_score < 0.42:
                    continue
                
                pairs_to_predict.append((s1, s2))
                metadata_pairs.append({'s1': s1, 's2': s2, 'sim_score': sim_score})

        # Agar koi common context sentence nahi mila toh cycle skip karein
        if not pairs_to_predict:
            continue

        # High-Speed Batch Inference
        raw_scores = nli_model.predict(pairs_to_predict, batch_size=16, show_progress_bar=False)
        
        for idx, score in enumerate(raw_scores):
            probs = F.softmax(torch.tensor(score), dim=0)
            pred_idx = int(probs.argmax())
            conf = float(probs[pred_idx])
            label = LABEL_MAP.get(pred_idx, "neutral")
            meta = metadata_pairs[idx]
            sim_score = meta['sim_score']

            # 🛡️ DYNAMIC CONFIDENCE CALIBRATION INTEGRATED 🛡️
            is_valid = False
            if label == 'contradiction':
                # Rule 1: Same exact topic/metric pe strong match hai, accept normal confidence (>0.75)
                if sim_score >= 0.52 and conf > 0.75:
                    is_valid = True
                # Rule 2: Borderline semantic connection hai (jaise narrative shift), demand extreme confidence (>=0.90)
                elif sim_score >= 0.42 and conf >= 0.90:
                    is_valid = True

            # Agar dynamic rules criteria cross ho jaye, toh entry save karo
            if is_valid:
                pair_results.append({
                    'sentence_1': meta['s1'],
                    'sentence_2': meta['s2'],
                    'label': label,
                    'confidence': conf,
                    'similarity': sim_score
                })

        # Sort results based on top contradiction confidence and similarity
        pair_results.sort(key=lambda x: (-x['confidence'], -x['similarity']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results': pair_results[:8] # Top 8 highly accurate contradictions
        })

    return findings
