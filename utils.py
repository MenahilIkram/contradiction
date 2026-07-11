import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st

# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """HIGH ACCURACY SOTA NLI Model - DeBERTa-v3-base"""
    return CrossEncoder('cross-encoder/nli-deberta-v3-base')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """HIGH ACCURACY Similarity Model - MPNET-base-v2"""
    return SentenceTransformer('all-mpnet-base-v2') 


# ─── Config & Labels ──────────────────────────────────────────────────────────

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {"contradiction": "❌", "entailment": "✅", "neutral": "🤷"}
LABEL_COLOR = {"contradiction": "#ff4b4b", "entailment": "#00c853", "neutral": "#888888"}


# ─── Text Processing ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences(text: str, max_sentences: int = 100) -> list:
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 4]
    
    unique_sentences = []
    for s in sentences:
        if s not in unique_sentences:
            unique_sentences.append(s)
            
    return unique_sentences[:max_sentences]


# ─── Main Analysis Engine ─────────────────────────────────────────────────────

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

        cosine_scores = util.cos_sim(art_i['embeddings'], art_j['embeddings'])
        
        pairs_forward = []
        metadata_pairs = []

        for idx_i, s1 in enumerate(art_i['sentences']):
            for idx_j, s2 in enumerate(art_j['sentences']):
                sim_score = float(cosine_scores[idx_i][idx_j])

                # 🛡️ RELAXED SIMILARITY FILTER (0.38)
                # Taake 'cheaper' vs 'viable' jese alag words miss na hon
                if sim_score < 0.38:
                    continue
                
                pairs_forward.append((s1, s2))
                metadata_pairs.append({'s1': s1, 's2': s2, 'sim_score': sim_score})

        if not pairs_forward:
            continue

        # Single batch processing (Super Fast)
        raw_scores_f = nli_model.predict(pairs_forward, batch_size=64, show_progress_bar=False)
        
        for idx in range(len(metadata_pairs)):
            probs_f = F.softmax(torch.tensor(raw_scores_f[idx]), dim=0)
            pred_idx_f = int(probs_f.argmax())
            conf_f = float(probs_f[pred_idx_f])
            label_f = LABEL_MAP.get(pred_idx_f, "neutral")

            # 🔥 THE BALANCED PRECISION FILTER (0.85 Sweet Spot)
            # Sahi wali 3 contradictions isse upar hain, 4th false positive automatic drop ho jayegi.
            if label_f == 'contradiction' and conf_f > 0.85:
                meta = metadata_pairs[idx]
                
                # Duplicate entries filter taake cross-matching na ho
                is_duplicate = False
                for existing in pair_results:
                    if existing['sentence_1'] == meta['s1'] or existing['sentence_2'] == meta['s2']:
                        if meta['sim_score'] < existing['similarity']:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    pair_results.append({
                        'sentence_1': meta['s1'],
                        'sentence_2': meta['s2'],
                        'label': 'contradiction',
                        'confidence': conf_f,
                        'similarity': meta['sim_score']
                    })

        # Sort by high confidence
        pair_results.sort(key=lambda x: (-x['confidence'], -x['similarity']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results': pair_results[:8]
        })

    return findings
