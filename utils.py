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


# ─── Universal Token Overlap Filter (No Hardcoding) ───────────────────────────

def get_clean_keywords(sentence: str) -> set:
    """Useless grammar words nikal kar meaningful continuous words dhoondta hai."""
    stop_words = {
        'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'is', 'are',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'was',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'were',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'should', 'now', 'by', 'due', 'releasing', 'experienced'
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
    return {w for w in words if w not in stop_words}

def compute_token_overlap(s1: str, s2: str) -> float:
    """Dono sentences ke beech core vocabulary overlap ratio calculate karta hai."""
    tokens1 = get_clean_keywords(s1)
    tokens2 = get_clean_keywords(s2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    # Jaccard index similarity logic
    return len(intersection) / len(union)


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

                # Relaxed threshold so everything passes to token checking phase
                if sim_score < 0.28:
                    continue
                
                pairs_forward.append((s1, s2))
                metadata_pairs.append({'s1': s1, 's2': s2, 'sim_score': sim_score})

        if not pairs_forward:
            continue

        raw_scores_f = nli_model.predict(pairs_forward, batch_size=64, show_progress_bar=False)
        
        candidate_pairs = []
        
        for idx in range(len(metadata_pairs)):
            probs_f = F.softmax(torch.tensor(raw_scores_f[idx]), dim=0)
            pred_idx_f = int(probs_f.argmax())
            conf_f = float(probs_f[pred_idx_f])
            label_f = LABEL_MAP.get(pred_idx_f, "neutral")

            meta = metadata_pairs[idx]

            if label_f == 'contradiction' and conf_f > 0.70:
                # Calculate dynamic overlap index
                overlap_ratio = compute_token_overlap(meta['s1'], meta['s2'])
                
                candidate_pairs.append({
                    's1': meta['s1'],
                    's2': meta['s2'],
                    'confidence': conf_f,
                    'similarity': meta['sim_score'],
                    'overlap': overlap_ratio
                })

        # Global priority sorting based on high model precision parameters
        candidate_pairs.sort(key=lambda x: (-x['confidence'], -x['similarity']))

        # 🔥 PURE MATHEMATICAL CLUSTER FILTER (Zero Hardcoding)
        used_s1 = set()
        used_s2 = set()

        for cand in candidate_pairs:
            s1_clean = cand['s1'].strip().lower()
            s2_clean = cand['s2'].strip().lower()

            # 🛡️ THE OVERLAP DEVIATION GUARD:
            # Agar dono sentences mein absolute cross-matching ho rahi hai (overlap < 4%),
            # par embedding high hai, toh yeh mismatch cluster hai. Isko filter out karein.
            # Lekin agar overlap range 5% se upar hai ya direct polar links hain, toh allow karein.
            if cand['overlap'] < 0.05 and cand['confidence'] < 0.999:
                continue

            if s1_clean in used_s1 and s2_clean in used_s2:
                continue

            used_s1.add(s1_clean)
            used_s2.add(s2_clean)

            pair_results.append({
                'sentence_1': cand['s1'],
                'sentence_2': cand['s2'],
                'label': 'contradiction',
                'confidence': cand['confidence'],
                'similarity': cand['similarity']
            })

    pair_results.sort(key=lambda x: (-x['confidence'], -x['similarity']))

    findings.append({
        'source_1': art_i['source'],
        'source_2': art_j['source'],
        'results': pair_results[:25]
    })

    return findings
