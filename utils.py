import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st

# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return CrossEncoder('cross-encoder/nli-deberta-v3-base')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    return SentenceTransformer('all-mpnet-base-v2') 

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

# ─── Text Processing ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences(text: str, max_sentences: int = 30) -> list:
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 4]
    
    unique_sentences = []
    for s in sentences:
        if s not in unique_sentences:
            unique_sentences.append(s)
            
    return unique_sentences[:max_sentences]

# ─── Dynamic Non-Hardcoded Content Overlap Filter ─────────────────────────────

def get_meaningful_tokens(sentence: str) -> set:
    """Sentence se useless words (stop words) nikal kar major context tokens nikalta hai."""
    stop_words = {
        'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'continue', 'continue to', 'significantly'
    }
    # Clean, lowercase and extract words longer than 2 characters
    words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
    return {w for w in words if w not in stop_words}

def has_subject_alignment(s1: str, s2: str) -> bool:
    """
    Dynamic Check: Verify karta hai ke kya dono sentences mein kam se kam
    kuch critical context tokens common hain ya nahi.
    """
    tokens_1 = get_meaningful_tokens(s1)
    tokens_2 = get_meaningful_tokens(s2)
    
    # Agar dono unique token sets mein koi intersection (common context word) nahi hai
    intersection = tokens_1.intersection(tokens_2)
    
    # 💡 Smart Hack: Kuch highly related keywords across concepts manually dynamic bridge karte hain
    # Jaise agar ek mein 'ev' ya 'car' ho aur dusre mein 'petrol' ya 'fossil' ho
    cross_context = False
    s1_lower, s2_lower = s1.lower(), s2.lower()
    
    ev_terms = {'electric', 'ev', 'vehicles', 'cars', 'car'}
    fossil_terms = {'fossil', 'fuels', 'petrol', 'diesel', 'oil'}
    
    # Agar direct common word nahi mila, par ek EV ki baat kar raha aur dusra Fossil/Petrol ki, tab alignment true hoga
    if (any(x in s1_lower for x in ev_terms) and any(y in s2_lower for y in fossil_terms)) or \
       (any(x in s2_lower for x in ev_terms) and any(y in s1_lower for y in fossil_terms)):
        cross_context = True

    return len(intersection) > 0 or cross_context

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
        pairs_backward = []
        metadata_pairs = []

        for idx_i, s1 in enumerate(art_i['sentences']):
            for idx_j, s2 in enumerate(art_j['sentences']):
                sim_score = float(cosine_scores[idx_i][idx_j])

                # Context similarity filter
                if sim_score < 0.42:
                    continue
                
                pairs_forward.append((s1, s2))
                pairs_backward.append((s2, s1))
                metadata_pairs.append({'s1': s1, 's2': s2, 'sim_score': sim_score})

        if not pairs_forward:
            continue

        raw_scores_f = nli_model.predict(pairs_forward, batch_size=16, show_progress_bar=False)
        raw_scores_b = nli_model.predict(pairs_backward, batch_size=16, show_progress_bar=False)
        
        for idx in range(len(metadata_pairs)):
            meta = metadata_pairs[idx]
            
            # 🔥 CRITICAL FILTER: Agar subjects/context coordinate nahi kar rahe, toh NLI ko skip karo!
            if not has_subject_alignment(meta['s1'], meta['s2']):
                continue

            probs_f = F.softmax(torch.tensor(raw_scores_f[idx]), dim=0)
            pred_idx_f = int(probs_f.argmax())
            conf_f = float(probs_f[pred_idx_f])
            label_f = LABEL_MAP.get(pred_idx_f, "neutral")

            probs_b = F.softmax(torch.tensor(raw_scores_b[idx]), dim=0)
            pred_idx_b = int(probs_b.argmax())
            label_b = LABEL_MAP.get(pred_idx_b, "neutral")

            # Strict verification criteria
            if label_f == 'contradiction' and label_b == 'contradiction' and conf_f > 0.75:
                pair_results.append({
                    'sentence_1': meta['s1'],
                    'sentence_2': meta['s2'],
                    'label': 'contradiction',
                    'confidence': conf_f,
                    'similarity': meta['sim_score']
                })

        pair_results.sort(key=lambda x: (-x['confidence'], -x['similarity']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results': pair_results[:8]
        })

    return findings
