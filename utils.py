import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """NLI CrossEncoder — Language aur Meaning ke liye."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """Sentence Transformer — Sirf performance boost (unrelated sentences skip karne) ke liye."""
    return SentenceTransformer('all-MiniLM-L6-v2')



# ─── Labels ───────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

LABEL_EMOJI = {
    "contradiction": "❌",
    "entailment":    "✅",
    "neutral":       "🤷"
}
LABEL_COLOR = {
    "contradiction": "#ff4b4b",
    "entailment":    "#00c853",
    "neutral":       "#888888"
}


# ─── Text Cleaning & Extraction ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    unique = []
    for sent in sentences:
        words_new = set(sent.lower().split())
        is_dup = False
        for existing in unique:
            words_ex = set(existing.lower().split())
            if not words_new or not words_ex: continue
            sim = len(words_new & words_ex) / len(words_new | words_ex)
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(sent)
    return unique

def extract_sentences(text: str, max_sentences: int = 15) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 4]
    sentences = sorted(sentences, key=lambda x: len(x.split()), reverse=True)
    unique_sentences = remove_duplicates(sentences)
    return unique_sentences[:max_sentences]


# ─── Pure AI "Number Masking" System ──────────────────────────────────────────

def normalize_numbers_for_nli(sentence: str) -> str:
    """
    Transformers ko bewakoof banne se rokne ke liye sab numbers ko '999' banata hai
    taake wo sirf English grammar/meaning pe focus kare.
    """
    s = sentence.lower()
    # 1. Years ko safe rakho (1900s, 2000s) taake context na toote
    s = re.sub(r'\b(in\s+)?(19|20)\d{2}\b', ' [YEAR] ', s)
    # 2. Saare actual numbers ko '999' se replace kardo
    s = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', '999', s)
    # 3. Scales aur modifiers uda do taake comparing asaan ho
    s = re.sub(r'\b(million|billion|trillion|thousand|hundred|around|about|approximately|nearly|over|under)\b', '', s)
    # Clean spaces
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_numbers_general(text: str) -> list:
    text = text.lower()
    text = re.sub(r'\b(in\s+)?(19|20)\d{2}\b', ' ', text) # Ignore years
    numbers = []
    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text): numbers.append(float(match.group(1)) * 1_000_000_000)
    for match in re.finditer(r'(\d+\.?\d*)\s*million', text): numbers.append(float(match.group(1)) * 1_000_000)
    text = re.sub(r'\b(\d{1,3})(?:,\d{3})+\b', lambda m: m.group(0).replace(',', ''), text)
    for match in re.finditer(r'\b(\d+\.?\d*)\b', text):
        try:
            val = float(match.group(1))
            if val != 0: numbers.append(val)
        except ValueError: continue
    return numbers

def detect_numerical_contradiction(nums1: list, nums2: list) -> tuple:
    if not nums1 or not nums2: return None, 0.0
    n1, n2 = max(nums1), max(nums2)
    diff_pct = abs(n1 - n2) / max(n1, n2)
    # 20% Variance Rule
    if diff_pct > 0.20:
        return 'contradiction', round(min(0.99, 0.65 + diff_pct), 2)
    return 'entailment', 0.85


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence

def are_same_topic(sim_model: SentenceTransformer, s1: str, s2: str, threshold: float = 0.25) -> bool:
    """Sirf bilkul unrelated sentences ko skip karta hai performance bachane ke liye (Loose Threshold)."""
    emb1 = sim_model.encode(s1, convert_to_tensor=True)
    emb2 = sim_model.encode(s2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)) >= threshold


# ─── Main Analysis Engine ─────────────────────────────────────────────────────

def analyze_articles(nli_model: CrossEncoder, sim_model: SentenceTransformer, articles: list) -> list:
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        parsed.append({'source': art['source'], 'sentences': sents})

    findings = []

    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]
        pair_results = []

        for s1 in art_i['sentences']:
            for s2 in art_j['sentences']:

                # Step 1: Broad Topic Match
                if not are_same_topic(sim_model, s1, s2):
                    continue

                # Step 2: AI's First Impression
                label, conf = classify_pair(nli_model, s1, s2)

                # Step 3: Extract Numbers
                nums1 = extract_numbers_general(s1)
                nums2 = extract_numbers_general(s2)

                # 🧠 THE LOGICAL GUARDRAIL 🧠
                if nums1 and nums2:
                    # Numbers ko 999 kardo taake model fair compare kare
                    s1_norm = normalize_numbers_for_nli(s1)
                    s2_norm = normalize_numbers_for_nli(s2)
                    norm_label, _ = classify_pair(nli_model, s1_norm, s2_norm)

                    if label == 'contradiction':
                        # Agar model ne pehle Contradiction bola, 
                        # par numbers same hone (999) ke baad usne Neutral keh diya:
                        # Iska matlab sentences totally different topics pe the (Killed vs Displaced)!
                        if norm_label == 'neutral':
                            label = 'neutral'
                            conf = 1.0 
                            
                    elif label in ['entailment', 'neutral']:
                        # Agar sentences structuraly exact same baaten kar rahe hain (Entailment),
                        # Toh ab hum math calculation se check karenge ke figures me contradiction to nahi.
                        if norm_label == 'entailment':
                            num_label, num_conf = detect_numerical_contradiction(nums1, nums2)
                            if num_label == 'contradiction':
                                label = 'contradiction'
                                conf = num_conf

                # Step 4: Filter Output
                if label == 'neutral':
                    continue  # Ignore neutral pairs so the output is clean

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort Results (Contradictions on top)
        pair_results.sort(key=lambda x: (x['label'] != 'contradiction', -x['confidence']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8] # Clean Top 8 most confident results
        })

    return findings
