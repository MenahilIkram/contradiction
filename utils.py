import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """NLI CrossEncoder — Language ke context aur meaning ko samajhta hai."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """Sentence similarity model — same topic check karta hai."""
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


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Universal Factual Check ──────────────────────────────────────────────────

def is_factual(sentence: str) -> bool:
    factual_patterns = [
        r'\d+', 
        r'\d+\.?\d*\s?%', 
        r'[\$£€₹¥]\s?\d+',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(kg|g|km|m|cm|mm|celsius|fahrenheit|kelvin|volts|watts|hz|pixels|miles|pounds|population)\b',
    ]
    s = sentence.lower()
    for pattern in factual_patterns:
        if re.search(pattern, s):
            return True
    return False


# ─── Vague Check ─────────────────────────────────────────────────────────────

def is_too_vague(sentence: str) -> bool:
    if len(sentence.split()) < 4:  # standard threshold
        return True
    vague_patterns = [
        r'^(he|she|they|it)\s+(is|was|are|were)\s+\w+\.$',
        r'^(authors?|writers?)\s+stated\s+that',
    ]
    s = sentence.lower().strip()
    for pattern in vague_patterns:
        if re.search(pattern, s):
            return True
    return False


# ─── Duplicate Removal ───────────────────────────────────────────────────────

def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    unique = []
    for sent in sentences:
        words_new = set(sent.lower().split())
        is_dup = False
        for existing in unique:
            words_ex = set(existing.lower().split())
            if len(words_new | words_ex) == 0:
                continue
            sim = len(words_new & words_ex) / len(words_new | words_ex)
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(sent)
    return unique


# ─── Information Richness Scoring ───────────────────────────────────────────

def score_sentence(sentence: str) -> int:
    """Sentence ko data aur heavy verbs ke hisab se rank karta hai."""
    score = 0
    s = sentence.lower()
    if re.search(r'\d+', s): score += 1
    if re.search(r'[\$£€₹¥]|%', s): score += 1
    # Strong action verbs check for semantic meaning
    if any(w in s for w in ['approved', 'rejected', 'banned', 'discovered', 'claimed', 'failed', 'won', 'lost', 'denied', 'confirmed']):
        score += 2
    return score


# ─── Sentence Extraction ─────────────────────────────────────────────────────

def extract_sentences(text: str, max_factual: int = 10, max_other: int = 10) -> list:
    """Ffactual aur pure text sentences ka ek perfect mix nikalta hai."""
    raw       = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if clean_text(s)]
    sentences = [s for s in sentences if not is_too_vague(s)]

    factual = [s for s in sentences if is_factual(s)]
    other   = [s for s in sentences if not is_factual(s)]

    factual = remove_duplicates(factual)
    other   = remove_duplicates(other)
    
    # Mix and sort by complexity/richness
    all_sents = factual[:max_factual] + other[:max_other]
    all_sents = sorted(all_sents, key=score_sentence, reverse=True)
    return all_sents


# ─── Topic Similarity Check ───────────────────────────────────────────────────

def are_same_topic(sim_model: SentenceTransformer, s1: str, s2: str, threshold: float = 0.35) -> bool:
    """Check karta hai ke kya dono sentences kisi common context par baat kar rahe hain."""
    emb1       = sim_model.encode(s1, convert_to_tensor=True)
    emb2       = sim_model.encode(s2, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    return similarity >= threshold


# ─── Dynamic Context Anchor Window ───────────────────────────────────────────

def extract_numerical_anchor_window(sentence: str) -> str:
    s = sentence.lower()
    s = re.sub(r'\b(in\s+)?(17|18|19|20)\d{2}\b', ' ', s)
    words = s.split()
    for i, word in enumerate(words):
        if any(c.isdigit() for c in word) or word in ['million', 'billion', 'thousand']:
            start_idx = max(0, i - 2)
            end_idx = min(len(words), i + 4)
            context_words = words[start_idx:i] + words[i+1:end_idx]
            if context_words:
                return " ".join(context_words)
    return ""


def extract_numbers_general(text: str) -> list:
    text = text.lower()
    text = re.sub(r'\b(in\s+)?(17|18|19|20)\d{2}\b', ' ', text)
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


def detect_numerical_contradiction(s1: str, s2: str) -> tuple:
    nums1 = extract_numbers_general(s1)
    nums2 = extract_numbers_general(s2)
    if not nums1 or not nums2: return None, 0.0
    n1, n2 = max(nums1), max(nums2)
    if n1 == 0 or n2 == 0: return None, 0.0
    diff_pct = abs(n1 - n2) / max(n1, n2)
    if diff_pct > 0.20:
        return 'contradiction', round(min(0.99, 0.65 + diff_pct), 2)
    return 'entailment', 0.85


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

                # Step 1: Semantic Topic Similarity (Context matching)
                if not are_same_topic(sim_model, s1, s2):
                    continue

                # Step 2: Base Transformer Prediction (Semantic analysis)
                label, conf = classify_pair(nli_model, s1, s2)

                # Step 3: Check for Numbers
                has_num1 = len(extract_numbers_general(s1)) > 0
                has_num2 = len(extract_numbers_general(s2)) > 0

                # ── MODE A: NUMERICAL GUARDRAIL MODE ──
                if has_num1 and has_num2:
                    anchor1 = extract_numerical_anchor_window(s1)
                    anchor2 = extract_numerical_anchor_window(s2)
                    
                    if anchor1 and anchor2:
                        emb_a1 = sim_model.encode(anchor1, convert_to_tensor=True)
                        emb_a2 = sim_model.encode(anchor2, convert_to_tensor=True)
                        anchor_sim = float(util.cos_sim(emb_a1, emb_a2))
                        
                        # Agar context different hai, to numbers different ho sakte hain (Override false positive)
                        if anchor_sim < 0.45 and label == 'contradiction':
                            label = 'neutral'
                            conf = 1.0
                        # Agar context same hai, to numbers test karo
                        elif anchor_sim >= 0.45:
                            num_label, num_conf = detect_numerical_contradiction(s1, s2)
                            if num_label is not None:
                                label = num_label
                                conf = num_conf

                # ── MODE B: PURE SEMANTIC MODE (NO NUMBERS) ──
                # Agar numbers nahi hain, to hum Transformer ke decision ko bilkul nahi charenge!
                # Wo khud antonyms, negatives (not), aur semantic clashes ko pakad rha hai.

                # Noise filtration
                if label == 'neutral' and conf < 0.75:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort results: Contradictions first
        pair_results.sort(key=lambda x: (x['label'] != 'contradiction', -x['confidence']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:10]  # Expanded to show more beautiful semantic insights
        })

    return findings

def classify_pair(nli_model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence
