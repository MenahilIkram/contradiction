import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """NLI CrossEncoder — contradiction/entailment/neutral classify karta hai."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """
    Sentence similarity model — check karta hai ke do sentences
    same topic pe hain ya nahi.
    Agar same topic nahi to NLI model ko call hi nahi karte — false positives avoid.
    """
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


# ─── Factual Check ────────────────────────────────────────────────────────────

def is_factual(sentence: str) -> bool:
    """
    Sentence mein factual claim hai ya nahi.
    Numbers, currency, death words, dates, measurements = factual.
    """
    factual_patterns = [
        r'\d+',
        r'\d+\.?\d*\s?%',
        r'[\$£€₹]\s?\d+',
        r'\d+\s?(million|billion|thousand|hundred)',
        r'\b(killed|died|dead|deaths|casualties|'
        r'injured|wounded|missing|arrested)\b',
        r'\b(january|february|march|april|may|june|'
        r'july|august|september|october|november|december)\b',
        r'\bin\s+\d{4}\b',
        r'\d+\s?(km|kg|meter|acre|hectare|liter)',
        r'\b(increased|decreased|rose|fell|dropped|'
        r'surged|declined)\s+by\s+\d+',
    ]
    s = sentence.lower()
    for pattern in factual_patterns:
        if re.search(pattern, s):
            return True
    return False


# ─── Vague Check ─────────────────────────────────────────────────────────────

def is_too_vague(sentence: str) -> bool:
    """
    Useless sentences filter karo.
    5 words se kam = incomplete.
    Generic filler = useless.
    """
    if len(sentence.split()) < 5:
        return True

    vague_patterns = [
        r'^(he|she|they|it)\s+(is|was|are|were)\s+\w+\.$',
        r'\b(very|quite|really|extremely)\s+(bad|good|important|serious)\b',
        r'^(officials?|authorities)\s+said\s+the\s+situation',
    ]
    s = sentence.lower().strip()
    for pattern in vague_patterns:
        if re.search(pattern, s):
            return True
    return False


# ─── Duplicate Removal ───────────────────────────────────────────────────────

def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    """
    Similar sentences remove karo using Jaccard similarity.
    common words / total unique words >= 85% = duplicate.
    """
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


# ─── Sentence Scoring ────────────────────────────────────────────────────────

def score_sentence(sentence: str) -> int:
    """
    Importance score.
    +3: death/casualty, +2: money/large numbers, +1: percentage/year/number
    """
    score = 0
    s = sentence.lower()
    if re.search(r'\b(killed|died|dead|deaths|casualties|injured)\b', s):
        score += 3
    if re.search(r'[\$£€₹]\s?\d+|\d+\s?(million|billion)', s):
        score += 2
    if re.search(r'\d+\.?\d*\s?%', s):
        score += 1
    if re.search(r'\bin\s+\d{4}\b', s):
        score += 1
    if re.search(r'\d+', s):
        score += 1
    return score


# ─── Sentence Extraction ─────────────────────────────────────────────────────

def extract_sentences(text: str,
                       max_factual: int = 5,
                       max_other: int = 4) -> list:
    """
    Article se top sentences nikalo — dono types:
    - Factual: numbers/currency/deaths (numeric contradictions)
    - Non-factual: pure language (semantic contradictions)
    """
    raw       = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if clean_text(s)]
    sentences = [s for s in sentences if not is_too_vague(s)]

    factual = [s for s in sentences if is_factual(s)]
    other   = [s for s in sentences if not is_factual(s)]

    factual = remove_duplicates(factual)
    other   = remove_duplicates(other)

    factual = sorted(factual, key=score_sentence, reverse=True)

    return factual[:max_factual] + other[:max_other]


# ─── Topic Similarity Check ───────────────────────────────────────────────────

def are_same_topic(sim_model: SentenceTransformer,
                   s1: str, s2: str,
                   threshold: float = 0.35) -> bool:
    """
    Check karo ke dono sentences same topic pe hain ya nahi.

    Cosine similarity use karta hai sentence embeddings pe.
    0.35 se kam = completely alag topic = NLI call skip karo.

    YE FIX KYUN ZAROORI THA:
    Pehle model "900 deaths" vs "UN $160M appeal" ko
    97% contradiction bol raha tha — kyunki dono mein
    alag numbers hain but topic hi same nahi tha.
    Ab ye filter pehle check karta hai — alag topic = skip.

    Threshold 0.35 kyun:
    - 0.0  = completely unrelated  (deaths vs cricket score)
    - 0.35 = loosely same topic    (deaths vs relief money — both flood related)
    - 0.7+ = almost same sentence
    0.35 pe rakh rahe hain taake same-event different-aspect
    sentences bhi compare hon.
    """
    emb1       = sim_model.encode(s1, convert_to_tensor=True)
    emb2       = sim_model.encode(s2, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    return similarity >= threshold


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder,
                  sent1: str, sent2: str) -> tuple:
    """
    CrossEncoder se classify karo.
    Returns (label, confidence).
    """
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


# ─── Main Analysis ───────────────────────────────────────────────────────────

def analyze_articles(nli_model: CrossEncoder,
                     sim_model: SentenceTransformer,
                     articles: list) -> list:
    """
    Har article pair ke beech contradictions dhundta hai.

    Flow:
    1. Har article se sentences nikalo
    2. Har article combination ke liye pairs banao
    3. PEHLE topic similarity check karo — alag topic = skip
    4. Same topic hain to NLI classify karo
    5. Sort karo — contradictions pehle, high confidence first
    6. Top 8 per pair return karo
    """
    # Step 1: Extract sentences
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        parsed.append({'source': art['source'], 'sentences': sents})

    findings = []

    # Step 2: Har article combination
    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]

        pair_results = []

        for s1 in art_i['sentences']:
            for s2 in art_j['sentences']:

                # Step 3: Topic check PEHLE
                # Alag topic = NLI call waste mat karo
                if not are_same_topic(sim_model, s1, s2):
                    continue

                # Step 4: Same topic hai — NLI classify karo
                label, conf = classify_pair(nli_model, s1, s2)

                # Low confidence neutral = noise, skip
                if label == 'neutral' and conf < 0.75:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Step 5: Sort
        pair_results.sort(
            key=lambda x: (x['label'] != 'contradiction', -x['confidence'])
        )

        # Step 6: Top 8 per pair
        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
