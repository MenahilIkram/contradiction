import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model():
    """Load NLI CrossEncoder model — cached so only loads once."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')


# Label order for this model: [contradiction, entailment, neutral]
LABEL_MAP   = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {"contradiction": "❌", "entailment": "✅", "neutral": "🤷"}
LABEL_COLOR = {"contradiction": "#ff4b4b", "entailment": "#00c853", "neutral": "#888888"}


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Factual Check ────────────────────────────────────────────────────────────

def is_factual(sentence: str) -> bool:
    """
    Check karta hai ke sentence mein koi factual claim hai ya nahi.
    Factual = numbers, currency, death words, dates, measurements.
    """
    factual_patterns = [
        r'\d+',
        r'\d+\.?\d*\s?%',
        r'[\$£€₹]\s?\d+',
        r'\d+\s?(million|billion|thousand|hundred)',
        r'\b(killed|died|dead|deaths|casualties|injured|wounded|missing|arrested)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\bin\s+\d{4}\b',
        r'\d+\s?(km|kg|meter|acre|hectare|liter)',
        r'\b(increased|decreased|rose|fell|dropped|surged|declined)\s+by\s+\d+',
    ]
    s = sentence.lower()
    for pattern in factual_patterns:
        if re.search(pattern, s):
            return True
    return False


# ─── Vague Check ─────────────────────────────────────────────────────────────

def is_too_vague(sentence: str) -> bool:
    """
    Vague sentences filter karta hai.
    5 words se kam = incomplete.
    Generic filler patterns = useless.
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


# ─── Duplicate Removal (Jaccard) ─────────────────────────────────────────────

def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    """
    Bohat similar sentences remove karta hai using Jaccard similarity.
    Jaccard = common words / total unique words
    85% se zyada similar = duplicate.
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
    Sentence ko importance score deta hai.
    +3: death/casualty words
    +2: currency/large numbers
    +1: percentage, year, any number
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

def extract_sentences(text: str, max_factual: int = 5, max_other: int = 4) -> list:
    """
    Article se top sentences nikalta hai — dono types:
    - Factual (numbers/currency/deaths): numeric contradictions ke liye
    - Non-factual (pure language): semantic contradictions ke liye
      e.g. "safe" vs "dangerous", "won" vs "lost"
    """
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if clean_text(s)]
    sentences = [s for s in sentences if not is_too_vague(s)]

    factual = [s for s in sentences if is_factual(s)]
    other   = [s for s in sentences if not is_factual(s)]

    factual = remove_duplicates(factual)
    other   = remove_duplicates(other)

    factual = sorted(factual, key=score_sentence, reverse=True)

    return factual[:max_factual] + other[:max_other]


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    """
    Do sentences ko NLI model se classify karo.
    Returns (label, confidence).
    Labels: contradiction / entailment / neutral
    """
    raw_scores = model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


# ─── Duplicate Pair Removal ───────────────────────────────────────────────────

def remove_duplicate_pairs(results: list) -> list:
    """
    Agar ek hi sentence multiple pairs mein aa rahi ho
    to sirf sabse confident wali pair rakho.

    Example:
    "1700 killed" vs "900 killed"   conf 0.91  <- rakho
    "1700 killed" vs "500 injured"  conf 0.65  <- hatao
    (sentence_1 repeat ho rahi hai, sirf best wali rakho)
    """
    seen_s1 = set()
    unique  = []
    for r in results:
        key = r['sentence_1'][:60]
        if key not in seen_s1:
            seen_s1.add(key)
            unique.append(r)
    return unique


# ─── Main Analysis ───────────────────────────────────────────────────────────

def analyze_articles(model: CrossEncoder, articles: list) -> list:
    """
    Har article pair ke beech contradiction/entailment/neutral dhundta hai.

    Flow:
    1. Har article se sentences nikalo
    2. Har article combination ke liye pairs banao
    3. Har pair classify karo
    4. Zyada results ko smart filter karo:
       - Contradictions: top 5 (high confidence, no duplicate sentences)
       - Entailments:    top 3
       - Neutral:        top 2
    """
    # Step 1: Sentences extract karo
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        parsed.append({'source': art['source'], 'sentences': sents})

    findings = []

    # Step 2: Har article pair ke liye
    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]

        all_pairs = []

        # Step 3: Har sentence pair classify karo
        for s1 in art_i['sentences']:
            for s2 in art_j['sentences']:
                label, conf = classify_pair(model, s1, s2)

                # Low confidence neutral = noise, skip
                if label == 'neutral' and conf < 0.75:
                    continue

                all_pairs.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort: contradictions first, phir confidence high se low
        all_pairs.sort(key=lambda x: (x['label'] != 'contradiction', -x['confidence']))

        # Step 4: Smart filtering — zyada results handle karna
        # Problem: 9x9 = 81 pairs ho sakte hain, sab dikhana = UI flood
        # Solution: alag alag bucket, duplicate sentences hatao, limit lagao

        contradictions = [r for r in all_pairs if r['label'] == 'contradiction']
        entailments    = [r for r in all_pairs if r['label'] == 'entailment']
        neutrals       = [r for r in all_pairs if r['label'] == 'neutral']

        # Duplicate sentences wali pairs hatao
        contradictions = remove_duplicate_pairs(contradictions)
        entailments    = remove_duplicate_pairs(entailments)

        # Per category limit
        final_results = (
            contradictions[:5] +   # top 5 contradictions
            entailments[:3]    +   # top 3 agreements
            neutrals[:2]           # top 2 neutral
        )

        findings.append({
            'source_1':             art_i['source'],
            'source_2':             art_j['source'],
            'results':              final_results,
            'total_pairs':          len(all_pairs),
            'total_contradictions': len(contradictions)
        })

    return findings
