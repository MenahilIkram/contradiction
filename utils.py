import re
import torch
import torch.nn.functional as F
from itertools import combinations  # TOP pe — sahi jagah
from sentence_transformers import CrossEncoder
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model():
    """Load NLI CrossEncoder model — cached so only loads once."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')


# Label order for this model: [contradiction, entailment, neutral]
LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {
    "contradiction": "❌",
    "entailment": "✅",
    "neutral": "🤷"
}
LABEL_COLOR = {
    "contradiction": "#ff4b4b",
    "entailment": "#00c853",
    "neutral": "#888888"
}


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_factual(sentence: str) -> bool:
    """
    Check karta hai ke sentence mein koi factual claim hai ya nahi.

    Factual claim hoti hai jab sentence mein ho:
    - Numbers (500, 1700)
    - Percentages (15%, 30%)
    - Currency ($30 billion, Rs 500)
    - Death/casualty words (killed, died, dead, injured)
    - Date indicators (in 2022, January)
    - Quantity words (million, billion, thousand)
    - Measurement words (km, kg, meter, acre)
    """
    factual_patterns = [
        r'\d+',
        r'\d+\.?\d*\s?%',
        r'[\$£€₹]\s?\d+',
        r'\d+\s?(million|billion|thousand|hundred)',
        r'\b(killed|died|dead|deaths|casualties|'
        r'injured|wounded|missing|arrested)\b',
        r'\b(january|february|march|april|may|june|'
        r'july|august|september|october|november|'
        r'december)\b',
        r'\bin\s+\d{4}\b',
        r'\d+\s?(km|kg|meter|acre|hectare|liter)',
        r'\b(increased|decreased|rose|fell|dropped|'
        r'surged|declined)\s+by\s+\d+',
    ]
    sentence_lower = sentence.lower()
    for pattern in factual_patterns:
        if re.search(pattern, sentence_lower):
            return True
    return False


def is_too_vague(sentence: str) -> bool:
    """
    Vague sentences filter karta hai jo contradiction ke liye useless hain.

    Vague sentence hoti hai jab:
    - 5 words se kam ho (genuinely incomplete)  ← FIX: 50 chars se 5 words
    - Sirf generic filler ho bina kisi claim ke
    """
    # FIX: pehle 50 char tha — galat tha
    # "Floods killed 1700" = 19 chars but factual hai
    # Ab 5 words check — zyada accurate
    if len(sentence.split()) < 5:
        return True

    vague_patterns = [
        r'^(he|she|they|it)\s+(is|was|are|were)\s+\w+\.$',
        r'\b(very|quite|really|extremely)\s+(bad|good|important|serious)\b',
        r'^(officials?|authorities)\s+said\s+the\s+situation',
    ]
    sentence_lower = sentence.lower().strip()
    for pattern in vague_patterns:
        if re.search(pattern, sentence_lower):
            return True
    return False


def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    """
    Bohat similar sentences remove karta hai.

    Jaccard similarity use karta hai:
    common words / total unique words
    85% se zyada similar = duplicate
    """
    unique = []
    for sent in sentences:
        words_new = set(sent.lower().split())
        is_duplicate = False
        for existing in unique:
            words_existing = set(existing.lower().split())
            if len(words_new | words_existing) == 0:
                continue
            similarity = len(words_new & words_existing) / \
                         len(words_new | words_existing)
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(sent)
    return unique


def score_sentence(sentence: str) -> int:
    """
    Sentence ko score deta hai — zyada score = zyada important.

    +3 : death/casualty words
    +2 : currency/large numbers
    +1 : percentage, year, any number
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


def extract_sentences(text: str,
                       max_factual: int = 5,
                       max_other: int = 4) -> list:
    """
    Article se top sentences nikalta hai — dono types:
    - Factual (numbers/currency/deaths): numeric contradictions
    - Non-factual (pure language): semantic contradictions
      e.g. "safe" vs "dangerous", "won" vs "lost"
    """
    # Step 1: Split
    raw = re.split(r'(?<=[.!?])\s+', text)

    # Step 2: Clean
    sentences = [clean_text(s) for s in raw if clean_text(s)]

    # Step 3: Vague filter
    sentences = [s for s in sentences if not is_too_vague(s)]

    # Step 4: Factual vs non-factual alag karo
    factual = [s for s in sentences if is_factual(s)]
    other   = [s for s in sentences if not is_factual(s)]

    # Step 5: Duplicates hatao dono se
    factual = remove_duplicates(factual)
    other   = remove_duplicates(other)

    # Step 6: Factual ko score se sort karo
    factual = sorted(factual, key=score_sentence, reverse=True)

    # Step 7: Dono se lo — numeric + semantic dono cover
    return factual[:max_factual] + other[:max_other]


def classify_pair(model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    """
    Do sentences classify karo.
    Returns (label, confidence).
    """
    raw_scores = model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


def analyze_articles(model: CrossEncoder, articles: list) -> list:
    """
    Har article pair ke beech contradictions dhundta hai.

    Flow:
    1. Har article se sentences nikalo
    2. Combinations se har article pair banao
    3. Har sentence pair classify karo
    4. Sort karo — contradictions pehle, confidence high se low
    5. Top 8 results return karo per pair
    """
    # Step 1: Extract
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        parsed.append({'source': art['source'], 'sentences': sents})

    findings = []

    # Step 2: Har article pair
    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]

        pair_results = []

        # Step 3: Classify har sentence pair
        for s1 in art_i['sentences']:
            for s2 in art_j['sentences']:
                label, conf = classify_pair(model, s1, s2)

                # Low confidence neutral = noise, skip
                if label == 'neutral' and conf < 0.75:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Step 4: Sort — contradictions first, high confidence first
        pair_results.sort(
            key=lambda x: (x['label'] != 'contradiction', -x['confidence'])
        )

        # Step 5: Top 8 rakho — sab dikhana = UI flood
        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
