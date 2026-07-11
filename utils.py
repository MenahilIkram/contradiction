import re
import torch
import torch.nn.functional as F
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
    - Date indicators (in 2022, January, on Monday)
    - Quantity words (million, billion, thousand, hundred)
    - Measurement words (km, kg, meter, acre)
    """

    factual_patterns = [
        r'\d+',                                          # koi bhi number: 500, 1700
        r'\d+\.?\d*\s?%',                               # percentage: 15%, 30.5%
        r'[\$£€₹]\s?\d+',                               # currency: $30, Rs500
        r'\d+\s?(million|billion|thousand|hundred)',     # large numbers: 30 million
        r'\b(killed|died|dead|deaths|casualties|'
        r'injured|wounded|missing|arrested)\b',         # death/incident words
        r'\b(january|february|march|april|may|june|'
        r'july|august|september|october|november|'
        r'december)\b',                                  # month names
        r'\bin\s+\d{4}\b',                              # year: in 2022
        r'\d+\s?(km|kg|meter|acre|hectare|liter)',      # measurements
        r'\b(increased|decreased|rose|fell|dropped|'
        r'surged|declined)\s+by\s+\d+',                # change with number
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
    - 5 words se kam ho (genuinely incomplete)
    - Sirf generic filler ho bina kisi claim ke
    """

    # 5 words se kam = incomplete sentence
    if len(sentence.split()) < 5:
        return True

    # Clearly useless patterns
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


def remove_duplicates(sentences: list[str], 
                       threshold: float = 0.85) -> list[str]:
    """
    Bohat similar sentences remove karta hai.
    
    Simple word-overlap method use karta hai:
    agar do sentences ke 85% words same hain to duplicate maano.
    """
    unique = []
    for sent in sentences:
        words_new = set(sent.lower().split())
        is_duplicate = False

        for existing in unique:
            words_existing = set(existing.lower().split())

            # Jaccard similarity: common words / total unique words
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
    Sentence ko score deta hai — zyada score = zyada factual/important.
    
    Scoring logic:
    +3  : death/casualty words (sabse important contradiction point)
    +2  : currency/money amount
    +2  : large number (million/billion)
    +1  : any number
    +1  : percentage
    +1  : date/year
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
                       max_other: int = 4) -> list[str]:
    """
    Main function — article text se top sentences nikalta hai.

    IMPORTANT: Dono types rakho —
    - Factual (numbers/currency/death words): numeric contradictions ke liye
    - Non-factual (pure language): semantic contradictions ke liye
      jaise "safe" vs "dangerous", "won" vs "lost", "helped" vs "damaged"

    Pipeline:
    1. Text ko sentences mein split karo
    2. Clean karo
    3. Vague/too-short sentences filter karo
    4. Factual aur non-factual alag karo
    5. Dono se duplicates hatao
    6. Factual ko score se sort karo
    7. Dono se alag alag limit le ke combine karo
    """

    # Step 1: Split into sentences
    raw = re.split(r'(?<=[.!?])\s+', text)

    # Step 2: Clean
    sentences = [clean_text(s) for s in raw if clean_text(s)]

    # Step 3: Remove vague sentences
    sentences = [s for s in sentences if not is_too_vague(s)]

    # Step 4: Separate factual vs non-factual
    factual = [s for s in sentences if is_factual(s)]
    other   = [s for s in sentences if not is_factual(s)]

    # Step 5: Remove duplicates from both
    factual = remove_duplicates(factual)
    other   = remove_duplicates(other)

    # Step 6: Sort factual by importance score
    factual = sorted(factual, key=score_sentence, reverse=True)

    # Step 7: Take from BOTH — dono types cover honge
    # Factual: numbers/currency/death = numeric contradictions
    # Other: language-based = semantic contradictions
    combined = factual[:max_factual] + other[:max_other]

    return combined


def classify_pair(model: CrossEncoder, sent1: str, sent2: str) -> tuple[str, float]:
    """
    Classify a sentence pair as contradiction / entailment / neutral.
    Returns (label, confidence_0_to_1).
    """
    raw_scores = model.predict([(sent1, sent2)])[0]          # shape: (3,)
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)  # normalize
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


def analyze_articles(model: CrossEncoder, articles: list[dict]) -> list[dict]:
    """
    Compare all article pairs and return list of results.
    Each article dict: {'source': str, 'text': str}
    Returns list of finding dicts.
    """
    # Extract sentences from each article
    parsed = []
    for art in articles:
        sents = extract_sentences(art['text'])
        parsed.append({'source': art['source'], 'sentences': sents})

    findings = []

    # Compare every pair of articles
    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]

        pair_results = []
        for s1 in art_i['sentences']:
            for s2 in art_j['sentences']:
                label, conf = classify_pair(model, s1, s2)
                # Only keep confident predictions (skip low-confidence neutral)
                if label == 'neutral' and conf < 0.75:
                    continue
                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort: contradictions first, then by confidence desc
        pair_results.sort(
            key=lambda x: (x['label'] != 'contradiction', -x['confidence'])
        )

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]   # top 8 per pair to keep it readable
        })

    return findings


from itertools import combinations  # ensure import at top level
