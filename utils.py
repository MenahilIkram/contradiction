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
    Sentence similarity model — same topic check karta hai.
    Alag topic hain to NLI call hi nahi karte — false positives avoid.
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
                      max_factual: int = 10,
                      max_other: int = 10) -> list:
    """
    Article se top sentences nikalo. Caps badha diye taake koi important claim miss na ho.
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
    Cosine similarity check. Threshold thoda kam (0.35) kiya taake core topics miss na hon.
    """
    emb1       = sim_model.encode(s1, convert_to_tensor=True)
    emb2       = sim_model.encode(s2, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    return similarity >= threshold


# ─── Numerical Contradiction Detector ────────────────────────────────────────

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'into',
    'through', 'during', 'from', 'up', 'down', 'out', 'and', 'or',
    'but', 'not', 'no', 'so', 'yet', 'both', 'either', 'that',
    'this', 'these', 'those', 'it', 'its', 'than', 'then', 'their',
    'they', 'them', 'he', 'she', 'we', 'us', 'our', 'also', 'just',
    'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
    'according', 'per', 'across', 'after', 'before', 'between',
    'each', 'further', 'here', 'how', 'i', 'me', 'my', 'over',
    'very', 'well', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'your', 'all', 'any', 'reported', 'said', 
    'noted', 'stated', 'confirmed', 'launched', 'seeking', 'including', 
    'around', 'approximately', 'nearly', 'least', 'total', 'related', 
    'due', 'continued', 'estimated',
    # Naye unit words taake inki wajah se false context matching na ho
    'dollars', 'dollar', 'million', 'billion', 'thousand', 'percent', 
    'percentage', 'people', 'country', 'provinces', 'districts', 'regions'
}


def extract_context_words(sentence: str) -> set:
    words = re.sub(r'[^a-zA-Z\s]', '', sentence.lower()).split()
    content = {w for w in words if w not in STOPWORDS and len(w) > 2}
    return content


def have_common_context(s1: str, s2: str, min_common: int = 1) -> bool:
    words1 = extract_context_words(s1)
    words2 = extract_context_words(s2)

    exact_common = words1 & words2
    if len(exact_common) >= min_common:
        return True

    for w1 in words1:
        for w2 in words2:
            if len(w1) >= 4 and len(w2) >= 4:
                if w1.startswith(w2[:4]) or w2.startswith(w1[:4]):
                    return True
    return False


def get_numerical_category(sentence: str) -> str:
    """
    Sentence ko metric-specific category mein classification karta hai.
    Sirf saheeh category ke sentences hi aapas mein compare honge.
    """
    s = sentence.lower()
    
    # 1. Casualties (Deaths/Injuries)
    if any(w in s for w in ['killed', 'died', 'dead', 'deaths', 'casualties', 'death toll', 'toll']):
        return 'casualty'
        
    # 2. Economic / Monetary Loss / Aid Appeal
    if any(w in s for w in ['dollar', '$', '£', '€', '₹', 'financial', 'economic', 'losses', 'damage cost', 'appeal']):
        return 'economic'
        
    # 3. Infrastructure / Houses Destroyed
    if any(w in s for w in ['residential', 'structures', 'buildings', 'houses', 'homes']) and not any(w in s for w in ['displaced', 'evacuated']):
        return 'infrastructure'
        
    # 4. Displaced / Affected Population
    if any(w in s for w in ['displaced', 'affected', 'evacuations', 'evacuated', 'homeless', 'citizens', 'humanitarian']):
        return 'population_affected'
        
    return 'other'


def extract_numbers_general(text: str) -> list:
    text = text.lower()
    
    # Years (e.g., 2022, 1999) ko filter out karo taake main figures kharab na hon
    text = re.sub(r'\b(in\s+)?(19|20)\d{2}\b', ' ', text)
    
    numbers = []

    # Scale conversions
    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text):
        numbers.append(float(match.group(1)) * 1_000_000_000)

    for match in re.finditer(r'(\d+\.?\d*)\s*million', text):
        numbers.append(float(match.group(1)) * 1_000_000)

    for match in re.finditer(r'(\d+\.?\d*)\s*thousand', text):
        numbers.append(float(match.group(1)) * 1_000)

    # Commas remove karo (1,700 -> 1700)
    text = re.sub(r'\b(\d{1,3})(?:,\d{3})+\b', lambda m: m.group(0).replace(',', ''), text)

    # Baki plain numbers aur decimals
    for match in re.finditer(r'\b(\d+\.?\d*)\b', text):
        try:
            val = float(match.group(1))
            if val >= 10:
                numbers.append(val)
        except ValueError:
            continue

    return numbers


def detect_numerical_contradiction(s1: str, s2: str) -> tuple:
    """
    Sirf apples-to-apples logic apply karta hai via smart grouping.
    > 20% difference = CONTRADICTION.
    < 20% difference = ENTAILMENT (Agreement).
    """
    cat1 = get_numerical_category(s1)
    cat2 = get_numerical_category(s2)
    
    # Agar domain category different hai ya generic hai to filter out karo
    if cat1 == 'other' or cat2 == 'other' or cat1 != cat2:
        return None, 0.0

    # Context verification double check karo
    if not have_common_context(s1, s2):
        return None, 0.0

    nums1 = extract_numbers_general(s1)
    nums2 = extract_numbers_general(s2)

    if not nums1 or not nums2:
        return None, 0.0

    n1 = max(nums1)
    n2 = max(nums2)

    if n1 == 0 or n2 == 0:
        return None, 0.0

    # Percentage farq calculate karein
    diff_pct = abs(n1 - n2) / max(n1, n2)

    if diff_pct > 0.20:
        confidence = min(0.99, 0.65 + diff_pct)
        return 'contradiction', round(confidence, 2)

    # Agar figures bohat close hain (e.g. 10B vs 10.5B) tou ye agreement hai!
    return 'entailment', 0.85


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder,
                  sent1: str, sent2: str) -> tuple:
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


# ─── Main Analysis ───────────────────────────────────────────────────────────

def analyze_articles(nli_model: CrossEncoder,
                     sim_model: SentenceTransformer,
                     articles: list) -> list:
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

                # Step 3: Topic similarity check
                if not are_same_topic(sim_model, s1, s2):
                    continue

                # Step 4: Linguistic NLI prediction
                label, conf = classify_pair(nli_model, s1, s2)

                # Step 5: Advanced Numerical override
                num_label, num_conf = detect_numerical_contradiction(s1, s2)
                if num_label == 'contradiction':
                    label = 'contradiction'
                    conf  = num_conf
                elif num_label == 'entailment' and label == 'neutral':
                    label = 'entailment'
                    conf  = num_conf

                # Noise control: Low confidence neutral pairs ko skip karo
                if label == 'neutral' and conf < 0.70:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sorting strategy: Contradictions sabse pehle, phir high confidence
        pair_results.sort(
            key=lambda x: (x['label'] != 'contradiction', -x['confidence'])
        )

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
