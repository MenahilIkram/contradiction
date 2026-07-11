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
                   threshold: float = 0.45) -> bool:
    """
    Cosine similarity se check karo ke dono sentences
    same topic pe hain ya nahi.
    0.45 se kam = alag topic = NLI call skip karo.
    """
    emb1       = sim_model.encode(s1, convert_to_tensor=True)
    emb2       = sim_model.encode(s2, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    return similarity >= threshold


# ─── Numerical Contradiction Detector ────────────────────────────────────────

# Generic stopwords — ye words "context" mein count nahi honge
# Sirf meaningful/content words se context decide hoga
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
    'who', 'whom', 'why', 'your', 'all', 'any', 'been', 'being',
    'reported', 'said', 'noted', 'stated', 'confirmed', 'launched',
    'seeking', 'including', 'around', 'approximately', 'nearly',
    'least', 'total', 'related', 'due', 'continued', 'estimated',
}


def extract_context_words(sentence: str) -> set:
    """
    Sentence se sirf meaningful/content words nikalo.

    GENERAL approach — koi hardcoded keywords nahi:
    - Stopwords hata do
    - Numbers hata do (alag handle hote hain)
    - Jo bache wo content words hain

    Example:
    "1700 people killed in Pakistan floods"
    → stopwords hata do: {people, killed, pakistan, floods}

    "900 flood deaths reported in country"
    → stopwords hata do: {flood, deaths, country}

    Common content words: {flood, deaths/killed}
    → Same topic confirm ✅
    """
    words = re.sub(r'[^a-zA-Z\s]', '', sentence.lower()).split()
    content = {w for w in words
               if w not in STOPWORDS and len(w) > 2}
    return content


def have_common_context(s1: str, s2: str,
                         min_common: int = 1) -> bool:
    """
    Dono sentences mein kam se kam 1 common content word hona chahiye.

    Ye ensure karta hai ke numbers same topic ke baare mein hain.

    Example:
    s1: "1700 people killed in floods"     → {people, killed, floods}
    s2: "900 flood deaths reported"        → {flood, deaths}
    Common: {} ... 'flood'/'floods' stem same hai but exact match nahi

    Isliye partial matching bhi karte hain:
    agar koi word dusre ka substring hai (flood/floods) to bhi match
    """
    words1 = extract_context_words(s1)
    words2 = extract_context_words(s2)

    # Exact common words
    exact_common = words1 & words2
    if len(exact_common) >= min_common:
        return True

    # Partial match — "flood" in "floods", "kill" in "killed"
    for w1 in words1:
        for w2 in words2:
            # Agar ek dusre ka prefix hai (stem matching)
            if len(w1) >= 4 and len(w2) >= 4:
                if w1.startswith(w2[:4]) or w2.startswith(w1[:4]):
                    return True

    return False


def extract_numbers_general(text: str) -> list:
    """
    Text se numbers nikalo — billion/million properly convert karo.

    GENERAL — koi domain specific logic nahi:
    "30 billion" → 30,000,000,000
    "15 million" → 15,000,000
    "1,700"      → 1700
    "15%"        → 15 (percentage)
    "1.5"        → 1.5
    """
    text = text.lower()
    numbers = []

    # Billion wale
    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text):
        numbers.append(float(match.group(1)) * 1_000_000_000)

    # Million wale
    for match in re.finditer(r'(\d+\.?\d*)\s*million', text):
        numbers.append(float(match.group(1)) * 1_000_000)

    # Thousand wale
    for match in re.finditer(r'(\d+\.?\d*)\s*thousand', text):
        numbers.append(float(match.group(1)) * 1_000)

    # Comma wale (1,700 → 1700)
    for match in re.finditer(r'\b(\d{1,3}(?:,\d{3})+)\b', text):
        numbers.append(float(match.group(1).replace(',', '')))

    # Plain numbers (sirf 10 se bade — chote numbers ignore)
    for match in re.finditer(r'\b(\d+\.?\d*)\b', text):
        val = float(match.group(1))
        if val >= 10:
            numbers.append(val)

    return numbers


def detect_numerical_contradiction(s1: str, s2: str) -> tuple:
    """
    Numbers wali sentences mein contradiction dhundta hai.

    NLI model numbers ka exact fark nahi samajhta —
    "1700 killed" vs "900 killed" dono ko related samajh ke
    neutral ya entailment bol deta hai.

    Ye function GENERAL rule-based approach use karta hai:

    Step 1: Dono sentences se numbers nikalo
    Step 2: Common context words check karo (same topic?)
    Step 3: Numbers ka fark calculate karo
    Step 4: 20% se zyada fark = CONTRADICTION
             20% se kam fark  = ENTAILMENT (same figure)

    Threshold 20% kyun:
    - Alag sources mein thodi variation normal hai (rounding)
    - 20% se zyada = genuinely different claim = contradiction
    - $10B vs $15B = 33% fark = CONTRADICTION
    - $10B vs $10.5B = 5% fark = same figure, ENTAILMENT
    """
    nums1 = extract_numbers_general(s1)
    nums2 = extract_numbers_general(s2)

    # Numbers nahi hain to kuch nahi kar sakte
    if not nums1 or not nums2:
        return None, 0.0

    # Common context check — same topic ke baare mein hain?
    if not have_common_context(s1, s2):
        return None, 0.0

    # Largest numbers compare karo (main claim wala number)
    n1 = max(nums1)
    n2 = max(nums2)

    if n1 == 0 or n2 == 0:
        return None, 0.0

    # Fark percentage mein
    diff_pct = abs(n1 - n2) / max(n1, n2)

    if diff_pct > 0.20:
        # Jitna zyada fark, utni zyada confidence
        # 20% fark → 0.70 confidence
        # 80% fark → 0.99 confidence
        confidence = min(0.99, 0.55 + diff_pct)
        return 'contradiction', round(confidence, 2)

    # Same figure — entailment
    return 'entailment', 0.75


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder,
                  sent1: str, sent2: str) -> tuple:
    """CrossEncoder se classify karo. Returns (label, confidence)."""
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
    3. Topic similarity check (0.45 threshold) — alag topic skip
    4. NLI model se classify karo
    5. NLI ne neutral/low confidence diya?
       → Numerical detector try karo (numbers wali sentences ke liye)
    6. Sort — contradictions pehle, high confidence first
    7. Top 8 per pair return karo
    """
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

                # Step 4: NLI classify
                label, conf = classify_pair(nli_model, s1, s2)

                # Step 5: NLI weak hai to numerical detector try karo
                # NLI model 1700 vs 900 ko neutral bol deta hai
                # Numerical detector specifically numbers ka fark check karta hai
                if label in ('neutral', 'entailment') or conf < 0.65:
                    num_label, num_conf = detect_numerical_contradiction(s1, s2)
                    if num_label == 'contradiction':
                        label = 'contradiction'
                        conf  = num_conf

                # Low confidence neutral = noise, skip
                if label == 'neutral' and conf < 0.75:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort: contradictions first, high confidence first
        pair_results.sort(
            key=lambda x: (x['label'] != 'contradiction', -x['confidence'])
        )

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
