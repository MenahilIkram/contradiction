import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    NLI Model: cross-encoder/nli-deberta-v3-base
    
    MiniLM se kyun better:
    - DeBERTa-v3 abhi NLI ka SOTA (State of the Art) model hai
    - SNLI + MultiNLI + FEVER pe trained hai
    - Disentangled attention mechanism use karta hai jo
      word relationships zyada accurately capture karta hai
    """
    return CrossEncoder('cross-encoder/nli-deberta-v3-base')


@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """
    Similarity Model: all-mpnet-base-v2
    
    all-MiniLM se kyun better:
    - MPNet (Masked and Permuted Pre-training) use karta hai
    - Sentence similarity benchmarks pe consistently better scores
    - Topic filtering ke liye zyada accurate embeddings
    """
    return SentenceTransformer('all-mpnet-base-v2')


# ─── Labels ───────────────────────────────────────────────────────────────────

# DeBERTa model ka label order: [contradiction, entailment, neutral]
LABEL_MAP   = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_EMOJI = {"contradiction": "❌", "entailment": "✅", "neutral": "🤷"}
LABEL_COLOR = {"contradiction": "#ff4b4b", "entailment": "#00c853", "neutral": "#888888"}


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Sentence Extraction ─────────────────────────────────────────────────────

def extract_sentences(text: str, max_sentences: int = 30) -> list:
    """
    Article ko sentences mein split karo.
    - 4 words se kam sentences skip karo (incomplete hote hain)
    - Exact duplicates remove karo
    """
    if not text:
        return []
    raw       = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw
                 if len(clean_text(s).split()) >= 4]

    # Exact duplicates remove karo (order preserve karo)
    unique = []
    for s in sentences:
        if s not in unique:
            unique.append(s)

    return unique[:max_sentences]


# ═══════════════════════════════════════════════════════════════════════════════
# TUMHARI CONTRIBUTION — Numerical Contradiction Detector
# ═══════════════════════════════════════════════════════════════════════════════
# NLI model (DeBERTa) language samajhta hai lekin exact numbers ka
# fark nahi pakadta:
# "1700 killed" vs "900 deaths" → NLI neutral bol deta hai
# Ye rule-based module specifically numbers compare karta hai.
# ═══════════════════════════════════════════════════════════════════════════════

# Number type keywords — sentence mein number KISKA baare mein hai
NUMBER_TYPES = {
    'deaths': [
        'killed', 'died', 'deaths', 'casualties',
        'dead', 'toll', 'fatalities', 'perished'
    ],
    'people_affected': [
        'displaced', 'evacuated', 'homeless',
        'assistance', 'humanitarian', 'affected'
    ],
    'people_count': [
        'citizens', 'residents', 'population',
        'individuals', 'persons', 'people'
    ],
    'homes': [
        'homes', 'houses', 'structures',
        'buildings', 'residential', 'properties'
    ],
    'money': [
        'billion', 'million', 'dollars', 'losses',
        'damage', 'cost', 'financial', 'economic',
        'rupees', 'euros', 'pounds'
    ],
    'percentage': [
        'percent', '%', 'rate', 'ratio', 'proportion'
    ],
    'area': [
        'area', 'land', 'territory', 'region',
        'acre', 'hectare', 'km', 'kilometer'
    ],
}


def get_number_type(sentence: str) -> str:
    """
    Sentence mein number kis cheez ka hai identify karo.

    Sirf same type ke numbers compare kiye jayenge:
    "1700 killed" (deaths) vs "900 deaths" (deaths)   → COMPARE
    "1700 killed" (deaths) vs "15M displaced" (people) → SKIP

    Ye fix isliye zaroori tha kyunke pehle:
    "1700 killed" vs "15M displaced" = 99% contradiction ← GALAT
    Ab type check karne se ye false positive nahi aata.
    """
    s = sentence.lower()
    for type_name, keywords in NUMBER_TYPES.items():
        if any(kw in s for kw in keywords):
            return type_name
    return 'general'


def extract_numbers(text: str) -> list:
    """
    Text se numbers nikalo — billion/million/thousand convert karo.

    "30 billion" → 30,000,000,000
    "15 million" → 15,000,000
    "1,700"      → 1700
    "6%"         → 6
    """
    text = text.lower()
    numbers = []

    for m in re.finditer(r'(\d+\.?\d*)\s*billion', text):
        numbers.append(float(m.group(1)) * 1_000_000_000)

    for m in re.finditer(r'(\d+\.?\d*)\s*million', text):
        numbers.append(float(m.group(1)) * 1_000_000)

    for m in re.finditer(r'(\d+\.?\d*)\s*thousand', text):
        numbers.append(float(m.group(1)) * 1_000)

    # Comma wale numbers: 1,700 → 1700
    for m in re.finditer(r'\b(\d{1,3}(?:,\d{3})+)\b', text):
        numbers.append(float(m.group(1).replace(',', '')))

    # Plain numbers — sirf 5 se bade (year/age confusion avoid)
    for m in re.finditer(r'\b(\d+\.?\d*)\b', text):
        val = float(m.group(1))
        if val >= 5:
            numbers.append(val)

    return numbers


def detect_numerical_contradiction(s1: str, s2: str) -> tuple:
    """
    Numbers wali sentences mein contradiction detect karo.

    Step 1: Type same hai? (deaths vs deaths, money vs money)
            Alag type → return None (compare nahi karein)

    Step 2: Numbers nikalo dono se

    Step 3: Fark 20% se zyada?
            Haan → CONTRADICTION
            Nahi → ENTAILMENT (same figure roughly)

    Threshold 20% kyun:
    - Sources mein thodi rounding normal hai (1700 vs 1750)
    - 20% se zyada = genuinely alag claim
    - $10B vs $15B = 33% fark = CONTRADICTION
    - $10B vs $10.5B = 5% fark = same figure
    """
    type1 = get_number_type(s1)
    type2 = get_number_type(s2)

    # Alag type = mat compare karo
    if type1 != type2:
        return None, 0.0

    # 'general' type — too vague to compare
    if type1 == 'general':
        return None, 0.0

    nums1 = extract_numbers(s1)
    nums2 = extract_numbers(s2)

    if not nums1 or not nums2:
        return None, 0.0

    n1 = max(nums1)
    n2 = max(nums2)

    if n1 == 0 or n2 == 0:
        return None, 0.0

    diff_pct = abs(n1 - n2) / max(n1, n2)

    if diff_pct > 0.20:
        # Jitna zyada fark, utni zyada confidence
        # 20% fark → 0.75 conf, 80% fark → 0.99 conf
        confidence = min(0.99, 0.55 + diff_pct)
        return 'contradiction', round(confidence, 2)

    return 'entailment', 0.75


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CODE — Google AI (Gemini) se liya, bug fix kiya, numerical detector add kiya
# Changes tumne kiye:
# 1. findings.append loop ke andar move kiya (bug fix)
# 2. Numerical detector integrate kiya as NLI fallback
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_articles(nli_model: CrossEncoder,
                     sim_model: SentenceTransformer,
                     articles: list) -> list:
    """
    Har article pair ke beech contradictions dhundta hai.

    Pipeline:
    1. Sentences extract karo
    2. Embeddings compute karo (batch mein — fast)
    3. Cosine similarity matrix — topic filter
    4. Filtered pairs → DeBERTa NLI classify karo (batch)
    5. NLI weak/neutral → Numerical detector try karo (tumhari contribution)
    6. Confidence + overlap filter lagao
    7. Results sort karke return karo
    """

    # Step 1 & 2: Extract + Embed
    parsed = []
    for art in articles:
        sents      = extract_sentences(art.get('text', ''))
        embeddings = sim_model.encode(
            sents, convert_to_tensor=True
        ) if sents else None
        parsed.append({
            'source':     art.get('source', 'Unknown'),
            'sentences':  sents,
            'embeddings': embeddings
        })

    findings = []

    for i, j in combinations(range(len(parsed)), 2):
        art_i = parsed[i]
        art_j = parsed[j]
        pair_results = []

        if art_i['embeddings'] is None or art_j['embeddings'] is None:
            continue

        # Step 3: Cosine similarity matrix — ek call mein sab pairs
        cosine_scores = util.cos_sim(
            art_i['embeddings'], art_j['embeddings']
        )

        pairs_to_classify = []
        metadata          = []

        for idx_i, s1 in enumerate(art_i['sentences']):
            for idx_j, s2 in enumerate(art_j['sentences']):
                sim_score = float(cosine_scores[idx_i][idx_j])

                # Topic filter: 0.28 se kam = bilkul alag topic
                if sim_score < 0.28:
                    continue

                pairs_to_classify.append((s1, s2))
                metadata.append({
                    's1': s1, 's2': s2,
                    'sim_score': sim_score
                })

        if not pairs_to_classify:
            # Koi pair nahi mila → empty result
            findings.append({
                'source_1': art_i['source'],
                'source_2': art_j['source'],
                'results':  []
            })
            continue

        # Step 4: Batch NLI classification — ek call mein sab
        raw_scores = nli_model.predict(
            pairs_to_classify,
            batch_size=16,
            show_progress_bar=False
        )

        candidates = []

        for idx, meta in enumerate(metadata):
            probs    = F.softmax(torch.tensor(raw_scores[idx]), dim=0)
            pred_idx = int(probs.argmax())
            conf     = float(probs[pred_idx])
            label    = LABEL_MAP.get(pred_idx, 'neutral')

            s1 = meta['s1']
            s2 = meta['s2']

            # Step 5: NLI weak/neutral → Numerical detector try karo
            # (TUMHARI CONTRIBUTION)
            # DeBERTa "1700 killed" vs "900 deaths" ko neutral bol deta hai
            # Numerical detector same-type numbers ka fark check karta hai
            if label in ('neutral', 'entailment') or conf < 0.70:
                num_label, num_conf = detect_numerical_contradiction(s1, s2)
                if num_label == 'contradiction':
                    label = 'contradiction'
                    conf  = num_conf

            # Sirf confident contradictions rakho
            if label != 'contradiction' or conf < 0.70:
                continue

            # Token overlap calculate karo
            # Overlap < 5% aur conf < 99.9% = likely false positive
            tokens1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s1.lower()))
            tokens2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s2.lower()))
            overlap = (len(tokens1 & tokens2) / len(tokens1 | tokens2)
                       if tokens1 | tokens2 else 0.0)

            if overlap < 0.05 and conf < 0.999:
                continue

            candidates.append({
                'sentence_1': s1,
                'sentence_2': s2,
                'label':      'contradiction',
                'confidence': conf,
                'similarity': meta['sim_score'],
                'overlap':    overlap
            })

        # Step 6: Sort by confidence
        candidates.sort(key=lambda x: (-x['confidence'], -x['similarity']))

        # Duplicate sentence filter — same sentence dobara nahi
        used_s1 = set()
        used_s2 = set()

        for cand in candidates:
            s1_key = cand['sentence_1'].strip().lower()
            s2_key = cand['sentence_2'].strip().lower()

            if s1_key in used_s1 and s2_key in used_s2:
                continue

            used_s1.add(s1_key)
            used_s2.add(s2_key)
            pair_results.append(cand)

        # Step 7: BUG FIX — findings.append loop KE ANDAR hona chahiye
        # Google AI wale code mein ye loop ke bahar tha —
        # sirf last pair ka result save hota tha
        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
