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


# ─── Factual Check ────────────────────────────────────────────────────────────

def is_factual(sentence: str) -> bool:
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

def extract_sentences(text: str, max_factual: int = 8, max_other: int = 6) -> list:
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

def are_same_topic(sim_model: SentenceTransformer, s1: str, s2: str, threshold: float = 0.40) -> bool:
    emb1       = sim_model.encode(s1, convert_to_tensor=True)
    emb2       = sim_model.encode(s2, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    return similarity >= threshold


# ─── Precise Sub-Metric Extraction ───────────────────────────────────────────

def get_sub_metric(sentence: str) -> str:
    """ Sentence ke exact data metric ko pakadta hai taake ghalti ka chance zero ho """
    s = sentence.lower()
    
    # 1. Deaths / Casualties
    if any(w in s for w in ['killed', 'died', 'dead', 'deaths', 'casualties', 'death toll', 'toll']):
        return 'deaths'
        
    # 2. Financial / Economic Losses
    if any(w in s for w in ['losses', 'financial damage', 'economic losses', 'total financial damage', 'damage was calculated at']):
        if any(w in s for w in ['billion', 'million', 'dollar', '$', 'euros', 'pounds']):
            return 'financial_loss'
            
    # 3. Infrastructure / Houses Damaged (Strictly physical structures, not displacement)
    if any(w in s for w in ['houses', 'homes', 'structures', 'residential structures', 'residential']) and any(w in s for w in ['destroyed', 'damage', 'destruction', 'sustained']):
        if 'displaced' not in s and 'homeless' not in s:
            return 'houses_destroyed'
            
    # 4. Displaced People (Explicitly forced to move / evacuated)
    if 'displaced' in s or 'homeless' in s:
        return 'people_displaced'
    if 'evacuations' in s or 'evacuated' in s:
        if 'affected over' in s or 'affected' in s:
            return 'people_affected'  # "affected 33M forcing evacuations" -> 33M refers to affected
        return 'people_displaced'
        
    # 5. General Affected Population (Impacted citizens)
    if 'affected' in s and any(w in s for w in ['citizens', 'people', 'population']):
        return 'people_affected'
        
    # 6. Humanitarian Assistance Needed (People requiring aid)
    if 'assistance' in s or 'humanitarian' in s or 'relief' in s:
        return 'need_assistance'
        
    return 'other'


def extract_numbers_general(text: str) -> list:
    text = text.lower()
    # Years filter out karo (e.g. 2022) ko hatao taake calculation kharab na ho
    text = re.sub(r'\b(in\s+)?(19|20)\d{2}\b', ' ', text)
    
    numbers = []
    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text):
        numbers.append(float(match.group(1)) * 1_000_000_000)
    for match in re.finditer(r'(\d+\.?\d*)\s*million', text):
        numbers.append(float(match.group(1)) * 1_000_000)
    for match in re.finditer(r'(\d+\.?\d*)\s*thousand', text):
        numbers.append(float(match.group(1)) * 1_000)

    # Commas hatayein (1,700 -> 1700)
    text = re.sub(r'\b(\d{1,3})(?:,\d{3})+\b', lambda m: m.group(0).replace(',', ''), text)

    for match in re.finditer(r'\b(\d+\.?\d*)\b', text):
        try:
            val = float(match.group(1))
            if val >= 10:  # Context ke chote numbers ko ignore karo
                numbers.append(val)
        except ValueError:
            continue
    return numbers


def detect_numerical_contradiction(s1: str, s2: str) -> tuple:
    nums1 = extract_numbers_general(s1)
    nums2 = extract_numbers_general(s2)

    if not nums1 or not nums2:
        return None, 0.0

    n1 = max(nums1)
    n2 = max(nums2)

    if n1 == 0 or n2 == 0:
        return None, 0.0

    diff_pct = abs(n1 - n2) / max(n1, n2)

    # 20% se zyada farq = Sach mein contradiction hai
    if diff_pct > 0.20:
        confidence = min(0.99, 0.65 + diff_pct)
        return 'contradiction', round(confidence, 2)

    # Agar farq 20% se kam hai to iska matlab figures agree kar rahe hain (Entailment)
    return 'entailment', 0.85


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


# ─── Main Analysis ───────────────────────────────────────────────────────────

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

                # Step 1: Topic Similarity
                if not are_same_topic(sim_model, s1, s2):
                    continue

                # Step 2: Exact Sub-Metric Extraction
                sub1 = get_sub_metric(s1)
                sub2 = get_sub_metric(s2)
                
                has_num1 = len(extract_numbers_general(s1)) > 0
                has_num2 = len(extract_numbers_general(s2)) > 0

                # Step 3: Base NLI Prediction
                label, conf = classify_pair(nli_model, s1, s2)

                # ── MACHINE LEARNING + RULE-BASED GUARDRAILS ──
                if has_num1 and has_num2:
                    # Case A: Metrics bilkul same hain aur factual hain (e.g., deaths vs deaths)
                    if sub1 != 'other' and sub1 == sub2:
                        num_label, num_conf = detect_numerical_contradiction(s1, s2)
                        if num_label is not None:
                            label = num_label
                            conf = num_conf
                    
                    # Case B: Metrics different hain (e.g., houses destroyed vs people displaced)
                    # Lekin model ne bewakoofi mein contradiction boldia. Override to Neutral!
                    else:
                        if label == 'contradiction':
                            label = 'neutral'
                            conf = 1.0

                # Low confidence neutral pairs ko filter out karein taake UI saaf rahe
                if label == 'neutral' and conf < 0.75:
                    continue

                pair_results.append({
                    'sentence_1': s1,
                    'sentence_2': s2,
                    'label':      label,
                    'confidence': conf
                })

        # Sort: Contradictions sabse pehle, high confidence ke sath
        pair_results.sort(key=lambda x: (x['label'] != 'contradiction', -x['confidence']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:8]
        })

    return findings
