import re
import torch
import torch.nn.functional as F
from itertools import combinations
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import streamlit as st


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """NLI CrossEncoder — Yeh har kism ki linguistic contradiction pakadta hai."""
    return CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """Similarity model ab sirf numbers ke false-positives rokne ke liye use hoga."""
    return SentenceTransformer('all-MiniLM-L6-v2')


# ─── Labels ───────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


# ─── Text Cleaning & Extraction ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_duplicates(sentences: list, threshold: float = 0.85) -> list:
    """Duplicate ya milti-julti lines ko filter karta hai taake processing fast ho."""
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
    """Bina kisi hardcoded rule ke, simply valid sentences nikalta hai."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    # Sirf un sentences ko rakho jinki length kam az kam 4 words ho (to avoid incomplete junk)
    sentences = [clean_text(s) for s in raw if len(clean_text(s).split()) >= 4]
    
    # Generic sorting: Bade sentences jinme zyada information hoti hai unko upar rakho
    sentences = sorted(sentences, key=lambda x: len(x.split()), reverse=True)
    
    unique_sentences = remove_duplicates(sentences)
    return unique_sentences[:max_sentences]


# ─── Masked Semantic Engine (General Number Guardrail) ───────────────────────

def get_masked_sentence(sentence: str) -> str:
    """Sentence se digits aur basic scales ko hata deta hai (No specific domains)."""
    s = sentence.lower()
    s = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', ' ', s) # Hatao digits
    s = re.sub(r'[\$£€₹¥%]', ' ', s) # Hatao symbols
    s = re.sub(r'\b(million|billion|trillion|thousand|hundred)\b', ' ', s) # Hatao scales
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_numbers_general(text: str) -> list:
    """Text se generic numbers nikalta hai taake unko compare kiya ja sake."""
    text = text.lower()
    numbers = []
    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text): numbers.append(float(match.group(1)) * 1_000_000_000)
    for match in re.finditer(r'(\d+\.?\d*)\s*million', text): numbers.append(float(match.group(1)) * 1_000_000)
    text = re.sub(r'\b(\d{1,3})(?:,\d{3})+\b', lambda m: m.group(0).replace(',', ''), text)
    
    for match in re.finditer(r'\b(\d+\.?\d*)\b', text):
        try:
            val = float(match.group(1))
            # Calendar years (19xx, 20xx) ko direct compare karne se bachne ke liye chota logic
            if not (1900 <= val <= 2099 and len(str(int(val))) == 4):
                numbers.append(val)
        except ValueError: continue
    return numbers


# ─── NLI Classification ──────────────────────────────────────────────────────

def classify_pair(nli_model: CrossEncoder, sent1: str, sent2: str) -> tuple:
    raw_scores = nli_model.predict([(sent1, sent2)])[0]
    probs      = F.softmax(torch.tensor(raw_scores), dim=0)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


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

                # 🚀 NO MORE BOTTLENECK!
                # Hum koi pre-filter (are_same_topic) nahi laga rahe. 
                # Direct CrossEncoder ko bol rahe hain "Tu bata, inka aapas me koi link ya clash hai?"
                label, conf = classify_pair(nli_model, s1, s2)

                # Agar NLI kehta hai ye sentences bilkul unrelated (Neutral) hain, toh skip kardo.
                if label == 'neutral':
                    continue

                # 🛡️ GENERAL NUMBER GUARDRAIL
                # NLI model numbers dekh kar kabhi ghalti karta hai (e.g., 100 cats vs 10 dogs ko contradiction keh dega).
                # Isko rokne ke liye hum masked similarity check karte hain.
                if label == 'contradiction':
                    nums1 = extract_numbers_general(s1)
                    nums2 = extract_numbers_general(s2)
                    
                    # Agar DONO sentences mein numbers hain, tabhi guardrail chalega
                    if nums1 and nums2:
                        masked_s1 = get_masked_sentence(s1)
                        masked_s2 = get_masked_sentence(s2)
                        
                        emb_m1 = sim_model.encode(masked_s1, convert_to_tensor=True)
                        emb_m2 = sim_model.encode(masked_s2, convert_to_tensor=True)
                        masked_sim = float(util.cos_sim(emb_m1, emb_m2))
                        
                        # Agar numbers hatane ke baad sentences ka aapas mein koi maqsad hi match nahi kar raha (e.g. deaths vs houses)
                        # Toh iska matlab ye actual contradiction NAHI thi, NLI ne bas numbers dekh ke jald bazi ki.
                        if masked_sim < 0.50:
                            label = 'neutral'
                            conf = 1.0

                # Agar ab bhi contradiction bachti hai, toh wo 100% solid hai!
                if label != 'neutral':
                    pair_results.append({
                        'sentence_1': s1,
                        'sentence_2': s2,
                        'label':      label,
                        'confidence': conf
                    })

        # Results ko sort karo (Contradictions sabse upar)
        pair_results.sort(key=lambda x: (x['label'] != 'contradiction', -x['confidence']))

        findings.append({
            'source_1': art_i['source'],
            'source_2': art_j['source'],
            'results':  pair_results[:10]
        })

    return findings
