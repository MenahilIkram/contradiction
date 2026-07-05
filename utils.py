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


def extract_sentences(text: str, max_sentences: int = 6) -> list[str]:
    """
    Split article into sentences and return top factual ones.
    Prioritizes sentences with numbers or proper nouns (more likely factual claims).
    """
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [clean_text(s) for s in raw if len(clean_text(s)) > 40]

    # Prefer factual sentences (numbers, capitalized entities)
    factual = [s for s in sentences if re.search(r'\d+|[A-Z][a-z]{2,}', s)]
    other   = [s for s in sentences if s not in factual]

    combined = factual + other
    return combined[:max_sentences]


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
