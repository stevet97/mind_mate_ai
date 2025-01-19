# toxic_filter/toxic_filter.py

from transformers import pipeline

# Create a pipeline for toxicity classification
toxicity_classifier = pipeline(
    "text-classification",
    model="textdetox/xlmr-large-toxicity-classifier"
)

def is_toxic(text, threshold=0.5):
    """
    Classify text using the pipeline. If label == 'TOXIC' and score >= threshold,
    we consider it toxic.
    """
    # For large texts, you might want to chunk or sample. Example:
    truncated_text = text[:1000]  # first 1000 chars
    results = toxicity_classifier(truncated_text)

    # Example result structure: [{"label": "TOXIC", "score": 0.99}]
    for r in results:
        if r["label"].upper() == "TOXIC" and r["score"] >= threshold:
            return True
    return False
