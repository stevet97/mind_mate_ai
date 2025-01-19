# toxic_filter/toxic_filter.py

from transformers import pipeline

toxicity_classifier = pipeline(
    "text-classification",
    model="textdetox/xlmr-large-toxicity-classifier"
)

def get_toxicity_score(text, max_length=1000):
    """
    Return a float in [0, 1], representing how 'toxic' the text is.
    If the classifier doesn't label the text as 'TOXIC', score = 0.0.
    Otherwise, it uses the 'score' from the model (probability).
    """
    # For large texts, we sample or truncate to 'max_length' chars
    truncated_text = text[:max_length]

    # The pipeline might return a list of dicts, e.g. [{"label": "TOXIC", "score": 0.99}]
    results = toxicity_classifier(truncated_text)

    # We'll assume the pipeline returns only one item in the list:
    for r in results:
        if r["label"].upper() == "TOXIC":
            return r["score"]
    return 0.0
