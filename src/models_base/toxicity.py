from transformers import AutoTokenizer, AutoModelForSequenceClassification


_MODEL_ID = "cointegrated/rubert-tiny-toxicity"


print("[toxicity] loading RuBERT-tiny…")
tok = AutoTokenizer.from_pretrained(_MODEL_ID)
mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
mdl.eval()
print("[toxicity] model ready ✓")

__all__ = ["tok", "mdl"]

