from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
mdl = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")
print("OK, model loaded")
