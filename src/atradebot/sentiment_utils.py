# sentiment analysis utils
# from https://github.com/culurciello/scratchy/

from nltk import tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# get NLP model to analyze sentiment
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_analyzer = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def get_sentiment(text, model, max_length=512):
    sentences = tokenize.sent_tokenize(text)
    # truncate sentences that are too long
    for i, s in enumerate(sentences):
        if len(s) > max_length:
            sentences[i] = sentences[i][:max_length]

    sentiment = model(sentences)
    sum, neutrals = 0, 0
    if len(sentiment) > 0:
        for r in sentiment: 
            sum += (r["label"] == "Positive")
            neutrals += (r["label"] == "Neutral")

        den = len(sentiment)-neutrals
        sentiment = sum/den if den > 0 else 1.0 # as all neutral
    return sentiment