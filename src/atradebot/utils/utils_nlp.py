#! /usr/bin/python3

# -*- coding: utf-8 -*-

# sentiment analysis utils
# from https://github.com/culurciello/scratchy/

# note: need to separately download punkt

from nltk import tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import re
import os
import math
import fitz # PyMuPDF
import io
from PIL import Image, ImageDraw
import tiktoken
import math
import layoutparser as lp
import cv2
from tqdm import tqdm


# get NLP model to analyze sentiment
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_analyzer = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)



MAX_TOKENS = 8192

def max_char(in_str, max_char=200):
    """return text with maximum number of characters

    :param in_str: input text
    :type in_str: str
    :param max_char: max number of char, defaults to 200
    :type max_char: int, optional
    :return: text with max number of characters
    :rtype: str
    """    
    txt = ''
    for w in in_str.split():
        if len(txt)+len(w) > max_char:
            break
        txt += ' ' + w
    return txt

def clean_text(in_str):
    ss = in_str.replace('-\n', '')  # connect linebreak
    for a in re.finditer(r'.\n.', ss):  # connect linebreak if not new paragraph
        ss = ss[:a.start() + 1] + ' ' + ss[a.start() + 2:]
    ss = in_str.replace('\n', ' ')
    return ss

def extract_pageimg(file, img_pages, out_dir = 'tmp_pages'):
    pages_img = {}
    pdf_document = fitz.open(file)
    for idx in img_pages:
        # Select the page by page_number
        pdf_page = pdf_document[idx]

        # Create a PNG image surface with the same dimensions as the page
        image = pdf_page.get_pixmap(alpha=False)
        # Export the image to a file
        output_image_path =  os.path.join(out_dir, f'page_{idx}.png')
        image.save(output_image_path)
        pages_img[idx] = output_image_path

    pdf_document.close()
    return pages_img

def text_split(txt, n_parts):
    text = txt.split()
    part_len = len(text) // n_parts
    parts = []
    for i in range(n_parts):
        if i == n_parts - 1: # the remaining characters
            parts.append(text[i * part_len:])
        else: # get `part_length` characters
            parts.append(text[i * part_len:(i+1) * part_len])
    parts = [' '.join(p) for p in parts]
    return parts
    
def breaktext(text):
    # split the text into chunks of a maximum number of tokens
    tokenizer = tiktoken.get_encoding('cl100k_base')
    if len(tokenizer.encode(text)) < MAX_TOKENS:
        return [text]

    sentences = text.split('. ')
    # Get the number of tokens for each sentence
    chunks = []
    chunk = []
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    tokens_so_far = 0
    for sentence, token in zip(sentences, n_tokens):
        if token > MAX_TOKENS:
            ltxt = text_split(sentence, math.ceil(token / MAX_TOKENS))
            for t in ltxt:
                chunks.append(t)
            continue
        # add the chunk to the list of chunks and reset
        if token + tokens_so_far > MAX_TOKENS:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
    if tokens_so_far > 0:
        chunks.append(". ".join(chunk) + ".")
    return chunks


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

if __name__ == "__main__":
    pass

