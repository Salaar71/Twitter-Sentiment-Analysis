import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


def input():
    nlp = spacy.load("en_core_web_md")
    with open ("Simple Text.txt", "r") as f:
        text = f.read()
    print(text)
    doc = nlp(text)
    tokenize(doc)
    
def tokenize(doc):
    print("\nStep 1: Tokenization")
    for token in doc:
        print(token)
    lemmatize(doc)

def lemmatize(doc):
    print("\nStep 2: Lemmatization")
    for token in doc:
        print(token.lemma_)
    POS(doc)
        
def POS(doc):
    print("\nStep 3: Parts of Speech Tagging")
    for token in doc:
        print(token.text, token.pos_)
    NER(doc)

def NER(doc):
    print("\nStep 4: Named Entity Recognition")
    for ent in doc.ents:
        print(ent.text,ent.label_)
    Sent_Analysis(doc)
        
def Sent_Analysis(doc):
    print("\nStep 5: Sentiment Analysis")
    comment = doc.text
    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']
    
    # sentiment analysis
    encoded_comment = tokenizer(comment, return_tensors='pt')
    #print(encoded_comment)
    
    # output = model(encoded_comment['input_ids'], encoded_comment['attention_mask'])
    output = model(**encoded_comment)
    #print(output)
    
    scores = output[0][0].detach().numpy()
    #print(scores)
    scores = softmax(scores)
    print("\n")
    for i in range(len(scores)):  
        l = labels[i]
        s = scores[i]
        print(l,s)
    if scores[0]>0.5:
        print("\nResult: The comment is negative.")
    elif scores[1]>0.5:
        print("\nResult: The comment is neutral.")
    elif scores[2]>0.5:
        print("\nResult: The comment is positive.")

if __name__ == "__main__":
    input()