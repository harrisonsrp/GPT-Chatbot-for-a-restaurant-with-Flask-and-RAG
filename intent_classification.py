import codecs
import json

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sentence_normalizer

#Gpt Requierments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
#Rag Requierments
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load the data from rag_data
with codecs.open(r'RAG_data\rag_embedded_dataset.json', 'r', encoding='utf-8') as f:
    rag_data = json.load(f)

def embed_sentence(sentence):
    tokenizer_path = "C:/Projects/GPTChatbot_rag/outputFineTune"
    model_path = "C:/Projects/GPTChatbot_rag/outputFineTune"
    # Get embedding for a word
    custom_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    custom_model = GPT2Model.from_pretrained(model_path)
    tokenized_input = custom_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        sentence_vec = custom_model(**tokenized_input)
    return sentence_vec


def predict_intent(input_sentence, rag_data=rag_data, distance_metric='euclidean'):
    input_sentence = sentence_normalizer.preprocess_main(input_sentence)
    input_vec = embed_sentence(input_sentence).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Extract patterns and corresponding tags
    X_train = []
    y_train = []
    for intent in rag_data['intents']:
        for pattern in intent['patterns']:
            X_train.append(np.array(pattern).flatten())
            y_train.append(intent['tag'])

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train KNN classifier with specified distance metric
    knn_classifier = KNeighborsClassifier(n_neighbors=1, metric=distance_metric)
    knn_classifier.fit(X_train, y_train)

    # Predict the intent for the input sentence
    predicted_intent = knn_classifier.predict([input_vec])[0]
    return predicted_intent