#for loading the tokenizer from my model
from transformers import GPT2Tokenizer, GPT2Model
import torch


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