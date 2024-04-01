
# Importing necessary libraries
import json  # Library for handling JSON data
import random  # Library for generating random numbers
import datetime  # Library for working with dates and times
import pymongo  # Library for interacting with MongoDB
import uuid  # Library for generating unique identifiers
import re
#infrence lib
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


#Load gpt model path
model_path = "C:\\Projects\\gptchatbot_rag\\outputFineTune"
max_length = 20




def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(sequence, model_path = model_path, max_length = max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    
    # Remove the user input from the sequence

    
    ids = tokenizer.encode(sequence, return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        pad_token_id=model.config.eos_token_id,
        top_k=100,
        early_stopping=True,  
        num_beams=5, 
        num_return_sequences=1,  
        max_length=max_length,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    
    # Remove the bot prefix "[bot]: " from the generated text
    # Define a regular expression pattern to match "[bot]:"
    pattern = r'\[bot\]:'
    # Replace the matched pattern with an empty string
    generated_text = re.sub(pattern, '', generated_text)
    generated_text = generated_text.split("\n")
    generated_text = generated_text[1]
    return generated_text




     