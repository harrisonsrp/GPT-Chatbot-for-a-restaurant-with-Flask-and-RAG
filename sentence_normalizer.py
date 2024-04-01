from nltk.tokenize import RegexpTokenizer  # Importing RegexpTokenizer for tokenization
from nltk.stem.wordnet import WordNetLemmatizer  # Importing WordNetLemmatizer for lemmatization



# List of stop words manually selected to be removed from the text
stop_words = ['now', 'then', 'here', 'there', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose', 'what',
                   'these', 'those', 'thus', 'than', 'that', 'so', 'such', 'shall', 'should', 'must', 'might', 
                   'may', 'will', 'would', 'can', 'could', 'ought', 'need', 'dare', 'ought', 'her', 'his', 'hers',
                   'he', 'she', 'they', 'them', 'it', 'its', 'their', 'theirs', 'our', 'ours', 'yours', 'whose',
                   'any', 'each', 'every', 'few', 'many', 'several', 'some', 'all', 'most', 'none', 'somebody',
                   'someone', 'something', 'anybody', 'anyone', 'anything', 'nobody', 'nothing', 'everybody',
                   'everyone', 'everything', 'other', 'another', 'others', 'another', 'both', 'either', 'neither',
                   'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third', 'last', 'next', 'own', 
                   'whole', 'same', 'much', 'more', 'many', 'little', 'less', 'least', 'enough', 'seems', 'seem',
                   'became', 'become', 'becomes', 'becoming', 'could', 'couldnt', 'do', 'does', 'doing', 'done', 
                   'did', 'had', 'has', 'having', 'make', 'made', 'makes', 'making', 'may', 'might', 'must', 'need',
                   'ought', 'shall', 'should', 'use', 'used', 'uses', 'using', 'try', 'trying', 'tried', 'tries',
                   'very', 'well', 'say', 'says', 'said', 'get', 'gets', 'got', 'getting', 'go', 'goes', 'gone',
                   'went', 'go', 'going', 'take', 'takes', 'took', 'taking', 'come', 'comes', 'came', 'coming', 'am', 'is', 'are','was', 'were', 'can', 
                   '\'t', 'we', 'they','whats', 'wanna', 'i', 'you']


# Function to lemmatize a list of tokens
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()  # Initializing WordNetLemmatizer
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatizing each token
    return lemmatized_tokens

# Function to tokenize a sentence and remove punctuation
def tokenize_and_remove_punctuation(sentence):
    tokenizer = RegexpTokenizer(r'\w+')  # Initializing RegexpTokenizer to remove punctuation
    tokens = tokenizer.tokenize(sentence)  # Tokenizing the sentence
    return tokens

# Function to remove stop words from a list of word tokens
def remove_stopwords(word_tokens):
    filtered_tokens = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_tokens.append(w)  # Adding non-stop words to filtered tokens list
    return filtered_tokens

# Main preprocessing function: converting to lower case, removing punctuation, lemmatizing, and removing stop words
def preprocess_main(sent):
    sent = sent.lower()  # Converting the sentence to lowercase
    tokens = tokenize_and_remove_punctuation(sent)  # Tokenizing the sentence and removing punctuation
    lemmatized_tokens = lemmatize_sentence(tokens)  # Lemmatizing the tokens
    orig = lemmatized_tokens  # Keeping a copy of the original lemmatized tokens
    filtered_tokens = remove_stopwords(lemmatized_tokens)  # Removing stop words from lemmatized tokens
    if len(filtered_tokens) == 0:
        filtered_tokens = orig  # If stop word removal removes everything, revert to original tokens
    normalized_sent = " ".join(filtered_tokens)  # Joining filtered tokens to form a normalized sentence
    return normalized_sent  # Returning the normalized sentence
