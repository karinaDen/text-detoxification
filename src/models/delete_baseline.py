import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd



def delete_baseline(sentence, bad_words):
    """
    This function takes a sentence and a list of bad words as input and returns a non-toxic sentence.
    @param sentence: A string representing the sentence that needs to be converted to non-toxic sentence.
    @param bad_words: A list of strings representing the bad words that need to be removed from the sentence.
    @return: A string representing the non-toxic sentence.
    """
    # Tokenize the sentence using NLTK
    tokens = tokenize_sentence(sentence.strip())

    # Remove bad words from the tokens
    non_toxic_tokens = [token for token in tokens if token.lower() not in bad_words]

    # detokenize the tokens
    non_toxic_sentence = TreebankWordDetokenizer().detokenize(non_toxic_tokens)
    
    return non_toxic_sentence

import nltk

def tokenize_sentence(sentence):
    """
    This function takes a sentence as input and returns a list of tokens.

    Args:
        sentence (str): The sentence to be tokenized.

    Returns:
        list: A list of tokens.
    """
    tokens = nltk.word_tokenize(sentence)
    return tokens

def predict_baseline(input_file, bad_words_file, output_file):
    """
    This function takes an input file, a bad words file and an output file as input and writes non-toxic sentences to the output file.
    @param input_file: A file containing toxic sentences.
    @param bad_words_file: A file containing bad words.
    @param output_file: A file where non-toxic sentences will be written.
    """

    # Read the toxic sentences from the input file
    with open(input_file, 'r') as f:
        toxic_sentences = f.readlines()

    # Read the bad words from the bad words dictionary
    with open(bad_words_file, 'r') as f:
        bad_words = f.read().splitlines()

    # Open the output file for writing non-toxic sentences
    with open(output_file, 'w+') as f:
        for sentence in toxic_sentences:
            
            non_toxic_sentence = delete_baseline(sentence, bad_words)
            # Write the non-toxic sentence to the output file
            f.write(non_toxic_sentence + '\n')

    print("Non-toxic sentences have been written to the output file.")

