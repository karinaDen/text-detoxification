import re
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer


def tokenize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens

def delete_baseline(input_file, bad_words_file, output_file):
    # Read the toxic sentences from the input file
    with open(input_file, 'r') as f:
        toxic_sentences = f.readlines()

    # Read the bad words from the bad words dictionary
    with open(bad_words_file, 'r') as f:
        bad_words = f.read().splitlines()

    # Open the output file for writing non-toxic sentences
    with open(output_file, 'w+') as f:
        for sentence in toxic_sentences:
            # Tokenize the sentence using NLTK
            tokens = tokenize_sentence(sentence.strip())

            # Remove bad words from the tokens
            non_toxic_tokens = [token for token in tokens if token.lower() not in bad_words]

            # detokenize the tokens
            non_toxic_sentence = TreebankWordDetokenizer().detokenize(non_toxic_tokens)

            # Write the non-toxic sentence to the output file
            f.write(non_toxic_sentence + '\n')

    print("Non-toxic sentences have been written to the output file.")

def main():
    #create output file
    delete_baseline('data/interim/reference.txt', 'data/interim/bad_words.txt', 'data/interim/baseline.txt')

if __name__ == "__main__":
    main()
