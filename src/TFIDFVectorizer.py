from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy


def main():
    text = 'Your text goes here.'

    # Tokenize sentences
    sentences = sent_tokenize(text.strip())

    vectorizer = TfidfVectorizer(norm=False, smooth_idf=False)
    # Generate sentence vectors
    sentence_vectors = vectorizer.fit_transform(sentences)

    print('Total sentences',len(sentence_vectors.toarray()))
    print('Total distinct words',len(vectorizer.get_feature_names()))
    print('Distinct words',vectorizer.get_feature_names())
    print('Sentence vectors shape',sentence_vectors.toarray().shape)
    print('Sentence vectors embeddings',sentence_vectors.toarray())

    # To generate word embeddings
    word_embeddings = []
    for i, word in enumerate(vectorizer.get_feature_names()):
        word_vectors = []
        for j, sentence in enumerate(sentences):
            # Get the count of word from each sentence
            word_vectors.append(sentence_vectors[j,i])
        word_embeddings.append(word_vectors)

    word_embeddings = numpy.array(word_embeddings)
    print('Word vectors shape',word_embeddings.shape)
    print('Word vectors embeddings',word_embeddings)


if __name__ == "__main__":
    main()