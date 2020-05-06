from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize


def main():
    text = 'Your text goes here.'

    sentences = sent_tokenize(text)
    print('Sentences',sentences)

    words = [word_tokenize(sent) for sent in sentences]
    print('Words',words)

    # To save model
    model = Word2Vec(words,min_count=1)
    model.save("word2vec_CBOW.model")
    print("Model Saved")

    # To load model
    model = Word2Vec.load("word2vec_CBOW.model")
    print("Model Loaded")

    keys = [word for word in model.wv.vocab]
    print('Words',keys)
    print('Words Counts',len(keys))

    print('Getting vector for the first 2 words.')
    for w in keys[:2]:
        print('Word - ',w)
        print('Word Vector - ',model.wv.get_vector(w))

    print('Similarity between first 2 words.')
    print(keys[0],keys[1],model.wv.similarity(keys[0],keys[1]))


if __name__ == "__main__":
    main()