import numpy as np

def tokenization_ftm(X,X_test,vocab):

    # tokenize the text for further calculations
    X['tokenized_final_text']=X['final_text'].str.split(' ')
    X_test['tokenized_final_text']=X_test['final_text'].str.split(' ')

    # document frequency(number of docs containing word w) and Inverse document frequency(measures rarity of each word)
    df = {}
    for k in vocab.keys():
        df[k] = np.sum(X['tokenized_final_text'].apply(lambda x: 1 if k in x else 0))

    # Now calculate the idf score of each word
    idf = {k: 1 + np.log((1 + X.shape[0] / (1 + v))) for k, v in df.items()}

    # tf * idf
    for elem in vocab.keys():
        X[elem] = X['tokenized_final_text'].apply(lambda x: x.count(elem) * idf[elem] if elem in x else 0)
    for elem in vocab.keys():
        X_test[elem] = X_test['tokenized_final_text'].apply(lambda x: x.count(elem) * idf[elem] if elem in x else 0)

    return X,X_test