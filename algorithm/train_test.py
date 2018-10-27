from sklearn.model_selection import train_test_split as sk_train_test_split

def train_test_split(data):
    X, X_test, y, y_test = sk_train_test_split(data.loc[:, 'text':], data['type'], test_size=0.3)
    return X, X_test, y, y_test