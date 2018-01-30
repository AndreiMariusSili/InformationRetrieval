import pickle
from sklearn.linear_model import LogisticRegression


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


if __name__ == '__main__':
    X_cols = ['tfidf', 'LDA', 'LSI', 'dp_mu_500', 'glm', 'doc_len', 'query_len']
    y_cols = ['relevance_label']

    training_data = load_pickle('../pickles/LTR_DF_Training.pkl')
    X_train = training_data[X_cols]
    y_train = training_data[y_cols].values.ravel()
    normalized_X_train = (X_train - X_train.mean()) / X_train.std()
    normalized_X_train = normalized_X_train.values

    validation_data = load_pickle('../pickles/LTR_DF_Validation.pkl')
    X_validate = validation_data[X_cols]
    y_validate = validation_data[y_cols].values.ravel()
    normalized_X_validate = (X_validate - X_validate.mean()) / X_validate.std()
    normalized_X_validate = normalized_X_validate.values

    log_reg = LogisticRegression()
    log_reg.fit(normalized_X_train, y_train)

    prediction = log_reg.predict(normalized_X_validate)

    print(prediction)
