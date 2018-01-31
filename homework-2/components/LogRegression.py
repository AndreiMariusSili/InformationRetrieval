import collections
import pickle
import pyndri
from sklearn.linear_model import LogisticRegressionCV
from components import Helper
from components import LTR_Process_Data


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


if __name__ == '__main__':
    X_cols = ['TF-IDF', 'LDA', 'LSI', 'dp_mu_500', 'GLM_top1000docs_sigma50_mu1000', 'doc_len', 'query_len']
    y_cols = ['relevance_label']

    # Load training data
    print("Loading training data... ", end='')
    try:
        training_data = load_pickle('../pickles/LTR_DF_Training.pkl')
    except FileNotFoundError:
        tfidf_data = dict(load_pickle('../pickles/prepro_doc_col_q50_top1000_tfidf.pkl'))
        index = pyndri.Index('../index/')

        models_files = ['TF-IDF', 'LDA', 'LSI', 'dp_mu_500', 'GLM']
        training_rel_file = '../ap_88_89/qrel_test'

        data_loader = LTR_Process_Data.TrainingDataLoader(ranked_data=tfidf_data, index=index, models=models_files,
                                                          rel_file=training_rel_file, doc_len=Helper.document_lengths,
                                                          int_to_ext_dict=Helper.int_to_ext_dict,
                                                          ext_to_int_dict=Helper.ext_to_int_dict,
                                                          queries=Helper.tokenized_queries)
        training_data = data_loader.data

    X_train = training_data[X_cols]
    y_train = training_data[y_cols].values.ravel()
    normalized_X_train = (X_train - X_train.mean()) / X_train.std()
    normalized_X_train = normalized_X_train.values
    print("Succes!")

    # Load validation data
    print("Loading validation data... ", end='')
    try:
        validation_data = load_pickle('../pickles/LTR_DF_Validation.pkl')
    except FileNotFoundError:
        tfidf_data = dict(load_pickle('../pickles/prepro_doc_col_q50_top1000_tfidf.pkl'))
        index = pyndri.Index('../index/')

        models_files = ['TF-IDF', 'LDA', 'LSI', 'dp_mu_500', 'GLM']
        validate_rel_file = '../ap_88_89/qrel_validation'

        data_loader = LTR_Process_Data.ValidatingDataLoader(ranked_data=tfidf_data, index=index, models=models_files,
                                                            rel_file=validate_rel_file, doc_len=Helper.document_lengths,
                                                            int_to_ext_dict=Helper.int_to_ext_dict,
                                                            ext_to_int_dict=Helper.ext_to_int_dict,
                                                            queries=Helper.tokenized_queries)

        validation_data = data_loader.data

    X_validate = validation_data[X_cols]
    y_validate = validation_data[y_cols].values.ravel()
    normalized_X_validate = (X_validate - X_validate.mean()) / X_validate.std()
    normalized_X_validate = normalized_X_validate.values
    print("Succes!")

    # Train Logistic Regression model on 10-fold cross validation
    print("Training model... ", end='')
    log_reg_cv = LogisticRegressionCV(cv=10)
    log_reg_cv.fit(normalized_X_train, y_train)
    print("Succes!")

    # Predict relevance labels for validation data and write to .run file
    print("Predicting values and saving to file... ", end='')
    validation_data['relevance_result'] = log_reg_cv.predict(normalized_X_validate)
    validation_data.sort_values(by=['query_id', 'relevance_result'], ascending=[True, False], inplace=True)

    data = collections.defaultdict(list)

    for idx, row in validation_data.iterrows():
        data[row['query_id']].append((row['relevance_result'], row['ext_doc_id']))

    with open('log_reg.run', 'w') as f_out:
        Helper.write_run(
            model_name='LogReg',
            data=data,
            out_f=f_out,
            max_objects_per_query=1000)
    print("Succes!")

