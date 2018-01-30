import numpy as np
import pandas as pd

from components.Helper import *


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


class DataLoader(object):
    def __init__(self, ranked_data: dict, index: pyndri.Index, doc_len: dict, int_to_ext_dict: dict,
                 ext_to_int_dict: dict, queries: list):
        self.tfidf_data = ranked_data
        self.query_ids = list(self.tfidf_data.keys())
        self.index = index
        self.df = None
        self.doc_len = doc_len
        self.int_to_ext_dict = int_to_ext_dict
        self.ext_to_int_dict = ext_to_int_dict
        self.queries = queries

    def load_data_from_file(self, model_name):
        """Load model scores from file.

        Args:
            model_name: name of the model.
        """
        print("Loading data for model {}".format(model_name))
        retrieval_start_time = time.time()

        with open('../retrievals/{}.run'.format(model_name)) as file:
            for line in file.readlines():
                query_id, _, ext_doc_id, __, score, model = line.split()
                idx = '~'.join((query_id, ext_doc_id))

                if idx in self.df.index:
                    self.df.loc[idx, model] = float(score)

        print("Data loaded in {} seconds.".format(time.time() - retrieval_start_time))

    def drop_rows_with_null(self):
        """Drop the rows containing null values."""
        i = 0
        for idx, row in self.df.iterrows():
            if row.isnull().any():
                i += 1
                self.df.drop(idx, inplace=True, axis=0)

        print("{} rows dropped. DataFrame length:".format(i), end="")
        print(self.data_length)

    def load_additional_features(self):
        """Add document and query length to feature list."""
        for idx, row in self.df.iterrows():
            self.df.loc[idx, 'doc_len'] = self.doc_len[self.ext_to_int_dict[row['ext_doc_id']]]
            self.df.loc[idx, 'query_len'] = len(self.queries[row['query_id']])

    def data_has_nulls(self):
        """Check whether df has any null values"""
        return self.df.isnull().any()

    def column_has_nulls(self, col_name):
        """Check whether a column has any null values.

        Args:
            col_name: name of the column.
        """
        return self.df[col_name].isnull().any()

    def count_null_values(self, col_name):
        """Retrieve the count of null values on a column.

        Args:
            col_name: name of the column
        """
        return np.sum(self.df[col_name].isnull())

    def save_dataframe(self, fpath):
        """Save DataFrame object to file.

        Args:
            fpath: file path to save.
        """
        with open(fpath, 'wb') as file:
            pickle.dump(self.df, file)

    @property
    def data_length(self):
        """Retrieve DataFrame object length."""
        return len(self.df)

    @property
    def data(self):
        """Retrieve DataFrame object."""
        return self.df


class TrainingDataLoader(DataLoader):
    def __init__(self, ranked_data: dict, index: pyndri.Index, models: list, rel_file: str,
                 doc_len: dict, int_to_ext_dict: dict, ext_to_int_dict: dict, queries: list):
        super(TrainingDataLoader, self).__init__(ranked_data, index, doc_len, int_to_ext_dict, ext_to_int_dict, queries)

        self.index_list = []
        self.full_index_list = []

        self.create_df()
        self.load_data(models_list=models, relevance_file=rel_file)

    def get_indices_lists(self):
        """Create the index list based on query ID and external document ID."""
        for query_id, int_doc_ids in self.tfidf_data.items():
            for int_doc_id in int_doc_ids:
                ext_doc_id, _ = self.index.document(int_doc_id)
                self.index_list.append('~'.join((str(query_id), str(ext_doc_id))))
                self.full_index_list.append('~'.join((str(query_id), str(int_doc_id), str(ext_doc_id))))

    def create_df(self):
        """Create initial DataFrame, populating it with useful data."""
        self.get_indices_lists()
        self.df = pd.DataFrame(index=self.index_list)
        self.df['idx'] = self.full_index_list
        self.df['query_id'] = self.df.idx.apply(lambda x: x.split('~')[0])
        self.df['int_doc_id'] = self.df.idx.apply(lambda x: x.split('~')[1])
        self.df['ext_doc_id'] = self.df.idx.apply(lambda x: x.split('~')[2])
        self.df.drop(['idx'], axis=1, inplace=True)
        print("DataFrame created.")

    def load_relevance_labels(self, file_path):
        """Load relevance labels from file.

        Args:
            file_path: path to the qrel_test file.
        """
        print("Loading relevance labels.")
        retrieval_start_time = time.time()

        with open(file_path) as file:
            for line in file.readlines():
                if line[:2] not in self.query_ids:
                    continue

                query_id, _, ext_doc_id, relevance = line.split()
                idx = '~'.join((query_id, ext_doc_id))

                if idx in self.df.index:
                    self.df.loc['~'.join((query_id, ext_doc_id)), 'relevance_label'] = int(relevance)

        self.df['relevance_label'].fillna(value=0, inplace=True)
        print("Labels loaded in {} seconds.".format(time.time() - retrieval_start_time))

    def load_data(self, models_list, relevance_file):
        """Wrapper method to load all models scores and relevance labels.

        Args:
            models_list: list of model names.
            relevance_file: path to the file with relevance labels
        """
        for model in models_list:
            self.load_data_from_file(model)
        self.drop_rows_with_null()
        self.load_additional_features()
        self.load_relevance_labels(relevance_file)


class TestingDataLoader(DataLoader):
    def __init__(self, ranked_data: dict, index: pyndri.Index, models: list, rel_file: str,
                 doc_len: dict, int_to_ext_dict: dict, ext_to_int_dict: dict, queries: list):
        super(TestingDataLoader, self).__init__(ranked_data, index, doc_len, int_to_ext_dict, ext_to_int_dict, queries)

        self.create_df(rel_file)
        self.load_data(models)

    def create_df(self, rel_file):
        self.df = pd.DataFrame(columns=['query_id', 'int_doc_id', 'ext_doc_id', 'relevance_label'])
        lookup_indices = []

        with open('../retrievals/tfidf.run', 'r') as file:
            for line in file.readlines():
                query_id, _, ext_doc_id, _, __, ___ = line.split()

                lookup_indices.append('~'.join((query_id, ext_doc_id)))

        with open(rel_file, 'r') as file:
            for line in file.readlines():
                query_id, _, ext_doc_id, relevance = line.split()

                idx = '~'.join((query_id, ext_doc_id))

                if idx in lookup_indices:
                    self.df.loc[idx, 'query_id'] = query_id
                    self.df.loc[idx, 'int_doc_id'] = ext_to_int_dict[ext_doc_id]
                    self.df.loc[idx, 'ext_doc_id'] = ext_doc_id
                    self.df.loc[idx, 'relevance_label'] = int(relevance)

    def load_data(self, models_list):
        """Wrapper method to load all models scores and relevance labels.

        Args:
            models_list: list of model names.
        """
        for model in models_list:
            self.load_data_from_file(model)
        self.load_additional_features()
        self.drop_rows_with_null()


if __name__ == '__main__':
    action = 'get_train_data'
    action = 'get_valdation_data'

    tfidf_data = dict(load_pickle('../pickles/prepro_doc_col_q10_top1000_tfidf.pkl'))
    index = pyndri.Index('../index/')

    models = ['tfidf', 'LDA', 'LSI', 'dp_mu_500', 'glm']
    training_rel_file = '../ap_88_89/qrel_test'
    validate_rel_file = '../ap_88_89/qrel_validation'

    if action == 'get_train_data':
        data_loader = TrainingDataLoader(ranked_data=tfidf_data, index=index, models=models,
                                         rel_file=training_rel_file, doc_len=document_lengths,
                                         int_to_ext_dict=int_to_ext_dict, ext_to_int_dict=ext_to_int_dict,
                                         queries=tokenized_queries)
        data_loader.save_dataframe('../pickles/LTR_DF_Training.pkl')
    elif action == 'get_valdation_data':
        data_loader = TestingDataLoader(ranked_data=tfidf_data, index=index, models=models,
                                        rel_file=validate_rel_file, doc_len=document_lengths,
                                        int_to_ext_dict=int_to_ext_dict, ext_to_int_dict=ext_to_int_dict,
                                        queries=tokenized_queries)
        data_loader.save_dataframe('../pickles/LTR_DF_Validation.pkl')
    else:
        print("Action not known")
