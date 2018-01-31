import collections
import time
import pickle
import os
import pyndri
import pyndri.compat
import numpy as np
import gensim
from components import Helper


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


class Sentences2Vec(pyndri.compat.IndriSentences):
    """IndriSentences own class implementation."""

    def __init__(self, index, dictionary, max_documents=None):
        super(Sentences2Vec, self).__init__(index, dictionary, max_documents)

    def __iter__(self):
        for int_doc_id in range(self.index.document_base(),
                                self._maximum_document()):
            ext_doc_id, doc = self.index.document(int_doc_id)
            tokens_bow = self.dictionary.doc2bow(doc)

            yield tuple(
                (token_id, weight)
                for (token_id, weight) in tokens_bow
                if token_id in self.dictionary and token_id > 0)


class LSMBaseClass:
    def __init__(self, index: pyndri.Index, dictionary: dict):
        self.index = index
        self.dictionary = dictionary
        pass

    @property
    def model_name(self):
        """Model name"""
        return ""

    def save(self, fpath: str):
        """Save current model to file for further use.

        Args:
            fpath: file path to save model.
        """
        self.model.save(fpath)
        print("Model saved.")

    def load_documents_representation(self):
        """Get and store document representations for future use."""
        try:
            print("Loading document representations from file...", end='')
            with open('../pickles/LSI_DocRepresentations.pkl', 'rb') as file:
                self.doc_representations_dict = pickle.load(file)
            print("Success!")
        except FileNotFoundError:
            print("Error!")
            print("Computing and loading documents' representations...")

            retrieval_start_time = time.time()
            for int_doc_id in range(self.index.document_base(), self.index.maximum_document()):
                ext_doc_id, doc = self.index.document(int_doc_id)
                self.doc_representations_dict[int_doc_id] = self.get_representation(doc)

            print("Documents successfully loaded in {} seconds.".format(time.time() - retrieval_start_time))

            with open('../pickles/LSI_DocRepresentations.pkl', 'wb') as file:
                pickle.dump(self.doc_representations_dict, file)

    def get_representation(self, tokens_list):
        """Build representation given list of token ids.

        Args:
            tokens_list: list of token ids.
        Return:
            List of the LSI representation.
        """
        tokens_bow = self.dictionary.doc2bow(tokens_list)
        doc_representation = [(token_id, weight)
                              for (token_id, weight) in tokens_bow
                              if token_id in self.dictionary and token_id > 0]
        lsi_repr = [x[1] for x in self.model[doc_representation]]
        return lsi_repr

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity of 2 vectors.

        Args:
            vec1: 1st vector.
            vec2: 2nd vector.
        Return:
            Cosine similarity.
        """
        dot_prod = np.dot(vec1, vec2)
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)
        return dot_prod / (norm_1 * norm_2)

    def run_retrieval(self, tfidf_data):
        """
        Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

        Args:
            tfidf_data: top-1000 query-document rankings from TF-IDF.
        """
        run_out_path = '{}.run'.format(self.model_name)

        if os.path.exists(run_out_path):
            print('RUN file already existing')
            return

        data = collections.defaultdict(list)

        print('Retrieving using {}'.format(self.model_name))
        retrieval_start_time = time.time()

        for query_id, doc_list in tfidf_data.items():
            query_representation = self.get_representation(Helper.tokenized_queries[query_id])

            for int_doc_id in doc_list:
                ext_doc_id, doc = self.index.document(int_doc_id)
                doc_representation = self.doc_representations_dict[int_doc_id]

                cos_similarity = self.cosine_similarity(query_representation, doc_representation)
                data[query_id].append((cos_similarity, ext_doc_id))

            data[query_id] = sorted(data[query_id], reverse=True)

        with open(run_out_path, 'w') as f_out:
            Helper.write_run(
                model_name=self.model_name,
                data=data,
                out_f=f_out,
                max_objects_per_query=1000)

        print('Retrieval run took {} seconds.'.format(time.time() - retrieval_start_time))


class LatentSemanticIndexing(LSMBaseClass):
    """Latent Semantic Indexing method implementation."""

    def __init__(self, index: pyndri.Index, dictionary: dict, num_topics=200, load_model=False, fname=""):
        super(LatentSemanticIndexing, self).__init__(index, dictionary)

        self.index = index
        self.dictionary = dictionary

        if load_model:
            if fname == "":
                raise ValueError('File path not provided.')
            self.load(fname)
        else:
            self.train(num_topics)

        self.doc_representations_dict = collections.defaultdict(list)
        self.load_documents_representation()

    @property
    def model_name(self):
        """Model name"""
        return "LSI"

    def load(self, fpath: str):
        """Load model from file.

        Args:
            fpath: file path to load model.
        """
        self.model = gensim.models.lsimodel.LsiModel.load(fpath)
        print("Model loaded.")

    def train(self, num_topics=200):
        """Train LSI model given the index and dictionary."""
        print("Training started...")
        retrieval_start_time = time.time()

        corpus = Sentences2Vec(self.index, self.dictionary)
        self.model = gensim.models.lsimodel.LsiModel(corpus=corpus,
                                                     id2word=self.dictionary.id2token,
                                                     num_topics=num_topics)
        print("Model trained in {} seconds.".format(time.time() - retrieval_start_time))


class LatentDirichletAllocation(LSMBaseClass):
    """Latent Dirichlet Allocation method implementation."""

    def __init__(self, index: pyndri.Index, dictionary: dict, num_topics=200, load_model=False, fname=""):
        super(LatentDirichletAllocation, self).__init__(index, dictionary)

        self.index = index
        self.dictionary = dictionary

        if load_model:
            if fname == "":
                raise ValueError('File path not provided.')
            self.load(fname)
        else:
            self.train(num_topics)

        self.doc_representations_dict = collections.defaultdict(list)
        self.load_documents_representation()

    @property
    def model_name(self):
        """Model name"""
        return "LDA"

    def load(self, fpath: str):
        """Load model from file.

        Args:
            fpath: file path to load model.
        """
        self.model = gensim.models.ldamodel.LdaModel.load(fpath)
        print("Model loaded.")

    def train(self, num_topics=200):
        """Train LDA model given the index and dictionary.

        Args:
            num_topics: number of topics for LDA Model
        """
        print("Training started...")
        retrieval_start_time = time.time()

        corpus = Sentences2Vec(self.index, self.dictionary)
        self.model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                     id2word=self.dictionary.id2token,
                                                     num_topics=num_topics,
                                                     update_every=1,
                                                     chunksize=10000,
                                                     passes=1)
        print("Model trained in {} seconds.".format(time.time() - retrieval_start_time))


if __name__ == '__main__':
    run_on_model = 'lsi'
    # run_on_model = 'lda'

    tfidf_data = dict(load_pickle('../pickles/prepro_doc_col_q150_top1000_tfidf.pkl'))

    if run_on_model == 'lsi':
        lsi_model = LatentSemanticIndexing(index=Helper.index, dictionary=Helper.dictionary,
                                           num_topics=len(Helper.tokenized_queries))
        lsi_model.run_retrieval(tfidf_data)
    elif run_on_model == 'lda':
        lda_model = LatentDirichletAllocation(index=Helper.index, dictionary=Helper.dictionary,
                                              num_topics=len(Helper.tokenized_queries))
        lda_model.run_retrieval(tfidf_data)
    else:
        print('Run on model not known.')
