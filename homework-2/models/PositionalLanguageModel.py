from components import Helper
import collections
import numpy as np
import pyndri
import pickle
import time
import math
import os


class PositionalLanguageModel:
    """Positional language model.

    Notes:
        The model uses query log likelihood to compute the scores at each position. We treat the
        vocabulary to be the query, since ultimately, other terms will have a score of 0.

        All five kernels are implemented: Gaussian, Triangle, Cosine, Circle, Passage.
        Only the best position strategy is used for document ranking.

    Attributes:
        index: pyndry index for the entire collection.
        inverted_index: dict of term frequencies per document.
        queries: dict of tokenized queries.
        doc_len: dict of documents lengths.
        col_freq: dict of term frequencies for the entire collection.
        col_size: size of the document collection (number of query terms in the collection)
        doc_len: dict of documents lengths.
        max_len: maximum length for the collection.
        sigma: sigma tuning parameter for the kernel.
        mu: tuning parameter to control Dirichlet smoothing interpolation.
        ker_type: the type of kernel to use.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), queries: dict,
                 col_freq: collections.defaultdict(int), doc_len: dict, max_len: int,
                 sigma: float, mu: float, ker_type: str):
        """Initialize positional language model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            queries: dict of tokenized queries.
            col_freq: dict of term frequencies for the entire collection.
            doc_len: dict of documents lengths.
            max_len: maximum length for the collection.
            sigma: sigma tuning parameter for the kernel.
            mu: tuning parameter to control Dirichlet smoothing interpolation.
            ker_type: the type of kernel to use.
        """
        self.index = index
        self.inverted_index = inverted_index
        self.queries = queries
        self.col_freq = col_freq
        self.col_size = sum(col_freq)
        self.doc_len = doc_len
        self.max_len = max_len
        self.sigma = sigma
        self.mu = mu
        self.ker_type = ker_type
        self.kernel_matrix = np.full(shape=(max_len, max_len), fill_value=np.nan, dtype=np.float)

    def score(self, int_doc_id: int, query: tuple) -> float:
        """Compute the score for a document and a query.

        Args:
            int_doc_id: the document id.
            query: tuple of query term ids.
        Return:
            Maximum position-based score of a query given a document.
        """
        ext_doc_id, doc = self.index.document(int_doc_id)

        # per position per term virtual counts (for each query term)
        term_virtual_counts = [collections.defaultdict(lambda: 0) for i in range(len(doc))]
        # per position total virtual counts (over all query terms)
        total_virtual_counts = [0] * len(doc)
        # log of interpolated probabilities according to virtual counts per position
        pos_scores = [0] * len(doc)
        for i in range(len(doc)):
            for query_term_id in query:
                bg_prob = self.bg_prob(query_term_id)
                for j in range(len(doc)):
                    doc_term_id = doc[j]
                    cwj = self.term_count(query_term_id, doc_term_id)
                    kij = self.ker(i, j)
                    a = cwj * kij
                    term_virtual_counts[i][query_term_id] += a
                    total_virtual_counts[i] += a
                vir_cwi = term_virtual_counts[i][query_term_id]
                zi = total_virtual_counts[i]
                pos_scores[i] += math.log((vir_cwi + self.mu * bg_prob) / (zi + self.mu))
        return max(pos_scores)

    def ker(self, i, j) -> float:
        """Return a value if already computed. Compute, store and return if not.

        Returns:
            the result of the kernel function.
        """
        if np.isnan(self.kernel_matrix[i, j]):
            if self.ker_type == 'gaussian':
                self.kernel_matrix[i, j] = math.exp(- ((i - j) ** 2) / (2 * self.sigma ** 2))
            elif self.ker_type == 'triangle':
                if abs(i - j) <= self.sigma:
                    self.kernel_matrix[i, j] = 1 - abs(i - j) / 2
                else:
                    self.kernel_matrix[i, j] = 0.0
            elif self.ker_type == 'cosine':
                if abs(i - j) <= self.sigma:
                    self.kernel_matrix[i, j] = 1 / 2 * (1 + math.cos(abs(i - j) * math.pi / self.sigma))
                else:
                    self.kernel_matrix[i, j] = 0.0
            elif self.ker_type == 'circle':
                if abs(i - j) <= self.sigma:
                    self.kernel_matrix[i, j] = math.sqrt(1 - (abs(i - j) / self.sigma) ** 2)
                else:
                    self.kernel_matrix[i, j] = 0.0
            elif self.ker_type == 'passage':
                if abs(i - j) <= self.sigma:
                    self.kernel_matrix[i, j] = 1
                else:
                    self.kernel_matrix[i, j] = 0
            else:
                raise ValueError('Kernel type not supported: {}'.format(self.ker_type))

        return self.kernel_matrix[i, j]

    # noinspection PyMethodMayBeStatic
    def term_count(self, query_term_id: int, doc_term_id: int) -> float:
        return int(query_term_id == doc_term_id)

    def bg_prob(self, query_term_id: int):
        return self.col_freq[query_term_id] / self.col_size

    def run(self, model_name: str, doc_col=None) -> collections.defaultdict or None:
        """
        Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.
        Optionally, accepts a collection of documents to re-rank for each query.
        Args:
            model_name: the name of the model (a string)
            doc_col: pass a preprocessed query-wise collection fo documents to be evaluated.
        Returns:
            Data written to a run file.
        """

        run_out_path = '../retrievals/{}.run'.format(model_name)

        if os.path.exists(run_out_path):
            print('Run file already exists. Aborting.')
            return

        data = collections.defaultdict(list)
        scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

        print('Retrieving using', model_name)
        retrieval_start_time = time.time()

        if doc_col is None:
            for int_doc_id in range(self.index.document_base(), self.index.maximum_document()):
                # noinspection PyArgumentList
                ext_doc_id, doc = self.index.document(int_doc_id)

                for query_id, query in self.queries.items():
                    doc_is_in_inverted_index = False
                    for query_term_id in query:
                        doc_term_freq = self.inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is not None:
                            doc_is_in_inverted_index = True
                    if doc_is_in_inverted_index:
                        scores[query_id][ext_doc_id] = self.score(int_doc_id, query)

                    if scores[query_id][ext_doc_id] != 0:
                        data[query_id].append((scores[query_id][ext_doc_id], ext_doc_id))
        else:
            for query_id, int_doc_ids in doc_col.items():
                query = self.queries[query_id]
                for int_doc_id in int_doc_ids:
                    # noinspection PyArgumentList
                    ext_doc_id, doc = self.index.document(int_doc_id)
                    doc_is_in_inverted_index = False
                    for query_term_id in query:
                        doc_term_freq = self.inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is not None:
                            doc_is_in_inverted_index = True

                    if doc_is_in_inverted_index:
                        scores[query_id][ext_doc_id] = self.score(int_doc_id, query)

                    if scores[query_id][ext_doc_id] != 0:
                        data[query_id].append((scores[query_id][ext_doc_id], ext_doc_id))

        with open(run_out_path, 'w') as f_out:
            Helper.write_run(
                model_name=model_name,
                data=data,
                out_f=f_out,
                max_objects_per_query=1000)

        print('Retrieval run took {} seconds.'.format(time.time() - retrieval_start_time))
        return data


if __name__ == '__main__':

    with open('../pickles/prepro_doc_col_q50_top1000_tfidf.pkl', 'rb') as file:
        doc_col = pickle.load(file)
        max_len = 0
        for query_id, int_doc_ids in doc_col.items():
            for int_doc_id in int_doc_ids:
                if Helper.document_lengths[int_doc_id] > max_len:
                    max_len = Helper.document_lengths[int_doc_id]

        positionalModel = PositionalLanguageModel(Helper.index, Helper.inverted_index, Helper.tokenized_queries,
                                                  Helper.collection_frequencies, Helper.document_lengths, max_len,
                                                  50, 1000, 'gaussian')
        positionalModel.run('plm_top1000_sigma50_mu1000', doc_col)
