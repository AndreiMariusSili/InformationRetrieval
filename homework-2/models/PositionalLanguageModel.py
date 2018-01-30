import collections
import math
import pyndri
import numpy as np


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
        doc_len: dict of documents lengths.
        col_freq: dict of term frequencies for the entire collection.
        col_size: size of the document collection (number of query terms in the collection)
        doc_len: dict of documents lengths.
        max_len: maximum length for the collection.
        sigma: sigma tuning parameter for the kernel.
        mu: tuning parameter to control Dirichlet smoothing interpolation.
        ker_type: the type of kernel to use.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict),
                 col_freq: collections.defaultdict(int), doc_len: dict, max_len: int,
                 sigma: float, mu: float, ker_type: str):
        """Initialize positional language model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            col_freq: dict of term frequencies for the entire collection.
            doc_len: dict of documents lengths.
            max_len: maximum length for the collection.
            sigma: sigma tuning parameter for the kernel.
            mu: tuning parameter to control Dirichlet smoothing interpolation.
            ker_type: the type of kernel to use.
        """
        self.index = index
        self.inverted_index = inverted_index
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

    def term_count(self, query_term_id: int, doc_term_id: int) -> float:
        return int(query_term_id == doc_term_id)

    def bg_prob(self, query_term_id: int):
        return self.col_freq[query_term_id] / self.col_size
