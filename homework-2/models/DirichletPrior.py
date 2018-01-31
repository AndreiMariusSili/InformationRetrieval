from models.LanguageModel import LanguageModel
from components import Helper
import collections
import pyndri
import math


class DirichletPrior(LanguageModel):
    """Multinomial language model with Dirichlet prior smoothing. Inherits from LanguageModel.

    Notes:
        We choose to calculate background probability such that all words contribute equally,
        for the convenience of computation. In principle, however, any other estimation of
        the background probability is possible depending on the desired focus.

    Attributes:
        mu: mu parameter to control interpolation.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), queries: dict,
                 col_freq: collections.defaultdict(int), doc_len: dict, doc_voc_size: dict, mu: int):
        """Initialize language model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            queries: dict of tokenized queries.
            col_freq: dict of term frequencies for the entire collection.
            doc_voc_size: dict of sizes of document vocabularies.
            mu: mu parameter to control interpolation.
        """
        super().__init__(index, inverted_index, queries, col_freq, doc_voc_size, doc_len)
        self.mu = mu

    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        """Compute the score for a document and a query term.

        Args:
            int_doc_id: the document id.
            query_term_id: the query term id (assuming you have split the query to tokens).
            doc_term_freq: the document term frequency of the query term.
        Return:
            Query log-likelihood with Dirichlet prior smoothing for a query term and a document.
        """
        doc_len = self.doc_len[int_doc_id]
        wtp = doc_len / (doc_len + self.mu) * self.compute_term_prob(int_doc_id, doc_term_freq)
        wbp = self.mu / (doc_len + self.mu) * self.compute_bg_prob(query_term_id)

        return math.log(wtp + wbp)

    def compute_term_prob(self, int_doc_id: int, doc_term_freq: int) -> float:
        """Inherited from LanguageModel."""
        return super().compute_term_prob(int_doc_id, doc_term_freq)

    def compute_bg_prob(self, query_term_id: int) -> float:
        """Inherited from LanguageModel."""
        return super().compute_bg_prob(query_term_id)

    def run(self, model_name: str, doc_col=None) -> collections.defaultdict or None:
        """Inherited from Language Model."""
        return super().run(model_name, doc_col)


if __name__ == '__main__':
    dirichletPrior = DirichletPrior(Helper.index, Helper.inverted_index, Helper.tokenized_queries,
                                    Helper.collection_frequencies, Helper.document_lengths,
                                    Helper.unique_terms_per_document, 1000)
    dirichletPrior.run('dp_mu_1000')
