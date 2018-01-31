from models.LanguageModel import LanguageModel
from components import Helper
import pyndri
import collections
import math


class AbsoluteDiscounting(LanguageModel):
    """Multinomial language model with absolute discounting smoothing. Inherits from LanguageModel.

    Notes:
        We choose to calculate background probability such that all words contribute equally,
        for the convenience of computation. In principle, however, any other estimation of
        the background probability is possible depending on the desired focus.

    Attributes:
        delta: mu parameter to control interpolation.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), queries: dict,
                 col_freq: collections.defaultdict(int), doc_len: dict, doc_voc_size: dict, delta: float):
        """Initialize language model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            col_freq: dict of term frequencies for the entire collection.
            queries: dict of tokenized queries.
            doc_len: dict of documents lengths.
            doc_voc_size: dict of sizes of document vocabularies.
            delta: delta parameter to control interpolation.
        """
        super().__init__(index, inverted_index, queries, col_freq, doc_voc_size, doc_len)
        self.delta = delta

    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        """Compute the score for a document and a query term.

        Args:
            int_doc_id: the document id.
            query_term_id: the query term id (assuming you have split the query to tokens).
            doc_term_freq: the document term frequency of the query term.
        Return:
            Query log-likelihood with absolute discounting prior smoothing for a query
            term and a document.
        """
        doc_len = self.doc_len[int_doc_id]
        wtp = 1 / doc_len * self.compute_term_prob(int_doc_id, doc_term_freq)
        wtb = self.doc_voc_size[int_doc_id] / doc_len * self.compute_bg_prob(query_term_id)

        return math.log(wtp + wtb)

    def compute_term_prob(self, int_doc_id: int, doc_term_freq: int) -> float:
        """Overrides LanguageModel."""

        return max(doc_term_freq - self.delta, 0)

    def compute_bg_prob(self, query_term_id: int) -> float:
        """Inherited from LanguageModel."""
        return super().compute_bg_prob(query_term_id)

    def run(self, model_name: str, doc_col=None) -> collections.defaultdict or None:
        """Inherited from Language Model."""
        return super().run(model_name, doc_col)


if __name__ == '__main__':
    absoluteDiscounting = AbsoluteDiscounting(Helper.index, Helper.inverted_index, Helper.tokenized_queries,
                                              Helper.collection_frequencies, Helper.document_lengths,
                                              Helper.unique_terms_per_document, 0.1)
    absoluteDiscounting.run('ad_delta_0.1')
