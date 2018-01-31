from models.VectorSpaceModel import VectorSpaceModel
from components import Helper
import collections
import pyndri


class BM25(VectorSpaceModel):
    """Scoring class for the BM25 method.

    Notes:
        The method neglects the term that normalizes query term frequencies because the
        queries presented here are typically short. Also, average document length is
        computed relative to the entire collection, as opposed to the average length
        of the documents that contain one or more query terms. this makes the score generalisable
        to all the collection.
    Attributes:
        k: tuning parameter that calibrates the document term frequency scaling.
        b: tuning parameter which calibrates the document length scaling.
        avg_len: average document length for the entire collection.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), queries: dict,
                 k: float, b: float, doc_len: dict, avg_len: float):
        """Initialize BM25 scoring method.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            queries: dict of tokenized queries.
            k: tuning parameter that calibrates the document term frequency scaling.
            b: tuning parameter which calibrates the document length scaling.
            doc_len: dict of document lengths.
            avg_len: average document length for the entire collection.

        """
        super().__init__(index, inverted_index, queries, doc_len)
        self.k = k
        self.b = b
        self.avg_len = avg_len

    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        """Compute the score for a document and a query term.

        Args:
            int_doc_id: the document id.
            query_term_id: the query term id (assuming you have split the query to tokens).
            doc_term_freq: the document term frequency of the query term.
        Return:
            bm25 score for a query term and a document.
        """
        wtf = self.wtf(int_doc_id, doc_term_freq)
        idf = self.compute_idf(query_term_id)

        return wtf * idf

    def wtf(self, int_doc_id: int, doc_term_freq: int) -> float:
        """Compute the term frequency term in the score.

        Args:
            int_doc_id: the document id.
            doc_term_freq: the document term frequency of the query term.

        Return:
            Term frequency weight.
        """
        return self.num(doc_term_freq) / self.denom(int_doc_id, doc_term_freq)

    def num(self, doc_term_freq: int) -> float:
        """Numerator of the first term.

        Args:
            doc_term_freq: the document term frequency of the query term.

        Return:
            Term frequency scaled by the `k` parameter.
        """
        return (self.k + 1) * doc_term_freq

    def denom(self, int_doc_id: int, doc_term_freq: int) -> float:
        """Denominator of the first term.

        Args:
            int_doc_id: the document id.
            doc_term_freq: the document term frequency of the query term.

        Return:
            term frequency normalized by document length according to parameters `k` and `b`.
        """
        return self.k * ((1 - self.b) + self.b * (self.doc_len[int_doc_id] / self.avg_len)) + doc_term_freq

    def compute_df(self, query_term_id: int) -> int:
        """Inherited from VectorSpaceModel."""
        return super().compute_df(query_term_id)

    def compute_idf(self, query_term_id: int) -> float:
        """Inherited from VectorSpaceModel."""
        return super().compute_idf(query_term_id)

    def run(self, model_name: str, doc_col=None) -> dict or None:
        """Inherited from VectorSpaceModel."""
        return super().run(model_name, doc_col)


if __name__ == '__main__':
    bm25 = BM25(Helper.index, Helper.inverted_index, Helper.tokenized_queries, 1.2, 0.75,
                Helper.document_lengths, Helper.avg_doc_length)
    bm25.run('BM25')
