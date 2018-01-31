from models.VectorSpaceModel import VectorSpaceModel
from components import Helper
import collections
import pyndri
import math


class TFIDF(VectorSpaceModel):
    """Scoring class for the tf-idf method.

    Notes:
        The implementation uses a logarithmic sublinear transformation for the term frequencies. The score of a
        query-document pair by summing over all query terms.

    Attributes:
        tf_transform: string denoting possible sublinear tf transformations. accepted values are: [log]
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), queries: dict,
                 doc_len: dict, tf_transform: str):
        """Initialize tf-idf scoring function.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            doc_len: dict of document lengths.
            tf_transform: string denoting possible sublinear tf transformations. accepted values are: `log`
        """
        super().__init__(index, inverted_index, queries, doc_len)
        self.tf_transform = tf_transform

    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        """Scoring method for a document and a query term.

        Args:
            int_doc_id: the document id.
            query_term_id: the query term id (assuming you have split the query to tokens).
            doc_term_freq: the document term frequency of the query term.
        Return:
            tf-idf score for a query term and a document.
        """
        if self.tf_transform == 'log':
            wtf = self.log_tf(doc_term_freq)
        else:
            raise ValueError('Unsupported term frequency transformation specified: {}'.format(self.tf_transform))
        idf = self.compute_idf(query_term_id)

        return wtf * idf

    # noinspection PyMethodMayBeStatic
    def log_tf(self, doc_term_freq: int) -> float:
        """Apply sublinear transformation to document query term frequency.

        Args:
            doc_term_freq: the document term frequency for the query term.

        Return:
            Log sublinear transformation.
        """
        return 1 + math.log(doc_term_freq)

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
    tf_idf = TFIDF(Helper.index, Helper.inverted_index, Helper.tokenized_queries, Helper.document_lengths, 'log')
    tf_idf.run('TF-IDF')
