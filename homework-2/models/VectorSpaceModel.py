from abc import ABC, abstractmethod
import pyndri
import collections
import math


class VectorSpaceModel(ABC):
    """Abstract class for vector space models.

    Notes:

    Attributes:
        index: pyndry index for the entire collection.
        inverted_index: dict of term frequencies per document.
        doc_len: dict of document lengths.
        col_size: number of documents in the collection.
    """

    @abstractmethod
    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), doc_len: dict):
        """Initialize vector space model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            doc_len: dict of document lengths.
        """
        self.index = index
        self.inverted_index = inverted_index
        self.doc_len = doc_len
        self.col_size = index.maximum_document() - index.document_base()

    @abstractmethod
    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        pass

    @abstractmethod
    def compute_df(self, query_term_id: int) -> int:
        """Calculate document frequency of query term.

        Args:
            query_term_id: pyndri query term id.
        Return:
            Length of the inverted index for the query.
        """
        return len(self.inverted_index[query_term_id])

    @abstractmethod
    def compute_idf(self, query_term_id: int) -> float:
        """Calculate inverted document frequency.

        Args:
            query_term_id: pyndri query term id.
        Return:
            Inverted document frequency.
        """
        return math.log(self.col_size) - math.log(self.compute_df(query_term_id))