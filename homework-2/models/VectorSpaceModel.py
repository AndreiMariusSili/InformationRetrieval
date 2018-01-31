from abc import ABC, abstractmethod
from components import Helper
import collections
import pyndri
import math
import time
import os


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
    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict),
                 queries: dict, doc_len: dict):
        """Initialize vector space model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            queries: dict of tokenized queries.
            doc_len: dict of document lengths.
        """
        self.index = index
        self.inverted_index = inverted_index
        self.queries = queries
        self.doc_len = doc_len
        self.col_size = index.maximum_document() - index.document_base()

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

    @abstractmethod
    def score(self, int_doc_id: int, query_term_id: int, doc_term_freq: int) -> float:
        pass

    @abstractmethod
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
                    for query_term_id in query:
                        doc_term_freq = self.inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is None:
                            continue
                        scores[query_id][ext_doc_id] += self.score(int_doc_id, query_term_id, doc_term_freq)

                    if scores[query_id][ext_doc_id] != 0:
                        data[query_id].append((scores[query_id][ext_doc_id], ext_doc_id))
        else:
            for query_id, int_doc_ids in doc_col.items():
                query = self.queries[query_id]
                for int_doc_id in int_doc_ids:
                    # noinspection PyArgumentList
                    ext_doc_id, doc = self.index.document(int_doc_id)
                    for query_term_id in query:
                        doc_term_freq = self.inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is None:
                            continue
                        scores[query_id][ext_doc_id] += self.score(int_doc_id, query_term_id, doc_term_freq)

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
