import collections
import io
import time
import math
import logging
import sys
import os
import pyndri
import numpy as np
from abc import ABC, abstractmethod
import pickle


def main():
    def write_run(model_name, data, out_f,
                  max_objects_per_query=sys.maxsize,
                  skip_sorting=False):
        """
        Write a run to an output file.
        Parameters:
            - model_name: identifier of run.
            - data: dictionary mapping topic_id to object_assesments;
                object_assesments is an iterable (list or tuple) of
                (relevance, object_id) pairs.
                The object_assesments iterable is sorted by decreasing order.
            - out_f: output file stream.
            - max_objects_per_query: cut-off for number of objects per query.
        """
        for subject_id, object_assesments in data.items():
            if not object_assesments:
                logging.warning('Received empty ranking for %s; ignoring.',
                                subject_id)

                continue

            # Probe types, to make sure everything goes alright.
            # assert isinstance(object_assesments[0][0], float) or \
            #     isinstance(object_assesments[0][0], np.float32)
            assert isinstance(object_assesments[0][1], str) or isinstance(object_assesments[0][1], bytes)

            if not skip_sorting:
                object_assesments = sorted(object_assesments, reverse=True)

            if max_objects_per_query < sys.maxsize:
                object_assesments = object_assesments[:max_objects_per_query]

            if isinstance(subject_id, bytes):
                subject_id = subject_id.decode('utf8')

            for rank, (relevance, object_id) in enumerate(object_assesments):
                if isinstance(object_id, bytes):
                    object_id = object_id.decode('utf8')

                out_f.write(
                    '{subject} Q0 {object} {rank} {relevance} '
                    '{model_name}\n'.format(
                        subject=subject_id,
                        object=object_id,
                        rank=rank + 1,
                        relevance=relevance,
                        model_name=model_name))

    def parse_topics(file_or_files,
                     max_topics=sys.maxsize, delimiter=';'):
        assert max_topics >= 0 or max_topics is None

        topics = collections.OrderedDict()

        if not isinstance(file_or_files, list) and \
                not isinstance(file_or_files, tuple):
            if hasattr(file_or_files, '__iter__'):
                file_or_files = list(file_or_files)
            else:
                file_or_files = [file_or_files]

        for f in file_or_files:
            assert isinstance(f, io.IOBase)

            for line in f:
                assert (isinstance(line, str))

                line = line.strip()

                if not line:
                    continue

                topic_id, terms = line.split(delimiter, 1)

                if topic_id in topics and (topics[topic_id] != terms):
                    logging.error('Duplicate topic "%s" (%s vs. %s).',
                                  topic_id,
                                  topics[topic_id],
                                  terms)

                topics[topic_id] = terms

                if 0 < max_topics <= len(topics):
                    break

        return topics

    def run_preprocess(model_name, preprocess_fn):
        """
           Preprocesses the collection to the top 1000 according to some scoring function.

            :param model_name: the name of the model
            :param preprocess_fn: the preprocess function
        """

        data = collections.defaultdict(list)
        scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

        print('Preprocessing using', model_name)
        preprocess_start_time = time.time()

        # The dictionary data should have the form: query_id --> [(document_score, external_doc_id)]
        for int_doc_id in range(index.document_base(), index.maximum_document()):
            ext_doc_id, doc = index.document(int_doc_id)

            for query_id, query in list(tokenized_queries.items()):
                for query_term_id in query:
                    doc_term_freq = inverted_index[query_term_id].get(int_doc_id)
                    if doc_term_freq is None:
                        continue
                    scores[query_id][ext_doc_id] += preprocess_fn(int_doc_id, query_term_id, doc_term_freq)

                if scores[query_id][ext_doc_id] != 0:
                    data[query_id].append(int_doc_id)

        for query_id, query in list(tokenized_queries.items()):
            data[query_id] = sorted(data[query_id], reverse=True)[0:1000]
        print('Preprocessing took {} seconds.'.format(time.time() - preprocess_start_time))
        print(data[list(data.keys())[0]])
        return data

    def run_retrieval(model_name, score_fn):
        """
        Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

        :param model_name: the name of the model (a string)
        :param score_fn: the scoring function (a function - see below for an example)
        """
        run_out_path = '{}.run'.format(model_name)

        if os.path.exists(run_out_path):
            return

        data = collections.defaultdict(list)
        scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

        print('Retrieving using', model_name)
        retrieval_start_time = time.time()

        # The dictionary data should have the form: query_id --> [(document_score, external_doc_id)]
        for int_doc_id in range(index.document_base(), index.maximum_document()):
            ext_doc_id, doc = index.document(int_doc_id)

            for query_id, query in list(tokenized_queries.items()):
                if model_name.startswith('plm'):
                    doc_is_in_inverted_index = False
                    for query_term_id in query:
                        doc_term_freq = inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is not None:
                            doc_is_in_inverted_index = True
                    if doc_is_in_inverted_index:
                        scores[query_id][ext_doc_id] = score_fn(int_doc_id, query)
                else:
                    for query_term_id in query:
                        doc_term_freq = inverted_index[query_term_id].get(int_doc_id)
                        if doc_term_freq is None:
                            continue
                        scores[query_id][ext_doc_id] += score_fn(int_doc_id, query_term_id, doc_term_freq)

                if scores[query_id][ext_doc_id] != 0:
                    data[query_id].append((scores[query_id][ext_doc_id], ext_doc_id))

        with open(run_out_path, 'w') as f_out:
            write_run(
                model_name=model_name,
                data=data,
                out_f=f_out,
                max_objects_per_query=1000)

        print('Retrieval run took {} seconds.'.format(time.time() - retrieval_start_time))

    with open('./ap_88_89/topics_title', 'r') as f_topics:
        queries = parse_topics([f_topics])

    index = pyndri.Index('index/')

    num_documents = index.maximum_document() - index.document_base()

    dictionary = pyndri.extract_dictionary(index)

    tokenized_queries = {
        query_id: [dictionary.translate_token(token)
                   for token in index.tokenize(query_string)
                   if dictionary.has_token(token)]
        for query_id, query_string in queries.items()}

    query_term_ids = set(
        query_term_id
        for query_term_ids in tokenized_queries.values()
        for query_term_id in query_term_ids)

    # inverted index creation.

    start_time = time.time()

    document_lengths = {}
    unique_terms_per_document = {}

    try:
        print('Trying to load statistics from file...', end='')
        with open('pickles/inverted_index.pkl', 'rb') as file:
            inverted_index = pickle.load(file)
        with open('pickles/collection_frequencies.pkl', 'rb') as file:
            collection_frequencies = pickle.load(file)
        with open('pickles/document_lengths.pkl', 'rb') as file:
            document_lengths = pickle.load(file)
        with open('pickles/unique_terms_per_document.pkl', 'rb') as file:
            unique_terms_per_document = pickle.load(file)
        with open('pickles/document_term_frequency.pkl', 'rb') as file:
            document_term_frequency = pickle.load(file)
        with open('pickles/avg_doc_length.pkl', 'rb') as file:
            avg_doc_length = pickle.load(file)
        print('Success!')
    except FileNotFoundError:
        print('Error!')
        print('Gathering statistics about', len(query_term_ids), 'terms.')

        inverted_index = collections.defaultdict(dict)
        collection_frequencies = collections.defaultdict(int)

        total_terms = 0

        for int_doc_id in range(index.document_base(), index.maximum_document()):
            ext_doc_id, doc_token_ids = index.document(int_doc_id)

            document_bow = collections.Counter(
                token_id for token_id in doc_token_ids
                if token_id > 0)
            document_length = sum(document_bow.values())

            document_lengths[int_doc_id] = document_length
            total_terms += document_length

            unique_terms_per_document[int_doc_id] = len(document_bow)

            for query_term_id in query_term_ids:
                assert query_term_id is not None

                document_term_frequency = document_bow.get(query_term_id, 0)

                if document_term_frequency == 0:
                    continue

                collection_frequencies[query_term_id] += document_term_frequency
                inverted_index[query_term_id][int_doc_id] = document_term_frequency

        avg_doc_length = total_terms / num_documents

        print('Inverted index creation took', time.time() - start_time, 'seconds.')

        print('Saving statistics for future use...', end='')
        with open('pickles/inverted_index.pkl', 'wb') as file:
            pickle.dump(inverted_index, file)
        with open('pickles/collection_frequencies.pkl', 'wb') as file:
            pickle.dump(collection_frequencies, file)
        with open('pickles/document_lengths.pkl', 'wb') as file:
            pickle.dump(document_lengths, file)
        with open('pickles/unique_terms_per_document.pkl', 'wb') as file:
            pickle.dump(unique_terms_per_document, file)
        with open('pickles/document_term_frequency.pkl', 'wb') as file:
            pickle.dump(document_term_frequency, file)
        with open('pickles/avg_doc_length.pkl', 'wb') as file:
            pickle.dump(avg_doc_length, file)
        print('Success!')

    tfidf = TFIDF(index, inverted_index, document_lengths, tf_transform='log')
    preprocessed = run_preprocess('tfidf', tfidf.score)
    with open('pickles/preprocessed_tfidf_collection.pkl', 'wb') as file:
        pickle.dump(preprocessed, file)

    # run_retrieval('plm', plm.score)


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


class TFIDF(VectorSpaceModel):
    """Scoring class for the tf-idf method.

    Notes:
        TODO: implement more sublinear transformations and compare results.

    Attributes:
        tf_transform: string denoting possible sublinear tf transformations. accepted values are: log
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict), doc_len: dict,
                 tf_transform: str):
        """Initialize tf-idf scoring function.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            doc_len: dict of document lengths.
            tf_transform: string denoting possible sublinear tf transformations. accepted values are: `log`
        """
        super().__init__(index, inverted_index, doc_len)
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
        col_freq: dict of term frequencies for the entire collection.
        doc_len: dict of documents lengths.
        doc_voc_size: dict of sizes of document vocabularies.
        col_size: the size of the entire collection of terms.
        sigma: sigma tuning parameter for the kernel.
        mu: tuning parameter to control Dirichlet smoothing interpolation.
        ker_type: the type of kernel to use.
    """

    def __init__(self, index: pyndri.Index, inverted_index: collections.defaultdict(dict),
                 col_freq: collections.defaultdict(int), doc_len: dict, doc_voc_size: dict,
                 sigma: float, mu: float, ker_type: str):
        """Initialize positional language model.

        Args:
            index: pyndry index for the entire collection.
            inverted_index: dict of term frequencies per document.
            col_freq: dict of term frequencies for the entire collection.
            doc_len: dict of documents lengths.
            doc_voc_size: dict of sizes of document vocabularies.
            sigma: sigma tuning parameter for the kernel.
            mu: tuning parameter to control Dirichlet smoothing interpolation.
            ker_type: the type of kernel to use.
        """
        self.index = index
        self.inverted_index = inverted_index
        self.col_freq = col_freq
        self.doc_len = doc_len
        self.doc_voc_size = doc_voc_size
        self.col_size = sum(col_freq)
        self.sigma = sigma
        self.mu = mu
        self.ker_type = ker_type
        self.kernel = self.compute_kernel(doc_len, ker_type)

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
                    kij = self.kernel[i, j]
                    a = cwj * kij
                    term_virtual_counts[i][query_term_id] += a
                    total_virtual_counts[i] += a
                vir_cwi = term_virtual_counts[i][query_term_id]
                zi = total_virtual_counts[i]
                pos_scores[i] += math.log((vir_cwi + self.mu * bg_prob) / (zi + self.mu))
        return max(pos_scores)

    def compute_kernel(self, doc_len: dict, t: str) -> np.ndarray:
        """Compute the kernel between two positions in the document.

        Args:
            doc_len: dict of document lengths
            t: the kernel type.

        Returns:
            the result of the kernel function.
        """
        print('Building kernel values...')

        max_len = max(doc_len)
        print('document max length:', max_len)
        kernel = np.empty(shape=(max_len, max_len), dtype=np.float)
        for i in range(max_len):
            if i == int(max_len/2):
                print("Halfway through... ", i)
            for j in range(max_len):
                if t == 'gaussian':
                    kernel[i, j] = math.exp(- ((i - j) ** 2) / (2 * self.sigma ** 2))
                elif t == 'triangle':
                    if abs(i - j) <= self.sigma:
                        kernel[i, j] = 1 - abs(i - j) / 2
                    else:
                        kernel[i, j] = 0.0
                elif t == 'cosine':
                    if abs(i - j) <= self.sigma:
                        kernel[i, j] = 1 / 2 * (1 + math.cos(abs(i - j) * math.pi / self.sigma))
                    else:
                        kernel[i, j] = 0.0
                elif t == 'circle':
                    if abs(i - j) <= self.sigma:
                        kernel[i, j] = math.sqrt(1 - (abs(i - j) / self.sigma) ** 2)
                    else:
                        kernel[i, j] = 0.0
                else:
                    raise ValueError('Kernel type not supported: {}'.format(t))

        print(kernel[0, :])
        return kernel

    def term_count(self, query_term_id: int, doc_term_id: int) -> float:
        return int(query_term_id == doc_term_id)

    def bg_prob(self, query_term_id: int):
        return self.col_freq[query_term_id] / self.col_size


if __name__ == '__main__':
    main()