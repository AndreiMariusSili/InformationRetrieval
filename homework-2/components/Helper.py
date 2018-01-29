import collections
import io
import time
import logging
import sys
import os
import pyndri
import pickle


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


with open('../ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

index = pyndri.Index('../index/')

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

document_lengths = {}
unique_terms_per_document = {}

try:
    print('Trying to load statistics from file...', end='')
    with open('../pickles/inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    with open('../pickles/collection_frequencies.pkl', 'rb') as f:
        collection_frequencies = pickle.load(f)
    with open('../pickles/document_lengths.pkl', 'rb') as f:
        document_lengths = pickle.load(f)
    with open('../pickles/unique_terms_per_document.pkl', 'rb') as f:
        unique_terms_per_document = pickle.load(f)
    with open('../pickles/avg_doc_length.pkl', 'rb') as f:
        avg_doc_length = pickle.load(f)
    print('Success!')
except FileNotFoundError:
    print('Error!')
    print('Gathering statistics about', len(query_term_ids), 'terms.')
    start_time = time.time()

    inverted_index = collections.defaultdict(dict)
    collection_frequencies = collections.defaultdict(int)

    total_terms = 0

    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)

        document_bow = collections.Counter(token_id for token_id in doc_token_ids if token_id > 0)
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
    with open('../pickles/inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    with open('../pickles/collection_frequencies.pkl', 'wb') as f:
        pickle.dump(collection_frequencies, f)
    with open('../pickles/document_lengths.pkl', 'wb') as f:
        pickle.dump(document_lengths, f)
    with open('../pickles/unique_terms_per_document.pkl', 'wb') as f:
        pickle.dump(unique_terms_per_document, f)
    with open('../pickles/avg_doc_length.pkl', 'wb') as f:
        pickle.dump(avg_doc_length, f)
    print('Success!')

