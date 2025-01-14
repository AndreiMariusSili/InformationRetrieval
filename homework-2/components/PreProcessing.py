from components import Helper
from models.TFIDF import TFIDF
import collections
import time
import pickle


def run_pre_process(model_name, pre_process_fn):
    """
       Pre-processes the collection to the top 1000 according to some scoring function. Can only handle simple
       scoring models, like TF-IDF and BM25, but no PLM.

        Args:
            model_name: the name of the model
            pre_process_fn: the pre-process function
    """

    data = collections.defaultdict(list)
    mapped_data = collections.defaultdict(list)
    scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    print('Pre-processing using', model_name)
    pre_process_start_time = time.time()

    # The dictionary data should have the form: query_id --> [(document_score, external_doc_id)]
    for int_doc_id in range(Helper.index.document_base(), Helper.index.maximum_document()):
        ext_doc_id, doc = Helper.index.document(int_doc_id)

        for query_id, query in list(Helper.tokenized_queries.items()):
            if 51 <= int(query_id) <= 100:
                for query_term_id in query:
                    doc_term_freq = Helper.inverted_index[query_term_id].get(int_doc_id)
                    if doc_term_freq is None:
                        continue
                    scores[query_id][ext_doc_id] += pre_process_fn(int_doc_id, query_term_id, doc_term_freq)

                if scores[query_id][ext_doc_id] != 0:
                    data[query_id].append((scores[query_id][ext_doc_id], ext_doc_id, int_doc_id))

    for query_id, query in list(Helper.tokenized_queries.items()):
        if 51 <= int(query_id) <= 100:
            data[query_id] = sorted(data[query_id], reverse=True)[0:1000]
            mapped_data[query_id] = list(map(lambda item: item[2], data[query_id]))

    print('Pre-processing took {} seconds.'.format(time.time() - pre_process_start_time))
    return dict(mapped_data)


if __name__ == '__main__':
    tfidf = TFIDF(Helper.index, Helper.inverted_index, Helper.tokenized_queries,
                  Helper.document_lengths, tf_transform='log')
    preprocessed = run_pre_process('tfidf', tfidf.score)
    with open('../pickles/prepro_doc_col_q50_top1000_tfidf.pkl', 'wb') as file:
        pickle.dump(preprocessed, file)