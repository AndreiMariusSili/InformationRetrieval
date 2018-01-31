from components.Helper import *
from models.PositionalLanguageModel import PositionalLanguageModel
from models.GeneralizedLanguageModel import GeneralizedLanguageModel


def run_retrieval(model_name, score_fn, doc_col=None):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example)
    """
    run_out_path = '../retrievals/{}.run'.format(model_name)

    if os.path.exists(run_out_path):
        print('Run file already exists. Aborting.')
        return

    data = collections.defaultdict(list)
    scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    print('Retrieving using', model_name)
    retrieval_start_time = time.time()

    # The dictionary data should have the form: query_id --> [(document_score, external_doc_id)]
    if doc_col is None:
        for int_doc_id in range(index.document_base(), index.maximum_document()):
            ext_doc_id, doc = index.document(int_doc_id)

            for query_id, query in tokenized_queries.items():
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
    else:
        #
        # For now we use only 50 queries with the top 100 documents from TF-IDF.
        #
        for query_id, int_doc_ids in list(doc_col.items())[0:50]:
            query = tokenized_queries[query_id]
            for int_doc_id in int_doc_ids[0:100]:
                ext_doc_id, doc = index.document(int_doc_id)

                if model_name.startswith('plm'):
                    scores[query_id][ext_doc_id] = score_fn(int_doc_id, query)
                else:
                    for query_term_id in query:
                        doc_term_freq = inverted_index[query_term_id].get(int_doc_id)
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


if __name__ == '__main__':
    name = None
    fn = None

    # PLM
    with open('../pickles/prepro_doc_col_q150_top1000_tfidf.pkl', 'rb') as file:
        doc_col = pickle.load(file)
        max_len = 0
        for query_id, int_doc_ids in doc_col.items():
            for int_doc_id in int_doc_ids:
                if document_lengths[int_doc_id] > max_len:
                    max_len = document_lengths[int_doc_id]

        plm = PositionalLanguageModel(index=index, inverted_index=inverted_index, col_freq=collection_frequencies,
                                      doc_len=document_lengths, max_len=max_len, sigma=50, mu=1000, ker_type='gaussian')

        name = 'plm_50queries_top100docs_sigma_50_mu_1000'
        fn = plm.score
        run_retrieval(model_name=name, score_fn=fn, doc_col=doc_col)
