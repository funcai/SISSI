import pandas as pd
#from model_factory import ModelFactory
from sentence_transformers import SentenceTransformer, util
from torch import topk as torch_topk
import sys
import numpy as np

def main(n_queries):

    print("Starting test for {} queries...".format(n_queries))

    #model_name = 'paraphrase-distilroberta-base-v1'
    model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    embedder = SentenceTransformer(model_name)
    # get query dev
    # db_queries = pd.read_csv('~/datasets/ms_marco/queries/queries.dev.tsv', sep='\t', names=['id', 'query'])

    # get passages
    #db_passages = pd.read_csv('~/datasets/ms_marco/collection.tsv', sep='\t', names=['id', 'text'])

    # get top 1000 dev
    print("Loading dataset\n")
    db_top1000 = pd.read_csv('~/datasets/ms_marco/top1000.dev', sep='\t', names=['qid', 'pid', 'query', 'passage'])
    db_qrels = pd.read_csv('~/datasets/ms_marco/qrels.dev.tsv', sep='\t', names=['qid', 'ignore1', 'pid', 'ignore2'])


    # get all query embeddings
    #db_passages = db_top1000['pid', 'passage'][]

    # get all passage embeddings

    #print("Embeddings done")

    overall_scores = {}

    for query_id in db_top1000.qid.unique()[:n_queries]:
    #query_id = db_top1000.qid.unique()[0]
        print("Starting query {}".format(query_id))
        query_data = db_top1000[db_top1000.qid == query_id]

        my_query = query_data['query'].iloc[0]
        query_embedding = embedder.encode(my_query, convert_to_tensor=True)
        #print("Done, got query embeddings with len {} for query {}".format(len(query_embeddings), my_query))

        passage_embeddings = embedder.encode(query_data.passage.values, convert_to_tensor=True)

        # save passage and check if embedding already exist


        #print("Done, got passage embeddings with len {}".format(len(passage_embeddings)))

        # cosin similarity and top results
        cos_scores = util.pytorch_cos_sim(query_embedding, passage_embeddings)[0]
        top_k = min(1000, len(query_data))

        top_results = torch_topk(cos_scores, k=top_k)

        # check with qrels to calculate score
        query_revelant_passage = db_qrels[db_qrels.qid == query_id]

        query_score = 0
        print("Query is: {}\n".format(my_query))
        for position, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            passage_id = query_data['pid'].iloc[idx.numpy()]
            passage_text = query_data['passage'].iloc[idx.numpy()]
            if position < 10 or position > 990:
                print("{} passage is: {}\n".format(position, passage_text))
            query_relevant_pids = query_revelant_passage.pid.values
            # check if relevant pids is > 0, otherwise skip
            if passage_id in query_relevant_pids:
                query_score += (1/(position+1))
        for rpids in query_relevant_pids:
            print("relevant passages would be {}".format(query_data['passage'][query_data.pid == rpids]))

        print('Query score is: {}\n'.format(query_score))

        overall_scores[str(query_id)] = query_score

    mrr = np.array(list(overall_scores.values())).mean()
    print('Test done, MRR is {}'.format(mrr))

    overall_scores['MRR'] = mrr

    results_df = pd.DataFrame(overall_scores, index=[0])

    csv_filename = 'results/' + str(n_queries) + '_model.csv'
    results_df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    n_queries = sys.argv[1] if len(sys.argv)>1 else 10
    main(int(n_queries))