#from transformers import AutoTokenizer
#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
#print(model("morgen"))

"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, util
import torch
import time
from random_word import RandomWords
r = RandomWords()

embedder = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

# Corpus with example sentences
corpus = ['Das feste, dunkle Äußere des Brotes heißt Kruste oder Rinde. Sie enthält Röstaromen, die durch die Maillard-Reaktion beim Backen entstehen. Durch Einschneiden der Brotoberfläche vor dem Backprozess oder durch das zufällig Aufreißen beim Gehen ergeben sich in der Kruste Ausbünde, die die Oberfläche des gebackenen Brotes vergrößern. Die Dicke der Kruste ergibt sich aus der Backdauer, und ihre Farbe ergibt sich aus der Backtemperatur.',
          'Spätestens ab der mittleren Altsteinzeit wurden wilder Hafer und Gerste zu Mehl vermahlen und wahrscheinlich gewässert und gekocht oder gebacken, um das Mehl genießbar zu machen. Am Fundplatz Shanidar, einer von Neanderthalern bewohnten Höhle im Nordirak, wurden über 40.000 Jahre alte Spuren von Wildgerste gefunden, die offenbar erhitzt worden waren.[9]',
          'Die Brezel entstand im 7. Jahrhundert als ein heiliges Symbol, das zum Gebet verschränkte Arme des Betenden darstellte.[19]',
          'Das Wort häkeln ist seit dem Ende des 17. Jahrhunderts bezeugt und hatte ursprünglich die allgemeinere Bedeutung „mit Haken fassen“; heute bezieht es sich nur noch auf die Arbeit mit der Häkelnadel.[1] Vermutlich handelt es sich um eine Ableitung von Haken mit l-Endung des Verbs (sogenannte Iterativbildung) oder um eine Ableitung von der Verkleinerungsform hækel, dem mittelhochdeutschen Wort für „Häkchen“.[2]',
          'Für eine feste Masche wird in eine vorhandene Masche eingestochen und eine Schlinge herausgezogen. Auf der Häkelnadel befinden sich jetzt zwei Schlingen. Nun wird das zum Knäuel führende Garn mit der Häkelnadel durch beide Schlingen gezogen. Durch dieses Abketten entsteht die feste Masche.',
          'Zunahme zur Verbreiterung eines Häkelteils erfolgt durch mehrfaches Einstechen an derselben Stelle, so dass mehrere Maschen auf eine Masche der Vorreihe gesetzt werden.',
          'Häkeln ist eine viel jüngere Technik als Stricken. Es sind keine gehäkelten Stücke bekannt, die nachweislich vor dem Jahr 1800 zu datieren sind, während Stricken, soweit bekannt, ab dem 13. Jahrhundert nördlich der Alpen praktiziert wurde.',
          'Traktoren werden neben der Landwirtschaft in der Forstwirtschaft, bei Kommunalbetrieben, im Gartenbau, im Einsatzwesen (Feuerwehr, THW), auf Flughäfen und im Bauwesen (Straßenbau, Erdbewegung, Garten- und Landschaftsbau) eingesetzt.',
          'Einen Entwicklungssprung weg von den schweren und kapitalintensiven Dampftraktoren gelang erstmals Ford Fordson in den USA mit dem 1917 vorgestellten Model F mit Vierzylinder-Ottomotor, bei dem erstmals die auch heute noch im Schlepperbau weit verbreitete rahmenlose Blockbauweise zur Anwendung kam.',
          'Wegweisend in der Traktorenentwicklung waren auch die Erfindungen der hinteren Dreipunktaufhängung mit Hydraulik (Dreipunkthydraulik) durch Harry Ferguson und der Zapfwelle, die sich ab ca. 1960 allgemein durchsetzten. Somit wurde aus der landwirtschaftlichen Zugmaschine ein sehr vielseitig nutzbarer Geräteträger.',
        ]
i = 0
while i < 1000000:
    corpus.append(str(i))
    i = i+1
print("done")
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
print("done embedding")
# Query sentences:
queries = ['Seit wann wird Brot gebacken?']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(3, len(corpus))

start = time.time()
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """

end = time.time()
print("scoring took {:.3f}s total".format(end - start))