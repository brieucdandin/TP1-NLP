"""
Questions 1.3.1 et 1.3.2 : validation de votre modèle avec NLTK

Dans ce fichier, on va comparer le modèle obtenu dans `mle_ngram_model.py` avec le modèle MLE fourni par NLTK.

Pour préparer les données avant d'utiliser le modèle NLTK, on pourra utiliser
>>> ngrams, words = padded_everygram_pipeline(n, corpus)
>>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle NLTK, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type Vocabulary.

On peut ensuite entraîner le modèle avec la méthode model.fit(ngrams). Attention, la documentation prête à confusion :
la méthode attends une liste de liste de n-grammes (`list(list(tuple(str)))` et non pas `list(list(str))`).
"""
from nltk.lm.models import MLE
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline

from itertools import chain

import preprocess_corpus as pre
import mle_ngram_model as mnm
import nltk_models as mod


def train_MLE_model(corpus, n):
    """
    Entraîne un modèle de langue n-gramme MLE de NLTK sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param n: l'ordre du modèle
    :return: un modèle entraîné
    """
    # Creation of the vocabulary from the given corpus
    flat_corpus = []
    for document in corpus:
        for word in document:
            flat_corpus.append(word)
    vocab = Vocabulary(flat_corpus, unk_cutoff=2)
    # Extraction of the n-grams
    n_grams = mnm.extract_ngrams(corpus, n)
    # Creation and training of the model on the corpus
    model = MLE(n, vocab)
    model.fit(n_grams)
    print("Modèle d'ordre", n, "généré par nltk.lm")
    return model


def compare_models(your_model, nltk_model, corpus, n):
    """
    Pour chaque n-gramme du corpus, calcule la probabilité que lui attribuent `your_model`et `nltk_model`, et
    vérifie qu'elles sont égales. Si un n-gramme a une probabilité différente pour les deux modèles, cette fonction
    devra afficher le n-gramme en question suivi de ses deux probabilités.

    À la fin de la comparaison, affiche la proportion de n-grammes qui diffèrent.

    :param your_model: modèle NgramModel entraîné dans le fichier 'mle_ngram_model.py'
    :param nltk_model: modèle nltk.lm.MLE entraîné sur le même corpus dans la fonction 'train_MLE_model'
    :param corpus: list(list(str)), une liste de phrases tokenizées à tester
    :return: float, la proportion de n-grammes incorrects
    """
    total_ngrams_nmb = 0
    wrong_ngrams_nmb = 0
    # List of possible contexts for n=1,2,3
    context = [ (), (corpus[0][0]), (corpus[0][0], corpus[0][1]) ]
    # Buffer for the first word of a trigram's context
    word_buffer = ''
    # List of incorrect n-grams
    wrong_ngrams_list = []
    # With n >= 2, testing the first n-1 words isn't useful (both model will return a probability of 0).
    # But these iterations enable to fill the context for the ones which follow accordingly.
    for document in corpus:
        for word in document:
            # Update both counts
            total_ngrams_nmb += 1
            # To print all n-grams and their probabilities
            # print("your_model.score(", word, ",", context[n-1], ") =", your_model.score(word, context[n-1]), "\t\tnltk_model.score(", word, ",", context[n-1], ") =", nltk_model.score(word, context[n-1]))
            if your_model.score(word, context[n-1]) != nltk_model.score(word, context[n-1]):
                wrong_ngrams_nmb += 1
                wrong_ngrams_list.append([word, context[n-1]])
            # Update the context used during the next iteration
            context[1] = word
            context[2] = (word_buffer, word)
            # Save this iteration's explored word for the iteration after the next one
            word_buffer = word

    print("Comparaison à l'ordre", n, "finie : les modèles", ("coïncident." if wrong_ngrams_nmb/total_ngrams_nmb == 0 else "ne coïncident pas. Liste des n-grammes dont les probabilités diffèrent :"))
    if wrong_ngrams_nmb/total_ngrams_nmb != 0:
        print('\t\t'.join(map(str, wrong_ngrams_list)))
    return wrong_ngrams_nmb/total_ngrams_nmb


if __name__ == "__main__":
    """
    Ici, vous devrez valider votre implémentation de `NgramModel` en la comparant avec le modèle NLTK. Pour n=1, 2, 3,
    vous devrez entraîner un modèle nltk `MLE` et un modèle `NgramModel` sur `shakespeare_train`, et utiliser la fonction 
    `compare_models `pour vérifier qu'ils donnent les mêmes résultats. 
    Comme corpus de test, vous choisirez aléatoirement 50 phrases dans `shakespeare_train`.
    """
    with open("data/shakespeare_train.txt", "r") as f:
       raw_data = f.read()
    # Corpus tokenisé
    corpus = pre.preprocessed_text(raw_data)[1]

    # Entrainement MLE avec n = 1, 2, 3
    lm_1 = train_MLE_model(corpus, 1)
    lm_2 = train_MLE_model(corpus, 2)
    lm_3 = train_MLE_model(corpus, 3)

    homemade_mle_1 = mod.train_LM_model(corpus, MLE, 1)
    homemade_mle_2 = mod.train_LM_model(corpus, MLE, 2)
    homemade_mle_3 = mod.train_LM_model(corpus, MLE, 3)

    compare_models(homemade_mle_1, lm_1, corpus, 1)
    compare_models(homemade_mle_2, lm_2, corpus, 2)
    compare_models(homemade_mle_3, lm_3, corpus, 3)
