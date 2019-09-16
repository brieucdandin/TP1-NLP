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
import preprocess_corpus as pre
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary
import 


def train_MLE_model(corpus, n):
    """
    Entraîne un modèle de langue n-gramme MLE de NLTK sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param n: l'ordre du modèle
    :return: un modèle entraîné
    """
    ngrams, words = padded_everygram_pipeline(n, corpus)
    vocab = Vocabulary(words, unk_cutoff=2)
    model = nltk.lm.models.MLE(n, vocab)
    model.fit(ngrams)
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
    ngrams, words = padded_everygram_pipeline(n, corpus)
    ngrams_list = []
    for ngram in ngrams:
        ngrams_list.append(ngram)
    words_list = []
    for word in words:
        words_list.append(word)

    nb_ngram = len(ngrams_list)
    nb_ngram_diff = 0
    for ngram in ngrams_list:
        word = ngram[-1]
        if n == 1:
            context = ()
        else:
            context = ngram[:n-1]
        if nltk_model.score(word, context) != your_model.proba(word, context):
            print("Probabilités différentes pour le ngram : ", ngram)
            print("proba nltk : ", nltk_model.score(word, context), "     notre proba : ", your_model.proba(word, context), "\n")
            nb_ngram_diff += 1
    return(nb_ngram_diff/nb_ngram)


if __name__ == "__main__":
    """
    Ici, vous devrez valider votre implémentation de `NgramModel` en la comparant avec le modèle NLTK. Pour n=1, 2, 3,
    vous devrez entraîner un modèle nltk `MLE` et un modèle `NgramModel` sur `shakespeare_train`, et utiliser la fonction 
    `compare_models `pour vérifier qu'ils donnent les mêmes résultats. 
    Comme corpus de test, vous choisirez aléatoirement 50 phrases dans `shakespeare_train`.
    """

    with open("data/shakespeare_train.txt", "r") as f:
       raw_data = f.read()
    corpus = pre.preprocessed_text(raw_data)[1]

    lm = train_MLE_model(corpus, 2)
