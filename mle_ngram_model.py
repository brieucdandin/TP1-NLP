"""
Question 1.2.1 à 1.2.7 : implémentation d'un modèle n-gramme MLE.

Les premières fonctions permettent de compter les n-grammes d'un corpus puis de calculer les estimées MLE; vous devrez
les compléter dans les questions 1.2.1 à 1.2.4. Elles sont ensuite encapsulées dans une classe `NgramModel` (déjà écrite),
vous pourrez alors entraîner un modèle n-gramme sur un corpus avec la commande :
>>> lm = NgramModel(corpus, n)

Sauf mention contraire, le paramètre `corpus` désigne une liste de phrases tokenizées
"""
from itertools import chain
import preprocess_corpus as pre
import nltk
from nltk.lm.preprocessing import pad_both_ends
from collections import defaultdict
import numpy as np
np.set_printoptions(precision=10)


def extract_ngrams_from_sentence(sentence, n):
    """
    Renvoie la liste des n-grammes présents dans la phrase `sentence`.

    >>> extract_ngrams_from_sentence(["Alice", "est", "là"], 2)
    [("<s>", "Alice"), ("Alice", "est"), ("est", "là"), ("là", "</s>")]

    Attention à la gestion du padding (début et fin de phrase).

    :param sentence: list(str), une phrase tokenizée
    :param n: int, l'ordre des n-grammes
    :return: list(tuple(str)), la liste des n-grammes présents dans `sentence`
    """
    padded_sentence = list(pad_both_ends(sentence, 2))
    ngrams = nltk.ngrams(padded_sentence, n)
    ngrams_list = []
    for ngram in ngrams:
        ngrams_list.append(ngram)
    return(ngrams_list)


def extract_ngrams(corpus, n):
    """
    Renvoie la liste des n-grammes présents dans chaque phrase de `corpus`.

    >>> extract_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2)
    [
        [("<s>", "Alice"), ("Alice", "est"), ("est", "là"), ("là", "</s>")],
        [("<s>", "Bob"), ("Bob", "est"), ("est", "ici"), ("ici", "</s>")]
    ]

    :param corpus: list(list(str)), un corpus à traiter
    :param n: int, l'ordre des n-grammes
    :return: list(list(tuple(str))), la liste contenant les listes de n-grammes de chaque phrase
    """
    res = []
    for sentence in corpus:
        res.append(extract_ngrams_from_sentence(sentence, n))
    return res


def count_ngrams(corpus, n):
    """
    Compte les n-grammes présents dans le corpus.

    Attention, le résultat de la fonction doit gérer correctement les n-grammes inconnus. Pour cela, la classe
    `collections.defaultdict` pourra être utile.

    >>> counts = count_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2)
    >>> counts[("est",)]["là"] # Bigramme connu
    1
    >>> counts[("est",)]["Alice"] # Bigramme inconnu
    0

    :param corpus: list(list(str)), un corpus à traiter
    :param n: int, l'ordre de n-grammes
    :return: mapping(tuple(str)->mapping(str->int)), l'objet contenant les comptes de chaque n-gramme
    """
    all_ngrams = extract_ngrams(corpus, n)
    res = defaultdict(lambda: 0)
    if n == 1:
        res[()] = defaultdict(lambda: 0)
        for sentence in all_ngrams:
            for unigram in sentence:
                if unigram[0] in res[()]:
                    res[()][unigram[0]] += 1
                else:
                    res[()][unigram[0]] = 1
    else:
        for sentence in all_ngrams:
            for ngram in sentence:
                context = ngram[:n-1]
                word = ngram[n-1]
                if context in res:
                    if word in context:
                        res[context][word] += 1
                    else:
                        res[context][word] = 1
                else:
                    res[context] = defaultdict(lambda: 0)
                    res[context][word] = 1
    return res



def compute_MLE(counts):
    """
    À partir de l'objet `counts` produit par la fonction `count_ngrams`, transforme les comptes en probabilités.

    >>> mle_counts = compute_MLE(counts)
    >>> mle_counts[("est",)]["là"] # 1/2
    0.5
    >>> mle_counts[("est",)]["Alice"] # 0/2
    0

    :param counts: mapping(tuple(str)->mapping(str->int))
    :return: mapping(tuple(str)->mapping(str->float))
    """
    res = counts.copy()
    if () in res:
        del res[()]["<s>"]
    for context in res.keys():
        tot = sum(res[context].values())
        for word in res[context].keys():
            res[context][word] /= tot
    return res



class NgramModel(object):
    def __init__(self, corpus, n):
        """
        Initialise un modèle n-gramme MLE à partir d'un corpus.

        :param corpus: list(list(str)), un corpus tokenizé
        :param n: int, l'ordre du modèle
        """
        counts = count_ngrams(corpus, n)
        self.n = n
        self.vocab = list(set(chain(["<s>", "</s>"], *corpus)))
        self.counts = compute_MLE(counts)

    def proba(self, word, context):
        """
        Renvoie P(word | context) selon le modèle.

        :param word: str
        :param context: tuple(str)
        :return: float
        """
        if context not in self.counts or word not in self.counts[context]:
            return 0.0
        return self.counts[context][word]

    def predict_next(self, context):
        """
        Renvoie un mot w tiré au hasard selon la distribution de probabilité P( w | context)

        :param context: tuple(str), un (n-1)-gramme de contexte
        :return: str
        """
        # Comment on gère <s> et </s>
        prob = []
        for word in self.vocab:
            prob.append(self.proba(word, context))
        if sum(prob) == 0:
            return "Le contexte n'existe pas."
        next = np.random.choice(self.vocab, p=prob)
        return next


if __name__ == "__main__":
    """
    Pour n=1, 2, 3:
    - entraînez un modèle n-gramme sur `shakespeare_train`
    - pour chaque contexte de `contexts[n]`, prédisez le mot suivant en utilisant la méthode `predict_next`

    Un exemple de sortie possible :
    >>> python mle_ngram_model.py
    n=1
    () --> the

    n=2
    ('King',) --> Philip
    ('I',) --> hope
    ('<s>',) --> Hence

    n=3
    ('<s>', '<s>') --> Come
    ('<s>', 'I') --> would
    ('Something', 'is') --> rotten
    ...
    """
    # Liste de contextes à tester pour n=1, 2, 3
    contexts = {
        1: [()],
        2: [("King",), ("I",), ("<s>",)],
        3: [("<s>", "<s>"), ("<s>", "I"), ("Something", "is"), ("To", "be"), ("O", "Romeo")]
        # 3: [("<s>", "I"), ("Something", "is"), ("To", "be"), ("O", "Romeo")]
    }

    with open("data/shakespeare_train.txt", "r") as f:
       raw_data = f.read()
    corpus = pre.preprocessed_text(raw_data)[1]

    lm_1 = NgramModel(corpus, 1)
    lm_2 = NgramModel(corpus, 2)
    lm_3 = NgramModel(corpus, 3)

    lm = [lm_1, lm_2, lm_3]

    for i in range(3):
        print("########### Test pour n = ", i, " ###########")
        for context in contexts[i+1]:
            print("Le mot qui suit le contexte ", context, " est :")
            print(lm[i].predict_next(context))
