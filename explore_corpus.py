"""
Questions 1.1.6 à 1.1.8 : calcul de différentes statistiques sur un corpus.

Sauf mention contraire, toutes les fonctions renvoient un nombre (int ou float).
Dans toutes les fonctions de ce fichier, le paramètre `corpus` désigne une liste de phrases tokenizées, par exemple :
>>> corpus = [
    ["Alice", "est", "là"],
    ["Bob", "est", "ici"]
]
"""
import preprocess_corpus as pre


def count_tokens(corpus):
    """
    Renvoie le nombre de mots dans le corpus
    """
    taille = 0
    for sentence in corpus:
        taille += len(sentence)
    return taille


def count_types(corpus):
    """
    Renvoie le nombre de types (mots distincts) dans le corpus
    """
    vocab = set()
    for sentence in corpus:
        vocab = vocab | set(sentence)
    return len(vocab)


def get_most_frequent(corpus, n):
    """
    Renvoie les n mots les plus fréquents dans le corpus, ainsi que leurs fréquences

    :return: list(tuple(str, float)), une liste de paires (mot, fréquence)
    """
    pass


def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """
    pass


def count_lemmas(corpus):
    """
    Renvoie le nombre de lemmes distincts
    """
    pass


def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    pass


def explore(corpus):
    """
    Affiche le résultat des différentes fonctions ci-dessus.

    Pour `get_most_frequent`, prenez n=15

    >>> explore(corpus)
    Nombre de tokens: 5678
    Nombre de types: 890
    ...
    Nombre de stems: 650

    """
    pass


if __name__ == "__main__":
    """
    Ici, appelez la fonction `explore` sur `shakespeare_train` et `shakespeare_test`. Quand on exécute le fichier, on 
    doit obtenir :

    >>> python explore_corpus
    -- shakespeare_train --
    Nombre de tokens: 5678
    Nombre de types: 890
    ...

    -- shakespeare_test --
    Nombre de tokens: 78009
    Nombre de types: 709
    ...
    """
    with open("data/shakespeare_train.txt", "r") as f:
        raw_data = f.read()
    corpus = pre.preprocessed_text(raw_data)[1]

    # nb_tokens = count_tokens(corpus)
    # print(nb_tokens)
    nb_types = count_types(corpus)
    print(nb_types)
