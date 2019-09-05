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
    frequency_table = {}
    for sentence in corpus:
        for word in sentence:
            if word not in frequency_table:
                frequency_table[word] = 1
            else:
                frequency_table[word] += 1
    frequency_table = sorted(frequency_table.items(), key=lambda t: t[1], reverse = True)
    return list(frequency_table[:n])


def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """
    return count_tokens(corpus)/count_types(corpus)


def count_lemmas(corpus):
    """
    Renvoie le nombre de lemmes distincts
    """
    lemmatized = pre.lemmatize(corpus)
    distinct_lemmas = set()
    for sentence in lemmatized:
        distinct_lemmas = distinct_lemmas | set(sentence)
    return len(distinct_lemmas)


def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    stems = pre.stem(corpus)
    distinct_stems = set()
    for sentence in stems:
        distinct_stems = distinct_stems | set(sentence)
    return len(distinct_stems)


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
    print("########## Question a ##########")
    print("Le nombre total de tokens est : ", count_tokens(corpus), "\n")

    print("########## Question b ##########")
    print("Le nombre total de mots distincts est : ", count_types(corpus), "\n")

    n = 15
    print("########## Question c ##########")
    print("Les ", 15, " mots les plus fréquents sont : ", get_most_frequent(corpus, n), "\n")

    print("########## Question d ##########")
    print("Le ratio token/type est : ", get_token_type_ratio(corpus), "\n")

    print("########## Question e ##########")
    print("Le nombre total de lemmes distincts est : ", count_lemmas(corpus), "\n")

    print("########## Question f ##########")
    print("Le nombre total de racines disctinctes est : ", count_stems(corpus), "\n")


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

    explore(corpus)

    # nb_tokens = count_tokens(corpus)
    # print(nb_tokens)
    # nb_types = count_types(corpus)
    # print(nb_types)
    # freq = get_most_frequent(corpus, 15)
    # print(freq)
    # ratio = get_token_type_ratio(corpus)
    # print(ratio)
    # print(count_lemmas(corpus))
    # print(count_stems(corpus))