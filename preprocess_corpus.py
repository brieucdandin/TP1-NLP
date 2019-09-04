"""
Questions 1.1.1 à 1.1.5 : prétraitement des données.
"""
import nltk

def segmentize(raw_text):
    """
    Segmente un texte en phrases.

    >>> raw_corpus = "Alice est là. Bob est ici"
    >>> segmentize(raw_corpus)
    ["Alice est là.", "Bob est ici"]

    :param raw_text: str
    :return: list(str)
    """
    return nltk.sent_tokenize(raw_text)


def tokenize(sentences):
    """
    Tokenize une liste de phrases en mots.

    >>> sentences = ["Alice est là", "Bob est ici"]
    >>> corpus = tokenize(sentences)
    >>> corpus_name
    [
        ["Alice", "est", "là"],
        ["Bob", "est", "ici"]
    ]

    :param sentences: list(str), une liste de phrases
    :return: list(list(str)), une liste de phrases tokenizées
    """
    res = []
    for sentence in sentences:
        res.append(nltk.word_tokenize(sentence))
    return res


def lemmatize(corpus):
    """
    Lemmatise les mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases lemmatisées
    """
    res = []
    lemmzer = nltk.WordNetLemmatizer()
    for sentence in corpus:
        res.append([lemmzer.lemmatize(token) for token in sentence])
    return res


def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    res = []
    stemmer = nltk.PorterStemmer()
    for sentence in corpus:
        res.append([stemmer.stem(lemme) for lemme in sentence])
    return res


def read_and_preprocess(filename):
    """
    Lit un fichier texte, puis lui applique une segmentation et une tokenization.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param filename: str, nom du fichier à lire
    :return: list(list(str))
    """
    with open(filename, "r") as f:
        raw_text = f.read()
    return tokenize(segmentize(raw_text))


def test_preprocessing(raw_text, sentence_id=0):
    """
    Applique à `raw_text` les fonctions segmentize, tokenize, lemmatize et stem, puis affiche le résultat de chacune
    de ces fonctions à la phrase d'indice `sentence_id`

    >>> trump = open("data/trump.txt", "r").read()
    >>> test_preprocessing(trump)
    Today we express our deepest gratitude to all those who have served in our armed forces.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,forces,.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,force,.
    today,we,express,our,deepest,gratitud,to,all,those,who,have,serv,in,our,arm,forc,.

    :param raw_text: str, un texte à traiter
    :param sentence_id: int, l'indice de la phrase à afficher
    :return: un tuple (sentences, tokens, lemmas, stems) qui contient le résultat des quatre fonctions appliquées à
    tout le corpus
    """
    sentence = [segmentize(raw_text)[sentence_id]]
    tokens = tokenize(sentence)
    lemmas = lemmatize(tokens)
    stems = stem(tokens)
    return (sentence, tokens, lemmas, stems)

def preprocessed_text(raw_data):
    sentences = segmentize(raw_data)
    tokens = tokenize(sentences)
    lemmas = lemmatize(tokens)
    stems = stem(tokens)
    return (sentences, tokens, lemmas, stems)


if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """
    with open("data/shakespeare_train.txt", "r") as f:
        raw_data = f.read()

    res_preprocessing = test_preprocessing(raw_data, 1)

    with open("output/shakespeare_train_phrases.txt", "w") as f:
        for phrase in res_preprocessing[0]:
            f.write(phrase + "\n")

    with open("output/shakespeare_train_mots.txt", "w") as f:
        for phrase in res_preprocessing[1]:
            for word in phrase:
                f.write(word + " ")
            f.write("\n")

    with open("output/shakespeare_train_lemmes.txt", "w") as f:
        for phrase in res_preprocessing[2]:
            for lemme in phrase:
                f.write(lemme + " ")
            f.write("\n")

    with open("output/shakespeare_train_stems.txt", "w") as f:
        for phrase in res_preprocessing[3]:
            for stem in phrase:
                f.write(stem + " ")
            f.write("\n")