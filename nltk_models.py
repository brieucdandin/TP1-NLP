"""
Questions 1.4.1 à 1.6.2 : modèles de langue NLTK

Dans ce fichier, on rassemble les fonctions concernant les modèles de langue NLTK :
- entraînement d'un modèle de langue sur un corpus d'entraînement, avec ou sans lissage
- évaluation d'un modèle sur un corpus de test
- génération de texte suivant un modèle de langue

Pour préparer les données avant d'utiliser un modèle, on pourra utiliser
>>> ngrams, words = padded_everygram_pipeline(n, corpus)
>>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type `Vocabulary`.

On peut ensuite entraîner le modèle avec la méthode `model.fit(ngrams)`
"""

import nltk.lm
import numpy
import matplotlib.pyplot as plt

from nltk.lm.models import MLE, Laplace, Lidstone
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline

import preprocess_corpus as pre
import mle_ngram_model as mnm


def train_LM_model(corpus, model, n, gamma=None, unk_cutoff=2):
    """
    Entraîne un modèle de langue n-gramme NLTK de la classe `model` sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param model: un des éléments de (MLE, Lidstone, Laplace)
    :param n: int, l'ordre du modèle
    :param gamma: float or None, le paramètre gamma (pour `model=Lidstone` uniquement). Si model=Lidstone, alors cet
    argument doit être renseigné
    :param unk_cutoff: le seuil au-dessous duquel un mot est considéré comme inconnu et remplacé par <UNK>
    :return: un modèle entraîné
    """
    #On veut condenser le corpus en une simple liste, pour pouvoir utiliser facilement Vocabulary
    flat_corpus = []
    for l in corpus:
        for w in l:
            flat_corpus.append(w)
    
    vocab = Vocabulary(flat_corpus, unk_cutoff)
    
    ngram_corpus = mnm.extract_ngrams(corpus,n)
    
    if (model == MLE):
        model_res = MLE(n)
        model_res.fit(ngram_corpus, vocab)
    
    if (model == Lidstone):
        model_res = Lidstone(gamma,n)
        model_res.fit(ngram_corpus, vocab)
    
    if (model == Laplace):
        model_res = Laplace(n)
        model_res.fit(ngram_corpus, vocab)
    
    return model_res


def evaluate(model, corpus):
    """
    Renvoie la perplexité du modèle sur une phrase de test.

    :param model: nltk.lm.api.LanguageModel, un modèle de langue
    :param corpus: list(list(str)), une corpus tokenizé
    :return: float
    """
    # On va renvoyer la perplexité moyenne par phrase du corpus
    ngram_corpus_test = mnm.extract_ngrams(corpus,model.order)
    res = 0
    for i in range (0,len(ngram_corpus_test)):
        res += model.perplexity(ngram_corpus_test[i])
    return res/(len(ngram_corpus_test))
    
    


def evaluate_gamma(gamma, train, test, n):
    """
    Entraîne un modèle Lidstone n-gramme de paramètre `gamma` sur un corpus `train`, puis retourne sa perplexité sur un
    corpus `test`.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param gamma: float, la valeur de gamma (comprise entre 0 et 1)
    :param train: list(list(str)), un corpus d'entraînement
    :param test: list(list(str)), un corpus de test
    :param n: l'ordre du modèle
    :return: float, la perplexité du modèle sur train
    """
    lm = train_LM_model(train, Lidstone, n, gamma=gamma)
    return evaluate(lm, test)


def generate(model, n_words, text_seed=None, random_seed=None):
    """
    Génère `n_words` mots à partir du modèle.

    Vous utiliserez la méthode `model.generate(num_words, text_seed, random_seed)` de NLTK qui permet de générer
    du texte suivant la distribution de probabilité du modèle de langue.

    Cette fonction doit renvoyer une chaîne de caractère détokenizée (dans le cas de Trump, vérifiez que les # et les @
    sont gérés); si le modèle génère un symbole de fin de phrase avant d'avoir fini, vous devez recommencer une nouvelle
    phrase, jusqu'à avoir produit `n_words`.

    :param model: un modèle de langue entraîné
    :param n_words: int, nombre de mots à générer
    :param text_seed: tuple(str), le contexte initial. Si aucun text_seed n'est précisé, vous devrez utiliser le début
    d'une phrase, c'est à dire respectivement (), ("<s>",) ou ("<s>", "<s>") pour n=1, 2, 3
    :param random_seed: int, la seed à passer à la méthode `model.generate` pour pouvoir reproduire les résultats. Pour
    ne pas fixer de seed, il suffit de laisser `random_seed=None`
    :return: str
    """   
    
    res = []

    # On veut connaitre le contexte courant.
    contexte_courant = []
    if (text_seed == None): 
        contexte_courant = tuple(["<s>"]*(model.order-1))
    else:
        contexte_courant = text_seed
    
    while (len(res) < n_words):
        word = model.generate( 1, contexte_courant, random_seed)
        while (word=="#" or word=="@"):
            word = word + model.generate( 1, tuple(list(contexte_courant) + list(word)), random_seed)
        # On ne veut pas conserver certains symboles inutiles puisqu'ils n'apportent pas d'information sémantique
        while (word == "__URL__" or word == "<UNK>"):
            word = model.generate( 1, contexte_courant, random_seed)
        # On veut tenir à jour le contexte
        contexte_courant = list(contexte_courant)
        contexte_courant.append(word)
        contexte_courant = tuple(contexte_courant)
        res.append(word)
        # On doit séparer le cas où l'on génère une nouvelle phrase
        if (word == "." or word == "!"):
           contexte_courant = tuple(["<s>"]*(model.order-1))

    # Conversion en une string
    text_res = " ".join(res)
    return text_res


if __name__ == "__main__":
    """
    Vous aurez ici trois tâches à accomplir ici :
    
    1)
    Dans un premier temps, vous devez entraîner des modèles de langue MLE et Laplace pour n=1, 2, 3 à l'aide de la 
    fonction `train_MLE_model` sur le corpus `shakespeare_train` (question 1.4.2). Puis vous devrez évaluer vos modèles 
    en mesurant leur perplexité sur le corpus `shakespeare_test` (question 1.5.2).

    2)
    Ensuite, on vous demande de tracer un graphe représentant le perplexité d'un modèle Lidstone en fonction du paramètre 
    gamma. Vous pourrez appeler la fonction `evaluate_gamma` (déjà écrite) sur `shakespeare_train` et `shakespeare_test` 
    en faisant varier gamma dans l'intervalle (10^-5, 1) (question 1.5.3). Vous utiliserez une échelle logarithmique en 
    abscisse et en ordonnée.
    
    Note : pour les valeurs de gamma à tester, vous pouvez utiliser la fonction `numpy.logspace(-5, 0, 10)` qui renvoie 
    une liste de 10 nombres, répartis logarithmiquement entre 10^-5 et 1.
    
    3)
    Enfin, pour chaque n=1, 2, 3, vous devrez générer 2 segments de 20 mots pour des modèles MLE entraînés sur Trump.
    Réglez `unk_cutoff=1` pour éviter que le modèle ne génère des tokens <UNK> (question 1.6.2).
    """
    with open("data/shakespeare_train.txt", "r") as f:
        raw_data_train = f.read()
    
    with open("data/shakespeare_test.txt", "r") as f:
        raw_data_test = f.read()
        
    with open("data/trump.txt", "r") as f:
        raw_data_trump = f.read()
        
    corpus_train = pre.preprocessed_text(raw_data_train)[1]
    corpus_test = pre.preprocessed_text(raw_data_test)[1]
    corpus_trump = pre.preprocessed_text(raw_data_trump)[1]
    
    # 1)
    
    # 3 modèles de MLE 
    
 
    MLE_model = []
    MLE_model.append(train_LM_model(corpus_train,MLE,1))
    MLE_model.append(train_LM_model(corpus_train,MLE,2))
    MLE_model.append(train_LM_model(corpus_train,MLE,3))

    # 3 modèles de Laplace
    LP_model = []
    LP_model.append(train_LM_model(corpus_train,Laplace,1))
    LP_model.append(train_LM_model(corpus_train,Laplace,2))
    LP_model.append(train_LM_model(corpus_train,Laplace,3))
    
    #liste contenant les scores de perplexite
    score_perplexite_MLE = []
    score_perplexite_LP = []
    score_perplexite_Lid = []
    
    
    for i in range (0,3):
        score_perplexite_MLE.append(evaluate(MLE_model[i], corpus_test))
        score_perplexite_LP.append(evaluate(LP_model[i], corpus_test))
    
    # Cohérent d'avoir une perplexité infini pour les ordres 2 et 3, car le produit des 1/P(wi|w1...wi-1)=+inf si un des P(wi|w1...wi-1)=0 , or certains 2-gram et 3-gram n'apparaissent pas dans le corpus de texte
    # Perplexité moyenne par phrase.
    print (score_perplexite_MLE)
    print (score_perplexite_LP)
    
    # 2)
    
    val_X = numpy.logspace(-5,0,10)
    val_y_1 = [evaluate_gamma(x,corpus_train,corpus_test,1) for x in val_X]
    val_y_2 = [evaluate_gamma(x,corpus_train,corpus_test,2) for x in val_X]
    val_y_3 = [evaluate_gamma(x,corpus_train,corpus_test,3) for x in val_X]
    
    plt.plot (val_X, val_y_1)
    plt.show()
    plt.plot (val_X, val_y_2)
    plt.show()
    plt.plot (val_X, val_y_3)
    plt.show()

    #1.6)

    # 2)
    # Génération du texte trump
    MLE_model_trump = []
    MLE_model_trump.append(train_LM_model(corpus_trump,MLE,1,unk_cutoff=1))
    MLE_model_trump.append(train_LM_model(corpus_trump,MLE,2,unk_cutoff=1))
    MLE_model_trump.append(train_LM_model(corpus_trump,MLE,3,unk_cutoff=1))
    
    text_trump = []
        
    text_trump.append(generate(MLE_model_trump[0], 20))
    text_trump.append(generate(MLE_model_trump[1], 20))
    text_trump.append(generate(MLE_model_trump[2], 20))
    
    for s in text_trump:
        print(s)
        print ("\n")
    


    
    
    
    
    