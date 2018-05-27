import pandas as pd
import os
from data.prepare_data import get_solutions, get_ps_data, get_vyjmenovana_slova_po_b
from similarities.text_similarities import levenshtein_similarity


def get_word2vec_items(model, data, verbose=False):
    """Returns word vectors and respective data for the word vectors.

    Additionally, adds 2 columns 'solution' and 'full_solution' to data because they will be needed for labeling, etc.

    Parameters
    ----------
    model : Word2Vec model
        Instantiated Word2Vec model you loaded before
    data : DataFrame
        Data that contain 'correct_answer' and 'question' attributes
    verbose : bool
        Parameter that specifies verbose mode for printing words that are not in our word2vec model

    Returns
    -------
    tuple of (list of word vectors, DataFrame)
        Word vectors and data based on the word2vec model
    """
    data = data.drop_duplicates(subset=['question_id'], keep='first').copy()
    data['solution'] = get_solutions(data, method='fillin')
    data['full_solution'] = get_solutions(data, method='full')
    X, ids = [], []

    for id, solution, question in zip(data['id'], data['solution'], data['question']):
        try:
            # in slova_po_b is the word 'bicí' a big outlier in visualization, i'd rather get rid of it
            if solution == 'bicí':
                continue
            X.append(model.wv[solution])
            ids.append(id)
        except KeyError:
            if verbose:
                print(
                    "Word '{}' in question '{}' is not in vocabulary of your model, "
                    "therefore it's not in the resulting data.".format(solution, question))
    return X, data[data['id'].isin(ids)].copy()


def create_word2vec_similarity_matrix(model, index, solutions):
    """Returns pairwise similarity matrix computed by using word2vec (between each pair of word vectors).

    In gensim library and in original papers of word2vec, cosine similarity is used for
    similarity computation between 2 word vectors.

    Parameters
    ----------
    model : Word2Vec model
        Instantiated Word2Vec model you loaded before
    index : list of int
        Index and columns of pairwise similarity matrix
    solutions : list of str
        Words that are used for similarity matrix computation

    Returns
    -------
    DataFrame
    """

    dataframe = pd.DataFrame(index=index, columns=index)
    for i, j in zip(solutions, index):
        for k, l in zip(solutions, index):
            dataframe.loc[j, l] = model.wv.similarity(i, k)
    return dataframe


def create_edit_similarity_matrix(index, solutions, similarity_function=levenshtein_similarity):
    """Returns pairwise similarity matrix computed by using Edit distance (between each pair of words).

    Parameters
    ----------
    index : list of int
        Index and columns of pairwise similarity matrix
    solutions : list of str
        Words that are used for similarity matrix computation
    similarity_function : function
        Function that is used for similarity computation between words
        Levenshtein similarity by default.

    Returns
    -------
    DataFrame
    """
    dataframe = pd.DataFrame(index=index, columns=index)
    for i, j in zip(solutions, index):
        for k, l in zip(solutions, index):
            dataframe.loc[j, l] = similarity_function(i, k)
    return dataframe


if __name__ == '__main__':
    # example of word2vec similarity matrix creation for concept 'Vyjmenovana slova B'
    os.chdir('/home/daniel/school/BP/pythesis')
    from gensim.models import Word2Vec
    model = Word2Vec.load('utils/word2vec.model')
    ps_data = get_ps_data()
    slova_po_b = get_vyjmenovana_slova_po_b(ps_data)
    X, data = get_word2vec_items(model, slova_po_b)
    word2vec_similarity_matrix = create_word2vec_similarity_matrix(model, data['question_id'], data['solution'])
    word2vec_similarity_matrix.to_csv('data/word2vec_similarity_matrix_slova-po-b.csv')
