import pandas as pd
import numpy as np
import os
from data.prepare_data import get_ps_data, get_vyjmenovana_slova_po_b
from similarities.text_similarities import levenshtein_similarity


def get_word2vec_items(model, data, verbose=False):
    """Returns list of word vectors for data which is only in the vocabulary of your model
    and the dataframe based on the model. 2 columns are added - 'solution' and 'full solution'.

    Parameter 'model' is instantiated Word2vecmodel you loaded before.
    Parameter 'data' has to have 'correct_answer', 'question' columns.
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
                print("Word '{}' in question '{}' is not in vocabulary of your model, therefore it's not in the resulting data.".format(
                solution, question))
    return X, data[data['id'].isin(ids)].copy()


def create_word2vec_similarity_matrix(model, full_solutions, solutions):
    """
    Parameter 'model' is the model which you want to build your word2vec matrix on. It's the instance which you loaded before.
    Parameter 'index' - index values for similarity matrix
    Parameter 'solutions' - words on which the similarity matrix is computed
    Similarity is computed only on the words which contain the "fill-in-the-blank" place.
    
    Example: 
    -'sb_rka známek' and 'b_lá barva' 
    -edit distance is counted on 'sbírka' and 'bíla'
    -index in the matrix are going to be the full solutions 'sbírka známek' and 'bíla barva'
    """
    dataframe = pd.DataFrame(index=full_solutions,columns=full_solutions)
    for i, j in zip(solutions, full_solutions):
        for k, l in zip(solutions, full_solutions):
                dataframe.loc[j, l] = model.wv.similarity(i, k)
    return dataframe


def create_edit_similarity_matrix(index, solutions, similarity_function=levenshtein_similarity):
    """
    Parameter 'index' - index values for similarity matrix
    Parameter 'solutions' - words on which the similarity matrix is computed
    Similarity is computed only on the words which contain the "fill-in-the-blank" place.
    
    Example: 
    -'sb_rka známek' and 'b_lá barva' 
    -edit distance is counted on 'sbírka' and 'bíla'
    -index in the matrix are going to be the full solutions 'sbírka známek' and 'bíla barva'
    """
    dataframe = pd.DataFrame(index=index,columns=index)
    for i, j in zip(solutions, index):
        for k, l in zip(solutions, index):
                dataframe.loc[j, l] = similarity_function(i, k)
    return dataframe

if __name__ == '__main__':
    os.chdir('/home/daniel/school/BP/pythesis')
    # example of word2vec similarity matrix creation for practice sets about 'Vyjmenovana slova B'
    from gensim.models import Word2Vec
    model = Word2Vec.load('utils/word2vec.model')
    ps_data = get_ps_data()
    slova_po_b = get_vyjmenovana_slova_po_b(ps_data)
    X, data = get_word2vec_items(model, slova_po_b)
    word2vec_similarity_matrix = create_word2vec_similarity_matrix(model, data['question_id'], data['solution'])
    word2vec_similarity_matrix.to_csv('data/word2vec_similarity_matrix_slova-po-b.csv')