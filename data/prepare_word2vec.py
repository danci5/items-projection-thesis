import pandas as pd
import os
from data.prepare_data import get_solutions


def get_word2vec_items_tuple(model, data):
    """Function to get all the items you need for manipulating items when working your word2vec model.
    Returns the data which is only in the vocabulary of your model.
    
    Parameter 'model' is instantiated Word2vecmodel you loaded before.
    Parameter 'data' has to have 'correct_answer', 'question' columns.
    
    Returns 4-tuple of word vectors array which can be then input into projection methods,
    labels which can annotate the point in the visualization, full solutions, 
    and solutions which contain only the 'fill-in-the-blank' word."""

    data = data.drop_duplicates(subset=['question'], keep='first')

    X, labels = [], []
    solutions_vocab, full_solutions_vocab = [], []
    solutions = get_solutions(data, method='fillin')
    full_solutions = get_solutions(data, method='full')

    for full_solution, solution, label in zip(full_solutions, solutions, data['question']):
        try:
            # in slova_po_b is the word 'bicí' a big outlier in visualization, i'd rather get rid of it
            if solution == 'bicí':
                continue
            X.append(model.wv[solution])
            labels.append(label)
            solutions_vocab.append(solution)
            full_solutions_vocab.append(full_solution)
        except KeyError as e:
            print("Word '{}' is not in vocabulary of your model, therefore it won't be in your visualization.".format(solution))
    return X, labels, solutions_vocab, full_solutions_vocab


def create_word2vec_similarity_matrix(model, full_solutions, solutions):
    """input parameters:
    'model' is the model which you want to build your word2vec matrix on. It's the instance which you loaded before.
    'solutions' contain only the 'fill-in-the-blank' word based on which is the similarity computed.
    'full_solutions' are for representing the assignment in matrix (same word can occur multiple times).
    
    Returns similarity matrix."""
    dataframe = pd.DataFrame(index=full_solutions,columns=full_solutions)
    for i, j in zip(solutions, full_solutions):
        for k, l in zip(solutions, full_solutions):
                dataframe.loc[j,l] = model.wv.similarity(i, k)
    return dataframe


if __name__ == '__main__':
    os.chdir('/home/daniel/school/BP/pythesis')
    # example of word2vec similarity matrix creation for practice sets about 'slova po b'
    from gensim.models import Word2Vec
    model = Word2Vec.load('utils/word2vec.model')
    slova_po_b = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')
    X, labels, solutions, full_solutions = get_word2vec_items_tuple(model, slova_po_b)
    word2vec_similarity_matrix = create_word2vec_similarity_matrix(model, full_solutions, solutions)
    word2vec_similarity_matrix.to_csv('data/processed/word2vec_words_similarity_matrix_slova_po_b.csv')