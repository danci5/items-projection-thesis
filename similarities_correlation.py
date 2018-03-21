from gensim.models import Word2Vec
from data.prepare_data import *
from data.prepare_word2vec import *

model = Word2Vec.load('utils/word2vec.model')
slova_po_b = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')
X, labels, solutions, full_solutions = get_word2vec_items_tuple(model, slova_po_b)

# word2vec_similarity_matrix = create_word2vec_similarity_matrix(model, full_solutions, solutions)
# word2vec_similarity_matrix.to_csv('data/processed/word2vec_words_similarity_matrix_slova_po_b2.csv')
word2vec_similarity_matrix = pd.read_csv('data/processed/word2vec_similarity_matrix_slova-po-b.csv', index_col=0)

# only words which are also in my word2vec model
vyjm_slova_filtered = slova_po_b.loc[slova_po_b['question'].isin(labels)]
correctness_matrix = reshape_to_correctness_matrix(vyjm_slova_filtered)
similarity_matrix = correctness_matrix_to_similarity_matrix('doublepearson', correctness_matrix)
similarity_matrix2 = correctness_matrix_to_similarity_matrix('pearson', correctness_matrix)

# similarity_matrix.columns = word2vec_similarity_matrix.columns
# similarity_matrix.index = word2vec_similarity_matrix.index
# similarity_matrix['sbírka známek'].corr(word2vec_similarity_matrix['sbírka známek'])

# one way to compare similarity measures
sim_matrix_doublepearvec = pd.Series(similarity_matrix.values.flatten())
sim_matrix_pearvec = pd.Series(similarity_matrix2.values.flatten())
sim_matrix_wordvec = pd.Series(word2vec_similarity_matrix.values.flatten())

print('doublepearson vs pearson: %s' % (sim_matrix_doublepearvec.corr(sim_matrix_pearvec)))
print('doublepearson vs word2vec: %s' % (sim_matrix_doublepearvec.corr(sim_matrix_wordvec)))
print('pearson vs word2vec: %s' % (sim_matrix_pearvec.corr(sim_matrix_wordvec)))
# correlation coefficient matrix

# second way to compare similarity measures
corrcoef_matrix = np.corrcoef([similarity_matrix.values.flatten(), similarity_matrix2.values.flatten(), word2vec_similarity_matrix.values.flatten()])

# third way to compare similarity measures
similarities = pd.DataFrame()
similarities['pearson'] = similarity_matrix.values.flatten()
similarities['dpearson'] = similarity_matrix2.values.flatten()
similarities['word2vec'] = word2vec_similarity_matrix.values.flatten()
similarities_final = similarities.corr()

# TODO: add similarities from the article
# Then I should maybe fill 0 instead of removing NaN values

# https://stackoverflow.com/questions/11620914/removing-nan-values-from-an-array
matrix_vec = correctness_matrix.values.flatten()
matrix_vec = matrix_vec[~np.isnan(matrix_vec)]
