import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from data.prepare_data import *
from data.prepare_word2vec_ed import *
from similarities.text_similarities import levenshtein_similarity

model = Word2Vec.load('utils/word2vec.model')
slova_po_b = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')
X, data = get_word2vec_items(model, slova_po_b)
# data - only words which are also in my word2vec model

### ED ###
edit1 = create_edit_similarity_matrix(data['full_solution'], data['solution'], levenshtein_similarity)
edit1 = edit1.astype(float)
edit1.to_csv('data/processsed/levenshtein_similarity_matrix_slova-po-b.csv')
#edit1 = pd.read_csv('data/processed/levenshtein_similarity_matrix_slova-po-b.csv', index_col=0)

### word2vec ###
word2vec = create_word2vec_similarity_matrix(model, data['full_solution'], data['solution'])
word2vec.to_csv('data/processed/word2vec_words_similarity_matrix_slova_po_b.csv')
#word2vec = pd.read_csv('data/processed/word2vec_similarity_matrix_slova-po-b.csv', index_col=0)

### pearson ###
vyjm_slova_filtered = slova_po_b.loc[slova_po_b['question'].isin(data['question'])]
correctness_matrix = reshape_to_correctness_matrix(vyjm_slova_filtered)
pearson1 = correctness_matrix_to_similarity_matrix('doublepearson', correctness_matrix)
pearson2 = correctness_matrix_to_similarity_matrix('pearson', correctness_matrix)

# correlation matrix
similarities = pd.DataFrame()
similarities['dpearson'] = pearson1.values.flatten()
similarities['pearson'] = pearson2.values.flatten()
similarities['word2vec'] = word2vec.values.flatten()
similarities['lev1'] = edit1.values.flatten().astype(float)
similarities_final = similarities.corr()

# matplotlib correlation matrix plot matshow
# plt.matshow(similarities_final.corr())
# plt.gcf().set_size_inches(6, 4)
# plt.xticks(range(len(similarities_final.columns)), similarities_final.columns)
# plt.yticks(range(len(similarities_final.columns)), similarities_final.columns)
# plt.colorbar()
# plt.savefig('visualizations/heatmap.png')
# plt.show()

# http://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(similarities_final, annot=True, fmt=".2f", vmin=0)
plt.savefig('visualizations/heatmap.png')
plt.show()
