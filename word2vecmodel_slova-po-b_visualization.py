import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import os

from data.prepare_word2vec_ed import get_word2vec_items
from data.prepare_data import label_data
from manual_labeling.labeled_data import SLOVA_PO_B
from projections.base_projection import Projection

os.chdir('/home/daniel/school/BP/pythesis')
slova_po_b_all = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')

# load model
# loading from word2vec C format is much slower
# + you can continue training with the loaded model!
model = Word2Vec.load('utils/word2vec.model')
X, data = get_word2vec_items(model, slova_po_b_all)
data = label_data(data, SLOVA_PO_B)

# ----------------------------------
# t-SNE
# ----------------------------------
# model = TSNE(perplexity=10, learning_rate=5, n_iter=100000)
# result = model.fit_transform(X)

# ----------------------------------
# PCA
# ----------------------------------
model = PCA(n_components=2)
result = model.fit_transform(X)

# create a scatter plot of the projections
x_positions = result[:, 0]
y_positions = result[:, 1]

projection = Projection(x_positions, y_positions, data.copy(), data_name='slova po b', model=model)
# projection.matplotlib_plot_with_manual_labels(save_path='visualizations/vis.png', annotate=True)
projection.plotly_with_manual_labels(save_path='visualizations/vis1.html', annotate=False)
