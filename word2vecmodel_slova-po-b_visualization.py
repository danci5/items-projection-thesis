import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import os

from data.prepare_word2vec import get_word2vec_items
from projections.base_projection import Projection

os.chdir('/home/daniel/school/BP/pythesis')
slova_po_b_all = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')

# load model
# loading from word2vec C format is much slower
# + you can continue training with the loaded model!
model = Word2Vec.load('utils/word2vec.model')
X, slova_po_b = get_word2vec_items(model, slova_po_b_all)

# TODO: extract this labeling from the script to some config
group1 = ["abych", "abychom", "abys", "kdybyste", "kdyby", "aby"]
group2 = ["bylina", "bylinkář", "bylinkový", "bylinka", "ruské pověsti se nazývají byliny"]
group3 = ["biologie", "biograf", "biotop", "biosféra", "biofyzika", "biografie", "biorytmus", "biochemik"]
group4 = ["zbytečný", "zbytek koláče", "zbytek", "zbylý", "zbylé látky odložila do krabice", "zbyl mi kousek dortu",
          "zbyly po něm dluhy", "měla zbytečné starosti"]
group5 = ["kobyla", "kobylka"]
group6 = ["obyvatelka", "obyvatelstvo", "výzva obyvatelstva k pořádku"]
group7 = ["dobytkářství", "zabýval se dobytkářstvím", "obchodoval s dobytkem", "dobytče", "dobytek"]
group8 = ["biblický příběh", "bible"]
groups = [group1, group2, group3, group4, group5, group6, group7, group8]

# TODO: extract this functionality to prepare_data.py
slova_po_b['manual_label'] = 0
for label, group in enumerate(groups, 1):
    slova_po_b.loc[slova_po_b['full_solution'].isin(group), 'manual_label'] = label

# ----------------------------------
# t-SNE
# ----------------------------------
# model = TSNE(perplexity=10, learning_rate=200, n_iter=200000)
# result = model.fit_transform(X)

# ----------------------------------
# PCA
# ----------------------------------
model = PCA(n_components=2)
result = model.fit_transform(X)

# create a scatter plot of the projections
x_positions = result[:, 0]
y_positions = result[:, 1]

projection = Projection(x_positions, y_positions, slova_po_b.copy(), data_name='slova po b', model=model)
projection.plot_with_manual_labels(save_path='visualizations/njuuu2.png', annotate=True)
