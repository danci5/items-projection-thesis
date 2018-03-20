import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import os
import re
os.chdir('/home/daniel/school/BP')


slova_po_b = pd.read_csv('data/processed/vyjmenovana_slova_po_b.csv')


class MoreUnderscoresError(Exception):
    """Raise when you get word with more underscores => more holes to fill in by student"""


def get_solutions(data):
    """
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.itertuples.html
    """
    solutions = []
    for index, row in data.iterrows():
        if row['question'].count('_') > 1:
            raise MoreUnderscoresError()
        # cutting the word, where is the underscore
        word = re.search("(\w*_\w*)", row['question']).group(1)
        # adding the word, where was the underscore
        # solutions.append(word.replace('_', '('+row[4]+')'))
        solutions.append(word.replace('_', row['correct_answer']))
    return solutions


# load model
# loading from word2vec C format is much slower
# + you can continue training with the loaded model!
model = Word2Vec.load('pythesis/utils/word2vec.model')

slova_po_b = slova_po_b.drop_duplicates(subset=['question'], keep='first')
X = []
labels = []
practice_sets = []
manual_labels = []
solutions = get_solutions(slova_po_b)

group1 = ["abych", "abychom", "abys", "kdybyste", "kdyby", "aby"]
group2 = ["bylina", "bylinkář", "bylinkový", "bylinka", "ruské pověsti se nazývají byliny"]
group3 = ["biologie", "biograf", "biotop", "biosféra", "biofyzika", "biografie", "biorytmus", "biochemik"]
group4 = ["zbytečný", "zbytek koláče", "zbytek", "zbylý", "zbylé látky odložila do krabice", "zbyl mi kousek dortu",
          "zbyly po něm dluhy", "měla zbytečné starosti"]
group5 = ["kobyla", "kobylka"]
group6 = ["obyvatelka", "obyvatelstvo", "výzva obyvatelstva k pořádku"]
group7 = ["dobytkářství", "zabýval se dobytkářstvím", "obchodoval s dobytkem", "dobytče", "dobytek"]
group8 = ["biblický příběh", "bible"]

slova_po_b['manual_label'] = 0
slova_po_b['solution'] = get_solutions(slova_po_b)
groups = [group1, group2, group3, group4, group5, group6, group7, group8]
for label, group in enumerate(groups, 1):
    slova_po_b.loc[slova_po_b['solution'].isin(group), 'manual_label'] = label

for solution, assignment, practice_set, label in zip(solutions, slova_po_b['question'], slova_po_b['ps'],
                                                     slova_po_b['manual_label']):
    try:
        if solution == 'bicí':
            continue
        X.append(model[solution])
        labels.append(assignment)
        practice_sets.append(practice_set)
        manual_labels.append(label)
    except KeyError:
        print(
            "Word {} is not in vocabulary of your model, therefore it won't be in your visualization.".format(solution))

# ----------------------------------
# t-SNE
# ----------------------------------
model = TSNE(perplexity=10, learning_rate=200, n_iter=200000)
result = model.fit_transform(X)

# ----------------------------------
# PCA
# ----------------------------------
# model = PCA(n_components=2)
# result = model.fit_transform(X)


# create a scatter plot of the projections
x_positions = result[:, 0]
y_positions = result[:, 1]
# plt.scatter(result[:, 0], result[:, 1])

colors = ['blue','green','red','darkcyan','magenta','yellow','darkgray','purple']
for i, label in enumerate(labels):
    if manual_labels[i] != 0:
        plt.scatter(x_positions[i], y_positions[i], c=colors[manual_labels[i]-1])
        plt.annotate(label, xy=(result[i, 0], result[i, 1]), color=colors[manual_labels[i]-1])
    else:
        plt.scatter(x_positions[i], y_positions[i], c='black')
        plt.annotate(label, xy=(result[i, 0], result[i, 1]), color='black')
plt.gcf().set_size_inches(40, 30)
plt.title("Used: 'slova po b', {}".format(model), color='black')
plt.savefig('visualizations/tsne_word_embeddings_vyjmenovana_slova_po_b_figure-40x30_with_corpus.png')
plt.show()
