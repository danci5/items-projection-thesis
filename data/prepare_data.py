"""
Script for preparing data and creating similarity matrix.
"""

import re
import pandas as pd
import numpy as np
import os
from similarities.performance_similarities import *


def cut_question(question):
    return re.search(",\"(.*?)\"]]", question).group(1)


def cut_answer(answer):
    return re.search(",\"(.*?)\"]]", answer).group(1)


def merge_logs_with_questions(logs, questions):
    """basic data, left join"""
    logs = logs.rename(columns={'question': 'question_id'})
    questions = questions.rename(columns={'id': 'question_id'})
    data = logs.join(questions.set_index('question_id'), on='question_id', rsuffix='_question')
    return data


def merge_data_with_practice_sets(logs_questions, practice_sets, ps_mapping):
    """ps data, left join"""

    data = logs_questions.join(practice_sets.set_index('problem'), on="question_id", rsuffix='_r')
    data = data.join(ps_mapping.set_index('id'), on='ps', rsuffix='_s')

    crucial_data = data[['id', 'user', 'correct', 'question_id', 'correct_question',
                         'question', 'url', 'ps', 'parent', 'exercise']]
    crucial_data = crucial_data.rename(columns={'correct_question': 'correct_answer', 'parent': 'parent_kc'})
    return crucial_data


def get_ps_data():
    logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
    questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
    system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
    system_ps = pd.read_csv('data/system_ps.csv', sep=';')
    system_kc = pd.read_csv('data/system_kc.csv', sep=';')
    basic_data = merge_logs_with_questions(logs, questions)
    ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps)
    return ps_data 


def get_data_by_knowledge_component(ps_data, kc_number):
    """
    ATTENTION: The knowledge component mapping is sometimes messy. They contain practice sets and
    words which they shouldn't contain. 
    Sometimes rather check the practice_set_numbers, which you want and use get_data_for_practice_sets. 
    
    input is - parameter ps_data from 'merge_data_with_practice_sets'
    parameter kc_number is the id of knowledge component from the dataset

    example: get_data_by_knowledge_component(ps_data, 26) would return
    the dataset of Vyjmenovaná slova B - 26 is its id
    """
    return ps_data[ps_data['parent_kc'] == kc_number].drop_duplicates(['user', 'question_id'], keep='first').drop_duplicates(['id'], keep='first')


def get_data_for_practice_sets(practice_sets_numbers, crucial_data):
    """
    input is from 'merge_data_with_practice_sets'
    Therefore the data should be already without duplicate answers for one particular question from one user.

    For example for 'vyjmenovana slova po b' it's get_data_for_practice_sets([383,384,385], crucial_data).
    """
    slova_po_b = crucial_data[crucial_data.ps.isin(practice_sets_numbers)]

    # if some practice sets share the questions, drop duplicates
    data = slova_po_b.drop_duplicates(['user', 'question_id'], keep='first')

    return data


def get_vyjmenovana_slova_po_b(crucial_data):
    """input is from 'merge_data_with_practice_sets'
    Therefore the data should be already without duplicate answers for one particular question from one user.
    """
    # ps_id, where are practice_sets for vyjmenovana slova po b:
    # 1, 2, 3, (85, 86, 87)-otazky, (169, 170)-diktat, 383, 384, 385
    # 1,2,3/85,86,87/383,384,385 seems like the same
    slova_po_b = crucial_data[crucial_data.ps.isin([383, 384, 385])]

    # if some practice sets share the questions
    data = slova_po_b.drop_duplicates(['user', 'question_id'], keep='first')

    return data


def reshape_to_correctness_matrix(data):
    """
        Reshapes data to matrix where the users are the indices(rows) and columns are the questions.
        The values in the matrix are specified by the correctness of the user's answer.
        """

    if pd.Series(['user','question_id','correct']).isin(data.columns).all():
        # we want only first occurrence
        # drop_duplicates - default is 'Drop duplicates except for the first occurrence'
        data = data.drop_duplicates(['user','question_id'])

        # alternative - handles duplicates
        # pd.pivot_table(data, values='correct', index='user',columns='question_id')
        similarity_matrix = data.pivot(index='user', columns='question_id', values='correct')
        return similarity_matrix
    else:
        print("Data are already pivoted or doesn't have the right structure.")
        return data


def correctness_matrix_to_similarity_matrix(method, matrix, no_nans=False):
    if method == 'pearson':
        similarity_matrix = pearson_similarity(matrix, no_nans)
    elif method == 'doublepearson':
        similarity_matrix = doublepearson_similarity(matrix, no_nans)

    return similarity_matrix


def get_labels_and_practice_sets_for_similarity_matrix(matrix, pre_matrix_data):
    """Input - similarity matrix and data in the state before similarity matrix.
    Returns tuple of labels and practice sets for every question from similarity matrix."""

    question_ids = matrix.index
    pre_matrix_data = pre_matrix_data.set_index('question_id')
    labels = np.array([pre_matrix_data.at[question_id, 'question'][0] for question_id in question_ids])
    practice_sets = np.array([pre_matrix_data.at[question_id, 'ps'][0] for question_id in question_ids])
    return labels, practice_sets


class MoreUnderscoresError(Exception):
    """Raise when you get word with more underscores => more holes to fill in by student"""


def get_solutions(data, method='fillin'):
    """
    method : {‘full’, ‘fillin’, ‘fillinextra’}
    
    example assignment: nab_t pušku
       - full : nabít pušku
       - fillin : nabít
       - fillinextra : nab(í)t
    """
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.itertuples.html

    solutions = []
    for index, row in data.iterrows():
        if row['question'].count('_') > 1:
            raise MoreUnderscoresError()
        
        if method == 'full':
            full_solution = row['question'].replace('_', row['correct_answer'])
            solutions.append(full_solution)
        elif method == 'fillin':
            # cutting the word, where is the underscore
            word = re.search("(\w*_\w*)", row['question']).group(1)
            # adding the word, where was the underscore
            solutions.append(word.replace('_', row['correct_answer']))
        elif method == 'fillinextra':
            word = re.search("(\w*_\w*)", row['question']).group(1)
            solutions.append(word.replace('_', '('+row['correct_answer']+')'))
    return solutions


def label_data(data, label_groups):
    """Adds column 'manual_label'.
    Data should have the 'full_solution' column to be labeled.
    """

    if 'full_solution' not in data.columns:
        raise ValueError("No column 'full_solution' in data.")
    data['manual_label'] = 0
    for label, group in enumerate(label_groups, 1):
        data.loc[data['full_solution'].isin(group), 'manual_label'] = label


if __name__ == '__main__':
    os.chdir('/home/daniel/school/BP/pythesis')

    logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
    questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
    system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
    system_ps = pd.read_csv('data/system_ps.csv', sep=';')
    system_kc = pd.read_csv('data/system_kc.csv', sep=';')

    # run this if the question and correct answer is in format like "[[""text"",""zab_dlený""]]"
    # questions['question'] = questions['question'].apply(cut_question)
    # questions['correct'] = questions['correct'].apply(cut_answer)
    # questions.to_csv('data/nova_doplnovacka_questions.csv', sep=';', index=False)

    basic_data = merge_logs_with_questions(logs, questions)
    # basic_data.to_csv('data/processed/basic_data.csv', index=False)
    ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps)
    # ps_data.to_csv('data/processed/ps_data.csv', index=False)
    # practice_sets_count = get_counts_of_answers_for_ps(ps_data, system_ps, True)

    # example
    # vyjm_slova = get_vyjmenovana_slova_po_b(ps_data)
    # solutions = get_solutions(vyjm_slova)
    # vyjm_slova.to_csv('data/processed/vyjmenovana_slova_po_b.csv', index=False)
    # correctness_matrix = reshape_to_correctness_matrix(vyjm_slova)
    # similarity_matrix = correctness_matrix_to_similarity_matrix('doublepearson', correctness_matrix)
