"""
Script for preparing data and creating similarity matrix.
"""

import re
import pandas as pd
import numpy as np
import os
from similarities.performance_similarities import pearson_similarity, doublepearson_similarity


def cut_answer_question(value):
    """Cuts the correct_answer or question from the Umime cesky dataset.

    >>> cut_answer_question('[["text","b_tost"]]')
    'b_tost'
    >>> cut_answer_question('[["text","y"]]')
    'y'
    """
    return re.search(",\"(.*?)\"]]", value).group(1)


def merge_logs_with_questions(logs, questions):
    """Merges logs of answers data with the questions data (left join).

    Parameters
    ----------
    logs : DataFrame
    questions : DataFrame

    Returns
    -------
    DataFrame
        logs and questions attributes together
    """
    logs = logs.rename(columns={'question': 'question_id'})
    questions = questions.rename(columns={'id': 'question_id'})
    data = logs.join(questions.set_index('question_id'), on='question_id', rsuffix='_question')
    return data


def merge_data_with_practice_sets(logs_questions, practice_sets, ps_mapping):
    """Merges all important data about questions together (into logs).

    Left join of logs and questions with practice sets, and then left join with practice sets information.

    Parameters
    ----------
    logs_questions : DataFrame
        Merged logs and questions coming from merge_logs_with_questions()
    practice_sets : DataFrame
        Mapping of questions to practice sets
    ps_mapping : DataFrame
        Information about practice sets and mapping of practice sets to knowledge components

    Returns
    -------
    DataFrame
    """
    data = logs_questions.join(practice_sets.set_index('problem'), on="question_id", rsuffix='_r')
    data = data.join(ps_mapping.set_index('id'), on='ps', rsuffix='_s')

    ps_data = data[['id', 'user', 'correct', 'question_id', 'correct_question',
                         'question', 'url', 'ps', 'parent', 'exercise']]
    ps_data = ps_data.rename(columns={'correct_question': 'correct_answer', 'parent': 'parent_kc'})
    return ps_data


def get_ps_data():
    """Returns DataFrame with all answers and their important attributes."""
    logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
    questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
    system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
    system_ps = pd.read_csv('data/system_ps.csv', sep=';')
    basic_data = merge_logs_with_questions(logs, questions)
    ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps)
    return ps_data 


def get_data_by_knowledge_component(ps_data, kc_number):
    """Returns DataFrame with the data for specified knowledge component.

    EXAMPLE:
        get_data_by_knowledge_component(ps_data, 26) would return the concept 'Vyjmenovaná slova B' - 26 is its id
    ATTENTION:
        The knowledge component mapping is sometimes messy. They contain practice sets and words
        that they shouldn't contain. Sometimes rather check the numbers of practice sets,
        which you want and use get_data_for_practice_sets().

    Parameters
    ----------
    ps_data : DataFrame
        Merged data coming from merge_data_with_practice_sets()
    kc_number : int
        Number of knowledge component that you want to get data from

    Returns
    -------
    DataFrame
    """
    return ps_data[ps_data['parent_kc'] == kc_number].drop_duplicates(['user', 'question_id'], keep='first').drop_duplicates(['id'], keep='first')


def get_data_for_practice_sets(ps_data, practice_sets_numbers):
    """Returns DataFrame with the data for specified practice sets.

    EXAMPLE:
        get_data_by_knowledge_component(ps_data, 26) would return the concept 'Vyjmenovaná slova B' - 26 is its id
    ATTENTION:
        The knowledge component mapping is sometimes messy. They contain practice sets and words
        that they shouldn't contain. Sometimes rather check the numbers of practice sets,
        which you want and use get_data_for_practice_sets().

    Parameters
    ----------
    ps_data : DataFrame
        Merged data coming from merge_data_with_practice_sets()
    practice_sets_numbers : list of int
        List of practice sets numbers that you want to get data from

    Returns
    -------
    DataFrame
    """
    ps_data = ps_data[ps_data.ps.isin(practice_sets_numbers)]
    data = ps_data.drop_duplicates(['user', 'question_id'], keep='first')
    return data


def get_vyjmenovana_slova_po_b(ps_data):
    """Returns DataFrame of concept 'Vyjmenovaná slova B'."""
    return get_data_for_practice_sets(ps_data, [383, 384, 385])


def reshape_to_correctness_matrix(data):
    """Reshapes data of answers to matrix where the users are the indices(rows) and columns are the questions.

    After reshaping, the values in the matrix are specified by the correctness of the user's answer,
    and row in the data is a binary vector of user's answers.

    Parameters
    ----------
    data : DataFrame
        Answers data that contain 'user', 'question_id', 'correct' attributes

    Returns
    -------
    DataFrame
        Correctness matrix
    """
    if pd.Series(['user','question_id','correct']).isin(data.columns).all():
        # we want only first occurrence
        data = data.drop_duplicates(['user','question_id'], keep='first')

        # alternative - handles duplicates (the result won't be dichotomous data)
        # pd.pivot_table(data, values='correct', index='user',columns='question_id')
        correctness_matrix = data.pivot(index='user', columns='question_id', values='correct')
        return correctness_matrix
    else:
        print("Data are already pivoted or doesn't have the right structure.")
        return data


def correctness_matrix_to_similarity_matrix(method, matrix, no_nans=False):
    """Returns similarity matrix for learners' data.

    Parameters
    ----------
    method : str
        Method/Measure used for computing similarities
        'pearson', 'doublepearson' can be used.
    data : DataFrame
        Correctness matrix (output of reshape_to_correctness_matrix())
    no_nans : bool
        Parameter that specifies, if the similarity matrix can contain NaN values or not

        For projections, clustering, etc. no NaN values should be in the matrix.
        For correlations there can be NaN values in the matrix, it excludes the NaN values for the correlation.

    Returns
    -------
    DataFrame
        Similarity matrix
    """
    if method == 'pearson':
        similarity_matrix = pearson_similarity(matrix, no_nans)
    elif method == 'doublepearson':
        similarity_matrix = doublepearson_similarity(matrix, no_nans)
    return similarity_matrix


def get_labels_and_practice_sets_for_similarity_matrix(matrix, pre_matrix_data):
    """Returns labels and practice sets for similarity matrix.

    Parameters
    ----------
    matrix : DataFrame
        Similarity matrix
    pre_matrix_data : DataFrame
        Data in the state before similarity matrix

    Returns
    -------
    tuple of (list of labels, list of practice_sets)
        Label and practice set for each question from the similarity matrix index
    """

    question_ids = matrix.index
    pre_matrix_data = pre_matrix_data.set_index('question_id')
    labels = np.array([pre_matrix_data.at[question_id, 'question'][0] for question_id in question_ids])
    practice_sets = np.array([pre_matrix_data.at[question_id, 'ps'][0] for question_id in question_ids])
    return labels, practice_sets


class UnderscoresError(Exception):
    """Raise this exception when you get word with 0 underscores or more underscores than 1 in fill-in exercises.
    NOTE: Now there are just questions with 0 or 1 underscore in the dataset."""


def get_solutions(data, method='fillin'):
    """Returns solutions for the dataset.

    EXAMPLE assignment: nab_t pušku
        - full : nabít pušku
        - fillin : nabít
        - fillinextra : nab(í)t

    Parameters
    ----------
    data : DataFrame
        Answers data that contain 'question' and 'correct_answer'
    method : str
        Method/Measure used for computing similarities
        ‘full’, ‘fillin’, ‘fillinextra’ can be used.

    Returns
    -------
    list of str
    """
    solutions = []
    for index, row in data.iterrows():
        if row['question'].count('_') != 1:
            raise UnderscoresError()
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
    """Labels the data according to manual labeling.

    Adds column 'manual_label' to DataFrame. Data should have the 'full_solution' column to be labeled.

    Parameters
    ----------
    data : DataFrame
        Answers data that contain 'full_solution' column.
    label_groups : list of lists of str
        List of manually determined groups in the data.

    Returns
    -------
    No return
    """
    if 'full_solution' not in data.columns:
        raise ValueError("No column 'full_solution' in data.")
    data['manual_label'] = 0
    for label, group in enumerate(label_groups, 1):
        data.loc[data['full_solution'].isin(group), 'manual_label'] = label


if __name__ == '__main__':
    # run this if the question and correct answer is in format like "[[""text"",""zab_dlený""]]"
    os.chdir('/home/daniel/school/BP/pythesis')
    questions = pd.read_csv('data/nova_doplnovacka_questions2.csv', sep=';')
    questions['question'] = questions['question'].apply(cut_answer_question)
    questions['correct'] = questions['correct'].apply(cut_answer_question)
    questions.to_csv('data/nova_doplnovacka_questions2.csv', sep=';', index=False)
