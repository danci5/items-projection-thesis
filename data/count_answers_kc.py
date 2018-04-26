# The counts contain all answers (no duplicated answers from user on one question) for knowledge components

import pandas as pd
import matplotlib.pyplot as plt
import os
from data.prepare_data import merge_logs_with_questions, merge_data_with_practice_sets


def get_counts_of_answers_for_kc(crucial_data, system_ps, export=False):
    """get practice sets with theirs counts of answers"""
    count_kcs = crucial_data['parent_kc'].value_counts()
    if export:
        practice_sets_answers_count.to_csv('processed/counts_of_answers_for_kcs.csv')
    return count_kcs


def plot_kcs_with_most_answers(data_ps, system_ps):
    df = get_counts_of_answers_for_kc(data_ps, system_ps).head(30)
    df.plot(kind='bar')
    plt.gcf().set_size_inches(15, 10)
    plt.title('Counts of answers for knowledge components (top30)', color='black')
    plt.xlabel('kc_id')
    plt.savefig('visualizations/matplotlib/counts_of_answers_for_knowledge_components.png')
    plt.show()


if __name__ == '__main__':
    os.chdir('/home/daniel/school/BP/pythesis')
    logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
    questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
    system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
    system_ps = pd.read_csv('data/system_ps.csv', sep=';')

    basic_data = merge_logs_with_questions(logs, questions).drop_duplicates(['user', 'question_id'], keep='first')
    ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps).drop_duplicates(['id', 'parent_kc'], keep='first')
    plot_kcs_with_most_answers(ps_data, system_ps)
