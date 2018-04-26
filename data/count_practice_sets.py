# The counts contain all answers (no duplicated answers from user on one question) for practice sets

import pandas as pd
import matplotlib.pyplot as plt
import os
from data.prepare_data import merge_logs_with_questions, merge_data_with_practice_sets


def get_counts_of_answers_for_ps(crucial_data, system_ps, export=False):
    """get practice sets with theirs counts of answers"""
    count_practice_sets = crucial_data['ps'].value_counts()
    practice_sets_table = system_ps[['id','url']].set_index('id')
    practice_sets_answers_count = count_practice_sets.to_frame().join(practice_sets_table)
    practice_sets_answers_count.columns = ['count', 'url']
    practice_sets_answers_count.index.names = ['ps_id']

    if export:
        practice_sets_answers_count.to_csv('processed/counts_of_answers_for_practice_sets.csv')

    return practice_sets_answers_count


def plot_practice_sets_with_most_answers(data_ps, system_ps):
    df = get_counts_of_answers_for_ps(data_ps, system_ps).head(15)
    df.plot(kind='bar')
    plt.gcf().set_size_inches(12, 8)
    plt.title('Counts of answers for practice sets (top15)', color='black')
    plt.xlabel('practice_set_id')
    plt.savefig('visualizations/matplotlib/counts_of_answers_for_practice_sets.png')
    plt.show()


if __name__ == '__main__':
    os.chdir('/home/daniel/school/BP/pythesis')
    logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
    questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
    system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
    system_ps = pd.read_csv('data/system_ps.csv', sep=';')

    basic_data = merge_logs_with_questions(logs, questions).drop_duplicates(['user', 'question_id'], keep='first')
    ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps)
    plot_practice_sets_with_most_answers(ps_data, system_ps)
