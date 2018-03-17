"""
Script for preparing data and creating similarity matrix.
Created based on the 'data_preparation' jupyter notebook.
"""

# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/home/daniel/school/BP')


logs = pd.read_csv('data/nova_doplnovacka_log.csv', sep=';')
questions = pd.read_csv('data/nova_doplnovacka_questions.csv', sep=';')
system_ps_problem = pd.read_csv('data/system_ps_problem.csv', sep=';')
system_ps = pd.read_csv('data/system_ps.csv', sep=';')
system_kc = pd.read_csv('data/system_kc.csv', sep=';')


def merge_logs_with_questions(logs, questions):
    """basic data, left join"""
    logs = logs.rename(columns={'question': 'question_id'})
    questions = questions.rename(columns={'id': 'question_id'})
    data = logs.join(questions.set_index('question_id'), on='question_id', rsuffix='_question')
    return data


# basic_data = merge_logs_with_questions(logs, questions)


def merge_data_with_practice_sets(logs_questions, practice_sets, ps_mapping):
    """ps data, left join"""

    data = logs_questions.join(practice_sets.set_index('problem'), on="question_id", rsuffix='_r')
    data = data.join(ps_mapping.set_index('id'), on='ps', rsuffix='_s')

    crucial_data = data[['id', 'user', 'correct', 'question_id', 'correct_question',
                         'question', 'url', 'ps', 'parent', 'exercise']]
    crucial_data.rename(columns={'correct_question': 'correct_answer', 'parent': 'parent_kc'}, inplace=True)
    return crucial_data


# ps_data = merge_data_with_practice_sets(basic_data, system_ps_problem, system_ps)


def get_counts_of_answers_for_ps(crucial_data, system_ps, export=False):
    """get practice sets with theirs counts of answers"""
    count_practice_sets = crucial_data['ps'].value_counts()
    practice_sets_table = system_ps[['id','url']].set_index('id')
    practice_sets_answers_count = count_practice_sets.to_frame().join(practice_sets_table)
    practice_sets_answers_count.columns = ['count', 'url']
    practice_sets_answers_count.index.names = ['ps_id']

    if export:
        practice_sets_answers_count.to_csv('data/processed/count_of_questions_for_practice_sets.csv')

    return practice_sets_answers_count


# practice_sets_count = get_counts_of_answers_for_ps(ps_data, system_ps)


def plot_practice_sets_with_most_answers(data_ps, system_ps):
    df = get_counts_of_answers_for_ps(data_ps, system_ps).head(15)
    df.plot(kind='bar')
    plt.gcf().set_size_inches(12, 8)
    plt.title('Count of answers for practice sets (top15)', color='black')
    plt.xlabel('practice_set_id')
    plt.savefig('visualizations/matplotlib/count_of_questions_for_practice_sets.png')
    plt.show()




def reshape_to_similarity_matrix(data):
    # reshape data to matrix where the users are the indices(rows) and columns are the questions
    # the values in the matrix are specified by the correctness of the user's answer

    # handles duplicates
    # pd.pivot_table(data, values='correct', index='user',columns='question_id').head()

    # drop_duplicates - default is 'Drop duplicates except for the first occurrence'
    # we want only first occurrence

    if pd.Series(['user','question_id','correct']).isin(data.columns).all():
        data = data.drop_duplicates(['user','question_id'])
        similarity_matrix = data.pivot(index='user', columns='question_id', values='correct')
        return similarity_matrix
    else:
        print("Data are already pivoted or doesn't have the right structure.")
        return data




def get_vyjmenovana_slova_po_b(crucial_data):
    """input 'crucial_data' should be already without duplicate answers for one particular question from one user"""
    # ps_id, where are practice_sets for vyjmenovana slova po b:
    # 1, 2, 3, (85, 86, 87)-otazky, (169, 170)-diktat, 383, 384, 385
    # 1,2,3/85,86,87/383,384,385 seems like the same
    slova_po_b = crucial_data[crucial_data.ps.isin([383, 384, 385])]

    # if some practice sets share the questions
    data = slova_po_b.drop_duplicates(['user', 'question_id'], keep='first')

    return data


# Can filter the created datasets like 'slova po b'
#   - e.g. consider only data where is the minimal count of answers for question or minimal count of answers for student