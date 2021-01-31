import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def file_to_df():
    if len(sys.argv) > 1:
        file = sys.argv[1]
        if os.path.isfile(file):
            df = pd.read_csv(file)
            return df
        else:
            raise Exception('There is no file with that name inside directory')
    else:
        raise Exception('Provide a filename')


def df_to_file(original_df, changed_df):
    original_df['RESULT'] = changed_df['RESULT']
    original_df.to_csv('result.csv')
    return original_df


def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(15, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature + " Distribution", color='DarkRed')
    plt.show()


def map_records_count(dataframe_name):
    demo_unique_id = dataframe_name['demo_id'].value_counts()
    # Number of unique demos
    number_unique_demos = len(dataframe_name['demo_id'].value_counts())
    print(f'ID: Numb of rec:\n{demo_unique_id}\nUnique demos: {number_unique_demos}')
    # ID and number of records
    demo_round_unique_id = dataframe_name['demo_round_id'].value_counts()
    # Number of unique rounds
    number_demo_round_unique = len(dataframe_name['demo_round_id'].value_counts())
    print(f'ID: Numb of rec:\n{demo_round_unique_id}\nUnique rounds: {number_demo_round_unique}')


def map_percentage_of_throws(dataframe_name, map_name):
    map_label_count = dataframe_name['LABEL'].value_counts()
    normalized_label_count = dataframe_name['LABEL'].value_counts(normalize=True)
    print(
        f'LABEL for {map_name}:\n{map_label_count}\n\nPercentage\n{normalized_label_count * 100}')


def map_compare_throws(dataframe_name, map_sucessful_throws, map_failed_throws, map_name):
    sns.countplot(x='LABEL', data=dataframe_name)
    plt.show()
    count_true = map_sucessful_throws['team'].value_counts().sort_index()
    normalized_count_true = map_sucessful_throws['team'].value_counts(normalize=True).sort_index()
    count_false = map_failed_throws['team'].value_counts()
    normalized_count_false = map_failed_throws['team'].value_counts(normalize=True)
    print(
        f'Successful throws on {map_name} for each team:\n{count_true}\n\nPercentage of sucessful throws\n{normalized_count_true * 100}\n'
        f'\nIncorrect throws on {map_name} for each team:\n{count_false}\n\nPercentage of failed throws\n{normalized_count_false * 100}')


def save_model(classifier, filename):
    pickle.dump(classifier, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


def standarize_data(data):
    return StandardScaler().fit_transform(data)
