import pandas as pd
from helpers import map_records_count, map_percentage_of_throws, map_compare_throws, draw_histograms, plt, sns, save_model, standarize_data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
# Random forest
from sklearn.ensemble import RandomForestClassifier

# Setting global parameters for plots
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
pd.set_option('precision', 2)

## Loading data
# de_inferno dataset -> Inferno
inferno = pd.read_csv(r"train-grenades-de_inferno.csv")
# de_mirage dataset -> Mirage
mirage = pd.read_csv(r"train-grenades-de_mirage.csv")


# Droping extra primary key
inferno.drop(columns="Unnamed: 0", inplace=True)
mirage.drop(columns="Unnamed: 0", inplace=True)

# Concat to one dataframe
df = pd.concat([inferno, mirage])
df = df.reset_index(drop=True)

## Checking for missing values
inferno.isnull().values.any()
mirage.isnull().values.any()

# Checking for duplicates
df.drop_duplicates(subset=None, inplace=True)

print(f'Inferno: {df.shape}, Mirage: {df.shape}')

## Data exploration
# Creating dataframe for successed and failed throws
inferno_successful_throws = inferno[inferno['LABEL'] == 1]
inferno_failed_throws = inferno[inferno['LABEL'] == 0]

mirage_sucessful_throws = mirage[mirage['LABEL'] == 1]
mirage_failed_throws = mirage[mirage['LABEL'] == 0]

# ID and number of records
map_records_count(mirage)
map_records_count(inferno)

## Analysis of throws
# de_inferno LABEL
map_percentage_of_throws(inferno, 'inferno')
# de_mirage LABEL
map_percentage_of_throws(mirage, 'mirage')

### Histogram of Correct and Incorrect throws for inferno

map_compare_throws(inferno, inferno_successful_throws, inferno_failed_throws, 'inferno')
map_compare_throws(mirage, mirage_sucessful_throws, mirage_failed_throws, 'mirage')

# # preprocesing on data
df.drop(columns=['demo_id', 'demo_round_id', 'weapon_fire_id'], inplace=True)
df['team'] = pd.get_dummies(df['team'])
df['map_name'] = pd.get_dummies(df['map_name'])
df = pd.concat([df, pd.get_dummies(df['TYPE'])], axis=1).drop(columns=['TYPE'])
df['LABEL'] = df['LABEL'].astype(int)

## Features distribution
# draw_histograms(df, df.columns, 8, 4)
df['throw_detonate_time'] = df['detonation_tick'] / 128 - df['throw_tick'] / 128
df = df.drop(index=[df[(df['flashbang'] == 1) & (df['throw_detonate_time'] > 3)].index[0]])
start_x_point = df['throw_from_raw_x']
start_y_point = df['throw_from_raw_y']
start_z_point = df['throw_from_raw_z']
end_x_point = df['detonation_raw_x']
end_y_point = df['detonation_raw_y']
end_z_point = df['detonation_raw_z']

traveled_length_3D = ((start_x_point - end_x_point) ** 2 + (end_y_point - start_y_point) ** 2 + (
            end_z_point - start_z_point) ** 2) ** 0.5

traveled_length_2D = ((start_x_point - end_x_point) ** 2 + (end_y_point - start_y_point) ** 2) ** 0.5

df['pseudo_angle'] = traveled_length_2D / traveled_length_3D

pseudo_velocity = traveled_length_3D / df['throw_detonate_time']
df['pseudo_velocity'] = pseudo_velocity
df['traveled_length_3D'] = traveled_length_3D
df = df[df['pseudo_velocity'] < 1200]
df.drop(columns=['detonation_raw_z', 'detonation_raw_y', 'detonation_raw_x', 'throw_from_raw_z', 'throw_from_raw_y',
                 'throw_from_raw_x'], inplace=True)
df.reset_index(inplace=True, drop=True)

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax, cmap='coolwarm')
plt.show()


## Modeling
X = df.drop(['LABEL', 'throw_tick', 'detonation_tick', 'throw_detonate_time'], 1)
y = df['LABEL']
n_samples = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_columns = X_train.columns
X_train_scaled = standarize_data(X_train)
X_test_scaled = standarize_data(X_test)
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 150, 170],
             'max_features': [4, 5, 6, 7],
             'min_samples_split': [i * 2 for i in range(1, 6)]}
rf_cv_model = GridSearchCV(rf, rf_params, cv=7, n_jobs=-1, verbose=1).fit(X_train_scaled, y_train)
best_s = rf_cv_model.best_score_
print('Score: ', best_s)
best_params = rf_cv_model.best_params_
print('Best param: ', best_params)
rf = RandomForestClassifier(max_features=best_params['max_features'],
                            min_samples_split=best_params['min_samples_split'],
                            n_estimators=best_params['n_estimators']).fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title('RF Confusion Matrix')
plt.savefig('rf_con_mat')
plt.show()

feature_imp = pd.Series(rf.feature_importances_,
                        index=X_columns).sort_values(ascending=False)
plt.figure(figsize=(15, 13))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Fetaure Importance')
plt.ylabel('Features')
plt.title('Levels of Feature Importances')
plt.show()
tree = DecisionTreeRegressor(splitter='random')
clf = tree.fit(X_train_scaled, y_train)

y_pred_tree = tree.predict(X_test_scaled)
print(classification_report(y_test, y_pred_tree.round()))

cm = confusion_matrix(y_test, y_pred_tree.round())
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title('Tree Confusion Matrix')
plt.savefig('tree_con_mat')
plt.show()

cv = 2
cv_results = cross_validate(clf, X_train_scaled, y_train, scoring=['accuracy', 'f1_weighted', 'recall'], cv=cv,
                            return_estimator=True)

accuracy = [round(x, 2) for x in cv_results['test_accuracy']]
recall = [round(x, 2) for x in cv_results['test_recall']]
f1 = [round(x, 2) for x in cv_results['test_f1_weighted']]

scores_frame = pd.DataFrame({'Acurracy': accuracy, 'Recall': recall, 'F1': f1})
best = f1.index(max(f1))
tree_cv = cv_results['estimator']
tree_cv = tree_cv[best]

y_pred_tree_cv = tree_cv.predict(X_test_scaled)
print(classification_report(y_test, y_pred_tree_cv))

cm = confusion_matrix(y_test, y_pred_tree_cv)
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title('Tree_cv Confusion Matrix')
plt.savefig('tree_cv_con_mat')
plt.show()

# get importance
importance = tree.feature_importances_
feature_names = X.columns
# summarize feature importance
for i, v in enumerate(importance):
    print(f'{i} Feature: {feature_names[i]}, Score: {v}')
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# save_model(tree_cv, 'tree_classifier.sav')

#
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)
#
# y_pred_tree_cv = loaded_model.predict(X_test)
# print(classification_report(y_test, y_pred_tree_cv))
#
# cm = confusion_matrix(y_test, y_pred_tree_cv)
# sns.heatmap(cm, annot=True, fmt="d", cbar=False)
# plt.title('Tree_cv Confusion Matrix')
# plt.savefig('tree_cv_con_mat')
# plt.show()

