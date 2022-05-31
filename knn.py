import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

input_data = pd.read_csv("data/processed.hungarian.data", header=None)
input_data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
input_data.head()
results = input_data.groupby(['age']).size()

ax = input_data.groupby(['age']).size().plot(kind='bar', title ="Age Distribution", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Counts", fontsize=12)
ax.get_legend().remove()
plt.show()

ax = input_data.groupby(['sex']).size().plot(kind='bar', title ="Sex Distribution", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Sex", fontsize=12)
ax.set_ylabel("Counts", fontsize=12)
ax.get_legend().remove()
plt.show()

#Replace missing values: "?" with -1
preprocessed_data = input_data.replace('?', -1)

#Drop features that contain missing values
x = preprocessed_data.drop(labels=['num'], axis=1)
y = preprocessed_data['num'].values

#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

#Number of neighbours = 4
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

predicted_values = knn.predict(x_test)
score = knn.score(x_test, y_test)
print("Prediction Score: ", score)

#Cross validation using k = 10
cv_scores = cross_val_score(knn, x, y, cv=10)
print("Cross Validation Scores: ",  cv_scores)
plt.figure(figsize=(16,7))
plt.title('Cross Validation results for KNN', fontsize=20)
plt.ylabel('Scores', fontsize=18)
plt.xlabel('Folds', fontsize=18)
plt.plot(cv_scores, marker='.', linewidth=2, markersize=16)
plt.show()


scores = precision_recall_fscore_support(y_true=y_test, y_pred=predicted_values, average='macro')
print("Precision: ", scores[0], " Recall: ", scores[1] ,"Accuracy: ", accuracy_score(y_test, predicted_values))



print(f'\nClassification Report for KNN S\n')


print('Micro Precision: {:.2f}'.format(precision_score(y_test, predicted_values, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, predicted_values, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, predicted_values, average='micro')))
print('Macro Precision: {:.2f}'.format(precision_score(y_test, predicted_values, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, predicted_values, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, predicted_values, average='macro')))
print('Weighted Precision: {:.2f}'.format(precision_score(y_test, predicted_values, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, predicted_values, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, predicted_values, average='weighted')))




cm = confusion_matrix(y_true=y_test, y_pred=predicted_values, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=['Value 0', 'Value 1'])
cmd.plot()
plt.tight_layout()
plt.show()

#create ROC curve
false_pos_rate, true_pos_rate, _ = metrics.roc_curve(y_test, predicted_values)
plt.plot(false_pos_rate, true_pos_rate)
plt.ylabel('True +ve rate')
plt.xlabel('False +ve rate')
plt.show()