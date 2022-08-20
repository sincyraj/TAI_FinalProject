import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

input_data = pd.read_csv("data/processed.cleveland.data", header=None)
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

# Replace missing values: "?" with median
preprocessed_data = input_data.fillna(input_data.median())
preprocessed_data = preprocessed_data.replace({'num': {2: 1, 3: 1, 4: 1}})

#Drop features that contain missing values
x = preprocessed_data.drop(labels=['num'], axis=1)
y = preprocessed_data['num'].values

#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

#Number of neighbours = 4
knn = KNeighborsClassifier(n_neighbors=3)
#knn = KNeighborsClassifier(n_neighbors=7, leaf_size=5, p=1)
knn.fit(x_train, y_train)

predicted_values = knn.predict(x_test)
print("Predicted values: ", predicted_values)
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

cm = confusion_matrix(y_true=y_test, y_pred=predicted_values, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=knn.classes_)
cmd.plot()
plt.tight_layout()
plt.show()

#create ROC curve
false_pos_rate, true_pos_rate, _ = metrics.roc_curve(y_test, predicted_values)
plt.plot(true_pos_rate, false_pos_rate)
plt.xlabel('True +ve rate')
plt.ylabel('False +ve rate')
plt.show()

leaf_size = list(range(1,20))
n_neighbors = list(range(1,20))
p=[1,2] # Manhattan or Euclidean distance
# Hyper parameters to be tuned
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#Create new KNN object
knn_tuned = KNeighborsClassifier()
#GridSearch using the hyperparameters
clf = GridSearchCV(knn_tuned, hyperparameters, cv=10)

#Fit the model
best_model = clf.fit(x,y)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])