import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

input_data = pd.read_csv("data/processed.hungarian.data", header=None)
input_data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
input_data.head()

#Number of neighbours = 4
knn = KNeighborsClassifier(n_neighbors=3)

#Replace missing values: "?" with -1
preprocessed_data = input_data.replace('?', -1)

#Drop features that contain missing values
x = preprocessed_data.drop(labels=['num'], axis=1)
y = preprocessed_data['num'].values

#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

knn.fit(x_train, y_train)
predicted_values = knn.predict(x_test)
score = knn.score(x_test, y_test)
print("Prediction Score: ", score)

#Cross validation using k = 10
cv_scores = cross_val_score(knn, x, y, cv=10)
print("Cross Validation Scores: ",  cv_scores)

scores = precision_recall_fscore_support(y_true=y_test, y_pred=predicted_values, average='macro')
print("Precision: ", scores[0], " Recall: ", scores[1])
print("Accuracy: ", accuracy_score(y_test, predicted_values))

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