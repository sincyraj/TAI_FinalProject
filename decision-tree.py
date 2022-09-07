# Load libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

input_data = pd.read_csv("data/processed.cleveland.data", header=None)
input_data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
input_data.head()

# Replace missing values: "?" with median
preprocessed_data = input_data.fillna(input_data.median())
preprocessed_data = preprocessed_data.replace({'num': {2: 1, 3: 1, 4: 1}})

#Split into features and target
feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
x = preprocessed_data[feature_cols]
y = preprocessed_data["num"]

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
dtc = DecisionTreeClassifier()

#Train Decision Tree Classifer
dtc = dtc.fit(x_train, y_train)

#Predict the response for test dataset
predicted_values = dtc.predict(x_test)
print("Predicted values: ", predicted_values)
score = dtc.score(x_test, y_test)
print("Prediction Score: ", score)

#Cross validation using k = 10
cv_scores = cross_val_score(dtc, x, y, cv=10)
print("Cross Validation Scores: ",  cv_scores)
plt.figure(figsize=(16,7))
plt.title('Cross Validation results for Decision tree', fontsize=20)
plt.ylabel('Scores', fontsize=18)
plt.xlabel('Folds', fontsize=18)
plt.plot(cv_scores, marker='.', linewidth=2, markersize=16)
plt.show()

scores = precision_recall_fscore_support(y_true=y_test, y_pred=predicted_values, average='macro')
print("Precision: ", scores[0], " Recall: ", scores[1])
print("Accuracy: ", accuracy_score(y_test, predicted_values))

print(f'\nClassification Report for Decision Tree \n')

cm = confusion_matrix(y_true=y_test, y_pred=predicted_values, normalize='all')
cmd = ConfusionMatrixDisplay(cm,  display_labels=dtc.classes_)
cmd.plot()
plt.tight_layout()
plt.show()

#create ROC curve
false_pos_rate, true_pos_rate, _ = metrics.roc_curve(y_test, predicted_values)
plt.plot(true_pos_rate, false_pos_rate)
plt.xlabel('True +ve rate')
plt.ylabel('False +ve rate')
plt.show()

#Plot Decision Tree
plot_tree(dtc, filled=True)
plt.title("Resulting decision tree")
plt.show()

hyperparameters = dict(max_depth = range(1,50,5), min_samples_leaf= range(5,100,5), criterion= ["gini", "entropy"])

#Create new KNN object
dtc_tuned = DecisionTreeClassifier()
#GridSearch using the hyperparameters
clf = GridSearchCV(dtc_tuned, hyperparameters, cv=10)

#Fit the model
best_model = clf.fit(x,y)
#Print The value of best Hyperparameters
print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
print('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
print('Best criterion:', best_model.best_estimator_.get_params()['criterion'])



