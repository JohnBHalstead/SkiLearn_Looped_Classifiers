#%%
# Requires your own X, y
# for loop to try 10 skilearn classification ML algrorithms
# measures accuracy on a validation and hold out set of data
# measures AUC on the validation data

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

my_model_results = []
for name, clf in zip(names, classifiers):
    clf.fit(X_tng, y_tr.values.ravel())
    score_val = clf.score(X_val, y_val.values.ravel())
    score_sc = clf.score(X_sc, y_sc.values.ravel()) # not needed in screening
    pred = clf.predict(X_val)  # not needed in screening
    fpr, tpr, thresholds = metrics.roc_curve(y_val.values.ravel(), pred, pos_label=2) # not needed in screening
    score_auc = metrics.auc(fpr, tpr) # not needed in screening
    my_model_results.append((name, score_val))
    print(my_model_results)

print("Modeling completed")
my_model_results = pd.DataFrame(my_model_results)
my_model_results.columns = ["Model", "Validation Data Accuracy"]
print(my_model_results.shape)
export2_csv = my_model_results.to_csv (r"/Users/jhalstead/Documents/data/project1Out/My_Model_Results.csv", header=True)
