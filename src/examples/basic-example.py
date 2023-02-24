from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from adaptee import HyperBRKGASearchCV

irisX, irisY = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(irisX, irisY, test_size=0.5, random_state=0)

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()
clf = HyperBRKGASearchCV(svc, parameters=parameters, data=X_train, target=y_train)

print("# Otimizando os hiperparâmetros para precisão\n")

clf.fit(X_train, y_train)

print("Melhor combinação de parâmetros encontrados no conjunto de treino:\n")
print(clf.cv_results_['best_param_decoded'])
print()
print("Scores do HyperBRKGA no conjunto de treino:\n")
print(clf.cv_results_)
