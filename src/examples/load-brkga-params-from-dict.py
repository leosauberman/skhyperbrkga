from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../../../hbrkga_adaptee')

from hbrkga.brkga_mp_ipr.types_io import load_configuration_from_dict
from src.adaptee import HyperBRKGASearchCV

irisX, irisY = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(irisX, irisY, test_size=0.5, random_state=0)

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()

brkga_params = load_configuration_from_dict({
    "population_size": 5,
    "elite_percentage": 0.25,
    "mutants_percentage": 0.1,
    "num_elite_parents": 1,
    "total_parents": 2,
    "bias_type": "LOGINVERSE",
    "num_independent_populations": 1,
    "pr_number_pairs": 0,
    "pr_minimum_distance": 0.15,
    "pr_type": "PERMUTATION",
    "pr_selection": "BESTSOLUTION",
    "alpha_block_size": 1.0,
    "pr_percentage": 1.0,
    "exchange_interval": 200,
    "num_exchange_indivuduals": 2,
    "reset_interval": 600,
})

clf = HyperBRKGASearchCV(svc, parameters=parameters, data=X_train, target=y_train, brkga_params=brkga_params)

print("# Otimizando os hiperparâmetros para precisão\n")

clf.fit(X_train, y_train)

print("Melhor combinação de parâmetros encontrados no conjunto de treino:\n")
print(clf.cv_results_['best_param_decoded'])
print()
print("Scores do HyperBRKGA no conjunto de treino:\n")
print(clf.cv_results_)
