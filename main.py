import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._search import BaseSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from adaptee import HyperBRKGASearchCV


def get_cv_time(method: BaseSearchCV):
    mean_fit_time = method.cv_results_['mean_fit_time']
    mean_score_time = method.cv_results_['mean_score_time']
    n_splits = method.n_splits_  # number of splits of training data
    n_iter = pd.DataFrame(method.cv_results_).shape[0]  # Iterations per split
    return np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter


def get_metrics(filename: str, hyperopt_method, is_hyperbrkga: bool, df: DataFrame):
    if is_hyperbrkga:
        score = hyperopt_method.cv_results_['best_param_score']
        time = hyperopt_method.cv_results_['total_time']
        params = str(hyperopt_method.cv_results_['best_param_decoded'])
    else:
        score = hyperopt_method.best_score_
        time = get_cv_time(hyperopt_method)
        params = str(hyperopt_method.best_params_)

    row = {
        'Score': score,
        'Time': time,
        'Param': params,
    }

    df = df.append(row, ignore_index=True)
    return df


def compute_mean_and_std(df, filename):
    means = {
        'Score': '',
        'Time': '',
        'Param': '',
    }
    for i in df.columns:
        if 'Param' not in i:
            means[i] = f'Média: {df[i].mean()}'

    std = {'Score': f'Desvio Padrão: {df["Score"].std()}'}

    df = df.append(means, ignore_index=True)
    df = df.append(std, ignore_index=True)

    df.to_csv(filename, mode='a')


def cred_preprocessing():
    # Definição dos nomes das variáveis (conforme a tabela contida no enunciado)
    colnames = ['ESCT', 'NDEP', 'RENDA', 'TIPOR', 'VBEM', 'NPARC',
                'VPARC', 'TEL', 'IDADE', 'RESMS', 'ENTRADA', 'CLASSE']
    # Leitura dos dados de treino
    arquivo = './experiments/credtrain.txt'
    data_train = pd.read_csv(arquivo, sep='\t', header=None, names=colnames)
    # Leitura dos dados de teste
    arquivo = './experiments/credtest.txt'
    data_test = pd.read_csv(arquivo, sep='\t', header=None, names=colnames)
    # Aplicação no conjunto de treinamento
    data_train_new = pd.get_dummies(data=data_train,
                                    prefix='ESCT',
                                    columns=['ESCT'],
                                    drop_first=True)
    data_test_new = pd.get_dummies(data=data_test, prefix='ESCT', columns=['ESCT'], drop_first=True)
    # Transformação da variável alvo do conjunto de treinamento e teste em vetor
    y_train = np.array(data_train_new['CLASSE'])
    y_test = np.array(data_test_new['CLASSE'])
    features_name_train = list(data_train_new.columns)  # nomes das colunas
    features_name_train.remove('CLASSE')  # remove variável "CLASSE"
    X_train = np.array(data_train_new.loc[:, features_name_train])  # Transformação em matriz de dados
    X_train = X_train.astype(float)
    # Transformação do conjunto de teste remanescente em matriz de dados
    features_name_test = list(data_test_new.columns)  # Nomes das colunas
    features_name_test.remove('CLASSE')  # Remoção da variável "CLASSE"
    X_test = np.array(data_test_new.loc[:, features_name_test])  # Transformação em matriz
    X_test = X_test.astype(float)
    # Criação do objeto para a padronização das features
    scaler = StandardScaler()
    # Ajustamento do StandardScaler ao conjunto de dados de treino e padronização dos dados de treino
    X_train = scaler.fit_transform(X_train)
    # Transformação dos dados de teste com os parâmetros ajustados a partir dos dados de treino
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def run_cred(data, algorithm: BaseEstimator, hyperopt_method, is_hyperbrkga: bool, parameters: dict, filename: str,
             df: DataFrame):
    X_train, y_train, X_test, y_test = data

    # Hyperparameter Optimization
    if is_hyperbrkga:
        hyper_opt = hyperopt_method(algorithm(), parameters=parameters, cv=5, data=X_train, target=y_train)
    else:
        hyper_opt = hyperopt_method(algorithm(), parameters, cv=5)

    # print(type(hyperopt_method), type(hyper_opt))
    hyper_opt.fit(X_train, y_train)

    return get_metrics(filename, hyper_opt, is_hyperbrkga, df)

    # algorithm.fit(X_train, y_train)
    # target_names = ['0', '1']
    # y_pred = algorithm.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    algorithms = [
        # LogisticRegression,
        # DecisionTreeClassifier,
        # RandomForestClassifier,
        # KNeighborsClassifier,
        XGBClassifier,
        MLPClassifier
    ]

    hyperparams = {
        "LogisticRegression": {
            'penalty': ('l1', 'l2', 'elasticnet', 'none'),
            'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        "DecisionTreeClassifier": {
            'criterion': ('gini', 'entropy', 'log_loss'),
            'splitter': ('best', 'random'),
            'max_depth': np.arange(0, 64),
            'min_samples_split': np.arange(0, 20),
            'min_samples_leaf': np.arange(0, 20)
        },
        "RandomForestClassifier": {
            'criterion': ('gini', 'entropy', 'log_loss'),
            'max_depth': np.arange(0, 64),
            'min_samples_split': np.arange(0, 20),
            'min_samples_leaf': np.arange(0, 20)
        },
        "KNeighborsClassifier": {
            'n_neighbors': np.arange(1, 30),
            'leaf_size': np.arange(1, 50),
            'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
            'p': [1, 2]
        },
        "XGBClassifier": {
            'max_depth': np.arange(0, 64),
            'max_leaves': np.arange(0, 64),
            'learning_rate': [0.001, 0.01, 0.125, 0.25, 0.5, 1],
            'grow_policy': ('depthwise', 'lossguide')
        },
        "MLPClassifier": {
            'hidden_layer_sizes': [(10, 30, 10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive']
        },
    }

    hyperopt_methods = [HyperBRKGASearchCV, GridSearchCV, RandomizedSearchCV]

    data = cred_preprocessing()

    for i in range(len(algorithms)):
        algorithm = algorithms[i]
        params = hyperparams[algorithm.__name__]
        for k in range(len(hyperopt_methods)):
            hom = hyperopt_methods[k]
            dataframe = DataFrame()
            filename = f"experiments/results/{algorithm.__name__}_{hom.__name__}_cred.csv"
            for _ in range(10):
                dataframe = run_cred(data, algorithm, hom, k == 2, params, filename, dataframe)

            compute_mean_and_std(dataframe, filename)
