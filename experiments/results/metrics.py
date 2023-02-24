import pandas as pd

files = [
    # "DecisionTreeClassifier_GridSearchCV_cred.csv",
     # "DecisionTreeClassifier_HyperBRKGASearchCV_cred.csv",
     # "DecisionTreeClassifier_RandomizedSearchCV_cred.csv",
     # "KNeighborsClassifier_GridSearchCV_cred.csv",
     # "KNeighborsClassifier_HyperBRKGASearchCV_cred.csv",
     # "KNeighborsClassifier_RandomizedSearchCV_cred.csv",
     # "LogisticRegression_GridSearchCV_cred.csv",
     # "LogisticRegression_HyperBRKGASearchCV_cred.csv",
     # "LogisticRegression_RandomizedSearchCV_cred.csv",
     # "RandomForestClassifier_GridSearchCV_cred.csv",
     # "RandomForestClassifier_HyperBRKGASearchCV_cred.csv",
     # "RandomForestClassifier_RandomizedSearchCV_cred.csv",
     # "XGBClassifier_HyperBRKGASearchCV_cred.csv",
     # "LinearRegression_GridSearchCV_diamond.csv",
     # "LinearRegression_HyperBRKGASearchCV_diamond.csv",
     # "LinearRegression_RandomizedSearchCV_diamond.csv",
     # "KNeighborsRegressor_GridSearchCV_diamond.csv",
     # "KNeighborsRegressor_HyperBRKGASearchCV_diamond.csv",
     # "KNeighborsRegressor_RandomizedSearchCV_diamond.csv",
     # "Lasso_GridSearchCV_diamond.csv",
     # "Lasso_HyperBRKGASearchCV_diamond.csv",
     # "Lasso_RandomizedSearchCV_diamond.csv",
     # "DecisionTreeRegressor_HyperBRKGASearchCV_diamond.csv",
     # "DecisionTreeRegressor_GridSearchCV_diamond.csv",
     # "DecisionTreeRegressor_RandomizedSearchCV_diamond.csv",
    # "MLPClassifier_GridSearchCV_cred.csv",
    # "MLPClassifier_HyperBRKGASearchCV_cred.csv",
    # "MLPClassifier_RandomizedSearchCV_cred.csv",
    # "XGBClassifier_GridSearchCV_cred.csv",
    # "XGBClassifier_HyperBRKGASearchCV_cred.csv",
    # "XGBClassifier_RandomizedSearchCV_cred.csv",
    "MLPRegressor_GridSearchCV_diamond.csv",
    "MLPRegressor_HyperBRKGASearchCV_diamond.csv",
    "MLPRegressor_RandomizedSearchCV_diamond.csv",
 ]


hoa_enum = {"GridSearchCV": 0, "HyperBRKGASearchCV": 1, "RandomizedSearchCV": 2}
dataframe = [["OptMethod", "Algorithm", "Dataset", "ScoreMean", "ScoreStd", "TimeMean"]]


for file in files:
    df = pd.read_csv(file)
    df.columns = ["Index", "Score", "Time", "Params"]
    df = df.drop(columns=["Index"])
    matrix = df.tail(2).to_numpy()
    [score_mean, time] = matrix[0][:2]
    score_std = matrix[1][0]
    [algorithm, opt_method, dataset] = file.removesuffix(".csv").split("_")

    score_mean = score_mean.removeprefix("Média: ")
    time = time.removeprefix("Média: ")
    score_std = score_std.removeprefix("Desvio Padrão: ")

    dataframe.append([hoa_enum[opt_method], algorithm, dataset, score_mean, score_std, time])


pd.DataFrame(dataframe).to_csv("anex.csv")
# print(pd.DataFrame(dataframe))
