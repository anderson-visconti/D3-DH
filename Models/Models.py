import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

class Modelos(object):

    def __init__(self):
        pass

    def fit_modelo(self):
        pass

    def eval_score(self):
        pass

    def add_task(self, pipe):
        pass


class LinSvmC(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=False,
            steps=[
                ('scale', StandardScaler()),
                ('LinSvm', LinearSVC(
                    verbose=True,
                    random_state=42,
                    max_iter=500,
                    #max_iter=1000
                )
                 ),
            ]
        )

        self.param_grid = {
            #'classify__kernel': ['linear'],
            #'dim_reduction__n_components': [i for i in range(2, 22)]

        }


class SvmC(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=True,
            steps=[
                ('scale', StandardScaler()),
                ('Svm', SVC(
                    verbose=False,
                    random_state=42,
                    max_iter=500,
                    #max_iter=1000
                )
                 ),
            ]
        )

        self.param_grid = {
            #'classify__kernel': ['linear'],
            #'dim_reduction__n_components': [i for i in range(2, 22)]

        }


class Logistic(Modelos):


    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=True,
            steps=[
                ('scale', StandardScaler()),
                ('Logistic', LogisticRegressionCV(verbose=0, random_state=42, n_jobs=-1, cv=5)),
            ]
        )

        self.param_grid = {
            #'classify__kernel': ['rbf', 'poly']
        }


class NaiveBayes(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=False,
            steps=[
                ('scale', QuantileTransformer(output_distribution='normal')),
                ('NaiveBayes', GaussianNB()),
            ]
        )

        self.param_grid = {
            #'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #'classify__kernel': ['rbf']
        }


class LightGbm(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=True,
            steps=[
                ('scale', StandardScaler()),
                ('LightGbm', LGBMClassifier(
                    n_jobs=7,
                    silent=True,
                    #n_estimators=50,
                    #learning_rate=0.1,

                )
                 ),
            ]
        )

        self.param_grid = {
            #'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'LightGbm__learning_rate': np.linspace(start=0.01, stop=0.5, num=10),
            'LightGbm__n_estimators': [50, 75, 100],
            'LightGbm__objective': ['binary'],
            'LightGbm__n_jobs': [7],

        }


class Gbm(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=False,
            steps=[
                ('scale', StandardScaler()),
                ('Gbm', GradientBoostingClassifier(verbose=0, n_estimators=100)),
            ]
        )

        self.param_grid = {
            #'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #'classify__kernel': ['rbf']
        }


class RandomForest(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=True,
            steps=[
                ('scale', StandardScaler()),
                ('RandomForest', ExtraTreesClassifier(n_estimators=50, verbose=False, n_jobs=7)),
            ]
        )

        self.param_grid = {
            #'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #'classify__kernel': ['rbf']
        }


class AdaBoost(Modelos):

    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline(
            verbose=False,
            steps=[
                ('scale', StandardScaler()),
                ('AdaBoost', AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=0.5)),
            ]
        )

        self.param_grid = {
            #'reduce_dim__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #'classify__kernel': ['rbf']
        }
