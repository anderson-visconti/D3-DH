from sklearn.utils import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Scheduler(object):
    def __init__(self):
        self.pipeline = list()
        self.scores =[
            'accuracy',
            'r2',
            'roc_auc'
        ]

        pass

    def add_task(self, pipeline):

        self.pipeline.append(pipeline)

    def execute_scheduler(self, pipeline, param_grid, X_train, y_train):

        with parallel_backend('threading'):
            grid = RandomizedSearchCV(
                verbose=0,
                estimator=pipeline,
                scoring=self.scores,
                refit='accuracy',
                param_distributions=param_grid,
                #param_grid=param_grid,
                n_jobs=7,
                cv=5,
                return_train_score=True,
            )

            grid.fit(X=X_train, y=y_train)

            #print('Erro do treino: {:.2%}'.format(grid.best_score_))

        return grid

