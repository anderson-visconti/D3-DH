import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split

from Handler.Handler import Handler
from FE.FE import FeatureEng
from Plotter.Plotter import Plotter
from Models.Models import *
from Models.Scheduler import Scheduler

# Definir se os dados
if __name__ == '__main__':
    models = dict(
        NaiveBayes=NaiveBayes(),
        LightGbm=LightGbm(),
        #AdaBoost=AdaBoost(),
        #RandomForest=RandomForest(),
        #RegLogistica=Logistic(),
    )


    handler = Handler()
    plot = Plotter()
    fe = FeatureEng()

    # Le dados de resmple - Só pra não demorar muito
    df_selected_features, df_target, X_test = handler.get_selected_features()
    df_target = np.ravel(df_target)

    # Feature Enge
    df_selected_features = fe.generate_feature(df=df_selected_features)
    X_test = fe.generate_feature(df=X_test)


    # Divisão do datset para validacao do modelo
    X_train, X_valid, y_train, y_valid = train_test_split(
        df_selected_features,
        df_target,
        random_state=42,
        train_size=0.8,
        test_size=0.2,
    )

    # Treinamento dos modelos
    scheduler = Scheduler()

    # Executa vários modelos - Stacking de Modelos

    df_params = pd.DataFrame()
    erros = dict()
    stack = ''
    n = 0
    for key, value in models.items():

        if len(stack) != 0:
            stack = stack + '_{:}'.format(key)

        else:
            stack = '{:}'.format(key)

        print('\nTunnando modelo {:}\n'.format(key))

        grid = scheduler.execute_scheduler(
            pipeline=value.pipeline,
            param_grid=value.param_grid,
            X_train=X_train,
            y_train=y_train,
        )

        aux = pd.DataFrame(data=grid.cv_results_)
        df_params = pd.DataFrame(pd.concat(objs=[df_params, aux]))
        df_params.to_csv(path_or_buf=r'Models/params_cv_{:}.csv'.format(stack), index=False)


        # Fazendo predicoes para X_train e X_valid e X_test
        y_hat_train = pd.DataFrame(
            data={'predict_{:}'.format(key): grid.best_estimator_.predict(X_train)},
            #data={'predict_{:}'.format(key): grid.best_estimator_.predict_proba(X_train)},
            index=X_train.index
        )

        y_hat_valid = pd.DataFrame(
            data={'predict_{:}'.format(key): grid.best_estimator_.predict(X_valid)},
            #data={'predict_{:}'.format(key): grid.best_estimator_.predict_proba(X_valid)},
            index=X_valid.index
        )

        y_hat_test = pd.DataFrame(
            data={
                'ID_code': ['test_{:}'.format(i) for i in range(X_test.shape[0])],
                'predict_{:}'.format(key): grid.best_estimator_.predict(X_test),
                #'predict_{:}'.format(key): grid.best_estimator_.predict_proba(X_test),
            },
            index=X_test.index,
        )

        erro = grid.best_estimator_.score(X_valid, y_valid)

        # Stacking de modelos
        erros.update({key: dict(train=grid.best_score_, valid=erro)})

        # Stack da previsao dos dados
        X_train = pd.concat(objs=[X_train, y_hat_train], axis=1, ignore_index=True)
        X_valid = pd.concat(objs=[X_valid, y_hat_valid], axis=1, ignore_index=True)
        X_test = pd.concat(objs=[X_test, y_hat_test['predict_{:}'.format(key)]], axis=1, ignore_index=True)

        # Print da curva de treino
        plot_config = dict(
            title=stack,
        )

        plot.create_learning_plot(
            estimator=grid.best_estimator_,
            X=X_train,
            y=y_train,
            cv=5,
            n_jobs=5,
            plot_config=plot_config
        )

        # Salva modelo treinado
        dump(value=grid.best_estimator_, filename=r'Pickle/{:}_{:}.joblib'.format(n, stack))
        n = n + 1
        print(erros)



    # Dados Finais

    #plot.plot_confusion_matrix(y_pred=y_hat_valid, y_true=y_valid, title=stack)

    # Previsao probabilistica
    y_proba = grid.best_estimator_.predict_proba(X_test.iloc[:, 0:-1])

    y_hat_test_proba = pd.DataFrame(
        data=dict(
            ID_code=['test_{:}'.format(i) for i in range(X_test.shape[0])],
            target=X_test.iloc[:, -1],
            p_0=y_proba[:, 0],
            p_1=y_proba[:, 1],
        )
    ).to_csv('proba.csv')

    # Salvando informações para submissão ao Kaggle
    df_submission = pd.DataFrame(
        data=dict(
            ID_code=['test_{:}'.format(i) for i in range(X_test.shape[0])],
            target=y_proba[:, 1],
        ),
    )

    df_submission.to_csv(path_or_buf=r'Data/submission_{:}.csv'.format(stack), index=False)

    print('Cabooooooooooooou')