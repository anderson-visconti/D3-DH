import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils import parallel_backend

class Plotter(object):
    def __init__(self):
        pio.renderers.default = 'browser'
        pass

    def create_bar_plot_h(self, df, x, y):

        fig = make_subplots(rows=1, cols=2)

        trace = go.Bar(
            x=df[y],
            y=['var_{:}'.format(i) for i in df[x]],
            orientation='h',
            name='Todas',
        )

        treshold = go.Scatter(
            x=df['threshold'],
            y=['var_{:}'.format(i) for i in df[x]],
            orientation='h',
            name='Threshold',
        )

        df_selected = pd.DataFrame(df.loc[df['importance'] >= df['threshold'].mean()])

        important = go.Bar(
            x=df_selected[y],
            y=['var_{:}'.format(i) for i in df_selected[x]],
            orientation='h',
            name='Selecionadas',
        )


        fig.add_trace(trace, row=1, col=1)
        fig.add_trace(treshold, row=1, col=1)
        fig.add_trace(important, row=1, col=2)

        fig.update_layout(title_text="Seleção de Variáveis")
        fig.show()
        pass


    def plot_histogram(self, df, feature_name):

        fig = px.histogram(
            data_frame=df,
            x=feature_name,
            y=feature_name,
            color=feature_name,
        )

        fig.show()


    def plot_feature_scatter(self, df, features=[]):

        fig = px.scatter_matrix(
            data_frame=df,
            dimensions=features
        )
        fig.show()


    def create_heatmap_plotly(self, data, config_plot=[]):
        data = pd.DataFrame(data)

        fig = go.Figure(
            data=go.Heatmap(
                x=data.columns,
                y=data.columns,
                z=data.values,
                type='heatmap',
                colorscale='RdBu',
                zmin=-0.2,
                zmax=0.2,
                zmid=0,
                #autocolorscale=False
            )
        )

        fig.show()
        return


    def create_learning_plot(self, estimator, X, y, cv, n_jobs, plot_config):

        plt.figure()
        plt.title(plot_config['title'])

        #if ylim is not None:
        #    plt.ylim(*ylim)

        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(
            verbose=1,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=np.linspace(0.1, 0.5, 5)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1,
            color="r"
        )

        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
        )

        plt.plot(
            train_sizes,
            train_scores_mean,
            'o-',
            color="r",
            label="Training score"
        )

        plt.plot(
            train_sizes,
            test_scores_mean,
            'o-',
            color="g",
            label="Cross-validation score"
        )

        plt.legend(loc="best")
        plt.savefig(r'Fig/training_plot_{:}.png'.format(plot_config['title']))


    def plot_confusion_matrix(self, y_pred, y_true, title):

        df = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred.iloc[:, 0]))

        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=[i for i in df.index],
                y=[i for i in df.columns],
                colorscale='RdBu',
                text=df.values,
                #textposition='center'
            )
        )

        fig.update_layout(title_text='Matriz de Confusão')
        fig.show()