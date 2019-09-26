import os
import numpy as np
import pandas as pd
import imblearn
import featuretools as ft
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureEng(object):
    def __init__(self):
        pass

    def create_over_sampling(self, x, y, mode='Random'):
        if mode == 'SMOTE':
            x_resample, y_resample = imblearn.over_sampling.SMOTE(n_jobs=os.cpu_count()).fit_resample(X=x, y=y)

        elif mode == 'Random':
            random = imblearn.over_sampling.RandomOverSampler(random_state=42)
            x_resample, y_resample = random.fit_resample(X=x, y=y)


        return x_resample, y_resample


    def generate_features_feature_tools(self, df):

        # Make an entityset and add the entity
        es = ft.EntitySet(id='santander')
        es.entity_from_dataframe(
            entity_id='data',
            dataframe=df,
            make_index=True,
            index='index'
        )

        # Run deep feature synthesis with transformation primitives
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_entity='data',
            n_jobs=os.cpu_count() - 3,
            agg_primitives=["mean", "max", "min", "std", "skew"],
            trans_primitives=['add_numeric'],
            max_depth=1,
            verbose=True,
            max_features=500
        )

        feature_matrix.to_csv('Data/features_matrix.csv')
        return feature_matrix


    def check_feature_importance(self, X, y):

        forest = ExtraTreesClassifier(
            verbose=True,
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )

        forest.fit(X, np.ravel(y))
        importances = forest.feature_importances_


        df_importances = pd.DataFrame(
            data=dict(
                variable=[i for i in range(200)],
                importance=importances
            )
        )

        df_importances['treshold'] = df_importances['importance'].mean()

        select = SelectFromModel(forest, prefit=True)
        selected_features = X.columns[(select.get_support())]

        df_selected_features = pd.DataFrame(
            data=select.transform(X),
            columns=selected_features
        )

        df_selected_features.to_csv(
            path_or_buf=r'Data/selected_features.csv',
            index=False
        )

        df_importances.to_csv(
            path_or_buf=r'Data/features_importances.csv',
            index=False

        )
        return df_selected_features, df_importances


    def generate_feature(self, df):
        df = pd.DataFrame(df)

        for i, col in enumerate(df.columns):

            uniques = df[col].value_counts()
            df['var_{:}_freq'.format(i)] = df[col].map(uniques).astype(np.int16)
            df['var_{:}_max'.format(i)] = df[col].max()
            df['var_{:}_min'.format(i)] = df[col].min()
            df['var_{:}_mean_c'.format(i)] = df[col].mean()
            df['var_{:}_skew'.format(i)] = df[col].skew()
            df['var_{:}_kurt'.format(i)] = df[col].kurtosis()

        return df


    def get_real_data_test(self, df_test):

        unique_samples = []
        unique_count = np.zeros_like(df_test)
        for feature in range(df_test.shape[1]):
            _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
            unique_count[index_[count_ == 1], feature] += 1

        # Samples which have unique values are real the others are fake
        real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
        synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

        print('Found', len(real_samples_indexes), 'real test')
        print('Found', len(synthetic_samples_indexes), 'fake test')

        ###################

        d = {}
        for i in range(df_test.shape[1]): d['var_' + str(i)] = 'float32'
        d['target'] = 'uint8'
        d['ID_code'] = 'object'

        return df_test