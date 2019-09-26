import numpy as np
import pandas as pd
from Handler.Handler import Handler
from FE.FE import FeatureEng
from Plotter.Plotter import Plotter

# Feature Selection
plot = Plotter()
handler = Handler()
fe = FeatureEng()

# Le dados de resmple - Só pra não demorar muito
df_train, df_target = handler.get_resample()
df_target = np.ravel(df_target)

# Usar metodos mais avançados
df_selected_features, df_importances = fe.check_feature_importance(X=df_train, y=df_target)

# Ordena dados
df_importances.sort_values(by=['importance'], inplace=True)
df_importances['threshold'] = df_importances['importance'].mean()

plot.create_bar_plot_h(df=df_importances, x='variable', y='importance')
