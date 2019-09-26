import pandas as pd
import numpy as np
from Handler.Handler import Handler
from EDA.EDA import Eda
from Plotter.Plotter import Plotter

plot = Plotter()
handler = Handler()
df_train, _, df_target = handler.get_data(frac=1.0)

df_target = pd.DataFrame(df_target)
df_train = pd.DataFrame(df_train)

# EDA
eda = Eda()

# Análise básica
d = eda.get_describe(df=df_train)
eda.check_imbalance(var_name='target', df=df_target)



