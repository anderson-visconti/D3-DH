import numpy as np
import pandas as pd
from Handler.Handler import Handler
from FE.FE import FeatureEng

handler = Handler()
fe = FeatureEng()

# Leitura dos dados
df_train, df_test, df_target = handler.get_data()
df_target = np.array(df_target).ravel()



# Fazendo Over-Sampling
fe = FeatureEng()
df_train, df_target = fe.create_over_sampling(x=df_train, y=df_target, mode='SMOTE')

# Salvando dados
pd.DataFrame(df_train).to_csv(
    path_or_buf='Data/X_resample.csv',
    index=False,
    header=['var_{:}'.format(i) for i in range(df_train.shape[1])]
)
pd.DataFrame(df_target).to_csv(
    path_or_buf='Data/y_resample.csv',
    index=False,
    header=['target']
)

print('f')
