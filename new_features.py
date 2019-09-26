from Handler.Handler import Handler
from FE.FE import FeatureEng

if __name__ == '__main__':
    handler = Handler()
    fe = FeatureEng()

    df_train, _, df_test = handler.get_selected_features()

    df_fe = fe.generate_feature(df=df_train)
    print(df_fe.head())
    print('d')