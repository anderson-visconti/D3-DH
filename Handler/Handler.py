import numpy as np
import pandas as pd


class Handler(object):

    def __init__(self):
        pass

    def get_data(self, frac=1.0):

        df_train = pd.read_csv(
            filepath_or_buffer=r'Data/train.csv',
            sep=',',
            decimal='.',
            dtype={'var_{:}'.format(k): np.float32 for k in range(200)}
        )

        df_target = pd.DataFrame(df_train[['target']])
        df_target['target'] = df_target['target'].astype(np.int8)

        df_test = pd.read_csv(
            filepath_or_buffer=r'Data/test.csv',
            sep=',',
            decimal='.',
            dtype={'var_{:}'.format(k): np.float32 for k in range(200)}
        )

        # Remove Id e Target
        df_train = df_train.drop(columns=['ID_code', 'target'])


        return df_train, df_test, df_target


    def get_resample(self):

        df_train = pd.read_csv(filepath_or_buffer=r'Data/X_resample.csv')
        df_target = pd.read_csv(filepath_or_buffer=r'Data/y_resample.csv')

        for col in df_train.columns:
            df_train[col] = df_train[col].astype(np.float32)

        for col in df_target.columns:
            df_target[col] = df_target[col].astype(np.int8)


        return df_train, df_target


    def get_selected_features(self):

        df_selected_features = pd.read_csv(filepath_or_buffer=r'Data/selected_features.csv')

        df_target = pd.read_csv(filepath_or_buffer=r'Data/y_resample.csv')

        df_test = pd.read_csv(filepath_or_buffer=r'Data/test.csv', usecols=df_selected_features.columns)

        # Reducao de memória
        for col in df_test.columns:

            df_test[col] = df_test[col].astype(np.float32)
            df_selected_features[col] = df_selected_features[col].astype(np.float32)

        for col in df_target.columns:
            df_target[col] = df_target[col].astype(np.int8)

        return df_selected_features, df_target, df_test


    def reduce_mem_usage(self, props):

        start_mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        NAlist = []  # Keeps track of columns that have missing values filled in.
        for col in props.columns:
            if props[col].dtype != object:  # Exclude strings

                # Print current column type
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", props[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all():
                    NAlist.append(col)
                    props[col].fillna(mn - 1, inplace=True)

                    # test if column can be converted to an integer
                asint = props[col].fillna(0).astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            props[col] = props[col].astype(np.uint8)
                        elif mx < 65535:
                            props[col] = props[col].astype(np.uint16)
                        elif mx < 4294967295:
                            props[col] = props[col].astype(np.uint32)
                        else:
                            props[col] = props[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props[col] = props[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props[col] = props[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props[col] = props[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props[col] = props[col].astype(np.int64)

                            # Make float datatypes 32 bit
                else:
                    props[col] = props[col].astype(np.float32)

                # Print new column type
                print("dtype after: ", props[col].dtype)
                print("******************************")

        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
        return props, NAlist