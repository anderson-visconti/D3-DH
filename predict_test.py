import os
import glob
import pandas as pd

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from Handler.Handler import Handler

handler = Handler()
# Load test data
_, X_test, _ = handler.get_data()

# Scale data
X_test = StandardScaler().fit(X=X_test)

for i, path in enumerate(glob.glob(pathname=os.path.join(r'Pickle/', '*.joblib'))):

    # load model
    model = joblib.load(path)

    y_hat_valid = pd.DataFrame(
        data={'predict_{:}'.format(os.path.basename(path=path)): model.predict(X_test)},
        index=X_test.index
    )
