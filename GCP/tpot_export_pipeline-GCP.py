import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was:-1615.966779868347
exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.5, loss="ls", max_depth=3, max_features=0.05, min_samples_leaf=3, min_samples_split=3, n_estimators=100, subsample=0.35000000000000003)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
