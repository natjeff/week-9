import pandas as pd
import numpy as np

class GroupEstimate:
    """
    A class to estimate a target variable (y)
    based on mean or median from data.
    """

    def __init__(self, estimate='mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.group_columns_ = None

    def fit(self, X, y):
        """
        Fits the GroupEstimate model by computing mean or median estimates for each group.
        """
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if y.isnull().any():
            raise ValueError("y contains missing values, which are not allowed.")

        df = pd.concat([X, y.rename('target')], axis=1)
        grouped = df.groupby(list(X.columns))['target']

        if self.estimate == 'mean':
            group_estimates = grouped.mean()
        else:
            group_estimates = grouped.median()

        self.group_estimates_ = group_estimates
        self.group_columns_ = list(X.columns)

        return self

    def predict(self, X_):
        """
        Predicts estimated values based on estimates.
        """
        if self.group_estimates_ is None:
            raise RuntimeError("You must fit the model before predicting.")

        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.group_columns_)

        if list(X_.columns) != self.group_columns_:
            raise ValueError("Input columns do not match those used during fit().")

        keys = [tuple(row) for row in X_.to_numpy()]
        predictions = []

        for key in keys:
            if key in self.group_estimates_.index:
                predictions.append(self.group_estimates_.loc[key])
            else:
                predictions.append(np.nan)

        predictions = np.array(predictions)
        missing_count = np.isnan(predictions).sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} observation(s) fell into missing group(s).")

        return predictions

# Load data and choose columns
df = pd.read_csv("Data_Science_Survey.csv")
categorical_cols = ['Primary streaming service', 'Fav genre']
target_col = 'Anxiety'

# Clean data
df_clean = df.dropna(subset=categorical_cols + [target_col])
X = df_clean[categorical_cols]
y = df_clean[target_col]
gm = GroupEstimate(estimate='mean')
gm.fit(X, y)

# Display results
print("Group-level estimates (mean anxiety by streaming service + genre):")
print(gm.group_estimates_.head(), "\n")
X_new = [
    ['Spotify', 'Pop'],
    ['Apple Music', 'Rap'],
    ['YouTube Music', 'Rock'],
    ['Pandora', 'Jazz']
]

# Predict and display new results
predictions = gm.predict(X_new)
print("Predictions for new observations:")
for obs, pred in zip(X_new, predictions):
    print(f"Input: {obs}  →  Estimated Anxiety: {pred}")
