import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
import json


class StackedModel:

    def __init__(self, df):

        self.string_cols = [c for c in list(df.columns) if df[c].dtype == 'object']
        self.features = [c for c in df.columns if c not in ['TARGET']]
        self.df = df

        columns_trans = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), self.string_cols)
            ],
            remainder='passthrough'
        )

        stacking_model = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_jobs=-1, class_weight=None)),
                ('lgbm', LGBMClassifier(verbose=-1, scale_pos_weight=5.935329462897532)),
                ('xgb', XGBClassifier(scale_pos_weight=7.5119405654163165))
            ],
            final_estimator=LogisticRegression()
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', columns_trans),
            ('classifier', stacking_model)
        ])

    
    def train_model(self, calibration_fraction=0.15):

        x_input = self.df[self.features]
        y_output = self.df["TARGET"]

        X_train, X_val, y_train, y_val = train_test_split(x_input, y_output, test_size=calibration_fraction, stratify=y_output)

        self.pipeline.fit(X_train, y_train.values.ravel())

        self.calibrated_pipeline = CalibratedClassifierCV(estimator=self.pipeline, method='isotonic', cv='prefit')
        self.calibrated_pipeline.fit(X_val, y_val)

        y_pred_proba_calibrated = self.calibrated_pipeline.predict_proba(X_val)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba_calibrated)

        beta = 1

        fbeta_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

        best_threshold = thresholds[np.argmax(fbeta_scores)]

        threshold_dict = {}
        threshold_dict['best_threshold'] = best_threshold

        with open("artifacts/threshold.json", "w") as json_file:
            json.dump(threshold_dict, json_file)

        return self.calibrated_pipeline
    
    def prediction_w_model(self, trained_model):

        threshold_dict = json.load(open('artifacts/threshold.json'))

        threshold = float(threshold_dict['best_threshold'])

        y_proba = trained_model.predict_proba(self.df[self.features])[:, 1]
        y_pred = np.where(y_proba >= threshold, 1, 0)

        return y_proba, y_pred
        



