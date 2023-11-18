import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Assuming 'data' is loaded
# data = pd.read_csv("path_to_your_data.csv")

# Advanced Time-Series Feature Engineering
data['lag1'] = data['transaction_amount'].shift(1)
data['lag7'] = data['transaction_amount'].shift(7)
data['rolling_median'] = data['transaction_amount'].rolling(window=48).median()
data['rolling_skew'] = data['transaction_amount'].rolling(window=48).skew()
data['rolling_kurt'] = data['transaction_amount'].rolling(window=48).kurt()

# SMOTE for Imbalance
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Hyperparameter Tuning
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
rf = RandomForestClassifier()
rf_cv = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1, random_state=42)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_

# Neural Network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

# Ensemble
clf1 = best_rf
clf2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)

def nn_proba_extractor(m):
    def predict_proba(X):
        return m.predict(X)
    return predict_proba
model.predict_proba = nn_proba_extractor(model)
clf3 = model

eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('nn', clf3)], voting='soft')
eclf.fit(X_train, y_train)

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_train)

# Model Evaluation
y_pred = eclf.predict(X_test)
y_pred_proba = eclf.predict_proba(X_test)
roc_value = roc_auc_score(y_test, y_pred_proba[:,1])
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_value)
