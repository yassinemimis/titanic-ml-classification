# Titanic Survival Prediction
# Author: Yassine KHERBOUCHE
# Dataset: Titanic - Kaggle

import os
import random
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# fix random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------------------------------------
# 1. load data
# -----------------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print("train shape:", train_df.shape)
print("test shape :", test_df.shape)
print("\nmissing values:\n", train_df.isnull().sum())
print("\nsurvival rate:", round(train_df['Survived'].mean(), 3))

# quick look at survival by sex
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('survival rate by sex')
plt.savefig(os.path.join(DATA_DIR, 'survival_by_sex.png'))
plt.close()

# -----------------------------------------------------------
# 2. preprocessing
# -----------------------------------------------------------
def preprocess(df, is_train=True):
    df = df.copy()

    # fill missing age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # fill missing embarked with most common value
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # fill missing fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # extract title from name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(
        {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # encode text columns
    le = LabelEncoder()
    df['Sex']      = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title']    = le.fit_transform(df['Title'])

    cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
            'FamilySize', 'IsAlone', 'Title']

    if is_train:
        return df[cols], df['Survived']
    return df[cols]


X, y         = preprocess(train_df, is_train=True)
X_test_final = preprocess(test_df,  is_train=False)

# -----------------------------------------------------------
# 3. train / validation split
# -----------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# -----------------------------------------------------------
# 4. train model
# I tried logistic regression and svm first but random forest
# gave better results on the validation set, especially for
# the minority class (survived=1). I also tuned max_depth
# between 3 and 10, and 6 gave the best f1.
# -----------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=4,
    random_state=SEED
)
model.fit(X_train_s, y_train)

# -----------------------------------------------------------
# 5. evaluate
# -----------------------------------------------------------
y_pred = model.predict(X_val_s)
y_prob = model.predict_proba(X_val_s)[:, 1]

print("\nvalidation results:")
print(classification_report(y_val, y_pred))

f1  = round(f1_score(y_val, y_pred), 4)
auc = round(roc_auc_score(y_val, y_prob), 4)
print(f"f1  : {f1}")
print(f"auc : {auc}")

# confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['not survived', 'survived'],
            yticklabels=['not survived', 'survived'])
plt.title('confusion matrix')
plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'))
plt.close()

# -----------------------------------------------------------
# 6. error analysis
# cases where the passenger survived but model predicted dead
# most of them are male passengers in class 3 traveling alone
# which makes sense because the model learned that pattern
# -----------------------------------------------------------
val_df = X_val.copy()
val_df['y_true'] = y_val.values
val_df['y_pred'] = y_pred

fn = val_df[(val_df['y_true'] == 1) & (val_df['y_pred'] == 0)]
print("\nfalse negatives (survived but predicted dead):")
print(fn[['Pclass', 'Sex', 'Age', 'FamilySize']].head(5))

# to fix this i tried lowering the decision threshold to 0.45
# it reduced false negatives but increased false positives
# so i kept the default 0.5 threshold

# -----------------------------------------------------------
# 7. cross validation
# -----------------------------------------------------------
cv_scores = cross_val_score(model, X_train_s, y_train,
                            cv=5, scoring='f1')
print(f"\ncv f1 scores : {cv_scores.round(4)}")
print(f"mean cv f1   : {cv_scores.mean().round(4)}")

# -----------------------------------------------------------
# 8. feature importance
# -----------------------------------------------------------
feat_imp = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='feature importance')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'feature_importance.png'))
plt.close()

# -----------------------------------------------------------
# 9. save model
# -----------------------------------------------------------
joblib.dump(model,  os.path.join(DATA_DIR, 'rf_titanic_v1.pkl'))
joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler_v1.pkl'))

print(f"\nmodel saved : rf_titanic_v1.pkl")
print(f"f1={f1} | auc={auc} | checkpoint=rf_titanic_v1.pkl")