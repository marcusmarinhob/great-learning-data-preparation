# Libraries to help with reading and manipulating

import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries to tune model, get different metric scores, and split data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Libraries to impute missing values
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline

# Libraries to build a logistic refression model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from xgboost import XGBClassifier

# Library to supress the warning
import warnings
warnings.filterwarnings('ignore')

# Reading CSV file
job = pd.read_csv('jobs_data.csv')

# copying data to another variable to avoid any changes to original data
data = job.copy()
