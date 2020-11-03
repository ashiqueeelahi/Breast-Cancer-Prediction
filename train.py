import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;

import keras;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential

a= pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv');
a.shape

a= pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv');
a.shape

sns.heatmap(a.corr(), annot = True)

x_a = a.drop(columns = 'diagnosis', axis = 1);
y_a = a[['diagnosis']];
x_train, x_test, y_train, y_test = train_test_split(x_a,y_a, test_size = 0.2, random_state = 55);

d = RandomForestClassifier()
d.fit(x_train, y_train)

d.score(x_test, y_test)

xx = XGBClassifier();
xx.fit(x_train, y_train)

xx.score(x_test, y_test)

sc =  StandardScaler();
sc.fit_transform(x_train);
x_train_sc = sc.fit_transform(x_train);
x_test_sc = sc.fit_transform(x_test);
sv = SVC(kernel='rbf');
sv.fit(x_train_sc, y_train)


SVC()

sv.score(x_test_sc,y_test)

