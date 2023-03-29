from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

def get_models():
    binary_model = joblib.load('binary_model.pkl')
    multiclass_model = joblib.load('multiclass_model.pkl')
    return binary_model, multiclass_model

def save_new_trained_models(training_data):
    binary_model, multiclass_model = train_new_models(training_data)
    joblib.dump(binary_model, 'binary_model.pkl')
    joblib.dump(multiclass_model, 'multiclass_model.pkl')

def train_new_models(master_dataset):
    # Load pre-processed data
    master = pd.read_csv(master_dataset)
    master = master.drop(columns=['Unnamed: 0'])
    master = master.dropna(axis=0)

    y = master['label']
    X = master.drop('label', axis=1)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.20, random_state=0)

    xx = master.copy()
    a = xx['label'].replace(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A12',
                             'A13', 'A14', 'A15', 'A16', 'A17', 'F1', 'F2', 'F3', 'F4',
                             'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13'],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            inplace=False)
    b = xx
    b['label'] = a

    y_binary = b['label']
    X_binary = b.drop('label', axis=1)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_binary)
    Xbinary_train, Xbinary_test, ybinary_train, ybinary_test = train_test_split(X_std, y_binary, test_size=0.20,
                                                                                random_state=0)


    binary_model = RandomForestClassifier(n_estimators=78, criterion="entropy", oob_score=True)
    binary_model.fit(Xbinary_train, ybinary_train)

    multiclass_model = LogisticRegression()
    multiclass_model.fit(X_train, y_train)

    return binary_model, multiclass_model


