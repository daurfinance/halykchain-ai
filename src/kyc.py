import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class KYCProcessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        # Пример обработки данных
        self.data.fillna(0, inplace=True)
        X = self.data.drop('label', axis=1)
        y = self.data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f'Model accuracy: {accuracy}')

    def predict(self, new_data):
        return self.model.predict(new_data)
