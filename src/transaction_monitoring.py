import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class TransactionMonitor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = IsolationForest(contamination=0.01)  # Пример модели для обнаружения аномалий

    def preprocess_data(self):
        # Пример обработки данных
        scaler = StandardScaler()
        self.data.fillna(0, inplace=True)
        X = self.data.drop('transaction_id', axis=1)  # Предполагаем, что у вас есть идентификатор транзакции
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def train_model(self):
        X = self.preprocess_data()
        self.model.fit(X)
        print("Transaction monitoring model trained.")

    def predict(self, new_data):
        return self.model.predict(new_data)

    def detect_anomalies(self):
        X = self.preprocess_data()
        predictions = self.model.predict(X)
        anomalies = self.data[predictions == -1]  # Изоляционный лес возвращает -1 для аномалий
        return anomalies
