# model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Load data
df = pd.read_csv('wine.csv')

# Feature columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']

# Prepare data for KNN (predict quality score)
X = df[features]
y_knn = df['quality']

# Prepare data for SVM (binary classification: quality >5)
y_svm = df['quality'] > 5

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_scaled, y_knn)

# Train SVM model
svm = SVC(C=1.0, kernel='rbf', gamma='auto', random_state=42)
svm.fit(X_scaled, y_svm)

# Save models and scaler
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(svm, 'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')