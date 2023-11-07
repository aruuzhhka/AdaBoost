# AdaBoost
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Загрузите ваш датасет
data = pd.read_csv("C:/Users/Lenovo/Documents/salarykz.csv")
# Замените запятые на точки в столбце 'Score' и преобразуйте его в числовой формат
data['Score'] = data['Score'].str.replace(',', '').astype(float)
# Выберите признаки (функции) и целевую переменную
X = data[['Rank', 'SubmissionCount']]
y = data['Score']
# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создайте модель AdaBoostRegressor
model = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)
# Обучите модель на обучающем наборе
model.fit(X_train, y_train)
# Сделайте прогнозы на тестовом наборе
y_pred = model.predict(X_test)
# Оцените качество модели
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
