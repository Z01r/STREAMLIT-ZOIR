import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Функция для построения ROC-кривой
def plot_roc_curve(y_test, y_pred_proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label='ROC-кривая (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Линия случайного выбора
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложноположительная ставка (FPR)')
    plt.ylabel('Истинноположительная ставка (TPR)')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Заголовок приложения
st.title("Модель классификации для выбора детского сада")

# Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "decision"]
data = pd.read_csv(url, header=None, names=columns)

# Отображение данных
st.subheader("Данные")
st.write(data)

# Кодирование категориальных признаков
data_encoded = pd.get_dummies(data, drop_first=True)

# Разделение на признаки и целевую переменную
X = data_encoded.drop('decision_recommend', axis=1)
y = data_encoded['decision_recommend']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Прогнозирование
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Оценка модели
st.subheader("Оценка модели")
st.write("Матрица ошибок:")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Отчет по классификации
st.write("Отчет по классификации:")
st.text(classification_report(y_test, y_pred))

# ROC AUC и визуализация ROC-кривой
if len(np.unique(y_pred)) > 1:  # Проверка наличия обоих классов
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"ROC AUC: {roc_auc:.2f}")
    plot_roc_curve(y_test, y_pred_proba)
else:
    st.write("Не удалось рассчитать ROC AUC, так как был предсказан только один класс.")
