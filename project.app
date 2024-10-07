import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv("nursery.data", header=None)
data.columns = ['class', 'parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
label_encoders = {column: LabelEncoder().fit(data[column]) for column in data.columns}
for column in data.columns:
    data[column] = label_encoders[column].transform(data[column])

X = data.drop('class', axis=1)
y = data['class']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

st.title("Рекомендация детского сада")
parents = st.selectbox("Положение родителей", label_encoders['parents'].classes_)
has_nurs = st.selectbox("Условия ухода", label_encoders['has_nurs'].classes_)
form = st.selectbox("Форма детского сада", label_encoders['form'].classes_)
children = st.number_input("Количество детей", min_value=1, max_value=10)
housing = st.selectbox("Условия проживания", label_encoders['housing'].classes_)
finance = st.selectbox("Финансовая ситуация", label_encoders['finance'].classes_)
social = st.selectbox("Социальные условия", label_encoders['social'].classes_)
health = st.selectbox("Состояние здоровья", label_encoders['health'].classes_)

input_data = np.array([label_encoders['parents'].transform([parents])[0],
                       label_encoders['has_nurs'].transform([has_nurs])[0],
                       label_encoders['form'].transform([form])[0],
                       children,
                       label_encoders['housing'].transform([housing])[0],
                       label_encoders['finance'].transform([finance])[0],
                       label_encoders['social'].transform([social])[0],
                       label_encoders['health'].transform([health])[0]]).reshape(1, -1)

if st.button("Получить рекомендацию"):
    prediction = model.predict(input_data)
    recommendation = label_encoders['class'].inverse_transform(prediction)
    st.write(f"Рекомендация: {recommendation[0]}")
