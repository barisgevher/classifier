import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


rad = st.sidebar.radio(
    "Menu", ["Ana Sayfa", "BMI", "Kalp Hastalığı", "Grafikler"])

if rad == "Ana Sayfa":
    st.title("Sınıflandırma")
    st.text("Aşağıdaki veri setleri üzerine karar destek sistemi oluşturulacaktır.")
    st.text("1. BMI Sınıflandırma")
    st.text("2. Kalp Hastalığı Tespiti")


df1 = pd.read_csv("bmi.csv")
df1['Gender'] = df1['Gender'].replace(['Male', 'Female'], [0, 1])
x1 = df1.drop("Index", axis=1)
y1 = df1["Index"]

x1_train, x1_text, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)


model1 = RandomForestClassifier()
model2 = DecisionTreeClassifier()
model3 = svm.SVC()
model4 = linear_model.LogisticRegression()


model1.fit(x1_train, y1_train)
model2.fit(x1_train, y1_train)
model3.fit(x1_train, y1_train)
model4.fit(x1_train, y1_train)

if rad == "BMI":
    st.header("BMI Test Ekrani")
    Gender = st.number_input(
        " Cinsiyet Değer aralığı 0-1 ", min_value=0, max_value=1, step=1, key="Gender")
    Height = st.number_input(" Uzunluk Değer aralığı 140-199 ",
                             min_value=140, max_value=199, step=1, key="Height")
    Weight = st.number_input(" Ağırlık Değer aralığı 50-160 ",
                             min_value=50, max_value=160, step=1,  key="Weight")
    tahmin1 = model1.predict([[Gender, Height, Weight]])[0]
    tahmin2 = model2.predict([[Gender, Height, Weight]])[0]
    tahmin3 = model3.predict([[Gender, Height, Weight]])[0]
    tahmin4 = model4.predict([[Gender, Height, Weight]])[0]

    if st.button("Random Forest İle Tahmin Et "):
        if tahmin1 == 0:
            st.warning("Aşırı Zayıf")
        elif tahmin1 == 1:
            st.success("Zayıf")
        elif tahmin1 == 2:
            st.success("Normal")
        elif tahmin1 == 3:
            st.success("Aşırı Kilolu")
        elif tahmin1 == 4:
            st.success("Obez")
        elif tahmin1 == 5:
            st.success("Aşırı  Obez")

    if st.button("Karar Ağacı İle Tahmin Et "):
        if tahmin2 == 0:
            st.warning("Aşırı Zayıf")
        elif tahmin2 == 1:
            st.success("Zayıf")
        elif tahmin2 == 2:
            st.success("Normal")
        elif tahmin2 == 3:
            st.success("Aşırı Kilolu")
        elif tahmin2 == 4:
            st.success("Obez")
        elif tahmin2 == 5:
            st.success("Aşırı  Obez")

    if st.button("SVM İle Tahmin Et "):
        if tahmin3 == 0:
            st.warning("Aşırı Zayıf")
        elif tahmin3 == 1:
            st.success("Zayıf")
        elif tahmin3 == 2:
            st.success("Normal")
        elif tahmin3 == 3:
            st.success("Aşırı Kilolu")
        elif tahmin3 == 4:
            st.success("Obez")
        elif tahmin3 == 5:
            st.success("Aşırı  Obez")

    if st.button("Lineer Regresyon İle Tahmin Et "):
        if tahmin4 == 0:
            st.warning("Aşırı Zayıf")
        elif tahmin4 == 1:
            st.success("Zayıf")
        elif tahmin4 == 2:
            st.success("Normal")
        elif tahmin4 == 3:
            st.success("Aşırı Kilolu")
        elif tahmin4 == 4:
            st.success("Obez")
        elif tahmin4 == 5:
            st.success("Aşırı  Obez")


df2 = pd.read_csv("heart.csv")

x1 = df2.drop("DEATH_EVENT", axis=1)
y1 = df2["DEATH_EVENT"]

x1_train, x1_text, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)


model1 = linear_model.LogisticRegression(max_iter=500)
model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()
model4 = svm.SVC()

model1.fit(x1_train, y1_train)
model2.fit(x1_train, y1_train)
model3.fit(x1_train, y1_train)
model4.fit(x1_train, y1_train)

if rad == "Kalp Hastalığı":
    st.header("Kalp Hastalığı  Test Ekrani")
    age = st.number_input("Yaş Değer aralığı 40-95",
                          min_value=40, max_value=95, step=1)
    anaemia = st.number_input(
        "anaemia Değer aralığı 0-1", min_value=0, max_value=1, step=1)
    creatinine_phosphokinase = st.number_input(
        "creatinine_phosphokinase Değer aralığı 23-7861", min_value=23, max_value=7861, step=1)
    diabetes = st.number_input(
        "Diyabet değer aralığı 0-1", min_value=0, max_value=1, step=1)
    ejection_fraction = st.number_input(
        "ejection_fraction aralığı 14-80", min_value=14, max_value=80, step=1)
    high_blood_pressure = st.number_input(
        "Yüksek kan basıncı aralığı 0-1", min_value=0, max_value=1)
    platelets = st.number_input(
        "platelets aralığı 251000-850000", min_value=251000, max_value=850000)
    serum_creatinine = st.number_input(
        "serum_creatinine aralığı 0.5-9.4", min_value=0.5, max_value=9.4)
    serum_sodium = st.number_input(
        "serum_sodium aralığı 113-148", min_value=113, max_value=148, step=1)
    sex = st.number_input("Cinsiyet değer aralığı 0-1",
                          min_value=0, max_value=1, step=1)
    smoking = st.number_input(
        "Sigara değişkeni değer aralığı 0-1", min_value=0, max_value=1, step=1)
    time = st.number_input("Zaman değer aralığı 0-1",
                           min_value=4, max_value=285, step=1)

    input_features = [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                       high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                       sex, smoking, time]]

    # Tahminler
    tahmin1 = model1.predict(input_features)[0]
    tahmin2 = model2.predict(input_features)[0]
    tahmin3 = model3.predict(input_features)[0]
    tahmin4 = model4.predict(input_features)[0]

    if st.button("Random Forest İle Tahmin Et "):
        if tahmin1 == 0:
            st.warning("Kalp hastalığı yok")
        elif tahmin1 == 1:
            st.success("Kalp hastalığı  var")

    if st.button("Karar Ağacı İle Tahmin Et "):
        if tahmin2 == 0:
            st.warning("Kalp hastalığı yok")
        elif tahmin2 == 1:
            st.warning("Kalp hastalığı var ")

    if st.button("SVM İle Tahmin Et "):
        if tahmin3 == 0:
            st.warning("Kalp hastalığı yok")
        elif tahmin3 == 1:
            st.success("Kalp hastalığı var")

    if st.button("Lineer Regresyon İle Tahmin Et "):
        if tahmin4 == 0:
            st.warning("Kalp hastalığı yok ")
        elif tahmin4 == 1:
            st.success("Kalp hastalığı yok")

if rad == "Grafikler":
    tip = st.selectbox("Grafik çıkarılacak Veri Setleri",
                       ["Home", "BMI", "Heart"])
    if tip == "BMI":
        fig = px.scatter(df1, x="Weight",
                         y="Index")
        st.plotly_chart(fig)
    elif tip == "Heart":
        fig = px.scatter(df2, x="ejection_fraction", y="DEATH_EVENT")
        st.plotly_chart(fig)
