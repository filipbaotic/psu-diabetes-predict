import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st

# odabir ulaznih velicina
data = pd.read_csv('dataset.csv') 
y = data.Outcome # uzimamo Outcome kao label za predvidjanje dijabetesa
X = data.drop('Outcome', axis=1) # za features uzimamo sve stupce osim Outcome koji je label
X = X.values # izbacuje upozorenje za StandardScaler ukoliko se ne stavi

# podjela na train i test tako da je 80% podataka train, a 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# skaliranje ulaznih podataka
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# izrada modela
model = sklearn.svm.SVC(kernel='linear') # vektorska klasifikacija
model.fit(X_train_scaler, y_train)

# evaluacija
y_predict_train = model.predict(X_train_scaler)
y_predict_test = model.predict(X_test_scaler)

# preciznost
train_accuracy = accuracy_score(y_predict_train, y_train)
test_accuracy = accuracy_score(y_predict_test, y_test)
# print('Train accuracy:', round(train_accuracy*100,2),'%')
# print('Test accuracy:', round(test_accuracy*100,2),'%')


# streamlit dio
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
st.title('Diabetes prediction')

col1, col2 = st.columns(2)

with st.form("form"):
  with col1:
    p1 = st.number_input('Number of pregnancies')
    p2 = st.number_input('Glucose')
    p3 = st.number_input('Blood pressure')
    p4 = st.number_input('Skin thickness')

  with col2:
    p5 = st.number_input('Insulin')
    p6 = st.number_input('BMI')
    p7 = st.number_input('Diabetes pedigree function')
    p8 = st.number_input('Age')

  submitted = st.form_submit_button("Submit")
  if submitted:
      # provjera
      input_data = (p1,p2,p3,p4,p5,p6,p7,p8)
      input = np.asarray(input_data) # input data - numpy array

      # oblikovanje 
      input = input.reshape(1,-1)

      scaled_data = scaler.transform(input)
      predict = model.predict(scaled_data)

      if (predict == 0):
        st.success('Not diabetic')
      else:
        st.error('Diabetic')


