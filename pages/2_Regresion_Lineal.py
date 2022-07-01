import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.title("Regresion Lineal")

#Cargo el archivo a utilizar
uploaded_file = st.file_uploader("Para comenzar la ejecucion, suba un archivo")


if uploaded_file is not None:
    type = uploaded_file.type

    if type == 'text/csv':
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    elif type == 'application/json':
        dataframe = pd.read_json(uploaded_file)
        dataframe.to_csv('prueba.csv')
        dataframe = pd.read_csv('prueba.csv')
        st.write(dataframe)
    elif type == 'application/vnd.ms-excel' or type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        dataframe = pd.read_excel(uploaded_file)
        dataframe.to_csv('prueba.csv')
        dataframe = pd.read_csv('prueba.csv')
        st.write(dataframe)




    #Pregunto parametro x
    var_x = st.selectbox('Parametro X', (dataframe.columns))

    #Pregunto parametro y
    var_y = st.selectbox('Parametro Y', (dataframe.columns))



    x = np.asarray(dataframe[var_x]).reshape(-1, 1)
    y = np.asarray(dataframe[var_y]).reshape(-1, 1)

    regr = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    r2 = r2_score(y, y_pred)

    st.write('y_pred: ', y_pred)
    st.write('coeficiente: ', regr.coef_[0][0])
    st.write('intercepcion: ', regr.intercept_[0])
    st.write('R^2: ', r2)
    st.write('Error Cuadratico: ', mean_squared_error(y, y_pred))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_pred, color='red')
    st.pyplot()

    numero = st.number_input('Numero a predecir')
    prediccion = regr.predict([[numero]])[0][0]
    st.write('Prediccion: ', prediccion)