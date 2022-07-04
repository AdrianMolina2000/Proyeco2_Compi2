import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


st.title("Regresion Polinomial")


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


    grado = st.number_input('Grado de la funcion polinomial')
    nb_degree=int(grado)
    pf = PolynomialFeatures(degree=nb_degree)
    x_transform = pf.fit_transform(x)

    regr = LinearRegression()
    regr.fit(x_transform, y)

    y_pred = regr.predict(x_transform)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    st.write('RMSE: ', rmse)
    st.write('R^2: ', r2)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_pred, color='red')
    st.pyplot()

    predNum = st.number_input('Numero a predecir')
    x_new_min = predNum
    x_new_max = predNum
    x_new = np.linspace(x_new_min, x_new_max, 1)
    x_new = x_new[:, np.newaxis]
    x_trans = pf.fit_transform(x_new)
    prediccion = regr.predict(x_trans)[0][0]
    st.write('Prediccion: ', prediccion)