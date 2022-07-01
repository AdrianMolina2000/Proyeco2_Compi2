import streamlit as st
import pandas as pd



st.title("Cargar Tabla")


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
