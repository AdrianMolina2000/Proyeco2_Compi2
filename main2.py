
from scipy.sparse import data
import streamlit as st

import pandas as pd
from PIL import Image
import base64
import pandas as pd
from io  import StringIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from streamlit.elements import selectbox
from fpdf import FPDF


st.markdown( '##  Proyecto 2 Compiladores 2, Machine Learning'  )


EXTENSION = st.radio(
    "Escoja la extension del archivo",
    ('csv', 'xls', 'json'))

uploaded_file = st.file_uploader("Para comenzar la ejecucion, suba un archivo")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    if EXTENSION == 'csv':
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    elif EXTENSION =='json':
        dataframe = pd.read_json(uploaded_file)
        dataframe.to_csv('prueba.csv')
        dataframe = pd.read_csv('prueba.csv')
        st.write(dataframe)
    elif EXTENSION == 'xls':
        dataframe = pd.read_excel(uploaded_file)
        dataframe.to_csv('prueba.csv')
        dataframe = pd.read_csv('prueba.csv')
        st.write(dataframe)


def Tendencia_Covid_Pais():


    if uploaded_file is not None:
        st.info("escoja Los campos que considere nescesarios para realizar la Tendencia de casos por pais")

        var = st.selectbox(
        'Seleccione el campo 1 ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)





        var1 = st.selectbox(
        'Seleccione el campo 2 ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        dataframe[var1]=dataframe[var1].fillna(0)




        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')


        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        if EXTENSION != 'JSON':
            st.write(casos_pais)



        tamanio=casos_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)




        X=np.asarray(arreglo).reshape(-1,1)


        Y=casos_pais[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)




        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='red')

        plt.title("TENDENCIA DE CASOS DEL PAIS:"+pais)
        plt.ylabel('CASOS_COVID')
        plt.xlabel('#')


        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        plt.savefig('TendendiaCovid_paisLineal.png')
        st.pyplot()
        image7 = Image.open('tendenciaa.png')

        st.image(image7, width=1200,use_column_width='auto')
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:

            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que  debido a las restricciones  los casos de covid 19 han ido disminuyendo considerablemente ')
            texto='La pendiente de una recta es negativa cuando la recta es decreciente , es decir que  debido a las restricciones  los casos de covid 19 han ido disminuyendo considerablemente '
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa   los casos en este pais  han ido aumentando considerablemente  alo largo de los ultimos reportes ')
            texto='La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa   los casos en este pais  han ido aumentando considerablemente  alo largo de los ultimos reportes '













        st.markdown('## Grafica Polinomial  ')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)
        X2=np.asarray(arreglo)
        Y2=casos_pais[var1]

        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]

        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X2)

        model= LinearRegression()
        model.fit(X_TRANSF,Y2)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW))

        r2=r2_score(Y2,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Tendecia de casos de COVID-19 en el pais "+pais+title)
        texto2="Grafica Polinomial Tendecia de casos de COVID-19 en el pais "+pais+title
        plt.xlabel('#')
        plt.ylabel('Casos de COVID-19')
        plt.savefig('TendendiaCovid_paisPolinomial.png')
        plt.show()
        st.pyplot()


        export_as_pdf = st.button("Export Report")
        if export_as_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_xy(0, 0)
            pdf.set_font('Times', 'B', 20)
            titulo_pd="Tendencia de casos  de Covid-19 en "+str(pais)
            pdf.multi_cell(200,10,txt=titulo_pd,align='J')

            pdf.set_font('Times', 'B', 12)





            pdf.image('TendendiaCovid_paisLineal.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
            pdf.multi_cell(200,10,txt=texto,align='J')

            pdf.add_page()
            pdf.multi_cell(200,10,txt=texto2,align='J')


            pdf.image('TendendiaCovid_paisPolinomial.png', x = None, y = None, w = 0, h = 0, type = '', link = '')

            pdf.output('TendendiaCovid_pais.pdf', 'F')

            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "TendendiaCovid_pais")
            st.markdown(html, unsafe_allow_html=True)




op = st.multiselect(
    'Escoja la opcion que prefiera',
    [
        'Inicio☄️', 'Tendencia de Covid por pais📈',
        'Predicción de Infectados en un País🧮',
        'Inidice de Progresión de la pandemia.🦠',
        'Predicción de mortalidad por COVID en un Departamento🧮',
        'Predicción de mortalidad por COVID en un Pais🧮',
        'Análisis del número de muertes por coronavirus en un País☠️',
        'Tendencia del número de infectados por día de un País.🗓️📈',
        'Predicción de casos de un país para un año🧮',
        'Tendencia de la vacunación de en un País💉📈',
        'Ánalisis Comparativo de Vacunación entre 2 paises💉',
        'Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo🙍🏻‍♂️',
        'Ánalisis Comparativo entres 2 o más paises o continentes🌎',
        'Muertes promedio por casos confirmados y edad de covid 19 en un País☠️',
        'Muertes según regiones de un país - Covid 19☠️',
        'Tendencia de casos confirmados de Coronavirus en un departamento de un País📈',
        'Porcentaje de muertes frente al total de casos en un país, región o continente.%📶☠️🌎',
        'Tasa de comportamiento de casos activos en relación al número de muertes en un continente☠️📈🦠',
        'Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.🦠',
        'Predicción de muertes en el último día del primer año de infecciones en un país.☠️',
        'Predicciones de casos y muertes en todo el mundo🧮',
        'Tasa de mortalidad por coronavirus (COVID-19) en un país📈☠️',
        'Factores de muerte por COVID-19 en un país.☠️',
        'Comparación entre el número de casos detectados y el número de pruebas de un país 💊💉',
        'Predicción de casos confirmados por día🧮'
    ]
)

st.write('You selected:', op)

if len(op)>0:
    if op[0] =='Inicio☄️':
        Inicio()
    elif op[0] =='Tendencia de Covid por pais📈':
        Tendencia_Covid_Pais()
    elif op[0] =='Predicción de Infectados en un País🧮':
        Prediccion_Infectados_Pais()
    elif op[0]=='Predicción de mortalidad por COVID en un Departamento🧮':
        Prediccion_Muertes_Departamento()
    elif op[0]=='Predicción de mortalidad por COVID en un Pais🧮':
        Prediccion_Muertes_Pais()
    elif op[0]=='Análisis del número de muertes por no en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Tendencia de la vacunación de en un País💉📈':
        Tendencia_Vacunancion_Pais()
    elif op[0]=='Ánalisis Comparativo de Vacunación entre 2 paises💉':
        Comparacion_Vacunacion_Pais()
    elif op[0]=='Tendencia de casos confirmados de Coronavirus en un departamento de un País📈':
        Tendencia_casos_Departamento()
    elif op[0]=='Ánalisis Comparativo entres 2 o más paises o continentes🌎':
        Analisis_Comparativo_entre2_pais_contienente()
    elif op[0]=='Muertes según regiones de un país - Covid 19☠️':
        Muertes_por_Region()
    elif op[0]=='Predicción de casos confirmados por día🧮':
        Prediccion_Muertes_dia()
    elif op[0]=='Tendencia del número de infectados por día de un País.🗓️📈':
        Tendencia_Infectados_dia()
    elif op[0]=='Comparación entre el número de casos detectados y el número de pruebas de un país 💊💉':
        Comparacion_Infectados_Vacunados_Pais()
    elif op[0]=='Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo🙍🏻‍♂️':
        Porcentaje_Hombres_Covid()
    elif op[0]=='Porcentaje de muertes frente al total de casos en un país, región o continente.%📶☠️🌎':
        porcentaje_muertes_p()
    elif op[0]=='Tasa de comportamiento de casos activos en relación al número de muertes en un continente☠️📈🦠':
        Tasa_Comportamiento_Muertes_Covid()
    elif op[0]=='Factores de muerte por COVID-19 en un país.☠️':
        Factores_Muertes()
    elif op[0]=='Predicciones de casos y muertes en todo el mundo🧮':
        prediccion_mundial()
    elif op[0]=='Muertes promedio por casos confirmados y edad de covid 19 en un País☠️':
        Muertes_Edad()
    elif op[0]=='Inidice de Progresión de la pandemia.🦠':
        indice_progresion()
    elif op[0]=='Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.🦠':
        Comportamiento_Casos_Municipio()
    elif op[0]=='Predicción de muertes en el último día del primer año de infecciones en un país.☠️':
        prediccion_ultimo_dia()
    elif op[0]=='Predicción de casos de un país para un año🧮':
        prediccion_casos_anio()
    elif op[0]=='Tasa de mortalidad por coronavirus (COVID-19) en un país📈☠️':
        Tasa_Mortalidad_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Anlisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()




