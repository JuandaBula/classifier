#importar librerias
import streamlit as st
import pickle
import pandas as pd
import joblib

#Extrar los archivos pickle (Modelos!!)
#with open('C:/Users/jdbul/OneDrive/Escritorio/Python/App_MachineLearning/lin_reg.pkl', 'rb') as li:
lin_reg = joblib.load("lin_reg.pkl")

#with open('C:/Users/jdbul/OneDrive/Escritorio/Python/App_MachineLearning/log_reg.pkl', 'rb') as lo:
log_reg = joblib.load("log_reg.pkl")

#with open('C:/Users/jdbul/OneDrive/Escritorio/Python/App_MachineLearning/svc_m.pkl', 'rb') as sv:
svc_m = joblib.load("svc_m.pkl")


#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    #titulo
    st.title('Modelamiento de Iris by Juan Daniel Bula')
    #titulo de sidebar
    st.sidebar.header('Ingresa los parámetros...')

    #funcion para poner los parametros en el sidebar
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('¿Cuál modelo quieres usar?', option)
# Lado derecho de la app:
    st.subheader('Parámetros del usuario:')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))


if __name__ == '__main__':
    main()
    
    
    
 # streamlit run "C:\Users\jdbul\OneDrive\Escritorio\Python\App_MachineLearning\irisWeb.py"   
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 