import joblib
import streamlit as st
import numpy as np
import pandas as pd

model_file_branco = 'modelos/modelo_branco.pkl'
model_file_tinto = 'modelos/modelo_tinto.pkl'
scaler = joblib.load('scaler.pkl')

modelo_branco = joblib.load(model_file_branco)
modelo_tinto = joblib.load(model_file_tinto)

def dar_nota(tipo, novo_dado):
    colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
               'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    
    novo_dado_df = pd.DataFrame([novo_dado], columns=colunas)
    
    novo_dado_scaled = scaler.transform(novo_dado_df)
    
    predicao = tipo.predict(novo_dado_scaled)
    
    return predicao[0]


st.set_page_config(page_title='Nota do Vinho', page_icon='🍷', layout='centered')

# Titulo----
st.title('🍷 Sistema Previsão de Qualidade - Vinho Tinto e Branco')
st.caption('Este sistema prevê a qualidade dos vinhos com base em suas propriedades físico-químicas, atribuindo uma nota de 1 a 5 estrelas.')

# Inputs----
st.header("Insira as informações do vinho:")

tipo_vinho = st.selectbox("Selecione o tipo de vinho", ("Vinho Tinto", "Vinho Branco"))

col1, col2 = st.columns(2)

with col1:
    acidez_fixa = st.number_input("Acidez Fixa", value=0.0, step=0.1)
    acidez_volatil = st.number_input("Acidez Volátil", value=0.00, step=0.01)
    acido_citrico = st.number_input("Ácido Cítrico", value=0.00, step=0.01)
    acucar_residual = st.number_input("Açúcar Residual", value=0.0, step=0.1)
    cloretos = st.number_input("Cloretos", value=0.000, step=0.001)
    dioxido_enxofre_livre = st.number_input("Dióxido de Enxofre Livre", value=0.0, step=1.0)

with col2:
    dioxido_enxofre_total = st.number_input("Dióxido de Enxofre Total", value=0.0, step=1.0)
    densidade = st.number_input("Densidade", value=0.00, step=0.01)
    ph = st.number_input("pH", value=0.0, step=0.01)
    sulfatos = st.number_input("Sulfatos", value=0.00, step=0.01)
    alcool = st.number_input("Álcool", value=0.0, step=0.1)

#Botao---
if st.button("Verificar a avaliação"):
    
    novo_dado = [acidez_fixa, acidez_volatil, acido_citrico, acucar_residual, cloretos,
                 dioxido_enxofre_livre, dioxido_enxofre_total, densidade, ph, sulfatos, alcool]
    
    if tipo_vinho == "Vinho Tinto":
        modelo = modelo_tinto
    else:
        modelo = modelo_branco
    
    #Chama a funcao----
    nota = dar_nota(modelo, novo_dado)
    
    #Mostrar o resultado----
    estrelas = '⭐' * int(nota)
    st.write(f"A qualidade do vinho é: {estrelas}")
