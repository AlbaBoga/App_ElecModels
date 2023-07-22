#--------------LIBRERÍAS--------------#
import numpy as np
import pandas as pd
from pycaret.classification import *
import streamlit as st
from PIL import Image
#--------------LIBRERÍAS--------------#

#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#
st.set_page_config(page_title='Predictor', page_icon='🖥️', layout='centered')
st.set_option('deprecation.showPyplotGlobalUse', False)
#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#
digital_class = pd.read_csv(r"data/digital_class.csv")

# -- img
image3 = Image.open(r'img/Predictor.png')
st.image(image3)
# -- img

# --pycaret
col1,col2=st.columns(2)
with col1:
    disponibilidad = st.selectbox("Disponibilidad", digital_class['Disponibilidad'].unique())
    estado = st.selectbox("Estado", digital_class['Estado'].unique())
    max_price = st.number_input("Precio máximo", min_value=0.00, max_value=7000.00, step=0.01)
    min_price = st.number_input("Precio mínimo", min_value=0.00, max_value=6000.00, step=0.01)
with col2:
    sale = st.selectbox('Rebajas',digital_class['prices.isSale'].unique())
    envio = st.selectbox('Envío',digital_class['Envios'].unique())
    cat_prim = st.selectbox('Categoría principal',digital_class['Categoria Principal'].unique())
    filtered_cat = digital_class[digital_class['Categoria Principal'] == cat_prim]['secondary category'].unique()
    cat_sec = st.selectbox("Categoría secundaria", filtered_cat)

def prediccion_tienda(max_price,min_price,disponibilidad,estado,sale,envio,cat_prim,cat_sec):
    data = pd.DataFrame({'prices.amountMax': [max_price], 'prices.amountMin': [min_price], 'Disponibilidad':[disponibilidad], 'Estado': [estado],
    'prices.isSale': [sale], 'Envios':[envio], 'Categoria Principal':[cat_prim],
    'secondary category':[cat_sec]})
    loaded_model = load_model('digital_class')
    prediction = predict_model(loaded_model, data=data)
    return prediction

prediction=prediccion_tienda(max_price,min_price,disponibilidad,estado,sale,envio,cat_prim,cat_sec)


if st.button('Tienda 👈'):
        
    if prediction.loc[0,'prediction_label']==0:
        st.write('Bestbuy')
        st.write('Accuracy: ', prediction.loc[0,'prediction_score'])
    elif prediction.loc[0,'prediction_label']==1:
        st.write('Bhphotovideo')
        st.write('Accuracy: ', prediction.loc[0,'prediction_score'])
    elif prediction.loc[0,'prediction_label']==2:
        st.write('Ebay')
        st.write('Accuracy: ', prediction.loc[0,'prediction_score'])
    else:
        st.write('Walmart')
        st.write('Accuracy: ', prediction.loc[0,'prediction_score'])

else:
    st.write('📝 Estimando ... ')


# --pycaret



if st.button('Volver 👈'):
    link = 'https://electronics.streamlit.app/Predictor'
    st.markdown(f'<a href="{link}">Volver</a>', unsafe_allow_html=True)



#--------------------------------------SIDEBAR-------------------------------------#
image1 = Image.open(r'img/logo.png')
st.sidebar.image(image1)
#--------------------------------------SIDEBAR-------------------------------------#