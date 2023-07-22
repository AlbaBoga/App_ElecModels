#--------------LIBRERÍAS--------------#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
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



# --NN

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
    data = pd.DataFrame(columns=['prices.amountMax', 'prices.amountMin', 'prices.isSale',
        'Disponibilidad_More on the Way',
        'Disponibilidad_Out of Stock', 'Disponibilidad_Retired',
        'Disponibilidad_Special Order', 'Estado_Refurbished', 'Estado_Used',
        'Envios_Free Standard Shipping', 'Envios_Minimum Order Free Shipping',
        'Envios_Paid Shipping', 'Categoria Principal_Computer and accessories',
        'Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories',
        'Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech',
        'secondary category_Audio and accessories',
        'secondary category_Computer',
        'secondary category_Computer and accessories',
        'secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech'])
        
    maxp_std=(max_price-1.0)/(6999.99-1.0)
    data.loc[0,'prices.amountMax']= maxp_std * (1-0) + 0
    minp_std=(min_price-1.0)/(5999.99-1.0)
    data.loc[0,'prices.amountMin']= minp_std * (1-0) + 0

    if sale==1:
        data.loc[0,'prices.isSale']=1
    else:
        data.loc[0,'prices.isSale']=0

    if disponibilidad=='In Stock':
        data.loc[0,['Disponibilidad_More on the Way','Disponibilidad_Out of Stock', 'Disponibilidad_Retired','Disponibilidad_Special Order']] =[0,0,0,0]
    elif disponibilidad=='Out of Stock':
        data.loc[0,['Disponibilidad_More on the Way','Disponibilidad_Out of Stock', 'Disponibilidad_Retired','Disponibilidad_Special Order']] =[0,1,0,0]
    elif disponibilidad=='Special Order':
        data.loc[0,['Disponibilidad_More on the Way','Disponibilidad_Out of Stock', 'Disponibilidad_Retired','Disponibilidad_Special Order']] =[0,0,0,1]
    elif disponibilidad=='More on the Way':
        data.loc[0,['Disponibilidad_More on the Way','Disponibilidad_Out of Stock', 'Disponibilidad_Retired','Disponibilidad_Special Order']] =[1,0,0,0]
    else:
        data.loc[0,['Disponibilidad_More on the Way','Disponibilidad_Out of Stock', 'Disponibilidad_Retired','Disponibilidad_Special Order']] =[0,0,1,0]

    if estado=='New':
        data.loc[0,['Estado_Refurbished', 'Estado_Used']]=[0,0]
    elif estado=='Refurbished':
        data.loc[0, ['Estado_Refurbished', 'Estado_Used']] = [1, 0]
    else:
        data.loc[0, ['Estado_Refurbished', 'Estado_Used']] = [0, 1]

    if envio=='Free Expedited Shipping':
        data.loc[0,['Envios_Free Standard Shipping', 'Envios_Minimum Order Free Shipping','Envios_Paid Shipping']]=[0,0,0]
    elif envio=='Free Standard Shipping':
        data.loc[0,['Envios_Free Standard Shipping', 'Envios_Minimum Order Free Shipping','Envios_Paid Shipping']]=[1,0,0]
    elif envio=='Minimum Order Free Shipping':
        data.loc[0,['Envios_Free Standard Shipping', 'Envios_Minimum Order Free Shipping','Envios_Paid Shipping']]=[0,1,0]
    else:
        data.loc[0,['Envios_Free Standard Shipping', 'Envios_Minimum Order Free Shipping','Envios_Paid Shipping']]=[0,0,1]
        
    if cat_prim=='Audio and accessories':
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[0,0,0,0,0]
    elif cat_prim=='Computer and accessories':
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[1,0,0,0,0]
    elif cat_prim=='Electronics':
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[0,1,0,0,0]
    elif cat_prim=='Phones and accessories':
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[0,0,1,0,0]
    elif cat_prim=='TV and accessories':
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[0,0,0,1,0]
    else:
        data.loc[0,['Categoria Principal_Computer and accessories','Categoria Principal_Electronics',
        'Categoria Principal_Phones and accessories','Categoria Principal_TV and accessories',
        'Categoria Principal_Wireless Tech']]=[0,0,0,0,1]
        
    if cat_sec=='Audio and accessories':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[1,0,0,0,0,0,0,0,0,0,0,0]
    elif cat_sec=='Computer':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,1,0,0,0,0,0,0,0,0,0,0]
    elif cat_sec=='Computer and accessories':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,1,0,0,0,0,0,0,0,0,0]
    elif cat_sec=='Digital camera and accessories':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,1,0,0,0,0,0,0,0,0]
    elif cat_sec=='Discos duros':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,1,0,0,0,0,0,0,0]
    elif cat_sec=='Electronics':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,1,0,0,0,0,0,0]
    elif cat_sec=='Home Theater':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,1,0,0,0,0,0]
    elif cat_sec=='Phones and accessories':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,1,0,0,0,0]
    elif cat_sec=='Smart-tech':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,0,1,0,0,0]
    elif cat_sec=='TV':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,0,0,1,0,0]
    elif cat_sec=='TV and accessories':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,0,0,0,1,0]
    elif cat_sec=='Wireless Tech':
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,0,0,0,0,1]
    else:
        data.loc[0,['secondary category_Audio and accessories','secondary category_Computer',
        'secondary category_Computer and accessories','secondary category_Digital camera and accessories',
        'secondary category_Discos duros', 'secondary category_Electronics',
        'secondary category_Home Theater',
        'secondary category_Phones and accessories',
        'secondary category_Smart-tech', 'secondary category_TV',
        'secondary category_TV and accessories',
        'secondary category_Wireless Tech']]=[0,0,0,0,0,0,0,0,0,0,0,0]
                        
    data = data.astype(float)
    X=data.values
    from tensorflow.keras.models import load_model
    modelo=load_model('digital_class_tf.h5')  
    predictions = modelo.predict(X)
        
    digito_predicho = np.argmax(predictions)

    if digito_predicho==0:
        tienda='Bestbuy'
    elif digito_predicho==1:
        tienda='Walmart'
    elif digito_predicho==2:
        tienda='Bhphotovideo'
    else:
        tienda='Ebay'


    return tienda

prediction=prediccion_tienda(max_price,min_price,disponibilidad,estado,sale,envio,cat_prim,cat_sec)

if st.button('Tienda 👈'):
    st.write(prediction)
else:
    st.write('📝 Estimando ... ')
# --NN



if st.button('Volver 👈'):
    link = 'https://electronics.streamlit.app/Predictor'
    st.markdown(f'<a href="{link}">Volver</a>', unsafe_allow_html=True)

    

#--------------------------------------SIDEBAR-------------------------------------#
image1 = Image.open(r'img/logo.png')
st.sidebar.image(image1)
#--------------------------------------SIDEBAR-------------------------------------#