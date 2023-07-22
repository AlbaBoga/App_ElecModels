# App_ElecModels

[Enlace a la aplicación de Streamlit](https://electronics.streamlit.app/)

## Contenido

[Enlace a la sección](https://github.com/AlbaBoga/DataAnalyticsPorfolio/tree/main/Project_ElectronicProducts)

* Programación en `Python`
* Análisis de los datos pertenecientes a la base de datos de productos de [Datafiniti](https://www.datafiniti.co/).
* Obtención de la muestra de los datos a través de [data.world](https://data.world/datafiniti/electronic-products-and-pricing-data).
* Preprocesamiento de los datos, buscando valores nulos, valores duplicados y limpieza de columnas pertinentes.
* Implemento un modelo de `Random Forest Classifier` a través de la librería `Scikit-Learn` para imputar los datos faltantes en la columna de tipo de envío.
* Utilización de la librería `plotly` para la visualización de los datos.
* `A/B testing` para determinar si características influyen en el precio de productos.
* Análisis de los precios de los productos visitados mensualmente (`estudio de serie temporal`).
* Implemento `modelo ARIMA` con hiperparámetros óptimos.
* Uso de `modelo Neural Prophet` para la predicción de valores futuros.
* Modelos Machine Learning para clasificación (estimación de tienda en función de las características del producto que busca el cliente):
  * Modelo `Extreme Gradient Boosting` a través de la librería `Pycaret`.
  * Modelo `Gradient Boosting Classifier` a través de la librería `Scikit-Learn` y usando la herramienta `Wandb` para búsqueda de hiperparámetros óptimos.
  * Modelo `Red Neuronal` a través de la librería `TensorFlow`.
* Resumen de los datos analizados mediante un `dashboard en PowerBI`
* Conclusiones de los datos.
* Utilización de la herramienta `Streamlit` para la visualización y explicación de los datos.

## Modelo de Clasificación

* Los datos utilizados son: el precio máximo y mínimo que ha podido tener el artículo, la disponibilidad, el estado, si tiene rebajas, el tipo de envío, categoría principal y categoría secundaria.
* Proporciona una tienda (Walmart, Bestbuy, Bhphotovideo o Ebay) que puede tener productos similares en base a las características que proporciona el cliente.
* Se tienen en cuenta valores atípicos.

### Extreme Gradient Boosting

[Enlace al modelo en Streamlit](https://electronicsmodels.streamlit.app/Pycaret_Extreme_Gradient_Boosting)

* Uso de la librería `pycaret`.
* Accuracy del `79%`.
* Se implementa el modelo en Streamlit.

### Gradient Boosting Classifier

[Enlace al modelo en Streamlit](https://electronicsmodels.streamlit.app/Gradient_Boostring_Classifier)

[Enlace al informe de Wandb](https://wandb.ai/alba-m-boga/project_digital4/reports/Modelo-de-clasificaci-n-de-tiendas--Vmlldzo0ODg2ODg4)

[Enlace a la búsqueda de parámetros mediante Wandb](https://github.com/AlbaBoga/DataAnalyticsPorfolio/blob/main/Project_ElectronicProducts/Project_digital_wandb.ipynb)

* Uso de la librería `Scikit-Learn` para determinar el modelo de clasificación que mejor se ajusta a los datos.
* Se utiliza la herramienta `Wandb` a través de una conexión API para la búsqueda de los parámetros más óptimos para implementar el modelo Gradient Boosting Classifier.
* Se obtienen los parámetros que proporcionan un accuracy del `78%` y se implementa el modelo.
* Se implementa el modelo en Streamlit.

### Red Neuronal

[Enlace al modelo en Streamlit](https://electronicsmodels.streamlit.app/Red_Neuronal)

[Enlace al código e implementación del modelo](https://github.com/AlbaBoga/DataAnalyticsPorfolio/blob/main/Project_ElectronicProducts/tensorflowdigital_class.ipynb)

* Uso de la librería `TensorFlow` para implementar una Red Neuronal.
* Se obtiene un accuracy final del `72%` y se implementa el modelo.
* Se implementa el modelo en Streamlit.

## Logos e imágenes

Logo: [vecteezy.com](https://www.vecteezy.com/)

Imágenes de la cabecera: [canva.com](https://www.canva.com/)
