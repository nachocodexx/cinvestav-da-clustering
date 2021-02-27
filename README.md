# Librerias
Se hace uso de un conjunto de librerias para el analisis de datos.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering,AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram,linkage
import time
import json
import codecs
from sklearn.cluster import KMeans
from functools import reduce
import operator 
from sklearn.metrics import calinski_harabasz_score as chs,davies_bouldin_score as dbs,silhouette_score as shs
from scipy.sparse.csgraph import laplacian
from numpy.random import RandomState,BitGenerator
import warnings
from scipy.sparse.csgraph import shortest_path
from validclust.validclust.indices import dunn
plt.rcParams["figure.figsize"] = (15,15)
warnings.filterwarnings("ignore")

ss = preprocessing.StandardScaler()
ac = AgglomerativeClustering()
```

# Funciones de utileria
Funciones las cuales se utilizan de una forma recurrente.


```python
def standarize(**kwargs):
    X = kwargs['X']
    scaler = ss.fit(X)
    return scaler.transform(X)
```

# Importar dataset
Se importan los datos utilizando Pandas


```python
df = pd.read_csv('./data/op01.csv')
oldCols =df.columns
columns = df.columns
useless_cols = ["OPERATION_ID","USER_ID","TYPE","TOPIC_ID","TOPIC_ROLE","USER_ROLE","FILE_ID"]
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPERATION_ID</th>
      <th>TYPE</th>
      <th>USER_ID</th>
      <th>PRODUCTION</th>
      <th>CONSUME</th>
      <th>CONSUME_OTHERS</th>
      <th>USER_VOLUME</th>
      <th>USER_DENSITY</th>
      <th>USER_ROLE</th>
      <th>FILE_ID</th>
      <th>FILE_SIZE</th>
      <th>TOPIC_ID</th>
      <th>TOPIC_VOLUME</th>
      <th>TOPIC_DENSITY</th>
      <th>TOPIC_ROLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>UPLOAD</td>
      <td>43114731</td>
      <td>0.01112</td>
      <td>0.00000</td>
      <td>0.233515</td>
      <td>1764</td>
      <td>9.966667</td>
      <td>Gamma</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>DOWNLOAD</td>
      <td>1405534545</td>
      <td>0.00000</td>
      <td>0.02224</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>DOWNLOAD</td>
      <td>1405534545</td>
      <td>0.00000</td>
      <td>0.02224</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
  </tbody>
</table>
</div>



# Visualizar informacion del dataset
Se muestra la informacion basica sobre el conjunto de datos, como los tipos de cada columna y su nombre y el tamaño en Megabytes


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8993 entries, 0 to 8992
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   OPERATION_ID    8993 non-null   int64  
     1   TYPE            8993 non-null   object 
     2   USER_ID         8993 non-null   int64  
     3   PRODUCTION      8993 non-null   float64
     4   CONSUME         8993 non-null   float64
     5   CONSUME_OTHERS  8993 non-null   float64
     6   USER_VOLUME     8993 non-null   int64  
     7   USER_DENSITY    8993 non-null   float64
     8   USER_ROLE       230 non-null    object 
     9   FILE_ID         8993 non-null   int64  
     10  FILE_SIZE       8993 non-null   int64  
     11  TOPIC_ID        8993 non-null   int64  
     12  TOPIC_VOLUME    8993 non-null   int64  
     13  TOPIC_DENSITY   8993 non-null   float64
     14  TOPIC_ROLE      8993 non-null   object 
    dtypes: float64(5), int64(7), object(3)
    memory usage: 1.0+ MB


# Resumen estadistico
Se realiza un resumen estadistico de las columnas, el cual contiene 8 columnas, entre las cuales se encuentran:
- **count**: Cantidad de registros 
- **mean** : La media aritmetica 
- **std**  : La desviacion estandar 
- **min**  : Valor minimo 
- **25%**  : Percentil 25
- **50%**  : Percentil 50 el cual es equivalente a la mediana
- **75%**  : Percentil 75
- **max**  : Valor maximo


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OPERATION_ID</th>
      <td>8993.0</td>
      <td>3.869080e+03</td>
      <td>2.309459e+03</td>
      <td>1.00</td>
      <td>1.829000e+03</td>
      <td>3.793000e+03</td>
      <td>5.875000e+03</td>
      <td>7.970000e+03</td>
    </tr>
    <tr>
      <th>USER_ID</th>
      <td>8993.0</td>
      <td>8.207837e+08</td>
      <td>8.367043e+08</td>
      <td>6161.00</td>
      <td>1.768264e+08</td>
      <td>3.749968e+08</td>
      <td>1.519623e+09</td>
      <td>2.887160e+09</td>
    </tr>
    <tr>
      <th>PRODUCTION</th>
      <td>8993.0</td>
      <td>4.129879e-04</td>
      <td>3.123724e-03</td>
      <td>0.00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>6.671856e-02</td>
    </tr>
    <tr>
      <th>CONSUME</th>
      <td>8993.0</td>
      <td>1.855354e-02</td>
      <td>3.828559e-02</td>
      <td>0.00</td>
      <td>1.111976e-02</td>
      <td>1.111976e-02</td>
      <td>1.111976e-02</td>
      <td>4.781497e-01</td>
    </tr>
    <tr>
      <th>CONSUME_OTHERS</th>
      <td>8993.0</td>
      <td>3.014688e-02</td>
      <td>3.318166e-01</td>
      <td>0.00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000778e+01</td>
    </tr>
    <tr>
      <th>USER_VOLUME</th>
      <td>8993.0</td>
      <td>4.739069e+03</td>
      <td>1.166155e+05</td>
      <td>0.00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.320000e+06</td>
    </tr>
    <tr>
      <th>USER_DENSITY</th>
      <td>8993.0</td>
      <td>1.297801e+02</td>
      <td>1.584705e+03</td>
      <td>0.00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.336748e+04</td>
    </tr>
    <tr>
      <th>FILE_ID</th>
      <td>8993.0</td>
      <td>5.624463e+01</td>
      <td>3.372575e+01</td>
      <td>1.00</td>
      <td>2.300000e+01</td>
      <td>5.700000e+01</td>
      <td>8.400000e+01</td>
      <td>1.140000e+02</td>
    </tr>
    <tr>
      <th>FILE_SIZE</th>
      <td>8993.0</td>
      <td>5.020572e+00</td>
      <td>2.005420e+00</td>
      <td>1.00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>TOPIC_ID</th>
      <td>8993.0</td>
      <td>1.157000e+01</td>
      <td>7.437687e+00</td>
      <td>1.00</td>
      <td>3.000000e+00</td>
      <td>1.200000e+01</td>
      <td>1.700000e+01</td>
      <td>2.500000e+01</td>
    </tr>
    <tr>
      <th>TOPIC_VOLUME</th>
      <td>8993.0</td>
      <td>4.942270e+06</td>
      <td>3.453246e+06</td>
      <td>7.00</td>
      <td>1.143521e+06</td>
      <td>5.167328e+06</td>
      <td>7.602267e+06</td>
      <td>1.003422e+07</td>
    </tr>
    <tr>
      <th>TOPIC_DENSITY</th>
      <td>8993.0</td>
      <td>4.839933e+04</td>
      <td>3.093134e+04</td>
      <td>5.05</td>
      <td>1.711632e+04</td>
      <td>4.662073e+04</td>
      <td>5.986430e+04</td>
      <td>1.056557e+05</td>
    </tr>
  </tbody>
</table>
</div>



# Productores
Se filtran los usuarios productores utilizando la columna **USER_ROLE** 


```python
producers = df.loc[df['USER_ROLE'].isna() != True]
producers.sort_values("USER_ROLE").head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPERATION_ID</th>
      <th>TYPE</th>
      <th>USER_ID</th>
      <th>PRODUCTION</th>
      <th>CONSUME</th>
      <th>CONSUME_OTHERS</th>
      <th>USER_VOLUME</th>
      <th>USER_DENSITY</th>
      <th>USER_ROLE</th>
      <th>FILE_ID</th>
      <th>FILE_SIZE</th>
      <th>TOPIC_ID</th>
      <th>TOPIC_VOLUME</th>
      <th>TOPIC_DENSITY</th>
      <th>TOPIC_ROLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2304</th>
      <td>1881</td>
      <td>UPLOAD</td>
      <td>19769622</td>
      <td>0.022240</td>
      <td>0.0</td>
      <td>3.091293</td>
      <td>332488</td>
      <td>7033.649979</td>
      <td>Alpha</td>
      <td>83</td>
      <td>7</td>
      <td>12</td>
      <td>10034220</td>
      <td>105655.700111</td>
      <td>Beta</td>
    </tr>
    <tr>
      <th>1265</th>
      <td>1053</td>
      <td>UPLOAD</td>
      <td>1574083</td>
      <td>0.033359</td>
      <td>0.0</td>
      <td>3.002335</td>
      <td>314550</td>
      <td>3371.733329</td>
      <td>Alpha</td>
      <td>61</td>
      <td>7</td>
      <td>18</td>
      <td>646877</td>
      <td>17116.316643</td>
      <td>Gamma</td>
    </tr>
    <tr>
      <th>1466</th>
      <td>1218</td>
      <td>UPLOAD</td>
      <td>189379530</td>
      <td>0.011120</td>
      <td>0.0</td>
      <td>1.667964</td>
      <td>157500</td>
      <td>6108.800001</td>
      <td>Alpha</td>
      <td>76</td>
      <td>7</td>
      <td>13</td>
      <td>7602267</td>
      <td>59864.299968</td>
      <td>Beta</td>
    </tr>
    <tr>
      <th>7460</th>
      <td>6536</td>
      <td>UPLOAD</td>
      <td>10581097</td>
      <td>0.011120</td>
      <td>0.0</td>
      <td>1.667964</td>
      <td>135000</td>
      <td>6312.999986</td>
      <td>Alpha</td>
      <td>50</td>
      <td>6</td>
      <td>24</td>
      <td>3794028</td>
      <td>34855.833363</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>7317</th>
      <td>6402</td>
      <td>UPLOAD</td>
      <td>218549275</td>
      <td>0.011120</td>
      <td>0.0</td>
      <td>1.667964</td>
      <td>135000</td>
      <td>6345.416682</td>
      <td>Alpha</td>
      <td>66</td>
      <td>6</td>
      <td>13</td>
      <td>7602267</td>
      <td>59864.299968</td>
      <td>Beta</td>
    </tr>
  </tbody>
</table>
</div>



## Distribucion de los datos
Se muestra la kernel density estimation (KDE) la estima la distribucion probabilistica de los datos


```python
Xp         = producers.drop(useless_cols,axis=1) 
Xp_columns = Xp.columns
for x in Xp_columns:
    sns.displot(data=Xp,x=x,height=5,kde=True,color='black')
```


    
![png](./images/output_13_0.png)
    



    
![png](./images/output_13_1.png)
    



    
![png](./images/output_13_2.png)
    



    
![png](./images/output_13_3.png)
    



    
![png](./images/output_13_4.png)

    



    
![png](./images/output_13_5.png)
    



    
![png](./images/output_13_6.png)
    



    
![png](./images/output_13_7.png)
    


## Elimando Outliers
Utilizando el rango intercuartil, los valores atípicos se definen como observaciones que caen por debajo de Q1 - 1.5 IQR o por encima de Q3 + 1.5 IQR.


```python
for col in Xp_columns:
    Q1 = Xp[col].quantile(0.25)
    Q3 = Xp[col].quantile(0.75)
    IQR = Q3 - Q1 
    filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 *IQR)
    Xp = Xp.loc[filter] 
#     sns.boxplot(x=Xp[col],width=.5,color="red",fliersize=20)
```

# Consumidores
Se filtran usuarios consumidores utilizando la columna **USER_ROLE**, la cual es nula en caso de ser consumidor.


```python
consumers     = df.loc[df['USER_ROLE'].isna()]
consumers.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPERATION_ID</th>
      <th>TYPE</th>
      <th>USER_ID</th>
      <th>PRODUCTION</th>
      <th>CONSUME</th>
      <th>CONSUME_OTHERS</th>
      <th>USER_VOLUME</th>
      <th>USER_DENSITY</th>
      <th>USER_ROLE</th>
      <th>FILE_ID</th>
      <th>FILE_SIZE</th>
      <th>TOPIC_ID</th>
      <th>TOPIC_VOLUME</th>
      <th>TOPIC_DENSITY</th>
      <th>TOPIC_ROLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>DOWNLOAD</td>
      <td>1405534545</td>
      <td>0.0</td>
      <td>0.022240</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>DOWNLOAD</td>
      <td>1405534545</td>
      <td>0.0</td>
      <td>0.022240</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>DOWNLOAD</td>
      <td>640881322</td>
      <td>0.0</td>
      <td>0.011120</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>DOWNLOAD</td>
      <td>1416963334</td>
      <td>0.0</td>
      <td>0.011120</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>DOWNLOAD</td>
      <td>2189419908</td>
      <td>0.0</td>
      <td>0.033359</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100</td>
      <td>4</td>
      <td>17</td>
      <td>5167328</td>
      <td>47838.233327</td>
      <td>Alpha</td>
    </tr>
  </tbody>
</table>
</div>



# Archivos
Se extraen los archivos los cuales son cargados por los usuarios productores.


```python
files   = producers[producers['TYPE']=='UPLOAD']
files   = files.sort_values('FILE_ID')
files   = files[['FILE_ID','FILE_SIZE','TOPIC_ID','TOPIC_VOLUME','TOPIC_DENSITY','TOPIC_ROLE']]
files.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FILE_ID</th>
      <th>FILE_SIZE</th>
      <th>TOPIC_ID</th>
      <th>TOPIC_VOLUME</th>
      <th>TOPIC_DENSITY</th>
      <th>TOPIC_ROLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3267</th>
      <td>1</td>
      <td>5</td>
      <td>13</td>
      <td>7602267</td>
      <td>59864.299968</td>
      <td>Beta</td>
    </tr>
    <tr>
      <th>4239</th>
      <td>2</td>
      <td>2</td>
      <td>16</td>
      <td>141050</td>
      <td>11242.366624</td>
      <td>Gamma</td>
    </tr>
    <tr>
      <th>6742</th>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>7602267</td>
      <td>59864.299968</td>
      <td>Beta</td>
    </tr>
    <tr>
      <th>4245</th>
      <td>4</td>
      <td>7</td>
      <td>7</td>
      <td>111926</td>
      <td>22003.833399</td>
      <td>Gamma</td>
    </tr>
    <tr>
      <th>2890</th>
      <td>5</td>
      <td>6</td>
      <td>2</td>
      <td>7498850</td>
      <td>46620.733299</td>
      <td>Alpha</td>
    </tr>
  </tbody>
</table>
</div>



# Temas
Se extraen los temas los cuales son asignados a los archivos


```python
topics  = files[['TOPIC_ID','TOPIC_VOLUME','TOPIC_DENSITY']] 
_topics = topics.drop_duplicates()
_topics = _topics.sort_values('TOPIC_ID')
_topics.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TOPIC_ID</th>
      <th>TOPIC_VOLUME</th>
      <th>TOPIC_DENSITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2494</th>
      <td>1</td>
      <td>3055415</td>
      <td>58781.099956</td>
    </tr>
    <tr>
      <th>2890</th>
      <td>2</td>
      <td>7498850</td>
      <td>46620.733299</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>3</td>
      <td>1143521</td>
      <td>11415.500000</td>
    </tr>
    <tr>
      <th>2493</th>
      <td>5</td>
      <td>204680</td>
      <td>12843.766668</td>
    </tr>
    <tr>
      <th>4245</th>
      <td>7</td>
      <td>111926</td>
      <td>22003.833399</td>
    </tr>
  </tbody>
</table>
</div>



# Spectral & Agglomerative Clustering

## Creacion del grafo
La generacion se lleva acabo utilizando **Neo4j**, una base de datos basada en grafos.


```python
from neo4j import GraphDatabase
```


```python
class Neo4J:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def getSession(self):
        return self.driver.session()
    def print_greeting(self, message):
        with self.getSession() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)
    def writeTransaction(self,fx,*args,**kwargs):
        return self.getSession().write_transaction(fx,*args,**kwargs)


def setToCluster(tx,data):
    query = """
        MATCH (n) WHERE id(n) IN $node_ids
            SET n: {}
            RETURN n
    """.format(data[1])
    return tx.run(query,node_ids=data[0])
def createConsumer(tx,data):
    user_id = data[0]
    consume = data[1]
    file_id = data[2]
    query ="""
    MATCH (f:File {file_id:$file_id})
    MERGE (c:Consumer {user_id: $user_id,consume:$consume })
    MERGE (f)<-[d:DOWNLOAD]-(c)
    ON CREATE SET d.count =1
    ON MATCH SET d.count = d.count+1
    RETURN f,d,c
    """
    
    return tx.run(query,user_id=user_id,consume=consume,file_id=file_id)

def createProducer(tx,data):
    user_id        = data[0]
    production     = data[1] 
    consume        = data[2]
    consume_others = data[3]
    user_volume    = data[4]
    user_density   = data[5]
    user_role      = data[6]
    file_id        = data[7]
    file_size      = data[8]
    topic_id       = data[9]
    topic_volume   = data[10]
    topic_density  = data[11]
    topic_role     = data[12]
    query = """
        MERGE (p:Producer {user_id:$user_id,production:$production,volume:$volume,density:$density,consume:$consume,consume_others:$consume_others,role:$role})
        MERGE (f:File {file_id:$file_id,size:$file_size})
        MERGE (t:Topic {topic_id: $topic_id,volume:$topic_volume,density: $topic_density,role:$topic_role})
        MERGE (p)-[u:UPLOAD]->(f)-[ht:HAS_TOPIC]->(t)
        RETURN p,u,f,ht,t
    """
    
    return tx.run(query,
                        user_id=user_id,
                        production=production,
                        volume=user_volume,
                        density=user_density,
                        consume=consume,
                        consume_others=consume_others,
                        role=user_role,
                        file_id=file_id,
                        file_size=file_size,
                        topic_id=topic_id,
                        topic_volume=topic_volume,
                        topic_density=topic_density,
                        topic_role=topic_role
                 )
db = Neo4J("bolt://localhost:7687", "neo4j", "password")
# x = db.writeTransaction(fx=getPersons)
# print(x,len(x))
# db.close()
```

# Crear Productores
Se crean los productores utilizando la API de **Neo4j**


```python
# producers_data = producers[['USER_ID','PRODUCTION','CONSUME','CONSUMER_OTHERS','USER_VOLUME','USER_DENSITY','USER_ROLE']]
producers_data  = producers.copy()
producer_data.sort_values("USER_ROLE").head()
producers_data  = producers_data.loc[producers_data['TYPE'] == 'UPLOAD']
producers_data  = producers_data.drop(["OPERATION_ID","TYPE"],axis=1)
producers_data  = producers_data.to_numpy().tolist()
# producers_data = producers_data.to_numpy()
# producers_data.head(5)
def runCreateProducers():
    for data in producers_data:
        db.writeTransaction(createProducer,data)
        print("Producer[{}] created".format(data[0]))
        time.sleep(.2)
    db.close()
```

Se muestra el resultado que se obtiene, utilizando la herramienta de visualizacion de **Neo4j**. 

![title](images/p0.png)

En la misma operacion se lleva acabo la creacion de los archivos y la relacion de carga con el usuario productor

![title](images/puf.png)

A continuacion se muestra una imagen de los productores, la realacion de carga de archivos y los temas a los que pertenecen los archivos

![title](images/pufhtt.png)

# Crear consumidores
Se crearon los consumidores


```python
consumers_data = consumers.copy()
consumers_data = consumers_data.drop(['TYPE','FILE_SIZE','TOPIC_ID','TOPIC_VOLUME','TOPIC_DENSITY','OPERATION_ID','CONSUME_OTHERS','USER_VOLUME','USER_DENSITY','USER_ROLE','PRODUCTION','TOPIC_ROLE'], axis=1)
# consumers_data[consumers_data['FILE_ID']==113]
consumers_data = consumers_data.to_numpy().tolist()[100:]

def runCreateConsumers(data):
    for c in data:
        db.writeTransaction(createConsumers,c)
        print("Consumer[{}]".format(c[0]))
        time.sleep(.2)
# runCreateConsumers(consumers_data)
# consumers_data[100:]['FILE_ID'].unique()
```

El resultado de los consumidores, unicamente se muestran 2000 nodos, pero en realidad existen 8000 consumidores. 

![title](images/c2.png)

# Creacion de matriz de adyacencia
En teoría de grafos y ciencias de la computación, una matriz de adyacencia es una matriz cuadrada que se usa para representar un gráfico finito.


```python
A = np.load('./data/am.npy')
SPA = shortest_path(A,directed = False)
SPA[SPA == np.inf] =0
SPA
```




    array([[ 0.,  1.,  2., ..., 19., 12., 11.],
           [ 1.,  0.,  1., ..., 18., 11., 10.],
           [ 2.,  1.,  0., ..., 17., 10.,  9.],
           ...,
           [19., 18., 17., ...,  0., 23., 22.],
           [12., 11., 10., ..., 23.,  0.,  1.],
           [11., 10.,  9., ..., 22.,  1.,  0.]])



# Matriz de grados
En teoría de grafos, la matriz de grados es una matriz diagonal que contiene información sobre el grado de cada vértice.


```python
D = np.diag(A.sum(axis=1))
D
```




    array([[ 1.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  2.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0., 15., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  2.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  1.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  2.]])



# Laplacian
La teoría de grafos, la matriz laplaciana, también llamada grafo laplaciano, matriz de admitancia, matriz de Kirchhoff o laplaciana discreta, es una representación matricial de un grafo. La matriz laplaciana se puede utilizar para encontrar muchas propiedades útiles de un gráfico.


```python
L = D-A
L
```




    array([[ 1., -1.,  0., ...,  0.,  0.,  0.],
           [-1.,  2., -1., ...,  0.,  0.,  0.],
           [ 0., -1., 15., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  2.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  1., -1.],
           [ 0.,  0.,  0., ...,  0., -1.,  2.]])



# Eigenvalues y Eigenvector


```python
vals, vecs = np.linalg.eigh(L)
# vals, vecs = np.linalg.eig(L)
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]
vecs.shape,vals.shape
fg,ax =plt.subplots()
ax.set_xlabel("Index")
ax.set_ylabel("Eigenvalue")
ax.scatter(range(vals.shape[0]),vals)
# vecs

```




    <matplotlib.collections.PathCollection at 0x7f1586f91d60>




    
![png](./images/output_45_1.png)
    



```python

def prepareScorePlot():
    fig, ax    = plt.subplots(len(score_names),1)
    def f(i,scores,labels):
        ax[i].plot(n_clusters,scores[0],'r--',label = labels[0])
        ax[i].plot(n_clusters,scores[1],'b--',label = labels[1])
        ax[i].set_xlabel("Num. Clusters")
        ax[i].set_ylabel("Score")
        ax[i].set_xticks(n_clusters)
        ax[i].legend(loc='best')
    return f
        
def neo4jPlot(data):
    for d in data:
        db.writeTransaction(setToCluster,d)
def plotScoresByScoreName(**kwargs):
    labels = kwargs['labels']
    data   = kwargs['data']
    score_names = kwargs['score_names']
    f = prepareScorePlot() 
    for i,key in enumerate(score_names):
        f(i,data[key],labels[key])
```


```python
# clusters = (2,5,10,15,20)
eigen_solvers =('arpack', 'lobpcg', 'amg') #The eigenvalue decomposition strategy to use
n_components  = 6 # Number of eigen vectors to use for the spectral embedding
random_satate = (RandomState(None),10) # A pseudo random number generator used for the initialization of the lobpcg eigen vectors decomposition when eigen_solver='amg' and by the K-Means initialization
n_init        = 10 # Number of time the k-means algorithm will be run with different centroid seeds. 
gamma         = 1.0 # Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels
affinity      = ('precomputed','nearest_neighbors','rbf','precomputed_nearest_neighbors')
n_neighbors   = 10 # Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method
eigen_tol     = 0.0 # Stopping criterion for eigendecomposition of the Laplacian matrix when eigen_solver='arpack'
assign_labels = ('kmeans','discretize') # The strategy to use to assign labels in the embedding space
degree        = 3 # Degree of the polynomial kernel. Ignored by other kernels
coef0         = 1 # Zero coefficient for polynomial and sigmoid kernels.
n_jobs        = -1 # -1 means using all processors.
verbose       = True
def runSC(**kwargs):
    #score_results =[[],[],[],[],[],[]]
    score_results = {'sh':([],[]),'ch':([],[]),'db':([],[]),'dun':([],[])}
    show_scores   = kwargs['show_scores']
#     spec
    visualize     = kwargs['visualize']
    X             = kwargs['X']
    sX            = standarize(X=X)
    clusters      = kwargs['n_clusters']
    for k in clusters:
        agglomerativeClus = AgglomerativeClustering(
            n_clusters=k,
            affinity='euclidean',
            connectivity=A).fit(sX)
        spectralClus = SpectralClustering(n_clusters=k,
                                n_components=k,
                                assign_labels=assign_labels[1],
                                random_state=random_satate[0],
                                affinity=affinity[1],
                                eigen_solver=eigen_solvers[0]
                               ).fit(X)
#         
        spectralClusLabels      = spectralClus.labels_
        agglomerativeClusLabels = agglomerativeClus.labels_
        scUniqueLabels          = np.unique(spectralClusLabels)
        acUniqueLabels          = np.unique(agglomerativeClusLabels)
        if(visualize):
            scColors =  list(map(lambda x: [np.where(spectralClusLabels==x)[0].tolist(),'C'+str(x),x],scUniqueLabels))
            neo4jPlot(scColors)
            input("Continue")
            agColors =  list(map(lambda x: [np.where(agglomerativeClusLabels==x)[0].tolist(),'C'+str(x),x],acUniqueLabels))
            neo4jPlot(agColors)
            input("Continue")
        
        if(show_scores):
            sillhoutte_scores = (shs(sX,spectralClusLabels), shs(sX,agglomerativeClusLabels))
            calinki_scores =(chs(sX,spectralClusLabels), chs(sX,agglomerativeClusLabels))
            davies_scores  = (dbs(sX,spectralClusLabels),dbs(sX,agglomerativeClusLabels))
            dunIndex_scores   = (dunn(sX,spectralClusLabels), dunn(sX,agglomerativeClusLabels))
            score_results['sh'][0].append(sillhoutte_scores[0])
            score_results['sh'][1].append(sillhoutte_scores[1])
            score_results['ch'][0].append(calinki_scores[0])
            score_results['ch'][1].append(calinki_scores[1])
            score_results['db'][0].append(davies_scores[0])
            score_results['db'][1].append(davies_scores[1])
            score_results['dun'][0].append(dunIndex_scores[0])
            score_results['dun'][1].append(dunIndex_scores[1])
    return score_results
n_clusters = list(range(3,20))
score_names =('sh','ch','db','dun')
plot_labels = {
    'sh':('Silhouette(Spectral Clustering)','Silhouette(Agglomerative)'),
    'ch':('Calinski–Harabasz(Spectral Clustering','Calinski–Harabasz(Agglomerative)'),
    'db': ('Davies–Bouldin(Spectral Clustering','Davies–Bouldin(Agglomerative)'),
    'dun': ("Dunn index(Spectral Clustering)","Dunn index(Agglomerative)")
}
scores     = runSC(X=SPA,
                   n_clusters=n_clusters,
                   show_scores=True,
                   visualize=False)
# print(scores)
plotScoresByScoreName(data=scores,labels=plot_labels,score_names=score_names)
# plt.savefig('./images/foo_1.jpg')
plt.show()
```


    
![png](./images/output_47_0.png)
    

