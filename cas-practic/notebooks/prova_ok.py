import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
import pandas as pd

def preparar_dades():
    # Cargar y limpiar el dataset
    df = sns.load_dataset("penguins")
    df = df.dropna()

    # Dividir en conjuntos de entrenamiento, validación y prueba
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=1)

    # Separar etiquetas
    y_train = df_train.species.values
    y_val = df_val.species.values

    del df_train['species']
    del df_val['species']

    # Definir columnas categóricas y numéricas
    categorical = ['island', 'sex']
    numerical = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Escalar datos numéricos
    scaler = StandardScaler()
    scaler.fit(df_train[numerical])
    df_train[numerical] = scaler.transform(df_train[numerical])

    # Vectorizar datos categóricos
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    # Función para transformar nuevos conjuntos de datos
    def transformar_dades(X):
        X[numerical] = scaler.transform(X[numerical])
        X_dict = X[categorical + numerical].to_dict(orient='records')
        return dv.transform(X_dict)

    # Transformar conjuntos de datos de entrenamiento y validación
    X_train = dv.transform(train_dict)
    X_val = transformar_dades(df_val)

    return X_train.shape[0], X_val.shape[0]  # Retornem la mida dels conjunts de dades

# def init_svc():
    
#     # Preparar dades
#     X_train, y_train, X_val, y_val, transformar_dades = preparar_dades()

#     # Model SVC
#     svc_model = SVC()
#     svc_model.fit(X_train, y_train)

#     y_pred = svc_model.predict(X_val)

#     # Comparar prediccions
#     comparar = pd.DataFrame(list(zip(y_pred, y_val)), columns=['y_pred', 'y_val'])
#     comparar["correct"] = comparar.apply(lambda x: 1 if x.y_pred == x.y_val else 0, axis=1)

#     # Retornar el percentatge d'encerts
#     accuracy = round((comparar.correct.mean()) * 100, 3)
#     return f"{accuracy}% d'acert a la SVC"