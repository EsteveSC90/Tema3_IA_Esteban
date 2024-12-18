import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def base(categorical, numerical, valors_especifics):
    # Càrrega de les dades
    df = sns.load_dataset("penguins")

    # Comprovar si ja existeix un fitxer CSV amb les dades processades
    csv_path = 'penguins_data.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)  # Llegir les dades existents des del CSV

    # Eliminar files NaN
    df = df.dropna()

    # Afegir els valors específics a df (per si n'hi ha)
    new_row = {}
    for col in categorical + numerical:
        if col in valors_especifics:
            new_row[col] = valors_especifics[col]

    # Crear un DataFrame amb la nova fila
    new_row_df = pd.DataFrame([new_row])

    # Afegir la nova fila al DataFrame original
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Guardar el DataFrame actualitzat al fitxer CSV
    df.to_csv(csv_path, index=False)

    # Preparam les dades per al model
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=1)
    y_train = df_train.species.values
    y_val = df_val.species.values

    del df_train['species']
    del df_val['species']

    # Escalat de columnes numèriques
    scaler = StandardScaler()
    scaler.fit(df_train[numerical])
    df_train[numerical] = scaler.transform(df_train[numerical])

    # Vectorització de les dades
    dv = DictVectorizer(sparse=False)
    dv.fit(df_train[categorical + numerical].to_dict(orient='records'))

    # Modificar df_val amb els valors específics si existeixen
    for col, val in valors_especifics.items():
        if col in numerical:
            df_val[col] = float(val)  # Convertir a número
        elif col in categorical:
            df_val[col] = val  # Assignar el valor categòric

    def transformar_dades(X):
        X[numerical] = scaler.transform(X[numerical])
        X_dict = X[categorical + numerical].to_dict(orient='records') 
        return dv.transform(X_dict)

    return df_train, y_train, df_val, y_val, transformar_dades, dv

def init(categorical, numerical, valors_especifics):
    df_train, y_train, df_val, y_val, transformar_dades, dv = base(categorical, numerical, valors_especifics)

    # Logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(dv.transform(df_train.to_dict(orient='records')), y_train)

    X_val = transformar_dades(df_val)
    y_pred = lr_model.predict(X_val)

    comparar = pd.DataFrame(list(zip(y_pred, y_val)), columns=['y_pred', 'y_val'])
    comparar["correct"] = comparar.apply(lambda x: 1 if x.y_pred == x.y_val else 0, axis=1)

    return f"{round((comparar.correct.mean()) * 100, 3)}% d'acert a la Regressió Logistica"

def init_svc(categorical, numerical, valors_especifics):
    df_train, y_train, df_val, y_val, transformar_dades, dv = base(categorical, numerical, valors_especifics)

    # Model SVC
    svc_model = SVC()
    svc_model.fit(dv.transform(df_train.to_dict(orient='records')), y_train)

    X_val = transformar_dades(df_val)
    y_pred = svc_model.predict(X_val)

    comparar = pd.DataFrame(list(zip(y_pred, y_val)), columns=['y_pred', 'y_val'])
    comparar["correct"] = comparar.apply(lambda x: 1 if x.y_pred == x.y_val else 0, axis=1)

    return f"{round((comparar.correct.mean()) * 100, 3)}% d'acert a la SVC"

def init_tree(categorical, numerical, valors_especifics):
    # Obtener los conjuntos de datos y transformaciones
    df_train, y_train, df_val, y_val, transformar_dades, dv = base(categorical, numerical, valors_especifics)

    # Modelo DecisionTreeClassifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(dv.transform(df_train.to_dict(orient='records')), y_train)

    X_val = transformar_dades(df_val)
    y_pred_tree = decision_tree_model.predict(X_val)

    # Comparar resultados del árbol de decisión
    comparar_tree = pd.DataFrame(list(zip(y_pred_tree, y_val)), columns=['y_pred', 'y_val'])
    comparar_tree["correct"] = comparar_tree.apply(lambda x: 1 if x.y_pred == x.y_val else 0, axis=1)

    accuracy_tree = round((comparar_tree.correct.mean()) * 100, 3)

    return f"{accuracy_tree}% d'encert amb el Árbol de Decisión"

def init_knn(categorical, numerical, valors_especifics, n_neighbors=5):
    """
    Inicializa y evalúa un modelo KNN con los parámetros proporcionados.

    Args:
        categorical (list): Lista de columnas categóricas.
        numerical (list): Lista de columnas numéricas.
        valors_especifics (dict): Valores específicos para ciertas columnas.
        n_neighbors (int): Número de vecinos (k) para el modelo KNN.

    Returns:
        str: Precisión del modelo en porcentaje.
    """
    # Obtener los conjuntos de datos y transformaciones
    df_train, y_train, df_val, y_val, transformar_dades, dv = base(categorical, numerical, valors_especifics)

    # Modelo KNN
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(dv.transform(df_train.to_dict(orient='records')), y_train)

    # Transformar los datos de validación y hacer predicciones
    X_val = transformar_dades(df_val)
    y_pred_knn = knn_model.predict(X_val)

    # Calcular la precisión usando accuracy_score
    accuracy = accuracy_score(y_val, y_pred_knn)

    return f"{accuracy * 100:.2f}% d'encert amb el model KNN"