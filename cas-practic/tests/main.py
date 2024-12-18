from flask import Flask, request
from notebooks.prova import init, init_svc, init_tree, init_knn  # Importa les funcions definides al mòdul
app = Flask(__name__)

@app.route('/rl')
def regresion_logistica():
    """
    Ruta per a la regressió logística amb paràmetres de columnes i valors específics.
    """
    # Obtenir les columnes categòriques i numèriques
    categorical = request.args.get('categorical', default="island,sex").split(",")
    numerical = request.args.get('numerical', default="bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g").split(",")

    # Recollir valors específics de la URL
    valors_especifics = {}
    
    # Coger valors per les variables numèriques
    for col in numerical:
        if col in request.args:
            valors_especifics[col] = request.args.get(col)

    # Coger valors per les variables categòriques
    for col in categorical:
        if col in request.args:
            valors_especifics[col] = request.args.get(col)

    # Cridar la funció init() amb paràmetres
    resultat = init(categorical, numerical, valors_especifics)
    return resultat

# http://127.0.0.1:5000/rl?bill_length_mm=0.254180&body_mass_g=0.643033&bill_depth_mm=-1.662317&flipper_length_mm=0.957524&island=Torgersen&sex=Male
# http://127.0.0.1:5000/rl?bill_length_mm=39.1&body_mass_g=3750.0&bill_depth_mm=-18.7&flipper_length_mm=181.0&island=Torgersen&sex=Female


@app.route('/svc')
def svc():
    """
    Ruta per al model SVC amb paràmetres de la URL.
    Crida la funció `init_svc` i retorna el resultat.
    """
    # Obtenir les columnes categòriques i numèriques des de la URL
    categorical = request.args.get('categorical', default="island,sex").split(",")
    numerical = request.args.get('numerical', default="bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g").split(",")

    # Recollir els valors específics de la URL per a les columnes numèriques i categòriques
    valors_especifics = {}
    for col in numerical + categorical:
        if col in request.args:
            valors_especifics[col] = request.args.get(col)

    # Cridar la funció init_svc() amb els paràmetres
    resultat = init_svc(categorical, numerical, valors_especifics)
    return resultat


# http://127.0.0.1:5000/svc?bill_length_mm=0.254180&body_mass_g=0.643033&bill_depth_mm=-1.662317&flipper_length_mm=0.957524&island=Torgersen&sex=Male
# http://127.0.0.1:5000/svc?bill_length_mm=39.1&body_mass_g=3750.0&bill_depth_mm=-18.7&flipper_length_mm=181.0&island=Torgersen&sex=Female


@app.route('/tree')
def tree():
    """
    Ruta per al model de l'Arbre de Decisió amb paràmetres de la URL.
    Crida la funció `init_tree` i retorna el resultat.
    """
    # Obtenir les columnes categòriques i numèriques des de la URL
    categorical = request.args.get('categorical', default="island,sex").split(",")
    numerical = request.args.get('numerical', default="bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g").split(",")

    # Recollir els valors específics de la URL per a les columnes numèriques i categòriques
    valors_especifics = {}
    for col in numerical + categorical:
        if col in request.args:
            valors_especifics[col] = request.args.get(col)

    # Cridar la funció init_tree() amb els paràmetres
    resultat = init_tree(categorical, numerical, valors_especifics)
    return resultat

@app.route('/knn')
def knn():
    """
    Ruta per al model KNN amb paràmetres de la URL.
    Crida la funció `init_knn` i retorna el resultat.
    """
    # Obtenir les columnes categòriques i numèriques des de la URL
    categorical = request.args.get('categorical', default="island,sex").split(",")
    numerical = request.args.get('numerical', default="bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g").split(",")

    # Recollir els valors específics de la URL per a les columnes numèriques i categòriques
    valors_especifics = {}
    for col in numerical + categorical:
        if col in request.args:
            valors_especifics[col] = request.args.get(col)

    # Obtenir el número de veïns (n_neighbors) des de la URL
    n_neighbors = int(request.args.get('n_neighbors', default=5))

    # Cridar la funció init_knn() amb els paràmetres
    resultat = init_knn(categorical, numerical, valors_especifics, n_neighbors)
    return resultat


if __name__ == '__main__':
    app.run(debug=True)