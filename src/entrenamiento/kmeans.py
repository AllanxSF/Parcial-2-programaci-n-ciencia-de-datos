import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import joblib
from pathlib import Path

# Cargar datos

ruta = Path(__file__).resolve().parent.parent.parent / "data" / "7_agro_intelligence.csv"

df = pd.read_csv(
    ruta,
    encoding="utf-8"
)

# ===============================
# PREPROCESAMIENTO
# ===============================

# Convertir variables categóricas a numéricas
df_modelo = pd.get_dummies(
    df,
    columns=['tipo_fertilizante', 'clima_estacion'],
    drop_first=True
)

# Eliminamos target porque KMeans NO la usa
X = df_modelo.drop('target_calidad', axis=1)

# ===============================
# ENTRENAMIENTO MODELO SIN ESCALADO
# ===============================

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df['Cluster_SinEscalar'] = kmeans.fit_predict(X)

print(df.head())

# ===============================
# ESCALADO
# ===============================

scaler = StandardScaler()

X_escalado = scaler.fit_transform(X)

# ===============================
# ENTRENAMIENTO - MODELO ESCALADO
# ===============================

kmeans_escalado = KMeans(
    n_clusters=3,
    random_state=42
)

df['Cluster'] = kmeans_escalado.fit_predict(X_escalado)

print(df.head())


# ===============================
# GUARDAR MODELO ENTRENADO
# ===============================

ruta_modelos = Path(__file__).resolve().parent.parent.parent / "models"

# Crear carpeta si no existe
ruta_modelos.mkdir(exist_ok=True)

# Guardar modelo KMeans
joblib.dump(
    kmeans_escalado,
    ruta_modelos / "modelo_kmeans.pkl"
)

# Guardar escalador
joblib.dump(
    scaler,
    ruta_modelos / "scaler_kmeans.pkl"
)

print("\nModelo y escalador guardados correctamente")


print(
    df[[
        'Cluster_SinEscalar',
        'Cluster'
    ]].head()
)