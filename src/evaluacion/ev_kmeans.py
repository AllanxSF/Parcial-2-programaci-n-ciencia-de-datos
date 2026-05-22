import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from src.entrenamiento.kmeans import df, X_escalado
import pandas as pd

# ===============================
# SILHOUETTE SCORE
# ===============================

score = silhouette_score(
    X_escalado,
    df['Cluster']
)

print("\nSilhouette Score:")
print(score)

# ===============================
# PROMEDIOS POR CLUSTER
# ===============================

resumen = df.groupby('Cluster')[[
    'ph_suelo',
    'nitrogeno',
    'fosforo',
    'potasio',
    'humedad_suelo',
    'diametro_fruto',
    'peso_promedio'
]].mean()

print("\nPromedios por cluster:")
print(resumen)

# ===============================
# VISUALIZACIÓN
# ===============================

plt.figure(figsize=(8,6))

plt.scatter(
    df['diametro_fruto'],
    df['peso_promedio'],
    c=df['Cluster'],
    cmap='viridis'
)

plt.xlabel("Diámetro fruto")
plt.ylabel("Peso promedio")
plt.title("Clusters encontrados")

plt.show()

# ===============================
# COMPARACIÓN CON TARGET
# ===============================

comparacion = pd.crosstab(
    df['Cluster'],
    df['target_calidad']
)

print("\nComparación con target:")
print(comparacion)