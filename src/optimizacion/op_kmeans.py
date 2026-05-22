from src.entrenamiento.kmeans import X_escalado

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ===============================
# MÉTODO DEL CODO
# ===============================

inercia = []

for k in range(1,11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42
    )

    modelo.fit(
        X_escalado
    )

    inercia.append(
        modelo.inertia_
    )

plt.figure(figsize=(8,6))

plt.plot(
    range(1,11),
    inercia,
    marker='o'
)

plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del Codo")

plt.show()

# ===============================
# COMPARACIÓN DE K MEDIANTE
# SILHOUETTE SCORE
# ===============================

print("\nComparación de K:")

for k in range(2,8):

    modelo = KMeans(
        n_clusters=k,
        random_state=42
    )

    grupos = modelo.fit_predict(
        X_escalado
    )

    score = silhouette_score(
        X_escalado,
        grupos
    )

    print(
        f"K={k}: {score:.3f}"
    )

print("\nOptimización completada correctamente")