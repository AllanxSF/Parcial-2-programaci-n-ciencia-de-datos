#Preparacion del entorno
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import sys
import os

#Cargar el dataset
sys.path.append(os.path.abspath(".."))
base_dir = Path(__file__).resolve().parents[2]
ruta = base_dir / "data" / "7_agro_intelligence.csv"
df = pd.read_csv(ruta, encoding="utf-8")

#copiamos el dataframe para no modificar el original
df = df.copy()

#preparacion de los datos
x = df[['ph_suelo', 'nitrogeno', 'fosforo', 'potasio', 'humedad_suelo',
        'radiacion_solar', 'riego_litros', 'tipo_fertilizante',
        'incidencia_plagas', 'diametro_fruto', 'peso_promedio',
        'clima_estacion']]
y = df['target_calidad']

columnas_categoricas = ['tipo_fertilizante', 'clima_estacion']

preprocesador = ColumnTransformer(
        transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas),
        ],
        remainder='passthrough'
)

#creamos el modelo de arbol de decision
modelo = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('clasificador', DecisionTreeClassifier(random_state=42, max_depth=3))
])

#dividimos para entrenar y evaluar despues con el mismo archivo guardado
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
)

#entrenamos el modelo
modelo.fit(x_train, y_train)

#guardamos el modelo entrenado
modelo_dir = base_dir / "models"
modelo_dir.mkdir(exist_ok=True)
joblib.dump(modelo, modelo_dir / "arbol_decision.pkl")
print("Modelo de árbol de decisión guardado en models/arbol_decision.pkl")

