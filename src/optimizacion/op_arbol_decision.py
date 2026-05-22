from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


base_dir = Path(__file__).resolve().parents[2]
ruta_datos = base_dir / "data" / "7_agro_intelligence.csv"
ruta_modelo = base_dir / "models" / "arbol_decision_optimizado.pkl"

df = pd.read_csv(ruta_datos, encoding="utf-8")

x = df[
	[
		"ph_suelo",
		"nitrogeno",
		"fosforo",
		"potasio",
		"humedad_suelo",
		"radiacion_solar",
		"riego_litros",
		"tipo_fertilizante",
		"incidencia_plagas",
		"diametro_fruto",
		"peso_promedio",
		"clima_estacion",
	]
]
y = df["target_calidad"]

x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size=0.2, random_state=42, stratify=y
)

columnas_categoricas = ["tipo_fertilizante", "clima_estacion"]

preprocesador = ColumnTransformer(
	transformers=[
		("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
	],
	remainder="passthrough",
)

pipeline = Pipeline(
	steps=[
		("preprocesamiento", preprocesador),
		("clasificador", DecisionTreeClassifier(random_state=42)),
	]
)

grid_parametros = {
	"clasificador__criterion": ["gini", "entropy", "log_loss"],
	"clasificador__max_depth": [3],
	"clasificador__min_samples_split": [2, 5, 10, 20],
	"clasificador__min_samples_leaf": [1, 2, 4, 8],
	"clasificador__max_features": [None, "sqrt", "log2"],
}

random_parametros = {
	"clasificador__criterion": ["gini", "entropy", "log_loss"],
	"clasificador__max_depth": randint(3, 4),
	"clasificador__min_samples_split": randint(2, 25),
	"clasificador__min_samples_leaf": randint(1, 12),
	"clasificador__max_features": [None, "sqrt", "log2"],
}

grid_search = GridSearchCV(
	pipeline,
	grid_parametros,
	cv=5,
	n_jobs=-1,
	scoring="accuracy",
	refit=True,
)

random_search = RandomizedSearchCV(
	pipeline,
	random_parametros,
	n_iter=30,
	cv=5,
	n_jobs=-1,
	scoring="accuracy",
	refit=True,
	random_state=42,
)

grid_search.fit(x_train, y_train)
random_search.fit(x_train, y_train)

mejor_busqueda = grid_search if grid_search.best_score_ >= random_search.best_score_ else random_search
modelo_optimizado = mejor_busqueda.best_estimator_

y_pred = modelo_optimizado.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()

resumen_busquedas = pd.DataFrame(
	[
		{
			"metodo": "GridSearchCV",
			"mejor_score_cv": grid_search.best_score_,
			"accuracy_test": accuracy_score(y_test, grid_search.best_estimator_.predict(x_test)),
			"mejores_parametros": grid_search.best_params_,
		},
		{
			"metodo": "RandomizedSearchCV",
			"mejor_score_cv": random_search.best_score_,
			"accuracy_test": accuracy_score(y_test, random_search.best_estimator_.predict(x_test)),
			"mejores_parametros": random_search.best_params_,
		},
	]
)

metrics = {
	"accuracy": accuracy,
	"confusion_matrix": cm,
	"classification_report": report_dict,
	"best_method": mejor_busqueda.__class__.__name__,
	"best_cv_score": mejor_busqueda.best_score_,
	"best_params": mejor_busqueda.best_params_,
}

modelo_dir = base_dir / "models"
modelo_dir.mkdir(exist_ok=True)
joblib.dump(modelo_optimizado, ruta_modelo)