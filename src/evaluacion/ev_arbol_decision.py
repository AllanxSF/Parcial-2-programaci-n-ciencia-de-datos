from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


base_dir = Path(__file__).resolve().parents[2]
ruta_datos = base_dir / "data" / "7_agro_intelligence.csv"
ruta_modelo = base_dir / "models" / "arbol_decision.pkl"

df = pd.read_csv(ruta_datos, encoding="utf-8")

x = df[['ph_suelo', 'nitrogeno', 'fosforo', 'potasio', 'humedad_suelo',
	'radiacion_solar', 'riego_litros', 'tipo_fertilizante',
	'incidencia_plagas', 'diametro_fruto', 'peso_promedio',
	'clima_estacion']]
y = df['target_calidad']

x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size=0.2, random_state=42, stratify=y
)

modelo = joblib.load(ruta_modelo)
y_pred = modelo.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()
metrics = {
	"accuracy": accuracy,
	"confusion_matrix": cm,
	"classification_report": report_dict,
}
