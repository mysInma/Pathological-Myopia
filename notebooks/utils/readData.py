import pandas as pd
from sklearn.model_selection import train_test_split


# Cargar los datos de entrenamiento desde un archivo CSV
train_data = pd.read_csv('datos_entrenamiento.csv')


# Cargar los datos de validación desde otro archivo CSV
val_data = pd.read_csv('datos_validacion.csv')

# Dividir los datos de entrenamiento en dos subconjuntos: uno para entrenamiento y otro para validación/ 20% para validación
X_train, X_val, y_train, y_val = train_test_split(train_data["imgPath"].values, train_data["label"].values, test_size=0.2, random_state=42)

# Obtener los datos de prueba y etiquetas correspondientes desde el archivo CSV de validación
X_test = val_data["imgPath"].values
y_test = val_data["label"].values





