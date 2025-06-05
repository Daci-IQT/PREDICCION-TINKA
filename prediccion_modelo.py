import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.set_page_config(page_title="Predicción Inteligente La Tinka", layout="centered")
st.title("🤖 Predicción Inteligente con Aprendizaje Automático")

# Cargar CSV
df = pd.read_csv("tinka.csv", sep=';')
df.columns = ["Fecha", "N1", "N2", "N3", "N4", "N5", "N6"]

# Crear dataset binario por número (1 si apareció, 0 si no)
todos_los_numeros = range(1, 51)
df_bin = pd.DataFrame(0, index=df.index, columns=todos_los_numeros)

for i, row in df.iterrows():
    for col in ["N1", "N2", "N3", "N4", "N5", "N6"]:
        num = int(row[col])
        df_bin.at[i, num] = 1

# Crear características (features)
frecuencia_total = df_bin.sum()
ultima_aparicion = df_bin[::-1].idxmax()  # última vez que salió
dias_sin_salir = len(df_bin) - ultima_aparicion
reciente = df_bin.tail(5).sum()

# Armar dataset para predicción
X_pred = pd.DataFrame({
    "Frecuencia": frecuencia_total,
    "DiasSinSalir": dias_sin_salir,
    "Reciente": reciente
})

# Normalizar características
X_pred = (X_pred - X_pred.mean()) / X_pred.std()

# Entrenar modelo por número y predecir probabilidad
X_train = []
y_train = []

for i in range(len(df_bin) - 1):  # entrenamiento usando todos menos el último
    stats = {
        "Frecuencia": df_bin.iloc[:i+1].sum(),
        "DiasSinSalir": (i + 1) - df_bin.iloc[:i+1][::-1].idxmax(),
        "Reciente": df_bin.iloc[max(0, i-4):i+1].sum()
    }
    for numero in todos_los_numeros:
        X_train.append([
            stats["Frecuencia"][numero],
            stats["DiasSinSalir"][numero],
            stats["Reciente"][numero]
        ])
        y_train.append(df_bin.iloc[i+1][numero])  # etiqueta real (siguiente sorteo)

X_train = pd.DataFrame(X_train, columns=["Frecuencia", "DiasSinSalir", "Reciente"])
X_train = (X_train - X_train.mean()) / X_train.std()  # normalizar
y_train = np.array(y_train)

# Entrenar el modelo logístico general
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir probabilidades para cada número (1 al 50)
probs = model.predict_proba(X_pred)[:, 1]
df_resultado = pd.DataFrame({"Número": todos_los_numeros, "Probabilidad": probs})
df_resultado = df_resultado.sort_values("Probabilidad", ascending=False)

st.subheader("📈 Probabilidad de aparición por número")
st.bar_chart(df_resultado.set_index("Número"))

# Generar hasta 5 combinaciones distintas de 6 números
st.subheader("🔮 Combinaciones sugeridas por el modelo")

predicciones = []
nums_disponibles = df_resultado["Número"].tolist()

for i in range(5):
    seleccion = sorted(np.random.choice(nums_disponibles[:20], size=6, replace=False))  # top 20 más probables
    predicciones.append(seleccion)

for idx, combinacion in enumerate(predicciones):
    lista_limpia = [int(num) for num in combinacion]  # 🔁 convertir np.int64 → int
    st.success(f"Predicción {idx + 1}: {lista_limpia}")

