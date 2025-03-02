import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from keras import regularizers
from keras.callbacks import EarlyStopping

# Calcular el peso de cada clase para balancear
class_weights = {0: 1., 1: 1.5, 2: 1.5}  # Ajusta estos pesos según el desequilibrio de las clases

def cargar_datos(archivo):
    preguntas = []
    respuestas = []
    with open(archivo, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                line = re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', line)  # Limpiar caracteres problemáticos
                pregunta, respuesta = line.strip().split(':')  # Dividir frase y etiqueta
                preguntas.append(pregunta.strip())
                respuestas.append(respuesta.strip())
            except ValueError:
                continue  # Ignorar líneas mal formateadas
    return np.array(preguntas), np.array(respuestas)

# Cargar los nuevos datos
archivo_datos = "datos_entrenamiento.txt"
preguntas, respuestas = cargar_datos(archivo_datos)

# Convertir las respuestas a valores numéricos (etiquetas)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(respuestas)

# Dividir los datos en entrenamiento y prueba
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    preguntas, y, test_size=0.2, random_state=42, stratify=y
)

# Crear y adaptar la capa de tokenización
text_vectorizer = tf.keras.layers.TextVectorization(output_mode='int', max_tokens=10000)
text_vectorizer.adapt(x_train_raw)

# Función para construir el modelo con más capas LSTM, Dropout y Regularización L2
def construir_modelo(text_vectorizer, num_clases=3):
    model = tf.keras.Sequential([
        text_vectorizer,
        tf.keras.layers.Embedding(
            input_dim=len(text_vectorizer.get_vocabulary()) + 1,
            output_dim=64,  # Tamaño mayor en la capa de embedding
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.3)),  # Usar Bidirectional LSTM
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Añadir capa densa con L2 Regularization
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Construir el modelo
modelo = construir_modelo(text_vectorizer, num_clases=len(label_encoder.classes_))

# Configurar el callback para EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo con más épocas y validación
modelo.fit(x_train_raw, y_train, epochs=30, batch_size=32, validation_data=(x_test_raw, y_test), 
            class_weight=class_weights, callbacks=[early_stopping])

# Evaluar el modelo
loss, accuracy = modelo.evaluate(x_test_raw, y_test, verbose=2)
print(f"Precisión del modelo: {accuracy}")

# Guardar el modelo entrenado
modelo.save("modelo_sentimiento", save_format="tf")

# Guardar el vocabulario del TextVectorization
vocabulario = text_vectorizer.get_vocabulary()
vocabulario = [re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', v) for v in vocabulario]

with open("tokenizer_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocabulario, f, ensure_ascii=False)

# Guardar las etiquetas de las respuestas (mapeo de números a etiquetas)
with open("etiquetas_respuestas.json", "w", encoding="utf-8") as f:
    json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False)

print("Modelo y tokenizer guardados exitosamente.")
