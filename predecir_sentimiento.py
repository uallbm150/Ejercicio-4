import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string

# Carga el modelo en formato TF o .keras
modelo = tf.keras.models.load_model("modelo_sentimiento")

# Mapeo de índices a etiquetas
etiquetas = {
    0: "Halago",
    1: "Información adicional",
    2: "Necesita apoyo técnico"
}

def predecir_sentimiento(model, texto):
    """ Recibe un texto y devuelve la clase de emoción."""
    entrada = [texto]
    probabilidades = model.predict(entrada)
    indice_pred = np.argmax(probabilidades, axis=1)[0]
    return etiquetas[indice_pred]

app = Flask(__name__)

# Plantilla HTML mejorada con los detalles mencionados
html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8"/>
    <title>Atención al usuario</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        /* Estilos base para un diseño moderno */
        body {
            background: linear-gradient(135deg, #6A82FB, #FC5C7D);
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            flex-direction: column;
        }

        header {
            background-color: #ffffff;
            color: #5c6bc0;
            text-align: center;
            padding: 1rem;
            width: 100%;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        h1 {
            font-size: 2rem;
            font-weight: 600;
            margin: 0;
        }

        .container {
            background-color: #fff;
            width: 100%;
            max-width: 450px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }

        label {
            display: block;
            margin-bottom: 12px;
            font-size: 1.1rem;
            font-weight: 500;
            color: #333;
        }

        textarea {
    width: 100%;
    padding: 16px;
    font-size: 1rem;
    border-radius: 8px;
    border: 1px solid #ccc;
    resize: none;
    min-height: 120px;
    outline: none;
    box-sizing: border-box;
}

textarea:focus {
    border-color: #5c6bc0;
    box-shadow: 0 0 10px rgba(92, 107, 192, 0.5);
}

input[type="text"], textarea {
    display: block;
    width: 100%;
    height: auto;
    margin-bottom: 10px;
}

input:focus, textarea:focus {
    border-color: #5c6bc0;
    box-shadow: 0 0 8px rgba(92, 107, 192, 0.5);
}


        .btn {
            width: 100%;
            padding: 15px;
            background-color: #5c6bc0;
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
            transition: background-color 0.3s ease-in-out, transform 0.2s ease-in-out;
        }

        .btn:hover {
            background-color: #3f4c9a;
            transform: scale(1.05);
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f2f2f2;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            text-align: left;
            color: #333;
        }

        .result p {
            margin: 5px 0;
            font-size: 1.1rem;
        }

        .result strong {
            color: #5c6bc0;
        }
    </style>
</head>
<body>

<header>
    <h1>Atención al usuario</h1>
</header>

<div class="container">
    <form method="POST">
        <label for="texto">Introduce una frase:</label>
        <textarea name="texto" id="texto" placeholder="Escribe aquí..." required></textarea><br>
        <button class="btn" type="submit">Predecir</button>
    </form>

    {% if resultado is not none %}
    <div class="result">
        <p><strong>Texto ingresado:</strong> {{ texto }}</p>
        <p><strong>Sentimiento predicho:</strong> {{ resultado }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    texto_usuario = ""
    if request.method == 'POST':
        texto_usuario = request.form.get('texto', '')
        resultado = predecir_sentimiento(modelo, texto_usuario)

    return render_template_string(html_template, resultado=resultado, texto=texto_usuario)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
