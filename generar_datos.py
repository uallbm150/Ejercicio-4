from parrot import Parrot
import re

def preprocesar_texto(texto):
    # Eliminar caracteres no alfanuméricos y números (excepto letras)
    texto = re.sub(r'[^a-zA-ZáéíóúüñÑ\s]', '', texto)  # Permitir caracteres especiales del español
    
    # Eliminar palabras de un solo carácter y limpiar espacios adicionales
    texto = re.sub(r'\b\w{1}\b', '', texto)
    
    # Eliminar espacios en blanco excesivos
    texto = ' '.join(texto.split())  # Elimina espacios extra entre palabras
    
    return texto

# Instanciamos el objeto Parrot
parrot = Parrot()

# Función para generar parafraseos
def generar_datos_sinteticos(frase_original, categoria, cantidad=5):
    # Preprocesamos el texto antes de parafrasearlo
    frase_procesada = preprocesar_texto(frase_original)
    
    frases_generadas = []
    
    # Generar parafraseos (con el modelo de Parrot para español, si es posible)
    parafraseos = parrot.augment(input_phrase=frase_procesada)
    
    # Verificamos si parafraseos es válido (no es None)
    if parafraseos is not None:
        # Recoger las frases generadas y devolverlas con la categoría
        for parafraseo in parafraseos:
            frases_generadas.append(f"{parafraseo}:{categoria}")
    
    return frases_generadas

# Función para registrar las interacciones de los usuarios y generar parafraseos
def registrar_interaccion_usuario(archivo="datos_entrenamiento.txt"):
    with open(archivo, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    
    # Iteramos sobre cada línea del archivo y generamos los parafraseos
    for linea in lineas:
        if ':' in linea:  # Aseguramos que la línea tenga la estructura esperada (frase:categoria)
            frase, categoria = linea.strip().split(':')
            frases_generadas = generar_datos_sinteticos(frase, categoria)
            
            # Guardar las frases generadas en el archivo
            with open(archivo, "a", encoding="utf-8") as f_out:
                for frase in frases_generadas:
                    f_out.write(f"{frase}\n")

# Ejecutamos el proceso para parafrasear todo el archivo
registrar_interaccion_usuario()
