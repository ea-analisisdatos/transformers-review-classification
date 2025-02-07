# 🏷️ Clasificación de Revisiones de Productos con Transformers 🛒📊

## 📌 Descripción del Proyecto

Este proyecto implementa un modelo **Transformer basado en BERT** para clasificar reseñas de productos en un ecommerce. El objetivo es determinar si un comentario de un cliente es **positivo** 😊 o **negativo** 😞, ayudando a las empresas a entender la satisfacción de sus clientes.

Se ha utilizado **ajuste fino (fine-tuning)** en un modelo **BERT entrenado en portugués**, optimizando su desempeño para clasificar reseñas reales extraídas del ecommerce **Olist**.

⚠️ **Nota sobre el uso de recursos:** Para este proyecto, se ha trabajado con **una muestra reducida del dataset** y se ha configurado **epochs=1** con el fin de reducir el tiempo de procesamiento, ya que no se contaba con muchos recursos computacionales. Si dispones de una máquina con mayor capacidad de cómputo, puedes **eliminar la parte del código que toma la muestra** y **aumentar los epochs (ejemplo: epochs=3 o más)** para mejorar los resultados.

## 🚀 Tecnologías Utilizadas

- **Python** 🐍
- **Hugging Face Transformers** 🤗
- **TensorFlow** 🔥
- **Kaggle API** 📡
- **Google Colab / Jupyter Notebook** 🖥️
- **GPU para aceleración (CUDA en NVIDIA)** ⚡
- **Gradio** 🖥️ (para pruebas interactivas del modelo)

## 🛠️ Configuración del Entorno Virtual

Para evitar conflictos de versiones entre librerías, se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
# Crear un entorno virtual (Windows)
python -m venv venv

# Activar el entorno virtual (Windows)
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 🔗 Descarga del Dataset desde Kaggle

Para descargar el dataset directamente desde Kaggle, primero debes configurar tu API de Kaggle:

### 1️⃣ **Generar tu API de Kaggle (`kaggle.json`)**
1. Accede a [Kaggle](https://www.kaggle.com/).
2. Ve a tu **perfil** (clic en tu foto de usuario, arriba a la derecha).
3. Dirígete a **Account (Cuenta)**.
4. En la sección `API`, haz clic en **"Create New API Token"**.
5. Se descargará un archivo llamado **`kaggle.json`**.

### 2️⃣ **Ubicación de `kaggle.json` en el Proyecto**
El archivo descargado debe colocarse en la carpeta raíz del proyecto.

```bash
# Mover `kaggle.json` a la raíz del proyecto
mv ~/Downloads/kaggle.json .
```

Luego, el código lo moverá automáticamente al directorio `.kaggle` para su uso:

```python
import os
import shutil

# Crear directorio .kaggle si no existe
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

# Mover kaggle.json
shutil.move("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
print("✅ kaggle.json movido correctamente")
```

Después de esto, puedes descargar el dataset usando:
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce
```

## ⚡ Requisitos y Configuración

Antes de ejecutar el proyecto, instala las dependencias necesarias utilizando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Si necesitas instalar las librerías manualmente, usa:

```bash
pip install transformers datasets kaggle accelerate torch gradio pandas tensorflow scikit-learn jupyter ipywidgets tf-keras
```

Si utilizas **GPU NVIDIA**, asegúrate de instalar los drivers de CUDA para aprovechar la aceleración por hardware.

## 🎛️ Uso de Gradio para Probar el Modelo
Para facilitar la prueba del modelo, se ha integrado **Gradio**, una herramienta que permite interactuar con la inferencia del modelo a través de una interfaz gráfica.

### ✨ **Ejecutar Gradio**
Ejecuta la siguiente celda en tu cuaderno Jupyter para iniciar la interfaz de usuario:

```python
import gradio as gr
import tensorflow as tf

def classify_review(text):
    """Función que clasifica una reseña usando el modelo BERT entrenado."""
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)[0]
    prediction = tf.nn.softmax(outputs, axis=1)
    label = "Positiva" if tf.argmax(prediction, axis=1).numpy()[0] == 1 else "Negativa"
    return label

iface = gr.Interface(
    fn=classify_review,
    inputs="text",
    outputs="label",
    title="🔍 Clasificación de Reseñas con BERT",
    description="Ingrese una reseña y el modelo la clasificará como Positiva o Negativa.",
    theme="default"
)

iface.launch()
```

Esto abrirá una interfaz web donde puedes ingresar una reseña y recibir una predicción de si es **positiva** o **negativa**.

## 📊 Resultados Esperados

Tras el entrenamiento, el modelo será capaz de predecir si una reseña es **positiva o negativa** con un alto grado de precisión. Se podrá usar en sistemas de ecommerce para **automatizar la detección de sentimientos** en los comentarios de los clientes.

## ⭐ Contribuciones

¡Cualquier mejora es bienvenida! Si tienes ideas para optimizar el modelo o mejorar el preprocesamiento, no dudes en hacer un **Pull Request**.

📢 **Dale una estrella ⭐ si te ha sido útil!** 😊

