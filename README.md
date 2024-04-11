# Streamlit Voice Wizard: Youtube2Text, Transcripción, Traducción y procesamiento de Audio
![image](https://github.com/demiurg0/Youtube2Text-Streamlit/assets/165735354/93c850c4-55e1-4bff-aed8-94ea44952054)


### Youtube download + Video2audio + Audio2text + Translate Spanish + PLN + Streamlit


Streamlit Voice Wizard es una herramienta versátil que simplifica el proceso de transcripción y traducción para archivos de audio y videos de YouTube. Con una interfaz intuitiva, los usuarios pueden cargar archivos de audio o ingresar URL de videos, que luego son transcritos utilizando el modelo Whisper o una API. Las transcripciones son posteriormente traducidas al español utilizando el modelo Opus-MT. Además, la herramienta realiza análisis lingüístico tanto en las transcripciones como en el texto traducido, extrayendo entidades y analizando el sentimiento. Además, obtiene información de los videos de YouTube, como detalles del canal, descripciones y títulos, para un análisis lingüístico adicional. Finalmente, los resultados se visualizan y muestran a través de la interfaz de Streamlit, ofreciendo a los usuarios una comprensión accesible y completa del contenido multimedia en diversos idiomas.
![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/4b13d644-67ca-437d-8250-148f16f73ea6)

## Características Principales:

- Transcripción instantánea de archivos de audio y videos de YouTube.
- Traducción automática de transcripciones al español.
- Análisis de entidades y sentimientos para comprender el contenido de manera más profunda.
- Visualización interactiva de los resultados para una fácil interpretación.

## Requisitos Previos:

- Python 3.x
- pip
- Instala las dependencias del archivo `requirements.txt`
- Añadir al codigo api key de huggingface y V3 de youtube
  
## Uso:

1. Ejecuta la aplicación ejecutando el siguiente comando en la terminal:
2. En la interfaz de usuario, carga un archivo de audio en formato .wav o ingresa la URL de un video de YouTube.
3. Selecciona el método de transcripción (API o Whisper).
4. Haz clic en el botón "Transcribir" para iniciar el proceso de transcripción y traducción.
5. Se mostrará la transcripción original, la traducción al español y los análisis de entidades y sentimientos.
6. Si cargaste un archivo de audio, también podrás reproducir el audio directamente desde la aplicación.

![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/5afbabc2-2071-45e9-900b-524c55d9bcd7)



## To-Do:
1. Añadir opcion de elgir a que idioma traducir
2. Añadir API a env o config
3. Añadir solucion para idioma arabe con Spacy ya que no hay modelo
4. Solucionar bug de index y columnas
5. Mejorar la gestion de archivos
6. Mejorar el output
7. Revisar y mejorar información del canal, intentar obtener y renderizar todos los videos, y pasar del video al canal en lugar de url, es decir; integrar como un buscador de youtube y render graf. de la inf.


## Licencia:

Este proyecto está bajo la [Licencia MIT](LICENSE).
