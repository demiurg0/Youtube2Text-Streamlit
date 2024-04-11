# Youtube2Text-Streamlit-Voice-Wizard
Streamlit Voice Wizard es una herramienta versátil que simplifica el proceso de transcripción,  traducción y procesamiento de texto para archivos de audio y videos de YouTube


# Streamlit Voice Wizard: Youtube2Text, Transcripción, Traducción y procesamiento de Audio

![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/4b13d644-67ca-437d-8250-148f16f73ea6)

### Youtube download + Video2audio + Audio2text + Translate Spanish + PLN + Streamlit


Streamlit Voice Wizard es una herramienta versátil que simplifica el proceso de transcripción y traducción para archivos de audio y videos de YouTube. Con una interfaz intuitiva, los usuarios pueden cargar archivos de audio o ingresar URL de videos, que luego son transcritos utilizando el modelo Whisper o una API. Las transcripciones son posteriormente traducidas al español utilizando el modelo Opus-MT. Además, la herramienta realiza análisis lingüístico tanto en las transcripciones como en el texto traducido, extrayendo entidades y analizando el sentimiento. Además, obtiene información de los videos de YouTube, como detalles del canal, descripciones y títulos, para un análisis lingüístico adicional. Finalmente, los resultados se visualizan y muestran a través de la interfaz de Streamlit, ofreciendo a los usuarios una comprensión accesible y completa del contenido multimedia en diversos idiomas.

## Características Principales:

- Transcripción instantánea de archivos de audio y videos de YouTube.
- Traducción automática de transcripciones al español.
- Análisis de entidades y sentimientos para comprender el contenido de manera más profunda.
- Visualización interactiva de los resultados para una fácil interpretación.

## Requisitos Previos:

- Python 3.x
- pip
- Instala las dependencias del archivo `requirements.txt`
- Añadir al codigo api key de huggingface y V3 de youtube --> TODO meterlo en un .env

## Uso:

1. Ejecuta la aplicación ejecutando el siguiente comando en la terminal:
2. En la interfaz de usuario, carga un archivo de audio en formato .wav o ingresa la URL de un video de YouTube.
3. Selecciona el método de transcripción (API o Whisper).
4. Haz clic en el botón "Transcribir" para iniciar el proceso de transcripción y traducción.
5. Se mostrará la transcripción original, la traducción al español y los análisis de entidades y sentimientos.
6. Si cargaste un archivo de audio, también podrás reproducir el audio directamente desde la aplicación.


[![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/3f128c95-867e-49a9-99ed-8e009e7336d2)
](https://private-user-images.githubusercontent.com/93614373/321261753-e10f081e-8fae-4e6f-9097-0ac6181f30b4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTI4Njg2ODAsIm5iZiI6MTcxMjg2ODM4MCwicGF0aCI6Ii85MzYxNDM3My8zMjEyNjE3NTMtZTEwZjA4MWUtOGZhZS00ZTZmLTkwOTctMGFjNjE4MWYzMGI0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDExVDIwNDYyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY4OTEzZGM3NmVlMGIwZmYzZTI0ZDAyMzY2YWUwMWYxYjJlMzM1MzBiYzc3MWQyNzY3MmFkNWQ5MDcyNmZkNzgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.VDlDb7ZfQzHy3Mdl4PbFBCapf4_DhR2OX5hO93yQ26c)

## To-Do:
1. Añadir opcion de elgir a que idioma traducir
2. Añadir solucion para idioma arabe con Spacy ya que no hay modelo
3. Solucionar bug de index y columnas
4. Mejorar la gestion de archivos
5. Mejorar el output
6. Revisar y mejorar información del canal, intentar obtener y renderizar todos los videos, y pasar del video al canal en lugar de url, es decir; integrar como un buscador de youtube y render graf. de la inf.


## Licencia:

Este proyecto está bajo la [Licencia MIT](LICENSE).
