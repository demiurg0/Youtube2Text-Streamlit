import pytest
from app import (
    youtube_to_wav, transcribe_with_whisper, transcribe_with_api,
    traducir_texto_espanol, analizar_sentimiento, analizar_entidades,
    obtener_detalles_canal, traducir_texto, obtener_id_canal,
    obtener_y_traducir_detalles_canal, obtener_informacion_canal
)

# Casos de prueba para la función youtube_to_wav
def test_youtube_to_wav():
    # Prueba con una URL válida de YouTube
    youtube_url = "https://www.youtube.com/watch?v="  # Cambiar por una URL real
    video_path, audio_path = youtube_to_wav(youtube_url)
    assert video_path is not None
    assert audio_path is not None

    # Prueba con una URL inválida de YouTube
    youtube_url_invalid = "invalid_url"
    with pytest.raises(Exception):
        video_path, audio_path = youtube_to_wav(youtube_url_invalid)

# Casos de prueba para la función transcribe_with_whisper
def test_transcribe_with_whisper():
    # Prueba con un archivo de audio válido
    audio_path = "audio.wav"  # Cambiar por un archivo de audio real
    transcription = transcribe_with_whisper(audio_path)
    assert isinstance(transcription, str)

    # Prueba con un archivo de audio inválido
    audio_path_invalid = "invalid_audio.wav"
    with pytest.raises(Exception):
        transcription = transcribe_with_whisper(audio_path_invalid)

# Casos de prueba para la función transcribe_with_api
def test_transcribe_with_api():
    # Prueba con un archivo de audio válido
    audio_path = "audio.wav"  # Cambiar por un archivo de audio real
    transcription = transcribe_with_api(audio_path)
    assert isinstance(transcription, str)

    # Prueba con un archivo de audio inválido
    audio_path_invalid = "invalid_audio.wav"
    with pytest.raises(Exception):
        transcription = transcribe_with_api(audio_path_invalid)

# Casos de prueba para la función traducir_texto_espanol
def test_traducir_texto_espanol():
    # Prueba con una transcripción válida y un idioma original conocido
    texto = "Hello, how are you?"  # Cambiar por una transcripción real
    idioma_original = "en"
    translation = traducir_texto_espanol(texto, idioma_original)
    assert isinstance(translation, str)

    # Prueba con una transcripción válida y un idioma original desconocido
    texto_desconocido = "Some text"  # Cambiar por una transcripción real
    idioma_original_desconocido = "unknown"
    with pytest.raises(Exception):
        translation = traducir_texto_espanol(texto_desconocido, idioma_original_desconocido)

# Casos de prueba para las funciones de análisis de sentimientos y entidades
def test_analisis_sentimiento_y_entidades():
    # Prueba con una transcripción válida
    texto = "This is a sample transcription."  # Cambiar por una transcripción real
    sentimiento = analizar_sentimiento(texto)
    entidades = analizar_entidades(texto)
    assert isinstance(sentimiento, dict)
    assert isinstance(entidades, list)

    # Prueba con una transcripción inválida
    texto_invalido = None
    with pytest.raises(Exception):
        sentimiento = analizar_sentimiento(texto_invalido)
        entidades = analizar_entidades(texto_invalido)

# Casos de prueba para la función obtener_detalles_canal
def test_obtener_detalles_canal():
    # Prueba con un ID de canal válido
    youtube_api = ""  # Cambiar por tu propia clave de API
    channel_id_valido = ""  # Cambiar por un ID de canal real
    detalles_canal = obtener_detalles_canal(youtube_api, channel_id_valido)
    assert isinstance(detalles_canal, dict)

    # Prueba con un ID de canal inválido
    channel_id_invalido = "invalid_channel_id"
    detalles_canal_invalido = obtener_detalles_canal(youtube_api, channel_id_invalido)
    assert detalles_canal_invalido == {}

def test_obtener_id_canal():
    # Prueba con una URL válida de YouTube
    youtube_api = ""  # Cambiar por tu propia clave de API
    youtube_url_valida = "https://www.youtube.com/watch?v="  # Cambiar por una URL real
    video_id = youtube_url_valida.split('v=')[1]
    channel_id = obtener_id_canal(youtube_api, video_id)
    assert isinstance(channel_id, str)

    # Prueba con una URL inválida de YouTube
    youtube_url_invalida = "invalid_url"
    with pytest.raises(Exception):
        video_id_invalido = youtube_url_invalida.split('v=')[1]
        channel_id = obtener_id_canal(youtube_api, video_id_invalido)

# Casos de prueba para la función traducir_texto
def test_traducir_texto():
    # Prueba con un texto válido y un idioma original conocido
    texto_valido = "Hello, how are you?"  # Cambiar por un texto real
    idioma_original_valido = "en"
    translation = traducir_texto(texto_valido, idioma_original_valido)
    assert isinstance(translation, str)

    # Prueba con un texto válido y un idioma original desconocido
    texto_desconocido = "Some text"  # Cambiar por un texto real
    idioma_original_desconocido = "unknown"
    with pytest.raises(Exception):
        translation = traducir_texto(texto_desconocido, idioma_original_desconocido)

# Casos de prueba para la función obtener_y_traducir_detalles_canal
def test_obtener_y_traducir_detalles_canal():
    # Prueba con una URL válida de YouTube
    youtube_api = ""  # Cambiar por tu propia clave de API
    youtube_url_valida = "https://www.youtube.com/watch?v="  # Cambiar por una URL real
    video_id = youtube_url_valida.split('v=')[1]
    detalles_traducidos = obtener_y_traducir_detalles_canal(youtube_api, video_id, 'en')
    assert isinstance(detalles_traducidos, dict)

    # Prueba con una URL inválida de YouTube
    youtube_url_invalida = "invalid_url"
    with pytest.raises(Exception):
        detalles_traducidos_invalidos = obtener_y_traducir_detalles_canal(youtube_api, youtube_url_invalida, 'en')

# Casos de prueba para la función obtener_informacion_canal
def test_obtener_informacion_canal():
    # Prueba con un objeto de YouTube válido
    youtube_api = ""  # Cambiar por tu propia clave de API
    informacion_videos = obtener_informacion_canal(youtube_api)
    assert isinstance(informacion_videos, list)
    assert len(informacion_videos) > 0

    # Prueba con un objeto de YouTube inválido
    youtube_api_invalido = None
    with pytest.raises(Exception):
        informacion_videos_invalido = obtener_informacion_canal(youtube_api_invalido)
