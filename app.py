import streamlit as st
import requests
import json
from pytube import YouTube
from moviepy.editor import AudioFileClip
from tempfile import NamedTemporaryFile
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np
from loguru import logger
import os
import librosa
from pathlib import Path
import subprocess
from langdetect import detect
import json
from transformers import MarianMTModel, MarianTokenizer
from googleapiclient.discovery import build
import spacy_streamlit
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nlp_es = spacy.load("es_core_news_sm")

st.set_page_config(page_title="Streamlit Voice Wizard", page_icon="👄")
st.title("Streamlit Voice Wizard")
st.write("""
- Sube un archivo .wav o ingresa una URL de YouTube para transcribirlo.
- Elige entre usar una API (hf) o el modelo Whisper para la transcripción.
- Debenes añadir en el codigo tu api de Huggingface y tu API de youtube V3
""")

logger.add("app.log", rotation="10 MB")

def youtube_to_wav(youtube_url: str):
    try:
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(only_audio=True).first()
        temp_video_file = NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_file_name = temp_video_file.name
        temp_video_file.close()  # Cerrar el archivo para que moviepy pueda escribir en él
        
        video_stream.download(filename=temp_video_file_name)
        logger.info("Video downloaded successfully.")

        clip = AudioFileClip(temp_video_file_name)
        temp_audio_file = NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_file_name = temp_audio_file.name
        temp_audio_file.close()
        clip.write_audiofile(temp_audio_file_name, codec='pcm_s16le')
        logger.info("Audio extracted and saved as WAV.")

        logger.info("Temporary files are kept for playback.")

        return temp_video_file_name, temp_audio_file_name  # Retorna ambos nombres de archivo
    except Exception as e:
        logger.error(f"Error in youtube_to_wav: {e}")
        raise

@st.cache_resource
def load_whisper_model(model_name: str = "openai/whisper-large"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)
    logger.info("Whisper model and processor loaded successfully.")
    return model, processor, device

def transcribir_audio(audio_path):
    model, processor, device = load_whisper_model()
    audio_input = processor(audio_path, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        outputs = model.generate(input_values=audio_input, max_length=512)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

def process_audio(audio_path: str, output_path: str, include_transcription: bool = False):
    transcription = transcribir_audio(audio_path)
    if include_transcription:
        detected_lang = detect(transcription)
        content_to_append = {"transcription": transcription, "language": detected_lang}
        append_to_json(output_path, content_to_append)


def transcribe_with_whisper(audio_path: str, segment_length: int = 30):
    model, processor, device = load_whisper_model()
    audio_signal, sampling_rate = librosa.load(audio_path, sr=16000)
    total_length_in_seconds = len(audio_signal) / sampling_rate
    segments = np.arange(0, total_length_in_seconds, segment_length)
    full_transcription = ""
    for start in segments:
        end = start + segment_length
        if end > total_length_in_seconds:
            end = total_length_in_seconds
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        segment_signal = audio_signal[start_sample:end_sample]
        input_features = processor(segment_signal, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=512)
        segment_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription += " " + segment_transcription
    
    return full_transcription

def traducir_texto_espanol(texto: str, idioma_original: str) -> str:
    if idioma_original == 'es':
        return texto  # Si el texto ya está en español, lo retorna tal cual
    try:
        modelo_traduccion_id = f"Helsinki-NLP/opus-mt-{idioma_original}-es"
        tokenizer = MarianTokenizer.from_pretrained(modelo_traduccion_id)
        model = MarianMTModel.from_pretrained(modelo_traduccion_id)
        texto_para_traducir = tokenizer.prepare_seq2seq_batch([texto], return_tensors="pt")
        texto_traducido = model.generate(**texto_para_traducir)
        transcripcion_traducida = tokenizer.decode(texto_traducido[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error durante la traducción: {e}")
        transcripcion_traducida = texto  # Retorna el texto original en caso de error
    return transcripcion_traducida

def append_to_json(output_path, content):
    with open(output_path, "a+", encoding="utf-8") as file:
        file.seek(0)
        try:
            data = json.load(file) if file.read(1) else []
        except json.JSONDecodeError:
            data = []
        data.append(content)
        file.seek(0)
        json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info("Content added to JSON file successfully.")

def transcribe_with_api(audio_path: str) -> str:
    api_token = ""  # Usar tu token real
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(API_URL, headers=headers, data=audio_file.read())
    return json.loads(response.content.decode("utf-8"))[0]['text']

def obtener_detalles_video(youtube_url):
    video_id = youtube_url.split('v=')[1]
    youtube = build('youtube', 'v3', developerKey='ADDAPIKEY')
    
    video_response = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    ).execute()
    
    video_details = video_response.get("items", [])[0]["snippet"]
    detalles = {
        "Título": video_details["title"],
        "Fecha de publicación": video_details["publishedAt"],
        "Canal": video_details["channelTitle"],
        "Descripción": video_details["description"],
    }
    
    return detalles

def obtener_y_traducir_detalles_canal(youtube, video_id, idioma_original):
    channel_id = obtener_id_canal(youtube, video_id)
    detalles_canal = obtener_detalles_canal(youtube, channel_id)
    detalles_canal_traducidos = {
        key: traducir_texto_espanol(value, idioma_original) if isinstance(value, str) else value
        for key, value in detalles_canal.items()
    }
    return detalles_canal_traducidos

def analizar_entidades(texto):
    palabras = word_tokenize(texto)
    etiquetas_pos = pos_tag(palabras)
    entidades_nombradas = ne_chunk(etiquetas_pos)
    return entidades_nombradas

def analizar_sentimiento(texto):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(texto)

def visualizar_analisis_nltk(entidades, sentimiento):
    entidades_str = ' '.join([' '.join(map(str, i)) for i in entidades])
    return {
        "Entidades": entidades_str,
        "Positivo": sentimiento.get('pos', 0),
        "Negativo": sentimiento.get('neg', 0),
        "Neutral": sentimiento.get('neu', 0),
        "Compuesto": sentimiento.get('compound', 0)
    }

def obtener_informacion_canal(youtube):
    informacion_videos = []
    lista_videos = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    for item in lista_videos['items']:
        playlist_id = item['contentDetails']['relatedPlaylists']['uploads']
        videos = youtube.playlistItems().list(playlistId=playlist_id, part='snippet', maxResults=50).execute()
        
        for video in videos['items']:
            video_id = video['snippet']['resourceId']['videoId']
            video_info = youtube.videos().list(id=video_id, part='snippet').execute()
            informacion_videos.append(video_info['items'][0]['snippet'])

    return informacion_videos

def obtener_id_canal(youtube, video_id):
    video_details = youtube.videos().list(id=video_id, part='snippet').execute()
    channel_id = video_details['items'][0]['snippet']['channelId']
    return channel_id

def obtener_detalles_canal(youtube, channel_id):
    channel_details = youtube.channels().list(id=channel_id, part='snippet,contentDetails,statistics').execute()
    return channel_details['items'][0]['snippet']

url_youtube = st.text_input("URL de YouTube", "")  # Assign user input to url_youtube
archivo_subido = st.file_uploader("O sube un archivo .wav", type=["wav"])
metodo_transcripcion = st.selectbox("Selecciona el Método de Transcripción", ["API", "Whisper"])
youtube_url = url_youtube
youtube_api = build('youtube', 'v3', developerKey='ADDAPIKEY')

if st.button("Transcribe"):
    video_path = None
    audio_path = None
    if youtube_url:
        with st.spinner("Downloading YouTube video..."):
            video_path, audio_path = youtube_to_wav(youtube_url)
    elif archivo_subido is not None:
        with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            temp_wav.write(archivo_subido.getvalue())
            audio_path = temp_wav.name
    else:
        st.warning("Please upload a file or enter a YouTube URL.")
        st.stop()
    transcription = ""
    translation = ""
    if metodo_transcripcion == "API":  # Corrected variable name
        transcription = transcribe_with_api(audio_path)
    else:
        transcription = transcribe_with_whisper(audio_path)
    detected_lang = detect(transcription)
    translation = traducir_texto_espanol(transcription, detected_lang)
    st.text_area("Transcription", transcription, height=250)
    st.text_area("Translation", translation, height=250)
    if video_path:  # Si tenemos un video, lo mostramos
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    if audio_path:  # Mostrar el audio
        audio_file = open(audio_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
    if youtube_url:
        video_id = youtube_url.split('v=')[1]
        channel_id = obtener_id_canal(youtube_api, video_id)
        detalles_canal = obtener_detalles_canal(youtube_api, channel_id)
        st.subheader("Información del Canal de YouTube")
        st.table(detalles_canal)
        video_id = youtube_url.split('v=')[1]
        detalles_canal_traducidos = obtener_y_traducir_detalles_canal(youtube_api, video_id, 'en') 
        st.subheader("Información del Canal de YouTube Traducida al Español")
        st.table(detalles_canal_traducidos)
    entidades_transcripcion = analizar_entidades(transcription)
    sentimiento_transcripcion = analizar_sentimiento(transcription)
    entidades_traduccion = analizar_entidades(translation)
    sentimiento_traduccion = analizar_sentimiento(translation)
    analisis_transcripcion = visualizar_analisis_nltk(entidades_transcripcion, sentimiento_transcripcion)
    analisis_traduccion = visualizar_analisis_nltk(entidades_traduccion, sentimiento_traduccion)
    st.table(analisis_transcripcion)
    st.table(analisis_traduccion)
    doc_es = nlp_es(translation)
    spacy_streamlit.visualize_ner(doc_es, labels=nlp_es.get_pipe("ner").labels)
    displacy.render(doc_es, style="ent") 
    nlp_es = spacy.load("es_core_news_sm")
    doc = nlp_es(translation)









