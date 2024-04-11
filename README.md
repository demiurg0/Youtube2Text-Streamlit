# Streamlit Voice Wizard: Youtube2Text, Transcription, Translation, and Audio Processing
![image](https://github.com/demiurg0/Youtube2Text-Streamlit/assets/165735354/93c850c4-55e1-4bff-aed8-94ea44952054)

### Youtube Download + Video2Audio + Audio2Text + Translate to Spanish + NLP + Streamlit

Streamlit Voice Wizard is a versatile tool that simplifies the process of transcription and translation for YouTube video and audio files. With an intuitive interface, users can upload audio files or enter video URLs, which are then transcribed using the Whisper model or an API. The transcriptions are subsequently translated into Spanish using the Opus-MT model. Furthermore, the tool performs linguistic analysis on both the transcriptions and the translated text, extracting entities and analyzing sentiment. Additionally, it retrieves information from YouTube videos, such as channel details, descriptions, and titles, for further linguistic analysis. Finally, the results are visualized and displayed through the Streamlit interface, offering users an accessible and comprehensive understanding of multimedia content in various languages.
![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/4b13d644-67ca-437d-8250-148f16f73ea6)

## Key Features:

- Instant transcription of YouTube videos and audio files.
- Automatic translation of transcriptions into Spanish.
- Entity and sentiment analysis to deeply understand the content.
- Interactive visualization of results for easy interpretation.

## Prerequisites:

- Python 3.x
- pip
- Install dependencies from the `requirements.txt` file.
- Add the Huggingface and Youtube V3 API keys to the code.
  
## Usage:

1. Run the application by executing the following command in the terminal:
2. In the user interface, upload an audio file in .wav format or enter a YouTube video URL.
3. Select the transcription method (API or Whisper).
4. Click the "Transcribe" button to start the transcription and translation process.
5. The original transcription, Spanish translation, and entity and sentiment analyses will be displayed.
6. If you uploaded an audio file, you will also be able to play the audio directly from the application.

![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/5afbabc2-2071-45e9-900b-524c55d9bcd7)

## To-Do:
1. Add option to choose the language for translation.
2. Add API to env or config.
3. Add solution for Arabic language with Spacy since there is no model.
4. Solve index and column bug.
5. Improve file management.
6. Improve the output.
7. Review and enhance channel information, try to retrieve and render all videos, and switch from video to channel instead of URL, i.e., integrate as a YouTube search engine and render graphics of the information.

## License:

This project is under the [MIT License](LICENSE).
