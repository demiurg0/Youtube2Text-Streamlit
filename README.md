# Streamlit Voice Wizard: Youtube2Text, Transcription, Translation, and Audio Processing

Streamlit Voice Wizard is an all-in-one application that transforms YouTube videos into actionable text insights through a streamlined process of downloading, transcribing, translating, and performing Natural Language Processing (NLP).

![Overview Image](https://github.com/demiurg0/Youtube2Text-Streamlit/assets/165735354/93c850c4-55e1-4bff-aed8-94ea44952054)

## Application Flow:

1. **Download:** Retrieve video content from YouTube.
2. **Video-to-Audio:** Convert video to audio format for processing.
3. **Audio-to-Text:** Transcribe the audio content into text.
4. **Translate:** Translate the text into Spanish or other languages.
5. **NLP Analysis:** Perform sentiment analysis and entity recognition.
6. **Visualization:** Display the processed data through an interactive Streamlit interface.

### Key Features

- **Transcription:** Quick and accurate conversion of video and audio content to text using AI models.
- **Translation:** Seamless translation of transcriptions to enhance understanding across languages.
- **Sentiment Analysis:** Gauge the emotional tone behind the text.
- **Entity Recognition:** Identify key entities within the text to extract meaningful insights.
- **Streamlit UI:** A user-friendly interface that provides a comprehensive view of the multimedia content.

### Getting Started

#### Prerequisites

- **Environment:** Python 3.x and pip installed.
- **Dependencies:** Install all necessary libraries from `requirements.txt`.
- **API Keys:** Secure Huggingface and Youtube V3 API keys included in the code.

#### Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory in the terminal.
3. Install dependencies:
4. Add API keys to the configuration file or environment variables as required.

#### Usage Guide

1. Start the application by running the command in the terminal:
2. Interface operations:
- Upload an audio file (.wav) or paste a YouTube video URL.
- Select the transcription method: API or Whisper Model.
- Click "Transcribe" to initiate the transcription and subsequent translation process.

3. View Results:
- Original transcription, translation, and analyses will be dynamically generated and displayed.
- Audio playback is available for uploaded files.

![image](https://github.com/demiurg0/Youtube2Text-Streamlit-Voice-Wizard/assets/165735354/5afbabc2-2071-45e9-900b-524c55d9bcd7)

## Future Enhancements

- **Language Options:** Incorporate multi-language support for translation.
- **Configuration Management:** Streamline API configuration through environment variables.
- **NLP Support:** Extend support for languages like Arabic within NLP analysis.
- **Usability Improvements:** Address UI bugs and enhance file management features.
- **Analytics Expansion:** Augment channel analytics and integrate advanced search capabilities.


## License:

This project is under the [MIT License](LICENSE).
