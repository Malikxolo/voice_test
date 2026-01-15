"""
STT Model Configuration
========================
Add your STT model config here. Each model just needs:
- name: Display name  
- env_key: The .env variable name for API key
- transcribe_fn: Function that takes (audio_bytes, api_key, sample_rate) and returns text

To add a new model:
1. Create a transcribe function
2. Add it to STT_MODELS dict
3. Add your API key to .env
"""

import os
import io
import tempfile
import requests
import base64


# ============== TRANSCRIBE FUNCTIONS ==============
# Each function takes (audio_bytes, api_key, sample_rate) -> returns transcribed text

def transcribe_openai(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """OpenAI Whisper API"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    return response.strip() if isinstance(response, str) else response.text.strip()


def transcribe_groq(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Groq Whisper API"""
    from groq import Groq
    client = Groq(api_key=api_key)
    
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    
    response = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="text"
    )
    return response.strip() if isinstance(response, str) else response.text.strip()


def transcribe_deepgram(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Deepgram Nova-3 API with multi-language support"""
    # Enable auto language detection for multi-language support (Hindi, English, etc.)
    url = f"https://api.deepgram.com/v1/listen?model=nova-3&detect_language=true&encoding=linear16&sample_rate={sample_rate}&channels=1"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    response = requests.post(url, headers=headers, data=audio_bytes)
    response.raise_for_status()
    result = response.json()
    
    # Handle case where no transcription is returned
    try:
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript if transcript else ""
    except (KeyError, IndexError):
        return ""


def transcribe_sarvam_saarika(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Sarvam AI Saarika - Speech to Text (transcribes in original language with auto-detection)"""
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": api_key
    }
    
    # Create multipart form data with audio file
    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
    }
    data = {
        "model": "saarika:v2.5",
        "language_code": "unknown"  # Auto-detect language (Hindi, English, etc.)
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    result = response.json()
    
    # Extract transcript from response
    try:
        transcript = result.get("transcript", "")
        return transcript if transcript else ""
    except (KeyError, AttributeError):
        return ""


def transcribe_sarvam_saaras(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Sarvam AI Saaras - Speech to Text Translation (auto-detects language, translates to English)"""
    url = "https://api.sarvam.ai/speech-to-text-translate"
    headers = {
        "api-subscription-key": api_key
    }
    
    # Create multipart form data with audio file
    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
    }
    data = {
        "model": "saaras:v2.5"
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    result = response.json()
    
    # Extract translated transcript from response
    try:
        transcript = result.get("transcript", "")
        return transcript if transcript else ""
    except (KeyError, AttributeError):
        return ""


def transcribe_assemblyai(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """AssemblyAI API"""
    import assemblyai as aai
    aai.settings.api_key = api_key
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_path)
        return transcript.text.strip() if transcript.text else ""
    finally:
        os.unlink(temp_path)


def transcribe_google(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Google Cloud Speech-to-Text (api_key = path to credentials.json)"""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key
    from google.cloud import speech
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    return " ".join([r.alternatives[0].transcript for r in response.results])


def transcribe_azure(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Azure Speech Services (api_key format: 'key|region')"""
    key, region = api_key.split("|")
    
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav",
    }
    params = {"language": "en-US"}
    
    response = requests.post(url, headers=headers, params=params, data=audio_bytes)
    response.raise_for_status()
    return response.json().get("DisplayText", "")


def transcribe_rev_ai(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Rev.ai API"""
    # Upload audio
    upload_url = "https://api.rev.ai/speechtotext/v1/jobs"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    files = {"media": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    response = requests.post(upload_url, headers=headers, files=files)
    response.raise_for_status()
    job_id = response.json()["id"]
    
    # Poll for completion
    import time
    status_url = f"https://api.rev.ai/speechtotext/v1/jobs/{job_id}"
    while True:
        status = requests.get(status_url, headers=headers).json()
        if status["status"] == "transcribed":
            break
        elif status["status"] == "failed":
            raise RuntimeError(f"Rev.ai failed: {status}")
        time.sleep(1)
    
    # Get transcript
    transcript_url = f"https://api.rev.ai/speechtotext/v1/jobs/{job_id}/transcript"
    headers["Accept"] = "text/plain"
    return requests.get(transcript_url, headers=headers).text.strip()


def transcribe_speechmatics(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Speechmatics API"""
    url = "https://asr.api.speechmatics.com/v2/jobs/"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    config = {
        "type": "transcription",
        "transcription_config": {"language": "en"}
    }
    
    files = {
        "data_file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
        "config": (None, str(config), "application/json")
    }
    
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.json().get("transcript", "")


def transcribe_whisper_cpp(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Local Whisper.cpp server (api_key = server URL like 'http://localhost:8080')"""
    url = f"{api_key}/inference"
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json().get("text", "").strip()


def transcribe_faster_whisper(audio_bytes: bytes, api_key: str, sample_rate: int = 16000) -> str:
    """Local Faster-Whisper (api_key = model size like 'base', 'small', 'medium', 'large-v3')"""
    from faster_whisper import WhisperModel
    
    model = WhisperModel(api_key, compute_type="int8")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        segments, _ = model.transcribe(temp_path)
        return " ".join([seg.text for seg in segments]).strip()
    finally:
        os.unlink(temp_path)


# ============== MODEL REGISTRY ==============
# Just add your model here! Format: "model_id": ("Display Name", "ENV_KEY_NAME", transcribe_function)

STT_MODELS = {
    "openai": ("OpenAI Whisper", "OPENAI_API_KEY", transcribe_openai),
    "groq": ("Groq Whisper", "GROQ_API_KEY", transcribe_groq),
    "deepgram": ("Deepgram Nova-3", "DEEPGRAM_API_KEY", transcribe_deepgram),
    "sarvam_saarika": ("Sarvam AI Saarika (Indian Languages)", "SARVAM_API_KEY", transcribe_sarvam_saarika),
    "sarvam_saaras": ("Sarvam AI Saaras (Translate to English)", "SARVAM_API_KEY", transcribe_sarvam_saaras),
    "assemblyai": ("AssemblyAI", "ASSEMBLYAI_API_KEY", transcribe_assemblyai),
    "google": ("Google Cloud STT", "GOOGLE_CREDENTIALS_PATH", transcribe_google),
    "azure": ("Azure Speech", "AZURE_SPEECH_KEY", transcribe_azure),  # Format: key|region
    "rev_ai": ("Rev.ai", "REV_AI_API_KEY", transcribe_rev_ai),
    "speechmatics": ("Speechmatics", "SPEECHMATICS_API_KEY", transcribe_speechmatics),
    "whisper_cpp": ("Whisper.cpp (Local)", "WHISPER_CPP_URL", transcribe_whisper_cpp),
    "faster_whisper": ("Faster-Whisper (Local)", "FASTER_WHISPER_MODEL", transcribe_faster_whisper),
}


def get_available_models() -> list:
    """Returns list of models that have API keys configured in .env"""
    from dotenv import load_dotenv
    load_dotenv()
    
    available = []
    for model_id, (name, env_key, _) in STT_MODELS.items():
        if os.getenv(env_key):
            available.append((model_id, name))
    return available


def get_model(model_id: str):
    """Get model config by ID. Returns (name, api_key, transcribe_fn)"""
    from dotenv import load_dotenv
    load_dotenv()
    
    if model_id not in STT_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(STT_MODELS.keys())}")
    
    name, env_key, transcribe_fn = STT_MODELS[model_id]
    api_key = os.getenv(env_key)
    
    if not api_key:
        raise ValueError(f"API key not found. Set {env_key} in .env file")
    
    return name, api_key, transcribe_fn
