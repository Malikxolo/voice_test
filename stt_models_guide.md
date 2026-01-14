# STT Models & API Keys Guide
This guide details the available Speech-to-Text (STT) models supported by this testing system, which API keys correspond to which, and what specific models are being used under the hood.
## 🟢 Groq (Fastest)
**API Key Variable:** `GROQ_API_KEY`
**Current Model Used:** `whisper-large-v3`
Groq provides extremely fast inference for OpenAI's Whisper models.
| Model ID | Description | Best For |
| :--- | :--- | :--- |
| `whisper-large-v3` | The most accurate Whisper model | General purpose, multilingual |
| `whisper-large-v3-turbo` | Faster, optimized version | Speed-critical applications |
| `distil-whisper-large-v3-en` | Distilled English-only model | extremely low latency (English only) |
*To change the model, edit `transcribe_groq` in `stt_config.py`.*
## 🟢 OpenAI
**API Key Variable:** `OPENAI_API_KEY`
**Current Model Used:** `whisper-1`
The standard Whisper API from OpenAI.
| Model ID | Description |
| :--- | :--- |
| `whisper-1` | The default and generally only model name exposed via their API, usually points to `large-v2` or `large-v3`. |
## 🟢 Deepgram
**API Key Variable:** `DEEPGRAM_API_KEY`
**Current Model Used:** `nova-2`
Deepgram specializes in audio intelligence and offers very low latency streaming limits.
| Model ID | Description | Best For |
| :--- | :--- | :--- |
| `nova-2` | Their latest, most accurate model | General purpose, speed & accuracy balance |
| `nova` | Previous generation | Legacy support |
| `enhanced` | Better accuracy than base | Difficult audio |
| `base` | Standard model | Cost efficiency |
*To change the model, edit `transcribe_deepgram` in `stt_config.py`.*
## 🟢 AssemblyAI
**API Key Variable:** `ASSEMBLYAI_API_KEY`
**Current Model Used:** Default (Best)
| Model ID | Description |
| :--- | :--- |
| `best` | Automatically selects their best available model (currently `nano` or similar for speed) |
| `nano` | Simple, fast model |
## 🟢 Google Cloud STT
**API Key Variable:** `GOOGLE_CREDENTIALS_PATH` (Path to JSON file)
**Current Model Used:** `default` (v1 API)
Google has extensive language support.
| Model ID | Description |
| :--- | :--- |
| `default` | Standard recognition model |
| `command_and_search` | Short queries (voice commands) |
| `phone_call` | Telephony audio (8khz) |
| `video` | Video audio |
| `chicrago` | Chirp (USM) models (v2 API required) |
## 🟢 Azure Speech
**API Key Variable:** `AZURE_SPEECH_KEY` (Format: `key|region`, e.g., `abcdef...|eastus`)
**Current Model Used:** `en-US` (Conversation)
## 🟢 Local Models (No API Cost)
These run on your machine.
### Whisper.cpp
**Variable:** `WHISPER_CPP_URL` (e.g., `http://localhost:8080`)
Requires running a local `whisper.cpp` server.
### Faster-Whisper
**Variable:** `FASTER_WHISPER_MODEL` (e.g., `base`, `small`, `medium`, `large-v3`)
Runs directly in Python. Larger models = better accuracy but slower.
---
## How to Switch Models in Code
To change specifically which model version is called (e.g., switching Groq from `whisper-large-v3` to `whisper-large-v3-turbo`), open `stt_config.py` and find the corresponding transcribe function:
```python
def transcribe_groq(...):
    # ...
    response = client.audio.transcriptions.create(
        model="whisper-large-v3",  # <--- CHANGE THIS
        # ...
    )
```