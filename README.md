# Rship Whisper

<h2 align="center">
  A nearly-live implementation of OpenAI's Whisper ... as an rship executor!
</h2>

This project is a real-time transcription application that uses the OpenAI Whisper model
to convert speech input into text output. It can be used to transcribe multiple audio input sources to individual targets in rship. 

## Installation
Requires Python 3.11 or newer (needed by current `rship_sdk` releases), PortAudio for PyAudio, and Tk for the window
(on Debian/Ubuntu: `apt-get install portaudio19-dev python3-tk`).

```bash
pip install -r requirements/client.txt
```

## Run

```bash
python rship_whisper.py --rship-host 192.168.1.10
```

| Flag | Default | |
|---|---|---|
| `--rship-host` | `localhost` | rship server address |
| `--rship-port` | `5155` | rship server port |
| `--machine-id` | this hostname | machine the instance is reported under in rship |
| `--whisper-host` | `localhost` | WhisperLive server address (also editable in the window) |
| `--whisper-port` | `9090` | WhisperLive server port |
| `--model` | `small` | Whisper model size requested from the server |
| `--lang` | `en` | language spoken, as a Whisper language code |

Click **Add Speaker**, give the speaker a name, pick its audio source and click **Start Transcription**.

## Targets and Emitters

Each speaker is a Target in the `whisper` service, with two Emitters:

| Emitter | Payload | Meaning |
|---|---|---|
| `Current` | `{ "message": string }` | the sentence being spoken; pulsed each time Whisper revises it |
| `Confirmed` | `{ "messages": string[] }` | the (up to two) sentences before it; pulsed when a sentence is completed |

### Setting up NVIDIA/TensorRT-LLM for TensorRT backend
- Please follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and for building Whisper-TensorRT engine.


## Whisper Live Server in Docker
it is required to have a version of the server running to use this application. we recomend the TensorRT variant as described below. 

- GPU
  - Faster-Whisper
  ```bash
  docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
  ```

  - TensorRT. 
  ```bash
  docker run -p 9090:9090 --gpus all --entrypoint /bin/bash -it ghcr.io/collabora/whisperlive-tensorrt

  # Build tiny.en engine
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en

  # Run server with tiny.en
  python3 run_server.py --port 9090 \
                        --backend tensorrt \
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en"
  ```

- CPU
```bash
docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
```
**Note**: By default we use "small" model size. To build docker image for a different model size, change the size in server.py and then build the docker image.

**Another Note**
just use the docker compose file by running `docker compose up -d`