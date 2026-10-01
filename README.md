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
| `--cue` | none | a word or phrase every speaker listens for; repeat for more (see [Transcript events](#transcript-events)) |
| `--voice-analysis` | off | also publish how each speaker sounds (see [Voice analysis](#voice-analysis)) |
| `--gate` | `-45` | voice analysis noise gate in dBFS; lower is more sensitive |

Click **Add Speaker**, give the speaker a name, pick its audio source and click **Start Transcription**.

## Targets and Emitters

Each speaker is a Target in the `whisper` service, with two Emitters:

| Emitter | Payload | Meaning |
|---|---|---|
| `Current` | `{ "message": string }` | the sentence being spoken; pulsed each time Whisper revises it |
| `Confirmed` | `{ "messages": string[] }` | the (up to two) sentences before it; pulsed when a sentence is completed |

## Transcript events

`Current` and `Confirmed` carry text, which is awkward to bind to. Each speaker also has emitters
a Binding can react to directly:

| Emitter | Payload | Meaning |
|---|---|---|
| `sentence` *(event)* | `{ "message": string, "words": number }` | a segment of speech was finished. Whisper's segments are usually a sentence or a clause. |
| `question` / `exclamation` *(event)* | `{ "message": string }` | the finished segment ended in `?` / `!` |
| `wordCount` | `{ "value": number }` | words spoken since transcription was started |
| `wordsPerMinute` | `{ "value": number }` | speaking rate over the last 15 seconds; falls to 0 in silence |
| `cue` *(event)* | `{ "cue": string }` | any cue was spoken |
| `cue_<words>` *(event)* | `{ "value": 1 }` | that cue was spoken, e.g. `cue_midnight`, `cue_open_the_door` |

**Cues** are words or phrases to listen for: "when the narrator says *midnight*, start the scene".
Give them at launch with `--cue midnight --cue "open the door"`, or set them for one speaker from
rship with its `setCues` action, which takes `{ "cues": "midnight, open the door" }` and replaces
that speaker's list. Matching ignores case and punctuation and is on whole words, so `night` does
not match *midnight*.

Whisper revises its text as it hears more. A cue fires once for each time it is spoken, when it
first appears, and not again as the text around it is revised. Expect it roughly a second after
the word, depending on the model and the machine running it: good for starting a scene, too loose
for a hit that has to land on the word. A cue phrase that Whisper happens to split across two
segments is missed.

`sentence` fires when Whisper moves on to the next segment, or after two seconds without change
if nothing follows, so the last thing said before a pause arrives about two seconds late.

## Voice analysis

Whisper reports *what* was said. Run with `--voice-analysis` and each speaker also reports *how*:
loudness, pitch, brightness, tempo, the length of phrases and silences, held notes and repetition.
It works on speech and on singing, runs on the audio already being recorded for Whisper, and needs
no extra server. It is off by default because it adds a steady stream of pulses: roughly 70 to 100
per second for each speaker while they are making sound, and about 12 per second in silence.

All payloads are `{ "value": number }`. Continuous emitters are pulsed about 12 times a second,
and only when their value has changed.

| Emitter | Meaning |
|---|---|
| `level` | loudness 0–1 above the noise gate |
| `pitch` / `pitchNorm` | fundamental in Hz / 0–1 across 80–800 Hz (held through unvoiced sounds) |
| `voiced` | 1 while there is a clear pitch (vowels, singing) |
| `brightness` | spectral centroid 0–1 (sibilance, breathiness, "ee" vs "oo") |
| `onsetRate` | syllables/onsets per second over the last 2 s — speech tempo |
| `soundSeconds` / `silenceSeconds` | how long the current phrase / pause has lasted |
| `sustainSeconds` | how long the current note has been held within a semitone |
| `repetition` | 0–1: how closely the last 1.2 s matches anything from an *earlier* phrase in the last minute (refrains, repeated words, a chorus). Decays when the speaker moves on. |
| `onset` *(event)* | each syllable/note attack; value = strength |
| `phraseStart` *(event)* | value = length of the silence before it |
| `phraseEnd` *(event)* | value = length of the phrase that just ended |

Actions on each speaker: `setGate` (dB) and `forget` (clear the repetition memory, e.g. at a new
chapter or song).

Audio reaches the executor in chunks of about a quarter of a second, so these pulses run about
a quarter of a second behind the voice. They are sent at their original spacing rather than in
bursts, so values driven from them move smoothly.

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