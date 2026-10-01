import argparse
import asyncio
import socket
import time
from typing import Callable
from whisper_live.client import TranscriptionClient
import transcript_events
import voice_analysis

import rship_sdk as rship
from rship_sdk import ActionArgs, EmitterProxy, InstanceProxy, InstanceArgs, EmitterArgs, TargetArgs, TargetProxy
import threading
from queue import Queue, Empty
import pyaudio

from tkinter import *
from tkinter import ttk
from customtkinter import *

from PIL import Image, ImageTk
import os

# Defaults; override from the command line (see parse_args).
rship_host = "localhost"
rship_port = 5155
machine_id = socket.gethostname()
whisper_host = "localhost"
whisper_port = 9090
whisper_model = "small"
whisper_lang = "en"
voice_analysis_enabled = False
voice_gate_db = -45.0
default_cues = []
service_id = "whisper"
service_name = "Whisper"
instance_name = "Whisper"
service_type_code = "Whisper"
color = "#4A006D"


class RshipLoop:
    """
    Owns the asyncio event loop that the rship websocket lives on.

    rship_sdk is async, but Tk and the WhisperLive client call in from ordinary threads,
    so everything that talks to rship is handed to this one long-lived loop.
    """
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def call(self, coro, timeout=10):
        """Run a coroutine on the rship loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def send(self, coro):
        """Schedule a coroutine on the rship loop without waiting for it."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        future.add_done_callback(self._report_failure)

    def send_later(self, delay, coro):
        """Schedule a coroutine on the rship loop to start after `delay` seconds."""
        self.loop.call_soon_threadsafe(self.loop.call_later, delay, self._start, coro)

    def _start(self, coro):
        self.loop.create_task(coro).add_done_callback(self._report_failure)

    @staticmethod
    def _report_failure(future):
        if not future.cancelled() and future.exception() is not None:
            print(f"Error sending to rship: {future.exception()!r}")


rship_loop: RshipLoop = None

# Speaker target id -> the Analyser currently listening to it. Action handlers look the
# analyser up here because rship_sdk keeps the first handler registered for an action,
# while each run of a speaker gets a new Analyser.
voice_analysers = {}

# The same goes for transcript events. Per speaker target id: the TranscriptTracker of the
# current run, the cues it listens for (so cues set from rship outlive stop / start), and its
# emitters, which grow as cues are added.
transcript_trackers = {}
speaker_cues = {}
transcript_emitters = {}


async def bootstrap_instance():
    print("Connecting to Rship at", rship_host, rship_port)

    try:
        client = rship.RshipExecClient(rship_host, rship_port)
        await client.connect()
        if not client.is_connected:
            return None, None

        # Keep reading so the connection is serviced for as long as the app runs.
        client.receive_task = asyncio.create_task(client.receive_messages())

        instance = await client.add_instance(
            InstanceArgs(
                name=instance_name,
                code=service_type_code,
                service_id=service_id,
                cluster_id=None,
                machine_id=machine_id,
                color=color,
                message=None
            )
        )

        print("Rship setup completed")
        return client, instance
    except Exception as e:
        print(f"Error during rship setup: {e}")
        return None, None

async def start_speaker(
        instance: InstanceProxy,
        name: str,
        source_id: int,
        url: str,
        port: int,
        stop_flag: threading.Event,
        transcript_queue: Queue
):
    speaker = await instance.add_target(TargetArgs(
        name=name,
        short_id=name.lower().replace(" ", "_"),
        category="Speaker",
    ))

    currently_updating = await speaker.add_emitter(
        EmitterArgs(
            name="Current",
            short_id="currently_updating",
            schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                }
            },
        )
    )

    confirmed = await speaker.add_emitter(
        EmitterArgs(
            name="Confirmed",
            short_id="confirmed",
            schema={
                "type": "object",
                "properties": {     
                    "messages": {
                        "type": "array", 
                        "items": {
                            "type": "string"
                        }
                    }      
                    
                }
            },
        )
    )

    audio_callback = await add_voice_analysis(speaker) if voice_analysis_enabled else None
    transcript_callback = await add_transcript_events(speaker, stop_flag)

    run_transcription(
        currently_updating, confirmed, transcript_queue, url, port, source_id, stop_flag, audio_callback,
        transcript_callback
    )

async def add_transcript_events(speaker: TargetProxy, stop_flag: threading.Event):
    """
    Publishes the transcript event emitters and the setCues action on a speaker.

    Returns the callback to hand each transcript update to.
    """
    target_id = speaker.id()
    tracker = transcript_events.TranscriptTracker()
    transcript_trackers[target_id] = tracker

    emitters = transcript_emitters.setdefault(target_id, {})
    for short_id, (name, schema) in transcript_events.EMITTERS.items():
        emitters[short_id] = await speaker.add_emitter(EmitterArgs(name=name, short_id=short_id, schema=schema))

    async def use_cues(cues):
        cues = transcript_trackers[target_id].set_cues(cues)
        speaker_cues[target_id] = cues
        # Each cue gets an emitter of its own, so a Binding can react to one cue without matching text.
        for cue in cues:
            short_id = transcript_events.cue_short_id(cue)
            emitters[short_id] = await speaker.add_emitter(EmitterArgs(
                name=f"Cue: {cue} (event)", short_id=short_id, schema=transcript_events.VALUE_SCHEMA
            ))
        print(f"{speaker.args.name}: listening for cues {cues}")

    async def set_cues(action, data):
        await use_cues(str(data.get("cues", "")).split(","))

    await speaker.add_action(ActionArgs(
        name="Set cue words (comma separated)", short_id="setCues",
        schema={"type": "object", "properties": {"cues": {"type": "string"}}}, handler=set_cues
    ))
    await use_cues(speaker_cues.get(target_id, default_cues))

    async def send_pulses(pulses):
        for short_id, payload in pulses:
            await emitters[short_id].pulse(payload)

    async def tick():
        # Some things happen when no transcript arrives: a sentence settles, the speaking rate falls.
        while not stop_flag.is_set():
            await asyncio.sleep(0.5)
            pulses = tracker.tick(time.monotonic())
            if pulses:
                await send_pulses(pulses)

    rship_loop.send(tick())

    def on_transcript(segments):
        # Runs on the WhisperLive thread.
        pulses = tracker.update(segments, time.monotonic())
        if pulses:
            rship_loop.send(send_pulses(pulses))

    return on_transcript

async def add_voice_analysis(speaker: TargetProxy):
    """
    Publishes the voice analysis emitters and actions on a speaker.

    Returns the callback to hand each chunk of recorded audio to.
    """
    target_id = speaker.id()
    previous = voice_analysers.get(target_id)
    # A gate set from rship should outlive stop / start.
    analyser = voice_analysis.Analyser(gate_db=previous.gate_db if previous else voice_gate_db)
    voice_analysers[target_id] = analyser

    emitters = {}
    for short_id, name in voice_analysis.EMITTERS.items():
        emitters[short_id] = await speaker.add_emitter(
            EmitterArgs(name=name, short_id=short_id, schema=voice_analysis.VALUE_SCHEMA)
        )

    async def set_gate(action, data):
        current = voice_analysers[target_id]
        current.gate_db = float(data.get("value", current.gate_db))
        print(f"{speaker.args.name}: noise gate set to {current.gate_db} dB")

    async def forget(action, data):
        voice_analysers[target_id].forget()
        print(f"{speaker.args.name}: repetition memory cleared")

    await speaker.add_action(ActionArgs(name="Set noise gate (dB, e.g. -45)", short_id="setGate",
                                        schema=voice_analysis.VALUE_SCHEMA, handler=set_gate))
    await speaker.add_action(ActionArgs(name="Forget (clear repetition memory)", short_id="forget",
                                        schema=voice_analysis.VALUE_SCHEMA, handler=forget))

    async def send_pulses(pulses):
        for short_id, payload in pulses:
            await emitters[short_id].pulse(payload)

    feed = voice_analysis.VoiceFeed(analyser)
    playhead = 0.0
    failed = False

    def on_audio(chunk):
        # Runs on the recording thread. A chunk arrives once its last sample is recorded, so its
        # pulses are replayed at their original spacing, one chunk late, not sent as one burst.
        nonlocal playhead, failed
        if failed:
            return
        try:
            now = time.monotonic()
            # Follow on from the previous chunk so pulses never overtake each other, unless
            # the schedule has fallen far behind the audio.
            start = playhead if now < playhead < now + 1.0 else now
            playhead = start + len(chunk) / voice_analysis.SR
            for offset, pulses in feed.feed(chunk):
                rship_loop.send_later(start + offset - now, send_pulses(pulses))
        except Exception as e:
            # Voice analysis is an extra: never let it take the transcription down with it.
            failed = True
            print(f"Voice analysis stopped for {speaker.args.name}: {e!r}")

    return on_audio

def get_input_options():
    """Returns the available audio inputs as a dict of display label -> PyAudio device index."""
    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    mic_options = {}

    for i in range(0, numdevices):
        device = p.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0:
            # The index is part of the label so that devices sharing a name stay distinct.
            mic_options[f"{device.get('index')}: {device.get('name')}"] = device.get('index')

    p.terminate()

    return mic_options

def add_speaker( instance: InstanceProxy, config_frame: CTkFrame, transcript_frame: CTkFrame, speaker_num: int, get_url: Callable[[], str], get_port: Callable[[], int]):
    font = CTkFont(family="Monospace", size=12, weight="bold")

    row_offset = speaker_num * 2 + 2

    name_label = CTkLabel(config_frame, font=font, text="Speaker Name").grid(column=0, row=row_offset, sticky="wn", padx=(0, 2))
    name_var = StringVar(value="Speaker")
    name_entry = CTkEntry(config_frame, textvariable=name_var)
    name_entry.grid(column=1, row=row_offset, sticky="we", padx=(0, 1), pady=(0, 1))

    audio_label = CTkLabel(config_frame, font=font, text="Audio Source").grid(column=0, row=row_offset + 1, sticky="wn")
    input_options = get_input_options()
    audio_selector = CTkComboBox(config_frame, values=list(input_options), state="readonly", width=100)
    audio_selector.grid(column=1, row=row_offset + 1, columnspan=2, sticky="we", pady=(0, 8))

    start_button = CTkButton(config_frame, font=font, text="Start Transcription")
    start_button.grid(column=2, row=row_offset, sticky="we", pady=(0, 1))

    # One transcript box per speaker, reused across stop / start.
    transcript_queue = Queue()
    print_transcript(transcript_frame, transcript_queue, speaker_num, name_entry.get)

    stop_event = None

    def on_button_click(e):
        nonlocal stop_event
        if stop_event is not None:
            stop_event.set()
            stop_event = None
            start_button.configure(text="Start Transcription")
        else:
            selected_source = audio_selector.get()
            if selected_source not in input_options:
                print("Select an audio source before starting transcription")
                return
            # A new event per run: a run that is still winding down must not see it cleared.
            stop_event = threading.Event()
            rship_loop.call(
                start_speaker(
                    instance,
                    name_entry.get(),
                    input_options[selected_source],
                    get_url(),
                    get_port(),
                    stop_event,
                    transcript_queue
                )
            )
            start_button.configure(text="Stop Transcription")
            print("Transcription started")

    start_button.bind("<Button-1>", on_button_click)

def print_transcript(frm: CTkFrame, transcript_queue: Queue, row: int, get_name: Callable[[], str]):

    transcript_content = CTkTextbox(frm, wrap="word")
    transcript_content.grid(column=0, row=row, sticky="nsew")
    transcript_content.configure(state=DISABLED)  # Make the text widget read-only
    frm.grid_rowconfigure(row, weight=1)

    def update_transcript():
        try:
            transcript = transcript_queue.get_nowait()
            transcript_formatted = f"{get_name()}: " + transcript.replace("\n", " ")
            transcript_content.configure(state=NORMAL)
            transcript_content.delete(1.0, END)
            transcript_content.insert(END, transcript_formatted)
            transcript_content.configure(state=DISABLED)
        except Empty:
            pass
        finally:
            frm.after(100, update_transcript)

    update_transcript()

def run_gui():
    global rship_loop
    rship_loop = RshipLoop()
    client, instance = rship_loop.call(bootstrap_instance())

    if client and instance:
        run_gui_sync(client, instance)
    else:
        print("Failed to setup rship")

def run_gui_sync(client, instance):
    app = CTk()
    app.geometry("400x600")
    app._set_appearance_mode("dark")
    app.title("rship-whisper")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons", "rocketship.ico")
    icon = ImageTk.PhotoImage(file=icon_path)

    app.wm_iconbitmap()
    app.iconphoto(True, icon)

    font = CTkFont(family="Monospace", size=12, weight="bold")

    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure((0, 1), weight=1)

    config_frame = CTkScrollableFrame(app)
    config_frame.grid(row=0, column=0, sticky="nswe")

    url_label = CTkLabel(config_frame, text="Whisper Server", font=font, fg_color="#4A006D", width=90).grid(column=0, row=0, sticky="wn", padx=(0, 5))
    url_var = StringVar(value=whisper_host)
    server_url = CTkEntry(config_frame, textvariable=url_var)
    server_url.grid(column=1, row=0, sticky="we", padx=(0, 2), pady=(0, 5))

    add_speaker_button = CTkButton(master=config_frame, font=font, text="Add Speaker", corner_radius=4, fg_color="#3C3C3C", hover_color="#646464")
    add_speaker_button.grid(row=0, column=2, sticky="w", pady=(0, 5))

    separator = CTkFrame(config_frame, height=3, border_color="white", border_width=1)
    separator.grid(row=1, column=0, columnspan=3, sticky="we", pady=(0, 2))
    
    transcript_frame = CTkFrame(app)    
    transcript_frame.grid(row=1, column=0, sticky="nsew")

    app.grid_rowconfigure(0, weight=100)
    transcript_frame.grid_columnconfigure(0, weight=100)

    speaker_num = 0

    def get_url():
        return server_url.get() 
    
    def get_port():
        return whisper_port

    def on_add_speaker(e):
        nonlocal speaker_num
        speaker_num += 1
        
        add_speaker(instance, config_frame, transcript_frame, speaker_num, get_url, get_port)

    add_speaker_button.bind("<Button-1>", on_add_speaker)

    app.mainloop()



def run_transcription(
        currently_updating: EmitterProxy, 
        confirmed: EmitterProxy, 
        transcript_queue: Queue, 
        url: str, 
        port: int, 
        source: int, 
        stop_flag: threading.Event,
        audio_callback: Callable = None,
        transcript_callback: Callable = None
):
    last_sent = None
    confirmed_sent = []

    def output_callback(data):
        nonlocal last_sent, confirmed_sent

        last = data[-1]
        confirmed_sentences = data[:-1]

        # The server resends its latest segments every time it processes more audio,
        # changed or not. Pulse only on change so Bindings fire once per update.
        # EmitterProxy.pulse is a coroutine, and this callback runs on the WhisperLive thread.
        if last != last_sent:
            last_sent = last
            last_data = {
                "message": last
            }
            print("Updating", last_data)
            rship_loop.send(currently_updating.pulse(last_data))

        if confirmed_sentences != confirmed_sent:
            confirmed_sent = confirmed_sentences
            confirmed_data = {
                "messages": confirmed_sentences
            }
            print("Rest", confirmed_data)
            rship_loop.send(confirmed.pulse(confirmed_data))

        if transcript_callback is not None:
            transcript_callback(data)

        transcript_queue.put(' '.join(data))

    print("Starting Transcription Client")

    client = TranscriptionClient(
        url,
        port,
        lang=whisper_lang,
        translate=False,
        model=whisper_model,
        use_vad=True,
        save_output_recording=False,                            # Only used for microphone input, False by Default
        output_recording_filename="./output_recording.wav",     # Only used for microphone input
        output_callback=output_callback,
        input_device_index=source,             
        stop_event=stop_flag,                                  # Only used for microphone input, 0 by Default
        audio_callback=audio_callback                          # Only used for microphone input
    )

    print("Starting transcription")

    transcript_thread = threading.Thread(target=client, daemon=True)
    transcript_thread.start()


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper transcription as an rship executor.")
    parser.add_argument("--rship-host", default=rship_host,
                        help="rship server address (default: %(default)s)")
    parser.add_argument("--rship-port", type=int, default=rship_port,
                        help="rship server port (default: %(default)s)")
    parser.add_argument("--machine-id", default=machine_id,
                        help="machine this instance is reported under in rship (default: this hostname)")
    parser.add_argument("--whisper-host", default=whisper_host,
                        help="WhisperLive server address, editable in the window (default: %(default)s)")
    parser.add_argument("--whisper-port", type=int, default=whisper_port,
                        help="WhisperLive server port (default: %(default)s)")
    parser.add_argument("--model", default=whisper_model,
                        help="Whisper model size to request from the server (default: %(default)s)")
    parser.add_argument("--lang", default=whisper_lang,
                        help="language spoken, as a Whisper language code (default: %(default)s)")
    parser.add_argument("--cue", action="append", default=[], metavar="WORDS",
                        help="a word or phrase every speaker listens for, firing an event when it is spoken; "
                             "repeat for more (also settable per speaker from rship)")
    parser.add_argument("--voice-analysis", action="store_true",
                        help="also publish how each speaker sounds: level, pitch, brightness, tempo, "
                             "phrase and silence lengths, sustain and repetition")
    parser.add_argument("--gate", type=float, default=voice_gate_db,
                        help="voice analysis noise gate in dBFS; lower is more sensitive (default: %(default)s)")
    return parser.parse_args()


def main():
    global rship_host, rship_port, machine_id, whisper_host, whisper_port, whisper_model, whisper_lang
    global voice_analysis_enabled, voice_gate_db, default_cues

    args = parse_args()
    rship_host = args.rship_host
    rship_port = args.rship_port
    machine_id = args.machine_id
    whisper_host = args.whisper_host
    whisper_port = args.whisper_port
    whisper_model = args.model
    whisper_lang = args.lang
    voice_analysis_enabled = args.voice_analysis
    voice_gate_db = args.gate
    default_cues = args.cue

    run_gui()


if __name__ == "__main__":
    main()