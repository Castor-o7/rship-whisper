"""Voice analysis for rship-whisper.

Whisper says *what* was said. This measures *how*: how loud, how high, how bright,
how dense, how long it has lasted, and how much it repeats what came before.

Feed an Analyser 16 kHz mono float audio in hops of HOP samples; read continuous
values from `state` and drain discrete events from `events`.
"""

import numpy as np

SR = 16000          # speech/singing content lives well below 8 kHz
HOP = 256           # 16 ms analysis hop
FRAME = 1024        # 64 ms analysis window
FEATURE_RATE = 10   # Hz; repetition memory is built from 100 ms summaries
PULSE_RATE = 15     # Hz; how often continuous emitters are sent to rship

N_MELS = 40
REP_WINDOW = 12     # 1.2 s phrase fingerprint
REP_MIN_LAG = 20    # ignore the last 2 s, otherwise everything "repeats" itself
REP_MEMORY = 600    # 60 s of history to compare against
REP_FLOOR = 0.4     # centred cosine: below this is "something new"
REP_CEIL = 0.85     # at or above this is "the same phrase again"

# short id -> display name. The first ten are continuous, the last three are events.
EMITTERS = {
    "level": "Level (0-1)",
    "pitch": "Pitch (Hz)",
    "pitchNorm": "Pitch (0-1, 80-800 Hz)",
    "voiced": "Voiced (0/1)",
    "brightness": "Brightness (0-1)",
    "onsetRate": "Syllables / onsets per second",
    "soundSeconds": "Sounding for (s)",
    "silenceSeconds": "Silent for (s)",
    "sustainSeconds": "Held note for (s)",
    "repetition": "Repetition (0-1)",
    "onset": "Onset (event)",
    "phraseStart": "Phrase start (event, value = preceding silence s)",
    "phraseEnd": "Phrase end (event, value = phrase length s)",
}
VALUE_SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}}}


def mel_filterbank(n_mels, n_fft, sr, fmin=50.0, fmax=7600.0):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        lo, mid, hi = bins[i - 1], bins[i], bins[i + 1]
        if mid > lo:
            fb[i - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def estimate_pitch(frame, sr, fmin=70.0, fmax=1000.0):
    """Normalised autocorrelation pitch; returns (hz, clarity 0-1)."""
    frame = frame - frame.mean()
    energy = np.dot(frame, frame)
    if energy < 1e-8:
        return 0.0, 0.0
    n = len(frame)
    spec = np.fft.rfft(frame, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac /= ac[0]
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 1)
    # The autocorrelation slopes away from 1 at lag 0, and for a low voice that slope is
    # still higher at `lo` than the true peak one period later. Look for the period only
    # after the slope has dropped below zero.
    below = np.flatnonzero(ac[:hi] < 0)
    if len(below) == 0:
        return 0.0, 0.0
    lo = max(lo, int(below[0]))
    lag = lo + int(np.argmax(ac[lo:hi]))
    clarity = float(ac[lag])
    if lo < lag < hi - 1:
        # parabolic interpolation for sub-sample accuracy
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        lag = lag + 0.5 * (a - c) / denom if denom != 0 else lag
    return sr / lag, max(0.0, clarity)


class Analyser:
    def __init__(self, gate_db=-45.0):
        self.gate_db = gate_db
        self.window = np.hanning(FRAME)
        self.melfb = mel_filterbank(N_MELS, FRAME, SR)
        self.freqs = np.fft.rfftfreq(FRAME, 1 / SR)
        self.buf = np.zeros(FRAME, dtype=np.float32)

        self.prev_mag = None
        self.flux_hist = []
        self.onset_times = []
        self.last_onset = 0.0

        self.started = False
        self.sound_start = None
        self.silence_start = None  # set on the first hop, in the caller's clock
        self.phrase_start = None
        self.last_sound = 0.0
        self.last_voiced = 0.0
        self.pitch_start = None
        self.pitch_ref = 0.0

        self.acc_mel, self.acc_voiced, self.acc_n = np.zeros(N_MELS), 0, 0
        self.memory = np.zeros((0, N_MELS))
        self.voiced_mem = np.zeros(0)
        self.times_mem = np.zeros(0)
        self.last_similarity = 0.0
        self.forget_requested = False

        self.state = dict(level=0.0, pitch=0.0, pitchNorm=0.0, voiced=0.0, brightness=0.0,
                          onsetRate=0.0, soundSeconds=0.0, silenceSeconds=0.0,
                          sustainSeconds=0.0, repetition=0.0)
        self.events = []  # (emitter_short_id, payload) discrete events

    def forget(self):
        """Clear the repetition memory, e.g. at a new chapter or song. Safe to call from another thread."""
        self.forget_requested = True

    def process_hop(self, hop, now):
        """Analyse the next HOP samples. `now` is the time in seconds at the end of the hop, on any steady clock."""
        self.buf = np.concatenate([self.buf[len(hop):], hop])
        if not self.started:
            self.started, self.silence_start, self.last_sound = True, now, now
        frame = self.buf
        rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-10)
        db = 20 * np.log10(rms)
        sounding = db > self.gate_db

        mag = np.abs(np.fft.rfft(frame * self.window))
        centroid = float((self.freqs * mag).sum() / (mag.sum() + 1e-10))

        # spectral flux onset detection with an adaptive threshold
        logmag = np.log1p(mag * 10)
        flux = 0.0 if self.prev_mag is None else float(np.maximum(logmag - self.prev_mag, 0).sum())
        self.prev_mag = logmag
        self.flux_hist = (self.flux_hist + [flux])[-40:]
        med = np.median(self.flux_hist)
        thresh = max(med + 1.5 * np.std(self.flux_hist), 2.0 * med) + 1.0
        if sounding and flux > thresh and now - self.last_onset > 0.08:
            self.last_onset = now
            self.onset_times.append(now)
            self.events.append(("onset", {"value": min(1.0, flux / (thresh * 2))}))
        self.onset_times = [t for t in self.onset_times if now - t < 2.0]

        hz, clarity = estimate_pitch(frame[-768:], SR) if sounding else (0.0, 0.0)
        voiced = sounding and clarity > 0.55

        # sound / silence durations, with a short hangover so gaps between words don't reset "sound"
        if sounding:
            if self.sound_start is None:
                self.sound_start = now
                self.phrase_start = now
                self.events.append(("phraseStart", {"value": now - self.silence_start}))
            self.silence_start = None
            self.last_sound = now
        elif self.sound_start is not None and now - self.last_sound > 0.35:
            duration = self.last_sound - self.sound_start
            self.events.append(("phraseEnd", {"value": duration}))
            self.sound_start = None
            self.silence_start = self.last_sound

        # a sustained note: pitch staying within ~a semitone
        if voiced and self.pitch_ref and abs(12 * np.log2(hz / self.pitch_ref)) < 1.0:
            self.pitch_ref = 0.9 * self.pitch_ref + 0.1 * hz
        elif voiced:
            self.pitch_start, self.pitch_ref = now, hz
        elif now - self.last_voiced > 0.15:
            self.pitch_start, self.pitch_ref = None, 0.0
        if voiced:
            self.last_voiced = now

        s = self.state
        level = np.clip((db - self.gate_db) / (-6 - self.gate_db), 0, 1)
        s["level"] = float(0.6 * s["level"] + 0.4 * level)
        if voiced:
            s["pitch"] = float(hz)
            # 80 Hz -> 0, 800 Hz -> 1 on a log scale
            s["pitchNorm"] = float(np.clip(np.log2(hz / 80) / np.log2(10), 0, 1))
        s["voiced"] = 1.0 if voiced else 0.0
        if sounding:  # hold brightness through silence rather than measuring room noise
            s["brightness"] = float(0.8 * s["brightness"] + 0.2 * np.clip(centroid / 4000, 0, 1))
        s["onsetRate"] = len(self.onset_times) / 2.0
        s["soundSeconds"] = now - self.sound_start if self.sound_start is not None else 0.0
        s["silenceSeconds"] = now - self.silence_start if self.silence_start is not None else 0.0
        s["sustainSeconds"] = now - self.pitch_start if self.pitch_start is not None else 0.0

        self.acc_mel += self.melfb @ (mag ** 2)
        self.acc_voiced += 1 if sounding else 0
        self.acc_n += 1
        if self.acc_n >= SR // HOP // FEATURE_RATE:
            self._feature_step(now)

    def _feature_step(self, now):
        mel = np.log(self.acc_mel / self.acc_n + 1e-9)
        mel = mel - mel.mean()  # loudness-independent spectral shape
        sounding = self.acc_voiced / self.acc_n > 0.5
        self.acc_mel[:], self.acc_voiced, self.acc_n = 0, 0, 0

        if self.forget_requested:
            self.forget_requested = False
            self.memory = np.zeros((0, N_MELS))
            self.voiced_mem = np.zeros(0)
            self.times_mem = np.zeros(0)

        self.memory = np.vstack([self.memory, mel])[-REP_MEMORY:]
        self.voiced_mem = np.append(self.voiced_mem, 1.0 if sounding else 0.0)[-REP_MEMORY:]
        self.times_mem = np.append(self.times_mem, now)[-REP_MEMORY:]
        self.state["repetition"] *= 0.9  # decays when there's nothing new to match
        n = len(self.memory)
        if n < REP_WINDOW + REP_MIN_LAG or self.voiced_mem[-REP_WINDOW:].mean() < 0.6:
            return

        # Remove the performer's average spectrum so what's compared is *what* is
        # being said/sung (pitches, vowels) rather than the fact that it's the same voice.
        voiced_frames = self.memory[self.voiced_mem > 0]
        centred = self.memory - voiced_frames.mean(axis=0)
        centred /= np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9

        # Only compare against material from *earlier* phrases, so a long held note
        # reads as sustain, not as repeating itself.
        phrase_start = self.phrase_start if self.phrase_start is not None else now
        cutoff = min(n - REP_WINDOW - REP_MIN_LAG, int(np.searchsorted(self.times_mem, phrase_start)) - REP_WINDOW)

        # Slide the latest 1.2 s fingerprint across the history; the best-matching
        # earlier moment (that was also sounding) is how much we're repeating.
        cur = centred[-REP_WINDOW:]
        best = 0.0
        for start in range(0, cutoff + 1):
            if self.voiced_mem[start:start + REP_WINDOW].mean() < 0.6:
                continue
            past = centred[start:start + REP_WINDOW]
            best = max(best, float(np.mean(np.sum(cur * past, axis=1))))
        self.last_similarity = best
        rep = float(np.clip((best - REP_FLOOR) / (REP_CEIL - REP_FLOOR), 0, 1))
        self.state["repetition"] = max(self.state["repetition"], rep)


class VoiceFeed:
    """
    Turns audio chunks of any size into the pulses an Analyser produces from them.

    Audio reaches the executor a chunk at a time (about a quarter of a second), well after
    most of it was spoken. `feed` reports how far into the chunk each pulse belongs, so the
    caller can send them at their original spacing rather than as one burst per chunk.
    """
    def __init__(self, analyser):
        self.analyser = analyser
        self.clock = 0.0            # seconds of audio analysed; the Analyser's clock
        self.last_pulse = 0.0
        self.sent = {}              # continuous emitter short id -> last value reported
        self.leftover = np.zeros(0, dtype=np.float32)

    def feed(self, chunk):
        """
        Analyse a chunk of 16 kHz mono float audio.

        Returns a list of (offset, pulses): `offset` is seconds from the start of the chunk,
        `pulses` is a list of (emitter short id, payload). Continuous emitters appear
        PULSE_RATE times a second, and only when their value has changed.
        """
        audio = np.concatenate([self.leftover, np.asarray(chunk, dtype=np.float32)])
        hops = len(audio) // HOP
        self.leftover = audio[hops * HOP:]

        out = []
        for i in range(hops):
            self.clock += HOP / SR
            self.analyser.process_hop(audio[i * HOP:(i + 1) * HOP], self.clock)

            pulses = list(self.analyser.events)
            self.analyser.events.clear()

            if self.clock - self.last_pulse >= 1 / PULSE_RATE:
                self.last_pulse = self.clock
                for short_id, value in self.analyser.state.items():
                    value = round(value, 4)
                    if self.sent.get(short_id) != value:
                        self.sent[short_id] = value
                        pulses.append((short_id, {"value": value}))

            if pulses:
                out.append((i * HOP / SR, pulses))
        return out
