"""Transcript events for rship-whisper.

WhisperLive sends the last few segments of the transcript over and over, revising the one
in progress as it hears more. A Binding needs something simpler to react to: an event when a
sentence is finished, an event when a cue word is spoken, a number for how fast someone is
talking. TranscriptTracker turns the stream of revisions into those.
"""

import re
import threading
from collections import deque

SETTLE_SECONDS = 2.0    # a segment in progress that stops changing for this long is finished
RATE_WINDOW = 15.0      # seconds of speech wordsPerMinute is measured over
RATE_MIN_WINDOW = 5.0   # ... and the shortest span it is extrapolated from, at the start of a run
MEMORY = 20             # completed segments remembered, to recognise ones that are sent again

# short id -> (display name, schema)
MESSAGE_SCHEMA = {"type": "object", "properties": {"message": {"type": "string"}}}
VALUE_SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}}}
EMITTERS = {
    "sentence": ("Sentence finished (event)", {
        "type": "object",
        "properties": {"message": {"type": "string"}, "words": {"type": "number"}},
    }),
    "question": ("Question asked (event)", MESSAGE_SCHEMA),
    "exclamation": ("Exclamation (event)", MESSAGE_SCHEMA),
    "wordCount": ("Words spoken this run", VALUE_SCHEMA),
    "wordsPerMinute": ("Words per minute", VALUE_SCHEMA),
    "cue": ("Cue spoken (event, any cue)", {"type": "object", "properties": {"cue": {"type": "string"}}}),
}


def words(text):
    """The words of a text, lowercased and without punctuation."""
    return re.findall(r"[\w']+", text.lower())


def normalise_cue(cue):
    """A cue as it is matched: lowercase words separated by single spaces. Empty if it has no words."""
    return " ".join(words(cue))


def cue_short_id(cue):
    """The short id of the emitter that fires for one particular cue."""
    return "cue_" + "_".join(words(cue))


def count_cue(cue_words, text_words):
    """How many times a cue (as a list of words) appears, as whole words, in a text (as a list of words)."""
    n = len(cue_words)
    return sum(1 for i in range(len(text_words) - n + 1) if text_words[i:i + n] == cue_words)


class TranscriptTracker:
    """
    Follows one speaker's transcript and reports what is new.

    `update` takes the texts WhisperLive last sent (completed segments, then usually the one in
    progress) and `tick` is called a couple of times a second; both return a list of
    (emitter short id, payload) to pulse. Times are seconds on any steady clock. Safe to call
    from different threads.
    """
    def __init__(self, cues=()):
        self.lock = threading.Lock()
        self.cues = {}                  # normalised cue -> its words
        self.completed = []             # texts of the last MEMORY completed segments, oldest first
        self.current = ""               # text of the segment in progress
        self.current_changed = 0.0      # when that text last changed
        self.current_reported = False   # whether `sentence` has fired for it
        self.cue_counts = {}            # cue -> times it has fired within the segment in progress
        self.completed_words = 0
        self.history = deque()          # (time, words spoken so far)
        self.sent = {}                  # wordCount / wordsPerMinute last reported
        self.set_cues(cues)

    def set_cues(self, cues):
        """Replaces the cues listened for. Returns them normalised, without blanks or repeats."""
        with self.lock:
            self.cues = {}
            for cue in cues:
                cue = normalise_cue(cue)
                if cue:
                    self.cues[cue] = cue.split(" ")
            return list(self.cues)

    def update(self, segments, now):
        with self.lock:
            segments = [s.strip() for s in segments if s.strip()]
            pulses = []
            fresh = self._fresh(segments)

            # Everything fresh but the last has been completed. The first of them is the segment
            # that was in progress, so the sentence it may already have reported is not repeated.
            # Whisper often completes that segment in pieces and carries on with the rest as a new
            # one, so cues it has already fired are matched off against each piece in turn and
            # whatever is left belongs to the new segment in progress.
            fired, rest = self.cue_counts, words(self.current)
            for i, text in enumerate(fresh[:-1]):
                was_current = i == 0 and self.current != ""
                pulses += self._cue_pulses(text, fired)
                piece = words(text)
                if len(piece) < len(rest) and rest[:len(piece)] == piece:
                    fired = {cue: fired[cue] - count_cue(cue_words, piece) for cue, cue_words in self.cues.items()
                             if fired.get(cue, 0) > count_cue(cue_words, piece)}
                    rest = rest[len(piece):]
                else:
                    fired, rest = {}, []
                if not (was_current and self.current_reported):
                    pulses += self._sentence_pulses(text)
                self.completed_words += len(words(text))
                self.completed = (self.completed + [text])[-MEMORY:]

            if len(fresh) > 1:      # the segment in progress was completed; a new one has begun
                self.current = ""
                self.cue_counts = fired
                self.current_reported = False
            # With nothing fresh the segment in progress is left as it was: during a pause the
            # server sends only completed segments, and brings the one in progress back afterwards.
            if fresh and fresh[-1] != self.current:
                self.current = fresh[-1]
                self.current_changed = now
            pulses += self._cue_pulses(self.current, self.cue_counts)

            return pulses + self._count_pulses(now)

    def tick(self, now):
        with self.lock:
            pulses = []
            if self.current and not self.current_reported and now - self.current_changed >= SETTLE_SECONDS:
                # The last thing said before a pause is never followed by another segment,
                # so it is reported once it has stopped changing.
                self.current_reported = True
                pulses += self._sentence_pulses(self.current)
            return pulses + self._count_pulses(now)

    def _fresh(self, segments):
        """The segments not yet seen as completed: newly completed ones, then the one in progress."""
        known = self.completed
        n = len(segments)
        # Sent again with nothing in progress, e.g. while the speaker pauses.
        if any(known[i:i + n] == segments for i in range(len(known) - n + 1)):
            return []
        for k in range(min(n, len(known)), 0, -1):
            if known[-k:] == segments[:k]:
                return segments[k:]
        return segments

    def _cue_pulses(self, text, counts):
        pulses = []
        text_words = words(text)
        for cue, cue_words in self.cues.items():
            count = count_cue(cue_words, text_words)
            # Whisper revises as it hears more; a cue that was already counted must not fire again.
            for _ in range(count - counts.get(cue, 0)):
                pulses.append(("cue", {"cue": cue}))
                pulses.append((cue_short_id(cue), {"value": 1}))
            counts[cue] = max(count, counts.get(cue, 0))
        return pulses

    def _sentence_pulses(self, text):
        pulses = [("sentence", {"message": text, "words": len(words(text))})]
        if text.endswith(("?", "？")):
            pulses.append(("question", {"message": text}))
        elif text.endswith(("!", "！")):
            pulses.append(("exclamation", {"message": text}))
        return pulses

    def _count_pulses(self, now):
        total = self.completed_words + len(words(self.current))
        if not self.history:
            if total == 0:
                return []       # nothing said yet
            self.history.append((now, 0))
        self.history.append((now, total))
        # Keep one sample at or before the start of the window to measure from.
        while len(self.history) > 1 and self.history[1][0] <= now - RATE_WINDOW:
            self.history.popleft()
        then, before = self.history[0]
        span = min(RATE_WINDOW, max(now - then, RATE_MIN_WINDOW))
        values = {"wordCount": total, "wordsPerMinute": round(max(0, total - before) * 60 / span)}

        pulses = []
        for short_id, value in values.items():
            if self.sent.get(short_id) != value:
                self.sent[short_id] = value
                pulses.append((short_id, {"value": value}))
        return pulses
