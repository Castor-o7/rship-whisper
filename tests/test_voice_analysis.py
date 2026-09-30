import unittest

import numpy as np

from voice_analysis import HOP, SR, Analyser, VoiceFeed


def tone(hz, seconds, amplitude=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def run(analyser, audio):
    """Feeds audio hop by hop on a sample clock; returns the events raised."""
    events = []
    for i in range(0, len(audio) - HOP + 1, HOP):
        analyser.process_hop(audio[i:i + HOP], (i + HOP) / SR)
        events += analyser.events
        analyser.events.clear()
    return events


class TestAnalyser(unittest.TestCase):
    def test_silence_is_quiet_and_raises_no_events(self):
        analyser = Analyser()
        events = run(analyser, silence(2.0))
        self.assertEqual(events, [])
        self.assertEqual(analyser.state["level"], 0.0)
        self.assertEqual(analyser.state["voiced"], 0.0)
        self.assertAlmostEqual(analyser.state["silenceSeconds"], 2.0, delta=0.05)

    def test_tone_is_voiced_at_its_pitch(self):
        analyser = Analyser()
        run(analyser, tone(220.0, 1.0))
        self.assertEqual(analyser.state["voiced"], 1.0)
        self.assertAlmostEqual(analyser.state["pitch"], 220.0, delta=2.0)
        self.assertGreater(analyser.state["level"], 0.5)
        self.assertAlmostEqual(analyser.state["sustainSeconds"], 1.0, delta=0.1)

    def test_pitch_norm_spans_80_to_800_hz(self):
        low, high = Analyser(), Analyser()
        run(low, tone(80.0, 0.5))
        run(high, tone(800.0, 0.5))
        self.assertAlmostEqual(low.state["pitchNorm"], 0.0, delta=0.02)
        self.assertAlmostEqual(high.state["pitchNorm"], 1.0, delta=0.02)

    def test_phrase_events_carry_durations(self):
        analyser = Analyser()
        events = run(analyser, np.concatenate([silence(1.0), tone(220.0, 2.0), silence(1.0)]))
        phrases = [(name, payload["value"]) for name, payload in events if name.startswith("phrase")]
        self.assertEqual([name for name, _ in phrases], ["phraseStart", "phraseEnd"])
        self.assertAlmostEqual(phrases[0][1], 1.0, delta=0.1)   # silence before the phrase
        self.assertAlmostEqual(phrases[1][1], 2.0, delta=0.1)   # length of the phrase

    def test_gate_decides_what_counts_as_sound(self):
        quiet = tone(220.0, 1.0, amplitude=0.002)   # about -57 dBFS
        default, sensitive = Analyser(), Analyser(gate_db=-70.0)
        run(default, quiet)
        run(sensitive, quiet)
        self.assertEqual(default.state["soundSeconds"], 0.0)
        self.assertGreater(sensitive.state["soundSeconds"], 0.5)

    def test_forget_clears_repetition_memory(self):
        analyser = Analyser()
        run(analyser, tone(220.0, 3.0))
        self.assertGreater(len(analyser.memory), 20)
        analyser.forget()
        run(analyser, tone(220.0, 0.2))
        self.assertLess(len(analyser.memory), 5)
        self.assertEqual(len(analyser.memory), len(analyser.times_mem))


class TestVoiceFeed(unittest.TestCase):
    def test_pulses_are_spread_across_the_chunk(self):
        feed = VoiceFeed(Analyser())
        out = feed.feed(tone(220.0, 4096 / SR))
        offsets = [offset for offset, _ in out]
        self.assertGreater(len(offsets), 2)
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreaterEqual(offsets[0], 0.0)
        self.assertLess(offsets[-1], 4096 / SR)

    def test_unchanged_values_are_not_repeated(self):
        feed = VoiceFeed(Analyser())
        feed.feed(silence(4096 / SR))
        pulsed = {short_id for _, pulses in feed.feed(silence(4096 / SR)) for short_id, _ in pulses}
        # In silence only the length of the silence is changing.
        self.assertEqual(pulsed, {"silenceSeconds"})

    def test_chunk_size_does_not_change_the_result(self):
        audio = np.concatenate([silence(0.5), tone(220.0, 1.0), silence(0.7)])

        def collect(chunk_size):
            feed = VoiceFeed(Analyser())
            pulses = []
            for i in range(0, len(audio), chunk_size):
                for _, chunk_pulses in feed.feed(audio[i:i + chunk_size]):
                    pulses += chunk_pulses
            return pulses

        self.assertEqual(collect(4096), collect(1000))


if __name__ == "__main__":
    unittest.main()
