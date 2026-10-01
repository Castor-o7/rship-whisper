import unittest

from transcript_events import SETTLE_SECONDS, TranscriptTracker, cue_short_id, normalise_cue


def named(pulses, short_id):
    return [payload for name, payload in pulses if name == short_id]


class TestCues(unittest.TestCase):
    def test_cue_fires_once_while_the_segment_is_revised(self):
        tracker = TranscriptTracker(cues=["midnight"])
        self.assertEqual(named(tracker.update([" It was"], 0.0), "cue"), [])
        self.assertEqual(named(tracker.update([" It was midnight"], 0.5), "cue"), [{"cue": "midnight"}])
        self.assertEqual(named(tracker.update([" It was midnight when"], 1.0), "cue"), [])
        # ... and not again when the segment is completed.
        self.assertEqual(named(tracker.update([" It was midnight when he left.", " Then"], 1.5), "cue"), [])

    def test_cue_does_not_fire_again_if_a_revision_drops_and_restores_it(self):
        tracker = TranscriptTracker(cues=["midnight"])
        tracker.update([" at midnight"], 0.0)
        tracker.update([" at mid night"], 0.5)
        self.assertEqual(named(tracker.update([" at midnight we"], 1.0), "cue"), [])

    def test_each_cue_has_its_own_emitter(self):
        tracker = TranscriptTracker(cues=["Open the door!"])
        pulses = tracker.update([" Please open the door"], 0.0)
        self.assertEqual(named(pulses, "cue"), [{"cue": "open the door"}])
        self.assertEqual(named(pulses, "cue_open_the_door"), [{"value": 1}])
        self.assertEqual(cue_short_id("Open the door!"), "cue_open_the_door")

    def test_cues_match_whole_words_whatever_the_case_or_punctuation(self):
        tracker = TranscriptTracker(cues=["night"])
        self.assertEqual(named(tracker.update([" Midnight, tonight."], 0.0), "cue"), [])
        self.assertEqual(named(tracker.update([" Midnight, tonight. NIGHT!"], 0.5), "cue"), [{"cue": "night"}])

    def test_a_cue_said_twice_fires_twice(self):
        tracker = TranscriptTracker(cues=["no"])
        self.assertEqual(len(named(tracker.update([" No, no"], 0.0), "cue")), 2)
        self.assertEqual(len(named(tracker.update([" No, no."], 0.5), "cue")), 0)
        self.assertEqual(len(named(tracker.update([" No, no.", " No"], 1.0), "cue")), 1)

    def test_cue_survives_the_segment_vanishing_during_a_pause(self):
        tracker = TranscriptTracker(cues=["midnight"])
        tracker.update([" One.", " Two.", " at midnight"], 0.0)
        # The server sends only completed segments while it waits for more audio ...
        self.assertEqual(named(tracker.update([" One.", " Two."], 1.0), "cue"), [])
        # ... then brings the segment in progress back.
        self.assertEqual(named(tracker.update([" One.", " Two.", " at midnight we"], 2.0), "cue"), [])

    def test_cue_does_not_fire_again_when_the_segment_is_split(self):
        # Seen from a real server: one segment in progress is completed in part, the rest carries on.
        tracker = TranscriptTracker(cues=["country", "ask not"])
        first = tracker.update([" And so, my fellow Americans, ask not what your country can do for you"], 0.0)
        self.assertEqual(len(named(first, "cue")), 2)
        split = tracker.update([" And so my fellow Americans", " ask not what your country can do for you ask"], 1.0)
        self.assertEqual(named(split, "cue"), [])
        more = tracker.update([" And so my fellow Americans",
                               " ask not what your country can do for you, ask what you can do for your country."], 2.0)
        self.assertEqual(named(more, "cue"), [{"cue": "country"}])

    def test_cue_does_not_fire_again_when_the_segment_is_split_in_three(self):
        tracker = TranscriptTracker(cues=["country", "ask not"])
        tracker.update([" And so, my fellow Americans, ask not what your country can do for you"], 0.0)
        split = tracker.update([" And so my fellow Americans", " Ask not", " what your country can do for you"], 1.0)
        self.assertEqual(named(split, "cue"), [])
        self.assertEqual([p["message"] for p in named(split, "sentence")], ["And so my fellow Americans", "Ask not"])

    def test_set_cues_normalises_and_replaces(self):
        tracker = TranscriptTracker(cues=["one"])
        self.assertEqual(tracker.set_cues([" Two ", "", "!!", "two", "Three  Four"]), ["two", "three four"])
        self.assertEqual(named(tracker.update([" one two"], 0.0), "cue"), [{"cue": "two"}])
        self.assertEqual(normalise_cue("  The   END. "), "the end")


class TestSentences(unittest.TestCase):
    def test_sentence_fires_when_the_next_segment_begins(self):
        tracker = TranscriptTracker()
        self.assertEqual(named(tracker.update([" Ask not"], 0.0), "sentence"), [])
        pulses = tracker.update([" Ask not what your country can do for you.", " Ask"], 0.5)
        self.assertEqual(named(pulses, "sentence"),
                         [{"message": "Ask not what your country can do for you.", "words": 9}])

    def test_resent_segments_do_not_fire_again(self):
        tracker = TranscriptTracker()
        tracker.update([" One.", " Two.", " Thr"], 0.0)
        self.assertEqual(named(tracker.update([" One.", " Two.", " Three"], 0.5), "sentence"), [])
        # The window slides, and slides back when only completed segments are sent.
        pulses = tracker.update([" Two.", " Three.", " Four"], 1.0)
        self.assertEqual([p["message"] for p in named(pulses, "sentence")], ["Three."])
        self.assertEqual(named(tracker.update([" One.", " Two.", " Three."], 1.5), "sentence"), [])
        self.assertEqual(named(tracker.update([" Two.", " Three.", " Four"], 2.0), "sentence"), [])

    def test_last_sentence_before_a_pause_fires_once_it_settles(self):
        tracker = TranscriptTracker()
        tracker.update([" One.", " And then nothing"], 0.0)
        self.assertEqual(named(tracker.tick(SETTLE_SECONDS - 0.1), "sentence"), [])
        self.assertEqual([p["message"] for p in named(tracker.tick(SETTLE_SECONDS), "sentence")],
                         ["And then nothing"])
        self.assertEqual(named(tracker.tick(SETTLE_SECONDS + 1), "sentence"), [])
        # It is not reported a second time when speech resumes and it is completed.
        pulses = tracker.update([" One.", " And then nothing.", " Until"], SETTLE_SECONDS + 2)
        self.assertEqual(named(pulses, "sentence"), [])

    def test_a_segment_still_changing_does_not_settle(self):
        tracker = TranscriptTracker()
        tracker.update([" And"], 0.0)
        tracker.update([" And then"], 1.5)
        self.assertEqual(named(tracker.tick(2.5), "sentence"), [])
        self.assertEqual(len(named(tracker.tick(3.5), "sentence")), 1)

    def test_questions_and_exclamations(self):
        tracker = TranscriptTracker()
        pulses = tracker.update([" Who goes there?", " Stop!", " Fine."], 0.0)
        pulses += tracker.update([" Stop!", " Fine.", " So"], 0.5)
        self.assertEqual(named(pulses, "question"), [{"message": "Who goes there?"}])
        self.assertEqual(named(pulses, "exclamation"), [{"message": "Stop!"}])
        self.assertEqual(len(named(pulses, "sentence")), 3)


class TestWordCounts(unittest.TestCase):
    def test_word_count_follows_the_transcript(self):
        tracker = TranscriptTracker()
        self.assertEqual(named(tracker.update([" One two"], 0.0), "wordCount"), [{"value": 2}])
        self.assertEqual(named(tracker.update([" One two"], 0.5), "wordCount"), [])
        self.assertEqual(named(tracker.update([" One two three.", " Four"], 1.0), "wordCount"), [{"value": 4}])

    def test_nothing_is_reported_before_anything_is_said(self):
        tracker = TranscriptTracker()
        self.assertEqual(tracker.tick(0.0), [])
        self.assertEqual(tracker.tick(10.0), [])

    def test_words_per_minute_rises_with_speech_and_decays_in_silence(self):
        tracker = TranscriptTracker()
        text, rate = "", 0
        for second in range(20):    # three words a second is 180 words a minute
            text += " la la la"
            for payload in named(tracker.update([text], float(second)), "wordsPerMinute"):
                rate = payload["value"]
        self.assertAlmostEqual(rate, 180, delta=15)
        for second in range(20, 40):
            for payload in named(tracker.tick(float(second)), "wordsPerMinute"):
                rate = payload["value"]
        self.assertEqual(rate, 0)


if __name__ == "__main__":
    unittest.main()
