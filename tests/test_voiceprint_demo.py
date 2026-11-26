from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.voiceprint_demo import BLOCK_SIZE, UtteranceBuffer
from src.voiceprint.engine import SAMPLE_RATE, VoiceprintEngine


def _signal(size):
    return (0.04 * np.sin(np.arange(size) * 0.15)).astype(np.float32)


def _engine(tmp_path):
    engine = VoiceprintEngine.__new__(VoiceprintEngine)
    engine._book_path = tmp_path / 'speakers.npz'
    engine._threshold = 0.40
    engine._vp = Mock()
    return engine


def test_continuous_speech_emits_at_length_limit():
    buffer = UtteranceBuffer()
    segments = [segment for _ in range(100) if (segment := buffer.feed(_signal(BLOCK_SIZE))) is not None]
    assert len(segments) == 1
    assert 6 * SAMPLE_RATE <= len(segments[0]) < 6 * SAMPLE_RATE + BLOCK_SIZE


def test_short_noise_does_not_count_as_speech():
    buffer = UtteranceBuffer()
    assert buffer.feed(_signal(BLOCK_SIZE)) is None
    segments = [segment for _ in range(12) if (segment := buffer.feed(np.zeros(BLOCK_SIZE))) is not None]
    assert len(segments) == 1 and segments[0].size == 0


@pytest.mark.parametrize(('scores', 'reason'), [
    ([0.39], 'below_threshold'), ([0.62, 0.59], 'ambiguous'),
    ([0.62, 0.49], 'matched'), ([0.62], 'matched'),
])
def test_identification_threshold_and_margin(tmp_path, scores, reason):
    engine = _engine(tmp_path)
    engine._vp.book = scores
    engine._vp.identify.return_value = [SimpleNamespace(name=f'person{i}', score=score)
                                        for i, score in enumerate(scores)]
    result = engine.identify_audio(_signal(SAMPLE_RATE))
    assert result['reason'] == reason
    assert result['name'] == ('person0' if reason == 'matched' else 'Unknown')
    assert result['accepted'] == (reason == 'matched')


def test_inconsistent_enrollment_does_not_modify_book(tmp_path):
    engine = _engine(tmp_path)
    engine._vp.embed_many.return_value = np.eye(2, dtype=np.float32)
    engine._book_path.write_bytes(b'existing')
    original = engine._vp.book
    assert engine.enroll_from_audio('test', [_signal(SAMPLE_RATE)] * 2) == 0
    assert engine._vp.book is original
    assert engine._book_path.read_bytes() == b'existing'


def test_enrollment_persists_and_reloads_template(tmp_path):
    store = pytest.importorskip('voicefingerprint.store')
    engine = _engine(tmp_path)
    engine._vp.book = store.SpeakerBook('test-model', 2)
    engine._vp.embed_many.return_value = np.array([[1., 0.], [0.99, 0.01]])
    assert engine.enroll_from_audio('测试', [_signal(SAMPLE_RATE)] * 2) == 2
    book = store.SpeakerBook.load(engine._book_path, expect_model='test-model')
    assert book.names == engine.list_speakers() == ['测试']
    assert book['测试'].n_utterances == 2
