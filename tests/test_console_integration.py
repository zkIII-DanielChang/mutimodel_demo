from datetime import datetime
from unittest.mock import Mock

from scripts.demo_console import CARD_GROUPS
from src.common.config_loader import load_config
from src.common.csv_logger import CSVLogger
from src.console.csv_tailer import estimate_fps_from_csv, latest_metrics, read_last_lines
from src.console.process_manager import ProcessManager


def _manager(tmp_path):
    return ProcessManager(load_config('demo_scenes')['modalities'], tmp_path, tmp_path)


def test_driving_scene_and_cards_reference_configured_modalities():
    cfg = load_config('demo_scenes')
    driving = next(scene for scene in cfg['scenes'] if scene['id'] == 2)
    assert driving['video_modality'] == 'dms'
    assert cfg['modalities']['dms']['command'] == ['python', 'scripts/run_dms.py']
    assert {key for _, group in CARD_GROUPS for key in group} == set(cfg['modalities'])


def test_camera_switch_does_not_start_if_previous_process_cannot_stop(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    fatigue, dms = (manager.modalities[key] for key in ['fatigue', 'dms'])
    monkeypatch.setattr(fatigue, 'is_running', lambda: True)
    stop = Mock(return_value=False)
    launch = Mock(return_value=True)
    monkeypatch.setattr(fatigue, 'stop', stop)
    monkeypatch.setattr(dms, 'start', launch)
    assert not manager.start('dms')
    stop.assert_called_once()
    launch.assert_not_called()


def test_voiceprint_releases_asr_microphone(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    asr, voiceprint = (manager.modalities[key] for key in ['asr', 'voiceprint'])
    operations = []
    monkeypatch.setattr(asr, 'is_running', lambda: True)
    monkeypatch.setattr(asr, 'stop', lambda: operations.append('stop ASR') or True)
    monkeypatch.setattr(voiceprint, 'start', lambda: operations.append('start voiceprint') or True)
    assert manager.start('voiceprint')
    assert operations == ['stop ASR', 'start voiceprint']


def test_nonzero_child_exit_reports_error(tmp_path):
    dms = _manager(tmp_path).modalities['dms']
    dms.process = Mock()
    dms.process.poll.return_value = 1
    dms.started_at = 123
    assert not dms.is_running()
    assert dms.status == 'error'
    assert dms.started_at is None


def test_console_filters_old_metrics_and_counts_state_frames(tmp_path):
    path = tmp_path / 'dms.csv'
    path.write_text(','.join(CSVLogger.FIELDS) + '\n'
                    '2026-01-01T12:00:00.000,dms,ear,0.3,\n'
                    '2026-01-01T12:00:01.000,dms,state,normal,\n'
                    '2026-01-01T12:00:01.500,dms,state,unavailable,\n', encoding='utf-8')
    since = datetime.fromisoformat('2026-01-01T12:00:01').timestamp()
    assert latest_metrics(path, ['state', 'ear'], since=since) == {'state': 'unavailable'}
    assert estimate_fps_from_csv(path, window_seconds=float('inf'), since=since, metric='state') == 2


def test_csv_tailer_does_not_consume_unfinished_multiline_record(tmp_path):
    path = tmp_path / 'asr.csv'
    path.write_text(','.join(CSVLogger.FIELDS) + '\n'
                    '2026-01-01T12:00:01.000,asr,text,"第一行\n', encoding='utf-8', newline='')
    assert read_last_lines(path) == []
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write('第二行",\n')
    assert latest_metrics(path, ['text']) == {'text': '第一行\n第二行'}
    with CSVLogger(path) as logger:
        logger.log('asr', 'text', '新消息')
    assert latest_metrics(path, ['text']) == {'text': '新消息'}
