"""Offline-only deterministic acceptance tests: no remote source is opened."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from calibrate import derive_thresholds, recalculate, reconcile_count, _save_atomic
from main import validate_config, monitor
from modules import events
from modules.detector import Change, StateTracker, brightness_score, check_frame_geometry, classify

ROOT = Path(__file__).resolve().parents[1]


def config_for(count: int) -> dict:
    sample = json.loads((ROOT / 'config.json.example').read_text())
    sample['sensor_count'] = count
    sample['stream']['url'] = ''
    sample['video']['reference_width'] = 640
    sample['video']['reference_height'] = 480
    sample['panel_roi'] = None
    sample['indicators'] = []
    for i in range(count):
        sample['indicators'].append({
            'id': f'SENSOR_{i+1:03}', 'name': f'Indicator {i+1}',
            'normal_state': 'OFF', 'roi': [10 + (i%8)*70, 20 + (i//8)*70, 17, 17],
            'reference_roi': None, 'off_threshold': 70, 'on_threshold': 150,
            'calibration_status': 'verified',
            'calibration_samples': {'ON':[210.0],'OFF':[30.0]}
        })
    return sample


@pytest.mark.parametrize('count', [1,5,12,20])
def test_dynamic_sensor_counts(count):
    conf = config_for(count)
    validate_config(conf)
    tracker = StateTracker(conf['indicators'], 2, 2.0)
    assert len(tracker.confirmed) == count
    frame = np.full((480,640,3), 30, dtype=np.uint8)
    check_frame_geometry(frame, conf)
    for sensor in conf['indicators']:
        assert classify(frame, sensor, .82).state == 'OFF'


def test_reject_mismatched_count_duplicate_ids_and_out_of_bounds():
    c = config_for(5)
    c['sensor_count'] = 6
    with pytest.raises(ValueError, match='sensor_count'):
        validate_config(c)
    c['sensor_count'] = 5
    c['indicators'][1]['id'] = c['indicators'][0]['id']
    with pytest.raises(ValueError, match='unique'):
        validate_config(c)
    c = config_for(5)
    c['indicators'][1]['roi'] = [636, 25, 17, 17]
    with pytest.raises(ValueError, match='outside'):
        validate_config(c)


def test_reject_obsolete_color_fields_and_missing_thresholds():
    c = config_for(1)
    c['indicators'][0]['hue_ranges'] = [[0,12]]
    with pytest.raises(ValueError, match='obsolete'):
        validate_config(c)
    del c['indicators'][0]['hue_ranges']
    c['indicators'][0]['on_threshold'] = None
    with pytest.raises(ValueError, match='both'):
        validate_config(c)
    c['indicators'][0]['off_threshold'] = None
    c['indicators'][0]['calibration_status'] = 'uncalibrated'
    validate_config(c, operational=False)
    with pytest.raises(ValueError, match='uncalibrated'):
        validate_config(c)


@pytest.mark.parametrize('bgr', [(0,0,220),(0,220,0),(220,0,0),(220,220,220)])
def test_color_independent_illumination(bgr):
    c = config_for(1)
    frame = np.zeros((480,640,3), np.uint8)
    frame[20:37,10:27] = bgr
    assert classify(frame, c['indicators'][0], .82).state == 'ON'
    frame[20:37,10:27] = (25,25,25)
    assert classify(frame, c['indicators'][0], .82).state == 'OFF'
    frame[20:37,10:27] = (100,100,100)
    assert classify(frame, c['indicators'][0], .82).state == 'UNKNOWN'


def test_reference_roi_and_p80_reject_single_bright_pixel():
    c = config_for(1)
    sensor = c['indicators'][0]
    sensor['reference_roi'] = [100,100,17,17]
    sensor['off_threshold'] = 35
    sensor['on_threshold'] = 80
    frame = np.full((480,640,3), 100, np.uint8)
    frame[20:37,10:27] = (170,0,0)
    score, bright, reference = brightness_score(frame, sensor, .82)
    assert score == pytest.approx(70)
    assert bright == pytest.approx(170)
    assert reference == pytest.approx(100)
    assert classify(frame, sensor, .82).state == 'UNKNOWN'
    frame[23,13] = 255
    assert classify(frame, sensor, .82).state == 'UNKNOWN'
    frame[20:37,10:27] = (210,0,0)
    assert classify(frame, sensor, .82).state == 'ON'


def tracked(count=1, normal='OFF'):
    c = config_for(count)
    for sensor in c['indicators']:
        sensor['normal_state'] = normal
    return StateTracker(c['indicators'], 2, 2.0), c['indicators']


def step(tracker, sensor, state, clock):
    return tracker.update(sensor['id'], state, '2026-10-24 12:00:00', clock,
                          np.full((17,17,3), 210 if state=='ON' else 30, dtype=np.uint8))


def test_changes_unknown_and_confirmation_both_directions():
    tracker, sensors = tracked()
    s = sensors[0]
    assert step(tracker, s, 'OFF', 0) is None
    assert step(tracker, s, 'OFF', 0.1) is None
    assert step(tracker, s, 'ON', .2) is None
    assert step(tracker, s, 'UNKNOWN', .25) is None
    assert step(tracker, s, 'ON', .3) is None
    change = step(tracker, s, 'ON', .4)
    assert change.kind == 'TRANSITION' and change.description == 'OFF -> ON'
    assert change.previous_crop is not None
    assert step(tracker, s, 'ON', .5) is None
    assert step(tracker, s, 'OFF', .6) is None
    change = step(tracker, s, 'OFF', .7)
    assert change.description == 'ON -> OFF'


def test_prolonged_uncertainty_and_stream_interruption_no_fabricated_transition():
    tracker, sensors = tracked()
    s = sensors[0]
    step(tracker,s,'OFF',0)
    step(tracker,s,'OFF',.1)
    step(tracker,s,'UNKNOWN',1)
    step(tracker,s,'UNKNOWN',3.1)
    step(tracker,s,'ON',3.2)
    event = step(tracker,s,'ON',3.3)
    assert event.kind == 'RESYNC_MISMATCH' and 'unobserved gap' in event.description
    tracker.invalidate()
    step(tracker,s,'OFF',4)
    event = step(tracker,s,'OFF',4.1)
    assert event.kind == 'RESYNC_MISMATCH'
    tracker.invalidate()
    step(tracker,s,'OFF',5)
    assert step(tracker,s,'OFF',5.1) is None


def test_startup_mismatch_has_evidence_event_not_transition():
    tracker, sensors = tracked(normal='ON')
    step(tracker,sensors[0],'OFF',0)
    event = step(tracker,sensors[0],'OFF',.1)
    assert event.kind == 'STARTUP_MISMATCH'
    assert 'expected ON' in event.description


def test_independent_simultaneous_events(tmp_path, capsys):
    tracker, sensors = tracked(5)
    frame = np.full((480,640,3), 30, np.uint8)
    for sensor in sensors:
        step(tracker,sensor,'OFF',0)
        step(tracker,sensor,'OFF',.1)
        step(tracker,sensor,'ON',.2)
    changes = [step(tracker,sensor,'ON',.3) for sensor in sensors]
    assert all(event is not None for event in changes)
    when = datetime(2026,10,24,14,30,tzinfo=timezone.utc)
    degraded = events.emit(changes, frame, when, tmp_path, 90, True, 2)
    assert not degraded
    screenshots = list(tmp_path.rglob('*.jpg'))
    assert len(screenshots) == 10  # independent originals and previews
    assert len({p.stem for p in screenshots}) == 10
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    for line in lines:
        path = Path(line.split(' | ')[-1])
        assert path.is_file() and path.stat().st_size > 0
        assert cv2.imread(str(path)).shape[:2] == (480,640)


def test_screenshot_failure_is_reported_no_success_path(tmp_path, capsys):
    change = Change('SENSOR_001','Indicator 1','OFF -> ON','TRANSITION',(10,20,17,17),None)
    with mock.patch.object(events,'_persist_jpeg',side_effect=OSError('Disk full')):
        degraded = events.emit([change],np.full((480,640,3),30,np.uint8),
                               datetime.now(timezone.utc),tmp_path,90,True,2)
    assert degraded
    assert 'SCREENSHOT_MISSING' in capsys.readouterr().out
    assert not list(tmp_path.rglob('*.jpg'))


def test_threshold_derivation_and_incomplete_samples():
    assert derive_thresholds([190,200],[25,40],12) == (90.0,140.0)
    with pytest.raises(ValueError,match='separation'):
        derive_thresholds([100],[99],12)
    c = config_for(1)
    sensor=c['indicators'][0]
    sensor['calibration_samples']={'ON': [190,200], 'OFF':[25,40]}
    assert 'VERIFIED' in recalculate(sensor,12)
    assert sensor['calibration_status']=='verified'
    sensor['calibration_samples']['ON']=[]
    assert 'required' in recalculate(sensor,12)
    assert sensor['off_threshold'] is None


def test_calibrate_reconciliation_preserves_sensor_configuration_and_requires_delete(monkeypatch):
    c = config_for(5)
    c['sensor_count']=12
    reconcile_count(c)
    assert len(c['indicators']) == 12
    assert c['indicators'][0]['id'] == 'SENSOR_001'
    assert c['indicators'][0]['off_threshold'] == 70
    assert len({i['id'] for i in c['indicators']}) == 12
    c['sensor_count']=1
    monkeypatch.setattr('builtins.input',lambda _:'NO')
    with pytest.raises(Exception,match='not authorized'):
        reconcile_count(c)
    assert len(c['indicators']) == 12
    reconcile_count(c,confirm_delete=True)
    assert len(c['indicators']) == 1


def test_save_atomic_keeps_config_and_backup(tmp_path):
    conf = config_for(1)
    path=tmp_path/'config.json'
    path.write_text(json.dumps(conf))
    conf['sensor_count']=1
    _save_atomic(path,conf)
    assert json.loads(path.read_text())['sensor_count'] == 1
    assert path.with_suffix('.json.bak').exists()


def test_migrated_config_preserves_legacy_rois():
    new = json.loads((ROOT/'config.json').read_text())
    expected = [
        ('RED_1', 'System Failure', 'OFF', [722,412,17,17]),
        ('GREEN', 'Status OK', 'ON', [722,428,17,17]),
        ('YELLOW', 'In Production', 'ON', [722,443,17,17]),
        ('RED_2', 'Pressure Failure', 'OFF', [1131,285,17,17]),
        ('RED_3', 'Machine Failure', 'OFF', [140,562,17,17]),
    ]
    assert new['sensor_count'] == len(expected)
    assert [(i['id'],i['name'],i['normal_state'],i['roi']) for i in new['indicators']] == expected
    validate_config(new, operational=False)
    with pytest.raises(ValueError,match='uncalibrated'):
        validate_config(new)


def test_cli_check_schema_does_not_open_stream(monkeypatch):
    from main import main
    from modules.stream import StreamReader
    def blocked_open(self):
        raise AssertionError("Network should never be opened for --check-schema")
    monkeypatch.setattr(StreamReader, 'open', blocked_open)
    assert main(['--config', str(ROOT / 'config.json'), '--check-schema']) == 0


def test_local_video_end_to_end_including_simultaneous_transitions(tmp_path, monkeypatch, capsys):
    from tools.generate_demo import main as make_demo
    from main import main
    folder = tmp_path/'demo'
    monkeypatch.setattr('sys.argv', ['generate_demo.py','--directory',str(folder),'--count','12'])
    make_demo()
    capsys.readouterr()
    assert main(['--config',str(folder/'config.json')]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if ' -> ' in line]
    assert len(lines) == 4
    assert sum('OFF -> ON' in line for line in lines) == 3
    assert sum('ON -> OFF' in line for line in lines) == 1
    originals = [x for x in (folder/'screenshots').rglob('*.jpg') if not x.stem.endswith('_preview')]
    assert len(originals) == 4
    assert all(cv2.imread(str(p)).shape == (480,640,3) for p in originals)
