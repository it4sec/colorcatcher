"""Deterministic, offline-only tests. No factory/remote endpoint is opened."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from calibrate import Calibrator, _save_atomic, capture_baseline, reconcile_count
from main import main, validate_config
from modules import events
from modules.detector import Change, StateTracker, brightness_score, check_frame_geometry, classify

ROOT = Path(__file__).resolve().parents[1]


def config_for(count: int = 1) -> dict:
    config = json.loads((ROOT / 'config.json.example').read_text())
    config['sensor_count'] = count
    config['stream']['url'] = ''
    config['video'].update(reference_width=640, reference_height=480)
    config['panel_roi'] = None
    config['indicators'] = [
        {'id': f'SENSOR_{i+1:03d}', 'name': f'Sensor {i+1}', 'normal_state': 'OFF',
         'baseline_state': 'OFF', 'roi': [10+(i%8)*70, 20+(i//8)*70, 17, 17],
         'reference_roi': None, 'baseline_brightness': 100.0,
         'baseline_reference_brightness': None}
        for i in range(count)]
    return config


def image(conf: dict, brightness: int = 100) -> np.ndarray:
    return np.full((conf['video']['reference_height'], conf['video']['reference_width'], 3),
                   brightness, np.uint8)


@pytest.mark.parametrize('count', [1, 5, 12, 20])
def test_dynamic_sensor_count(count):
    conf = config_for(count)
    validate_config(conf)
    assert len(StateTracker(conf['indicators'], 2, 2).confirmed) == count
    check_frame_geometry(image(conf), conf)
    for indicator in conf['indicators']:
        assert classify(image(conf), indicator, 10).state == 'OFF'


def test_reject_invalid_sensor_lists_and_coordinates():
    conf = config_for(5)
    conf['sensor_count'] = 6
    with pytest.raises(ValueError, match='sensor_count'):
        validate_config(conf)
    conf['sensor_count'] = 5
    conf['indicators'][1]['id'] = conf['indicators'][0]['id']
    with pytest.raises(ValueError, match='unique'):
        validate_config(conf)
    conf = config_for(5)
    conf['indicators'][0]['roi'] = [639, 20, 17, 17]
    with pytest.raises(ValueError, match='outside'):
        validate_config(conf)


@pytest.mark.parametrize('value', [None, -1, 1001, False, float('nan')])
def test_reject_bad_percent(value):
    conf = config_for()
    conf['detection']['brightness_change_percent'] = value
    with pytest.raises((ValueError, TypeError)):
        validate_config(conf)


def test_no_baseline_blocks_operational_monitoring():
    conf = config_for()
    conf['indicators'][0]['baseline_brightness'] = None
    validate_config(conf, operational=False)
    with pytest.raises(ValueError, match='baseline missing'):
        validate_config(conf)



def test_true_arithmetic_mean_uses_every_pixel():
    conf = config_for()
    sensor = conf['indicators'][0]
    frame = image(conf, 0)
    x,y,w,h = sensor['roi']
    frame[y:y+h, x:x+w] = 100
    frame[y, x] = 200
    # All 289 pixels count, no elliptical masking or percentile/maximum.
    expected = (288 * 100 + 200) / 289
    measured, ref = brightness_score(frame, sensor)
    assert measured == pytest.approx(expected, abs=0.01)
    assert ref is None
    capture_baseline(sensor, frame)
    assert sensor['baseline_brightness'] == pytest.approx(expected, abs=.001)
    assert classify(frame, sensor, 10).state == 'OFF'


@pytest.mark.parametrize('bgr', [(0,0,220), (0,220,0), (220,0,0), (220,220,220)])
def test_detection_not_hue_dependent(bgr):
    conf = config_for()
    sensor = conf['indicators'][0]
    frame = image(conf)
    x,y,w,h = sensor['roi']
    frame[y:y+h, x:x+w] = bgr
    brightness, _ = brightness_score(frame, sensor)
    expected_gray = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2GRAY)[0,0]
    assert brightness == float(expected_gray)
    assert classify(frame, sensor, 10).state == ('ON' if abs(brightness-100)>10 else 'OFF')


@pytest.mark.parametrize('current,expected', [
    (100, 'OFF'), (110, 'OFF'), (111, 'ON'), (89, 'ON'),
    (90, 'OFF'), (109, 'OFF'), (70, 'ON'), (131, 'ON')])
def test_symmetric_percent_threshold(current, expected):
    conf=config_for()
    assert classify(image(conf, current), conf['indicators'][0], 10).state == expected


def test_initially_on_sensor_reverse_classification():
    conf = config_for()
    sensor = conf['indicators'][0]
    sensor['normal_state'] = 'ON'
    sensor['baseline_state'] = 'ON'
    assert classify(image(conf, 100), sensor, 10).state == 'ON'
    assert classify(image(conf, 89), sensor, 10).state == 'OFF'
    assert classify(image(conf, 111), sensor, 10).state == 'OFF'


def test_zero_baseline_is_safe():
    conf = config_for()
    sensor = conf['indicators'][0]
    sensor['baseline_brightness'] = 0.0
    assert classify(image(conf, 0), sensor, 10).state == 'OFF'
    assert classify(image(conf, 1), sensor, 10).state == 'ON'


def test_ambient_compensation_for_uniform_lighting_shift():
    conf = config_for()
    sensor = conf['indicators'][0]
    sensor['reference_roi'] = [110, 110, 17, 17]
    frame = image(conf, 100)
    capture_baseline(sensor, frame)
    validate_config(conf)
    uniform = image(conf, 120)
    assert classify(uniform, sensor, 10).state == 'OFF'
    uneven = image(conf, 120)
    x,y,w,h = sensor['roi']
    uneven[y:y+h,x:x+w] = 140
    assert classify(uneven, sensor, 10).state == 'ON'


def test_baseline_change_resets_when_roi_moved(tmp_path):
    conf = config_for()
    screen = image(conf)
    sensor = conf['indicators'][0]
    gui = Calibrator(conf, [screen], tmp_path/'config.json', 8)
    gui.display_size = (640,480)
    gui.toolbar_height = 60
    gui._mouse(cv2.EVENT_LBUTTONDOWN, 200, 200+60, 0, None)
    assert sensor['baseline_brightness'] == pytest.approx(100)
    assert sensor['roi'] == [192,192,17,17]


def test_save_calibration_automatically_fills_all_baselines(tmp_path, capsys):
    conf = config_for(5)
    for sensor in conf['indicators']:
        sensor['baseline_brightness'] = None
    target = tmp_path/'config.json'
    target.write_text(json.dumps(conf))
    screen = image(conf, 60)
    gui = Calibrator(conf, [screen], target, 8)
    assert gui._action('save') is True
    saved = json.loads(target.read_text())
    validate_config(saved)
    assert [i['baseline_brightness'] for i in saved['indicators']] == [60.0]*5
    assert 'brightness baselines: 5/5' in capsys.readouterr().out


def test_baseline_state_toggle_tracks_true_calibration_state(tmp_path):
    conf=config_for()
    gui=Calibrator(conf, [image(conf)], tmp_path/'config.json',8)
    gui._action('toggle_state')
    sensor=conf['indicators'][0]
    assert sensor['baseline_state']=='ON'
    assert classify(image(conf, 100), sensor,10).state=='ON'
    gui._action('toggle_state')
    assert sensor['baseline_state']=='OFF'



def test_count_reconcile_keeps_existing_and_requires_confirmation(monkeypatch):
    conf=config_for(5)
    conf['sensor_count']=12
    reconcile_count(conf)
    assert len(conf['indicators'])==12
    assert len({i['id'] for i in conf['indicators']})==12
    conf['sensor_count']=1
    monkeypatch.setattr('builtins.input',lambda _:'NO')
    with pytest.raises(Exception,match='not authorized'):
        reconcile_count(conf)
    reconcile_count(conf,confirm_delete=True)
    assert len(conf['indicators'])==1


def test_state_tracking_reverses_and_unknowns():
    conf=config_for()
    indicator=conf['indicators'][0]
    tracker=StateTracker(conf['indicators'],2,2.0)
    def step(level,clock):
        frame=image(conf,level)
        obs=classify(frame,indicator,10)
        x,y,w,h=indicator['roi']
        return tracker.update(indicator['id'],obs.state,str(clock),clock,frame[y:y+h,x:x+w])
    assert step(100,0) is None
    assert step(100,.1) is None
    assert step(111,.2) is None
    change=step(111,.3)
    assert change.kind=='TRANSITION'
    assert change.description=='OFF -> ON'
    assert step(109,.4) is None
    change=step(100,.5)
    assert change.description=='ON -> OFF'


def test_resync_disallows_fabricated_transition():
    conf=config_for()
    sensor=conf['indicators'][0]
    tracker=StateTracker(conf['indicators'],2,2.0)
    roi=np.full((17,17,3),100,np.uint8)
    for clock in (0, .1):
        tracker.update(sensor['id'],'OFF',str(clock),clock,roi)
    tracker.invalidate()
    assert tracker.update(sensor['id'],'ON','1',1,roi) is None
    change=tracker.update(sensor['id'],'ON','1.1',1.1,roi)
    assert change.kind=='RESYNC_MISMATCH'


def test_parallel_evidence_and_failed_storage(tmp_path,capsys):
    frame=np.full((480,640,3),100,np.uint8)
    changes=[Change(f'S{i}',f'Sensor {i}','OFF -> ON','TRANSITION',(10+i*22,20,17,17),None)
             for i in range(3)]
    when=datetime(2026,10,24,12,tzinfo=timezone.utc)
    assert events.emit(changes,frame,when,tmp_path,90,True,2) is False
    paths=[p for p in tmp_path.rglob('*.jpg') if not p.stem.endswith('_preview')]
    assert len(paths)==3
    assert all(p.stat().st_size>0 for p in paths)
    capsys.readouterr()
    with mock.patch.object(events,'_persist_jpeg',side_effect=OSError('disk full')):
        assert events.emit(changes,frame,when,tmp_path,90,True,2) is True
    assert 'SCREENSHOT_MISSING' in capsys.readouterr().out


def test_atomic_config_save_backups(tmp_path):
    conf=config_for()
    path=tmp_path/'config.json'
    path.write_text(json.dumps(conf))
    _save_atomic(path, conf)
    assert path.with_suffix('.json.bak').exists()
    assert json.loads(path.read_text())['schema_version']==3


def test_provided_config_still_needs_baselines():
    conf=json.loads((ROOT/'config.json').read_text())
    assert conf['sensor_count']==5
    assert [i['roi'] for i in conf['indicators']]==[[722,412,17,17],[722,428,17,17],
        [722,443,17,17],[1131,285,17,17],[140,562,17,17]]
    validate_config(conf,operational=False)
    with pytest.raises(ValueError,match='baseline missing'):
        validate_config(conf)


def test_schema_check_never_opens_stream(monkeypatch):
    from modules.stream import StreamReader
    monkeypatch.setattr(StreamReader,'open',lambda self: (_ for _ in ()).throw(
        AssertionError('No stream should open')))
    assert main(['--config',str(ROOT/'config.json'),'--check-schema'])==0


def test_local_end_to_end(tmp_path, monkeypatch, capsys):
    from tools.generate_demo import main as make_demo
    folder=tmp_path/'demo'
    monkeypatch.setattr('sys.argv',['generate_demo.py','--directory',str(folder),'--count','12'])
    make_demo()
    capsys.readouterr()
    assert main(['--config',str(folder/'config.json')])==0
    lines=[line for line in capsys.readouterr().out.splitlines() if ' -> ' in line]
    assert len(lines)==4
    assert sum('OFF -> ON' in line for line in lines)==3
    assert sum('ON -> OFF' in line for line in lines)==1
    paths=[p for p in (folder/'screenshots').rglob('*.jpg') if not p.stem.endswith('_preview')]
    assert len(paths)==4
    assert all(cv2.imread(str(p)).shape ==(480,640,3) for p in paths)


def test_no_legacy_schema_support_at_runtime():
    conf = config_for()
    conf['schema_version'] = 2
    with pytest.raises(ValueError, match='schema_version must be 3'):
        validate_config(conf, operational=False)


def test_no_legacy_schema_support_in_calibrator(tmp_path, monkeypatch):
    from calibrate import main as calibration_main
    conf = config_for()
    conf['schema_version'] = 2
    path = tmp_path/'config.json'
    path.write_text(json.dumps(conf))
    # Invalid schema must be rejected before reading any images or network sources.
    monkeypatch.setattr('calibrate._images', lambda *_: (_ for _ in ()).throw(
        AssertionError('Must not open image or stream for invalid config')))
    assert calibration_main(['--config', str(path)]) == 2
