"""Create a fully local synthetic video and operational config for offline testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ColorCatcher offline test video")
    parser.add_argument('--directory', type=Path, default=Path('offline_demo'))
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error('--count must be 1..20 for this test asset generator')
    folder = args.directory.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    configuration = json.loads((root/'config.json.example').read_text())
    configuration['sensor_count'] = args.count
    configuration['stream']['url'] = 'demo.avi'
    configuration['video'].update({'reference_width':640,'reference_height':480,'detection_fps':None})
    configuration['panel_roi'] = None
    configuration['indicators'] = []
    colors = [(0,0,230), (0,230,0), (230,0,0), (230,230,230)]
    for i in range(args.count):
        configuration['indicators'].append({
            'id':f'SENSOR_{i+1:03d}', 'name':f'Demo Light {i+1}',
            'normal_state':'OFF', 'baseline_state':'OFF',
            'roi':[20+(i%8)*75,20+(i//8)*70,17,17],
            'reference_roi':None,
            'baseline_brightness':30.0,
            'baseline_reference_brightness':None,
        })
    video_file = folder/'demo.avi'
    writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'MJPG'), 12.0, (640,480))
    if not writer.isOpened():
        raise RuntimeError('Cannot initialize local MJPG video writer')
    try:
        # Base OFF; sensor 1 ON then OFF; sensors 2 and 3 ON simultaneously.
        for frame_index in range(48):
            image = np.full((480,640,3),30,np.uint8)
            for i, indicator in enumerate(configuration['indicators']):
                on = (i == 0 and 12 <= frame_index < 30) or (
                     i in (1,2) and frame_index >= 30)
                if on:
                    x,y,w,h = indicator['roi']
                    image[y:y+h,x:x+w] = colors[i%len(colors)]
            writer.write(image)
    finally:
        writer.release()
    (folder/'config.json').write_text(json.dumps(configuration,indent=2)+'\n')
    print(f'Offline video: {video_file}')
    print(f'Baseline simulation config: {folder/"config.json"}')
    print('Run: python main.py --config '+str(folder/'config.json'))


if __name__ == '__main__':
    main()
