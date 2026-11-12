"""Build Label Studio timelapse tasks, one task per well.

Layout: <dataset_dir>/<dhhmmss>/<00001.jpg ...>; every `--slices` consecutive images are the slices of one well.
Output tasks match the `$timelapse[{{idx}}].image_urls` + `time` label config.

    python scripts/timelapse_wells.py ../dataset/OneDrive_4_11-11-2026 -o .data/tasks/OneDrive_4_11-11-2026.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def build_tasks(dataset_dirs, slices=11, url_prefix='dataset'):
    wells = defaultdict(list)
    n_wells = None
    for dataset_dir in map(Path, dataset_dirs):
        for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and re.fullmatch(r'\d{7}', p.name)):
            images = sorted(f.name for f in folder.glob('*.jpg'))
            if not images or len(images) % slices:
                raise ValueError(f'{folder}: {len(images)} images is not a multiple of {slices} slices')
            if n_wells is not None and len(images) // slices != n_wells:
                raise ValueError(f'{folder}: {len(images) // slices} wells, expected {n_wells}')
            n_wells = len(images) // slices
            n = folder.name  # dhhmmss
            time = f'{n[0]}:{n[1:3]}:{n[3:5]}:{n[5:7]}'
            for w in range(n_wells):
                urls = [
                    f'/data/local-files/?d={url_prefix}/{dataset_dir.name}/{folder.name}/{img}'
                    for img in images[w * slices : (w + 1) * slices]
                ]
                wells[w].append({'time': time, 'image_urls': urls})
    names = ','.join(Path(d).name for d in dataset_dirs)
    return [
        {'well': f'W{w + 1}', 'dataset': names, 'timelapse': sorted(frames, key=lambda f: f['time'])}
        for w, frames in sorted(wells.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_dirs', nargs='+')
    parser.add_argument('--slices', type=int, default=11)
    parser.add_argument('--url-prefix', default='dataset', help='dir of the datasets under LOCAL_FILES_DOCUMENT_ROOT')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    try:
        tasks = build_tasks(args.dataset_dirs, args.slices, args.url_prefix)
    except ValueError as e:
        sys.exit(f'error: {e}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(tasks, ensure_ascii=False))
    frames = len(tasks[0]['timelapse']) if tasks else 0
    print(f'{len(tasks)} wells x {frames} frames x {args.slices} slices -> {args.output}')


if __name__ == '__main__':
    main()
