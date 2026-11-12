import pytest
from timelapse_wells import build_tasks


def make(root, folder, n):
    d = root / 'day4' / folder
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        (d / f'{i:05d}.jpg').touch()
    return root / 'day4'


def test_wells_split_every_slices_images(tmp_path):
    make(tmp_path, '4161757', 22)
    ds = make(tmp_path, '4131756', 22)
    tasks = build_tasks([ds], slices=11)
    assert [t['well'] for t in tasks] == ['W1', 'W2']
    w2 = tasks[1]['timelapse']
    assert [f['time'] for f in w2] == ['4:13:17:56', '4:16:17:57']
    assert w2[0]['image_urls'][0] == '/data/local-files/?d=dataset/day4/4131756/00012.jpg'
    assert len(w2[0]['image_urls']) == 11


def test_rejects_partial_well(tmp_path):
    with pytest.raises(ValueError, match='not a multiple'):
        build_tasks([make(tmp_path, '4131756', 23)])
