import cv2
import numpy as np
from crop_timelapse_wells import crop_dataset, embryo_centre


def well_image(centre=(520, 600), seed=0):
    """964x1288 gray well: dark microwell wall and a textured embryo at `centre` (x, y)."""
    rng = np.random.default_rng(seed)
    img = np.full((1288, 964), 128, np.uint8)
    cv2.circle(img, (480, 640), 370, 30, 25)
    blob = np.zeros_like(img, bool)
    cv2.circle(blob.view(np.uint8), centre, 150, 1, -1)
    img[blob] = rng.integers(60, 200, blob.sum()).astype(np.uint8)
    return img


def test_embryo_centre_finds_the_textured_blob_not_the_wall():
    x, y = embryo_centre(well_image())
    assert abs(x - 520) <= 20 and abs(y - 600) <= 20


def test_embryo_centre_ignores_a_haloed_wall_that_is_not_the_darkest_part():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 60, (1288, 964)).astype(np.uint8)  # dark noisy floor outside the microwell
    cv2.circle(img, (480, 640), 330, 128, -1)  # smooth microwell interior
    for r, v in ((345, 250), (360, 70), (375, 250), (390, 70)):  # bright halo and dark wall rings
        cv2.circle(img, (480, 640), r, v, 8)
    blob = np.zeros_like(img)
    cv2.circle(blob, (520, 600), 120, 1, -1)
    img[blob > 0] = rng.integers(90, 170, int(blob.sum())).astype(np.uint8)
    x, y = embryo_centre(img)
    assert abs(x - 520) <= 25 and abs(y - 600) <= 25


def test_crop_dataset_keeps_names_and_one_box_per_well(tmp_path):
    src, dst = tmp_path / 'day4', tmp_path / 'day4_crop600'
    for f, shift in (('4131756', 0), ('4133256', 6)):
        (src / f).mkdir(parents=True)
        for i in range(1, 12):  # one well, 11 slices
            cv2.imwrite(str(src / f / f'{i:05d}.jpg'), well_image((520 + shift, 600), seed=i))
    boxes = crop_dataset(src, dst, slices=11, size=600)
    assert len(boxes) == 1 and boxes[0][2] - boxes[0][0] == 600 and boxes[0][3] - boxes[0][1] == 600
    out = sorted(p.relative_to(dst).as_posix() for p in dst.rglob('*.jpg'))
    assert len(out) == 22 and out[0] == '4131756/00001.jpg'
    assert cv2.imread(str(dst / '4133256' / '00011.jpg'), cv2.IMREAD_GRAYSCALE).shape == (600, 600)
