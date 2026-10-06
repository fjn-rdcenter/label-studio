import cv2
import numpy as np

from opencv_segmentation.service import _simplify_contour


def test_simplify_contour_reduces_vertices_without_changing_shape_significantly():
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = 100 + np.where(np.arange(360) % 2 == 0, 1, -1)
    contour = np.column_stack((150 + radii * np.cos(angles), 150 + radii * np.sin(angles)))
    contour = np.rint(contour).astype(np.int32).reshape(-1, 1, 2)

    simplified = _simplify_contour(contour)
    original_area = abs(cv2.contourArea(contour))
    simplified_area = abs(cv2.contourArea(simplified))

    assert 8 <= len(simplified) <= 40
    assert abs(simplified_area - original_area) / original_area < 0.03
