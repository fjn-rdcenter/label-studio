import ipaddress
import heapq
import io
import math
import re
import socket
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from defusedxml import ElementTree
from label_studio_converter.brush import mask2rle
from PIL import Image, UnidentifiedImageError
from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter

from .config import Settings


class InputError(ValueError):
    pass


class ImageFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class LabelSpec:
    from_name: str
    to_name: str
    image_expression: str
    uses_value_list: bool
    labels: tuple[str, ...]
    polygon_from_name: str | None = None
    polygon_labels: tuple[str, ...] = ()


def _tag_name(element: Any) -> str:
    return element.tag.rsplit("}", 1)[-1]


def parse_label_config(label_config: str | None) -> LabelSpec:
    if not label_config:
        return LabelSpec("brush", "image", "$image", False, ("Object",))
    try:
        root = ElementTree.fromstring(label_config)
    except ElementTree.ParseError as exc:
        raise InputError(f"Invalid label_config XML: {exc}") from exc

    images = {node.attrib.get("name"): node for node in root.iter() if _tag_name(node).lower() == "image"}
    brushes = [node for node in root.iter() if _tag_name(node).lower() == "brushlabels"]
    polygons = [node for node in root.iter() if _tag_name(node).lower() == "polygonlabels"]
    if not brushes and not polygons:
        raise InputError("label_config must contain a BrushLabels or PolygonLabels tag")

    output = brushes[0] if brushes else polygons[0]
    from_name = output.attrib.get("name", "brush")
    to_name = (output.attrib.get("toName") or output.attrib.get("toname") or "image").split(",")[0].strip()
    image = images.get(to_name)
    if image is None and images:
        to_name, image = next(iter(images.items()))
    value_list = image.attrib.get("valueList") or image.attrib.get("valuelist") if image is not None else None
    image_expression = value_list or (image.attrib.get("value", "$image") if image is not None else "$image")
    labels = tuple(
        node.attrib["value"]
        for node in output.iter()
        if _tag_name(node).lower() == "label" and node.attrib.get("value")
    )
    polygon = polygons[0] if polygons else None
    polygon_labels = tuple(
        node.attrib["value"]
        for node in polygon.iter()
        if _tag_name(node).lower() == "label" and node.attrib.get("value")
    ) if polygon is not None else ()
    return LabelSpec(
        from_name,
        to_name,
        image_expression or "$image",
        value_list is not None,
        labels or ("Object",),
        polygon.attrib.get("name") if polygon is not None else None,
        polygon_labels,
    )


def _is_allowed_host(hostname: str | None, allowed_hosts: frozenset[str]) -> bool:
    if not hostname:
        return False
    host = hostname.lower().rstrip(".")
    if host in allowed_hosts:
        return True
    return any(entry.startswith(".") and host.endswith(entry) for entry in allowed_hosts)


def _reject_unlisted_ip(hostname: str, allowed_hosts: frozenset[str]) -> None:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return
    if hostname.lower().rstrip(".") not in allowed_hosts:
        raise ImageFetchError("Image IP address is not explicitly allowlisted")


def _is_label_studio_media_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in ("/data/local-files", "/data/upload"))


def _resolve_image_url(source: str, settings: Settings) -> tuple[str, bool]:
    if source.startswith("/"):
        if not _is_label_studio_media_path(urlparse(source).path):
            raise ImageFetchError("Only Label Studio local-files and upload URLs are supported")
        if not settings.label_studio_url:
            raise ImageFetchError("LABEL_STUDIO_URL is required for relative image URLs")
        source = urljoin(settings.label_studio_url + "/", source.lstrip("/"))

    parsed = urlparse(source)
    if (
        settings.label_studio_url
        and _is_label_studio_media_path(parsed.path)
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        internal = urlparse(settings.label_studio_url)
        parsed = parsed._replace(scheme=internal.scheme, netloc=internal.netloc)
        source = parsed.geturl()
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ImageFetchError("Image URL must be an http(s) URL without embedded credentials")
    if not _is_allowed_host(parsed.hostname, settings.allowed_image_hosts):
        raise ImageFetchError(f"Image host is not allowlisted: {parsed.hostname or '<missing>'}")
    _reject_unlisted_ip(parsed.hostname or "", settings.allowed_image_hosts)
    return source, _is_label_studio_media_path(parsed.path)


def _pin_external_url(url: str, settings: Settings) -> tuple[str, str | None]:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    label_studio_host = urlparse(settings.label_studio_url).hostname
    if label_studio_host and hostname == label_studio_host:
        return url, None

    try:
        addresses = [
            item[4][0]
            for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        ]
    except socket.gaierror as exc:
        raise ImageFetchError(f"Unable to resolve image host: {hostname}") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ImageFetchError("External image host resolves to a non-public IP address")

    address = next((item for item in addresses if ipaddress.ip_address(item).version == 4), addresses[0])
    pinned_host = f"[{address}]" if ipaddress.ip_address(address).version == 6 else address
    if parsed.port:
        pinned_host = f"{pinned_host}:{parsed.port}"
    # HostHeaderSSLAdapter uses this value for TLS hostname verification, so an
    # explicit HTTPS port must not be included in the certificate hostname.
    host_header = hostname if parsed.scheme == "https" or not parsed.port else f"{hostname}:{parsed.port}"
    return parsed._replace(netloc=pinned_host).geturl(), host_header


def fetch_image(source: str, settings: Settings) -> np.ndarray:
    url, is_label_studio_media = _resolve_image_url(source, settings)

    session = requests.Session()
    session.trust_env = False
    session.mount("https://", HostHeaderSSLAdapter())
    try:
        for _ in range(4):
            headers = {"Accept": "*/*"}
            ls_host = urlparse(settings.label_studio_url).hostname
            current_host = urlparse(url).hostname
            if is_label_studio_media and settings.label_studio_api_key and current_host == ls_host:
                headers["Authorization"] = f"Token {settings.label_studio_api_key}"
            request_url, host_header = _pin_external_url(url, settings)
            if host_header:
                headers["Host"] = host_header
            response = session.get(
                request_url,
                headers=headers,
                timeout=settings.request_timeout,
                stream=True,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                target = response.headers.get("Location")
                response.close()
                if not target:
                    raise ImageFetchError("Image redirect has no Location header")
                url, is_label_studio_media = _resolve_image_url(urljoin(url, target), settings)
                continue
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > settings.max_image_bytes:
                raise ImageFetchError("Image exceeds MAX_IMAGE_BYTES")
            chunks = []
            size = 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > settings.max_image_bytes:
                    raise ImageFetchError("Image exceeds MAX_IMAGE_BYTES")
                chunks.append(chunk)
            response.close()
            break
        else:
            raise ImageFetchError("Too many image redirects")
    except requests.RequestException as exc:
        raise ImageFetchError(f"Unable to fetch image: {exc}") from exc
    finally:
        session.close()

    content = b"".join(chunks)
    try:
        with Image.open(io.BytesIO(content)) as image_header:
            width, height = image_header.size
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ImageFetchError("Downloaded content is not a safe supported image") from exc
    if height * width > settings.max_image_pixels:
        raise ImageFetchError("Image header exceeds MAX_IMAGE_PIXELS")

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageFetchError("Downloaded content is not a supported image")
    height, width = image.shape[:2]
    if height * width > settings.max_image_pixels:
        raise ImageFetchError("Decoded image exceeds MAX_IMAGE_PIXELS")
    return image


def _context_results(context: Any) -> list[dict[str, Any]]:
    if isinstance(context, list):
        return [item for item in context if isinstance(item, dict)]
    if not isinstance(context, dict):
        return []
    results = context.get("result", context.get("results", []))
    return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


def _template_index(template: str, actual: str) -> int | None:
    if "{{idx}}" not in template:
        return None
    parts = template.split("{{idx}}")
    pattern = "^" + r"(\d+)".join(re.escape(part) for part in parts) + "$"
    match = re.fullmatch(pattern, actual)
    if not match or len(set(match.groups())) != 1:
        return None
    return int(match.group(1))


def _parse_item_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        item_index = int(value)
    except (TypeError, ValueError) as exc:
        raise InputError("item_index must be a non-negative integer") from exc
    if isinstance(value, bool) or item_index < 0:
        raise InputError("item_index must be a non-negative integer")
    return item_index


def _resolve_indices(spec: LabelSpec, context: Any) -> tuple[int | None, int | None]:
    dynamic_index = None
    item_index = _parse_item_index(context.get("item_index")) if isinstance(context, dict) else None
    for result in _context_results(context):
        result_to_name = result.get("to_name")
        matched_index = None
        if isinstance(result_to_name, str):
            matched_index = _template_index(spec.to_name, result_to_name)
            if "{{idx}}" in spec.to_name and matched_index is None:
                continue
            if "{{idx}}" not in spec.to_name and result_to_name != spec.to_name:
                continue
        elif "{{idx}}" in spec.to_name:
            continue

        if matched_index is not None:
            if dynamic_index is not None and dynamic_index != matched_index:
                continue
            dynamic_index = matched_index
        result_item_index = _parse_item_index(result.get("item_index", (result.get("value") or {}).get("item_index")))
        if result_item_index is not None:
            item_index = result_item_index
            break

    if "{{idx}}" in spec.to_name and dynamic_index is None:
        raise InputError(f"Unable to infer template index from prompt to_name matching '{spec.to_name}'")
    return dynamic_index, item_index


def _render_template(value: str, dynamic_index: int | None) -> str:
    if "{{idx}}" not in value:
        return value
    if dynamic_index is None:
        raise InputError(f"Unable to resolve dynamic expression '{value}'")
    return value.replace("{{idx}}", str(dynamic_index))


def _resolve_data_expression(data: dict[str, Any], expression: str, dynamic_index: int | None) -> Any:
    expression = _render_template(expression, dynamic_index)
    if not expression.startswith("$"):
        return expression

    path = expression[1:]
    root = re.match(r"[A-Za-z_][A-Za-z0-9_]*", path)
    if not root:
        raise InputError(f"Unsupported image expression '{expression}'")
    value: Any = data
    tokens: list[str | int] = [root.group(0)]
    position = root.end()
    token_pattern = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
    while position < len(path):
        match = token_pattern.match(path, position)
        if not match:
            raise InputError(f"Unsupported image expression '{expression}'")
        tokens.append(match.group(1) if match.group(1) is not None else int(match.group(2)))
        position = match.end()

    for token in tokens:
        try:
            if isinstance(token, int):
                if not isinstance(value, (list, tuple)):
                    raise TypeError
                value = value[token]
            else:
                if not isinstance(value, dict):
                    raise TypeError
                value = value[token]
        except (KeyError, IndexError, TypeError) as exc:
            raise InputError(f"Image expression '{expression}' does not resolve against task data") from exc
    return value


def _resolve_image_source(data: dict[str, Any], spec: LabelSpec, dynamic_index: int | None, item_index: int | None) -> tuple[str, int | None]:
    value = _resolve_data_expression(data, spec.image_expression, dynamic_index)
    if spec.uses_value_list:
        if not isinstance(value, (list, tuple)):
            raise InputError(f"Image valueList expression '{spec.image_expression}' must resolve to a list")
        item_index = 0 if item_index is None else item_index
        try:
            value = value[item_index]
        except IndexError as exc:
            raise InputError(f"item_index {item_index} is outside image valueList") from exc
    if not isinstance(value, str):
        raise InputError(f"Image expression '{spec.image_expression}' must resolve to a URL string")
    return value, item_index


def _percent_point(value: dict[str, Any], width: int, height: int) -> tuple[int, int] | None:
    try:
        x = int(round(float(value["x"]) * width / 100.0))
        y = int(round(float(value["y"]) * height / 100.0))
    except (KeyError, TypeError, ValueError):
        return None
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def parse_prompts(
    context: Any,
    width: int,
    height: int,
    to_name: str | None = None,
    item_index: int | None = None,
) -> tuple[tuple[int, int, int, int], list, list, list[str]]:
    rectangle = (0, 0, width, height)
    positive_points: list[tuple[int, int]] = []
    background_points: list[tuple[int, int]] = []
    prompt_labels: list[str] = []

    for result in _context_results(context):
        if to_name and result.get("to_name") not in {None, to_name}:
            continue
        result_item_index = _parse_item_index(result.get("item_index", (result.get("value") or {}).get("item_index")))
        if item_index is not None and result_item_index not in {None, item_index}:
            continue
        value = result.get("value") or {}
        result_type = str(result.get("type", "")).lower()
        labels = value.get("rectanglelabels") or value.get("keypointlabels") or value.get("labels") or []
        prompt_labels.extend(str(label) for label in labels)
        if result_type in {"rectangle", "rectanglelabels"}:
            try:
                x1 = int(np.floor(float(value["x"]) * width / 100.0))
                y1 = int(np.floor(float(value["y"]) * height / 100.0))
                x2 = int(np.ceil((float(value["x"]) + float(value["width"])) * width / 100.0))
                y2 = int(np.ceil((float(value["y"]) + float(value["height"])) * height / 100.0))
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                rectangle = (x1, y1, x2, y2)
        elif result_type in {"keypoint", "keypointlabels"}:
            point = _percent_point(value, width, height)
            if point is None:
                continue
            label_text = " ".join(str(label).lower() for label in labels)
            is_positive = result.get("is_positive", value.get("is_positive", value.get("positive", True)))
            if is_positive is False or any(token in label_text for token in ("background", "negative", "exclude")):
                background_points.append(point)
            else:
                positive_points.append(point)
    return rectangle, positive_points, background_points, prompt_labels


def parse_polygon_prompt(
    context: Any,
    width: int,
    height: int,
    to_name: str | None = None,
    item_index: int | None = None,
) -> tuple[list[tuple[int, int]], list[str]]:
    for result in reversed(_context_results(context)):
        if str(result.get("type", "")).lower() not in {"polygon", "polygonlabels"}:
            continue
        if to_name and result.get("to_name") not in {None, to_name}:
            continue
        result_item_index = _parse_item_index(result.get("item_index", (result.get("value") or {}).get("item_index")))
        if item_index is not None and result_item_index not in {None, item_index}:
            continue

        value = result.get("value") or {}
        points = []
        for point in value.get("points") or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = int(round(float(point[0]) * width / 100.0))
                y = int(round(float(point[1]) * height / 100.0))
            except (TypeError, ValueError):
                continue
            points.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))
        labels = [str(label) for label in value.get("polygonlabels") or value.get("labels") or []]
        if len(points) >= 3:
            return points, labels
    return [], []


def _least_cost_path(cost: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = cost.shape
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    margin = min(max(16, int(distance * 0.35)), 96)
    x1 = max(min(start[0], end[0]) - margin, 0)
    y1 = max(min(start[1], end[1]) - margin, 0)
    x2 = min(max(start[0], end[0]) + margin + 1, width)
    y2 = min(max(start[1], end[1]) + margin + 1, height)
    roi = cost[y1:y2, x1:x2]

    # Bound pathological prompts while retaining the straight segment as a safe fallback.
    if roi.size > 350_000:
        line = np.linspace(start, end, max(int(distance), 2), dtype=np.int32)
        return [(int(point[0]), int(point[1])) for point in line]

    sx, sy = start[0] - x1, start[1] - y1
    ex, ey = end[0] - x1, end[1] - y1
    distances = np.full(roi.shape, np.inf, dtype=np.float64)
    parents = np.full((roi.shape[0], roi.shape[1], 2), -1, dtype=np.int32)
    distances[sy, sx] = 0
    queue = [(0.0, sx, sy)]
    neighbours = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

    while queue:
        current_distance, x, y = heapq.heappop(queue)
        if current_distance != distances[y, x]:
            continue
        if x == ex and y == ey:
            break
        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= roi.shape[1] or ny >= roi.shape[0]:
                continue
            step = 1.41421356 if dx and dy else 1.0
            next_distance = current_distance + step * float((roi[y, x] + roi[ny, nx]) * 0.5)
            if next_distance >= distances[ny, nx]:
                continue
            distances[ny, nx] = next_distance
            parents[ny, nx] = (x, y)
            heapq.heappush(queue, (next_distance, nx, ny))

    path = []
    x, y = ex, ey
    while x >= 0 and y >= 0:
        path.append((x + x1, y + y1))
        if x == sx and y == sy:
            break
        x, y = parents[y, x]
    if not path or path[-1] != start:
        return [start, end]
    path.reverse()
    return path


def trace_polygon(image: np.ndarray, anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)
    cost = 1.0 + 24.0 * (1.0 - gradient)

    path = []
    for index, start in enumerate(anchors):
        end = anchors[(index + 1) % len(anchors)]
        segment = _least_cost_path(cost, start, end)
        path.extend(segment[:-1])

    contour = np.asarray(path, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, max(0.75, perimeter * 0.001), True).reshape(-1, 2)
    points = [(int(x), int(y)) for x, y in simplified]
    return points if len(points) >= 3 else anchors


def _candidate_score(mask: np.ndarray, positives: list, backgrounds: list) -> float:
    area_ratio = float(np.count_nonzero(mask)) / mask.size
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    border_ratio = float(np.count_nonzero(border)) / border.size
    score = 2.0 * (1.0 - border_ratio) - abs(area_ratio - 0.35)
    score += 5.0 * sum(mask[y, x] > 0 for x, y in positives)
    score += 4.0 * sum(mask[y, x] == 0 for x, y in backgrounds)
    if area_ratio < 0.005 or area_ratio > 0.98:
        score -= 20.0
    return score


def _keep_prompted_components(mask: np.ndarray, positives: list, backgrounds: list) -> np.ndarray:
    count, components = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if count <= 1:
        return mask
    positive_ids = {int(components[y, x]) for x, y in positives if components[y, x]}
    negative_ids = {int(components[y, x]) for x, y in backgrounds if components[y, x]}
    keep = positive_ids or (set(range(1, count)) - negative_ids)
    return np.where(np.isin(components, list(keep)), 255, 0).astype(np.uint8)


def _threshold_mask(image: np.ndarray, rectangle: tuple[int, int, int, int], positives: list, backgrounds: list) -> np.ndarray:
    x1, y1, x2, y2 = rectangle
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(3, min(9, (min(gray.shape) // 40) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    candidates = []
    local_positive = [(x - x1, y - y1) for x, y in positives if x1 <= x < x2 and y1 <= y < y2]
    local_background = [(x - x1, y - y1) for x, y in backgrounds if x1 <= x < x2 and y1 <= y < y2]
    for candidate in (bright, cv2.bitwise_not(bright)):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidates.append((_candidate_score(candidate, local_positive, local_background), candidate))
    local_mask = max(candidates, key=lambda item: item[0])[1]
    local_mask = _keep_prompted_components(local_mask, local_positive, local_background)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = local_mask
    return mask


def _grabcut_mask(image: np.ndarray, rectangle: tuple[int, int, int, int], initial: np.ndarray, positives: list, backgrounds: list) -> np.ndarray:
    x1, y1, x2, y2 = rectangle
    gc_mask = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    gc_mask[initial > 0] = cv2.GC_PR_FGD
    radius = max(2, min(image.shape[:2]) // 150)
    for point in positives:
        cv2.circle(gc_mask, point, radius, cv2.GC_FGD, -1)
    for point in backgrounds:
        cv2.circle(gc_mask, point, radius, cv2.GC_BGD, -1)
    try:
        cv2.grabCut(image, gc_mask, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return initial
    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def segment(image: np.ndarray, rectangle: tuple[int, int, int, int], positives: list, backgrounds: list) -> np.ndarray:
    mask = _threshold_mask(image, rectangle, positives, backgrounds)
    x1, y1, x2, y2 = rectangle
    roi = mask[y1:y2, x1:x2]
    occupancy = float(np.count_nonzero(roi)) / roi.size
    misses_positive = any(mask[y, x] == 0 for x, y in positives)
    if occupancy < 0.01 or occupancy > 0.90 or misses_positive:
        mask = _grabcut_mask(image, rectangle, mask, positives, backgrounds)
    return mask


def make_ring(mask: np.ndarray, width: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = np.zeros_like(mask)
    if contours:
        cv2.drawContours(outer, contours, -1, 255, cv2.FILLED)
    kernel_size = max(3, width * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv2.erode(outer, kernel, iterations=1)
    return cv2.subtract(outer, inner)


def _selected_label(labels: tuple[str, ...], prompt_labels: list[str], context: Any) -> str:
    requested = context.get("label") if isinstance(context, dict) else None
    for candidate in [requested, *prompt_labels]:
        if not isinstance(candidate, str):
            continue
        for configured_label in labels:
            if candidate.casefold() == configured_label.casefold():
                return configured_label
    return labels[0]


def predict_task(task: dict[str, Any], spec: LabelSpec, context: Any, settings: Settings, model_version: str) -> dict:
    data = task.get("data")
    if not isinstance(data, dict):
        raise InputError("Task data must be an object")
    dynamic_index, item_index = _resolve_indices(spec, context)
    image_source, item_index = _resolve_image_source(data, spec, dynamic_index, item_index)
    from_name = _render_template(spec.from_name, dynamic_index)
    to_name = _render_template(spec.to_name, dynamic_index)
    image = fetch_image(image_source, settings)
    height, width = image.shape[:2]
    rectangle, positives, backgrounds, prompt_labels = parse_prompts(context, width, height, to_name, item_index)
    polygon_anchors, polygon_prompt_labels = parse_polygon_prompt(context, width, height, to_name, item_index)

    if polygon_anchors and spec.polygon_from_name:
        polygon_labels = spec.polygon_labels or spec.labels
        selected_label = _selected_label(polygon_labels, polygon_prompt_labels, context)
        points = trace_polygon(image, polygon_anchors)
        result = [{
            "id": uuid.uuid4().hex[:10],
            "from_name": _render_template(spec.polygon_from_name, dynamic_index),
            "to_name": to_name,
            "type": "polygonlabels",
            "value": {
                "points": [[x * 100.0 / width, y * 100.0 / height] for x, y in points],
                "polygonlabels": [selected_label],
                "closed": True,
            },
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
        }]
        if item_index is not None:
            result[0]["item_index"] = item_index
        return {"result": result, "score": 1.0, "model_version": model_version}

    selected_label = _selected_label(spec.labels, prompt_labels, context)
    mask = segment(image, rectangle, positives, backgrounds)

    ring = settings.default_ring or selected_label.casefold() in {"membrane", "ring"}
    ring_width = settings.ring_width
    if isinstance(context, dict):
        if "ring" in context:
            ring = bool(context["ring"])
        elif "membrane" in context:
            ring = bool(context["membrane"])
        try:
            ring_width = min(max(int(context.get("ring_width", ring_width)), 1), 100)
        except (TypeError, ValueError):
            raise InputError("ring_width must be an integer")
    if ring:
        mask = make_ring(mask, ring_width)

    result = []
    if np.any(mask):
        result.append(
            {
                "id": uuid.uuid4().hex[:10],
                "from_name": from_name,
                "to_name": to_name,
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": mask2rle(mask.astype(np.uint8)),
                    "brushlabels": [selected_label],
                },
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
            }
        )
        if item_index is not None:
            result[-1]["item_index"] = item_index
    return {"result": result, "score": 1.0 if result else 0.0, "model_version": model_version}
