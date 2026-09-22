# OpenCV Interactive Segmentation Backend

A small, standalone Label Studio ML backend for interactive image segmentation. Rectangle and keypoint prompts produce `BrushLabels` RLE masks. Smart polygon prompts use OpenCV image gradients to trace a `PolygonLabels` contour between user-supplied anchor points.

The repository includes two ready-to-use labeling configurations:

- `label-config.xml` for tasks with a single `$image`.
- `label-config-timelapse.xml` for the Fujinet `$timelapse[{{idx}}].image_urls` slice format. It preserves the existing `kp-{{idx}}`/`PN` and `pl-{{idx}}`/`PN Borderline` controls so existing annotations remain compatible.

Both configurations are available in the project creation UI as **OpenCV Semi-Auto Segmentation** and **OpenCV Timelapse Segmentation**. They include manual rectangle, polygon, ellipse, keypoint, and brush tools. A separate **OpenCV Detect Polygon** toolbar button produces an editable edge-traced polygon without mixing it into Label Studio's Auto-Detect menu.

## Protocol

- `GET /health`
- `POST /setup` with Label Studio's `schema`; returns `model_version`
- `POST /validate` with `label_config` or `schema`
- `POST /predict` with `tasks`, `label_config`, and optional `params.context`

The label config is parsed at request time. The first `BrushLabels` control, its target `Image`, the complete `value` or `valueList` expression, and available labels are discovered automatically. Defaults (`brush`, `image`, `$image`, `Object`) are used only when no config is supplied.

Segmentation first evaluates both polarities of an Otsu-thresholded, morphologically cleaned ROI. Positive and background keypoints influence candidate selection and connected components. Weak masks fall back to GrabCut. Labels named `Membrane` or `Ring` (case-insensitive) automatically produce a membrane ring built from filled external contours minus an eroded interior. An explicitly supplied `params.context.ring` overrides this behavior in either direction; `ring_width` controls thickness in pixels. Other labels such as `Object` produce the full mask unless ring mode is explicitly enabled or `DEFAULT_RING` is set.

## Multi-image and timelapse data

Standard Label Studio `valueList` is supported. The backend reads `item_index` from the matching prompt result (or from context), selects that image, and includes the same `item_index` in its brush result:

```xml
<Image name="scan" valueList="$images"/>
<BrushLabels name="brush" toName="scan"><Label value="Object"/></BrushLabels>
```

The custom timelapse template can use a dynamic index in tag names and nested expressions:

```xml
<Image name="timelapse_{{idx}}" valueList="$timelapse[{{idx}}].image_urls"/>
<BrushLabels name="membrane_{{idx}}" toName="timelapse_{{idx}}">
  <Label value="Object"/><Label value="Membrane"/>
</BrushLabels>
```

For a prompt with `"to_name": "timelapse_2"` and `"item_index": 3`, the backend resolves `$timelapse[2].image_urls[3]` and returns `from_name="membrane_2"`, `to_name="timelapse_2"`, and `item_index=3`. Expressions are parsed as data paths; they are never evaluated as Python or JavaScript.

Prompt coordinates use Label Studio percentages:

```json
{
  "tasks": [{"id": 1, "data": {"image": "https://images.example.test/sample.png"}}],
  "label_config": "<View><Image name=\"image\" value=\"$image\"/><BrushLabels name=\"brush\" toName=\"image\"><Label value=\"Object\"/></BrushLabels></View>",
  "params": {
    "context": {
      "ring": true,
      "ring_width": 4,
      "result": [
        {"type": "rectanglelabels", "value": {"x": 20, "y": 20, "width": 60, "height": 60}},
        {"type": "keypointlabels", "value": {"x": 50, "y": 50, "keypointlabels": ["Object"]}},
        {"type": "keypointlabels", "value": {"x": 22, "y": 22, "keypointlabels": ["Background"]}}
      ]
    }
  }
}
```

## Run locally

Python 3.10 is the supported runtime.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
ALLOWED_IMAGE_HOSTS=localhost,images.example.test LABEL_STUDIO_URL=http://localhost:8080 .venv/bin/gunicorn --bind 0.0.0.0:9090 opencv_segmentation.app:app
pytest -q
```

On Windows, activate the virtual environment and run Flask's development server for local debugging:

```powershell
$env:ALLOWED_IMAGE_HOSTS = "localhost,images.example.test"
$env:LABEL_STUDIO_URL = "http://localhost:8080"
$env:FLASK_APP = "opencv_segmentation.app:app"
flask run --port 9090
pytest -q
```

## Docker

From the repository root, start Label Studio and this backend together:

```bash
docker compose -f docker-compose.yml -f docker-compose.opencv.yml up --build
```

For local-file images, put a Label Studio API token in the git-ignored `.data/label_studio_data/opencv.env` file:

```dotenv
LABEL_STUDIO_API_KEY=your-token
```

Then run the same Compose command. The optional env file is loaded only by the OpenCV service.

Then open the project's **Settings > Machine Learning**, add `http://opencv-segmentation:9090`, and enable **Use for interactive preannotations**. The Compose service is intentionally available only inside the Docker network because the ML protocol has no authentication. Publish or bind port `9090` to loopback separately only when Label Studio itself runs outside Docker.

To use the tools:

1. Select a polygon label.
2. Select the separate **OpenCV Detect Polygon** tool or press `I`.
3. Place coarse anchors around the object and close the polygon.
4. Adjust the generated edge-traced polygon before submitting.

The OpenCV service automatically treats the `Membrane` and `Ring` labels as ring output. `DEFAULT_RING_WIDTH` controls its default thickness in source-image pixels.

To run only the backend container:

```bash
docker build -t opencv-segmentation .
docker run --rm -p 9090:9090 \
  -e LABEL_STUDIO_URL=http://host.docker.internal:8080 \
  -e LABEL_STUDIO_API_KEY=your-token \
  -e ALLOWED_IMAGE_HOSTS=host.docker.internal,images.example.test \
  opencv-segmentation
```

`/data/local-files/?d=...` and `/data/upload/...` paths are resolved against `LABEL_STUDIO_URL`. The API key is sent as `Authorization: Token ...` only for Label Studio media requests to that same host. Redirect destinations are validated against the same host allowlist.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LABEL_STUDIO_URL` | empty | Label Studio base URL; its host is automatically allowlisted |
| `LABEL_STUDIO_API_KEY` | empty | Token for Label Studio local-files image requests |
| `ALLOWED_IMAGE_HOSTS` | empty | Comma-separated exact hosts; a leading dot permits subdomains |
| `IMAGE_FETCH_TIMEOUT` | `10` | HTTP connect/read timeout in seconds |
| `MAX_IMAGE_BYTES` | `20971520` | Maximum downloaded compressed image size |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded width times height |
| `MAX_TASKS_PER_REQUEST` | `16` | Batch limit |
| `MAX_REQUEST_BYTES` | `2097152` | JSON request limit |
| `DEFAULT_RING` | `false` | Enable ring output unless overridden by context |
| `DEFAULT_RING_WIDTH` | `3` | Default ring thickness in pixels |
| `WEB_CONCURRENCY` | `2` | Gunicorn workers in the container |
| `GUNICORN_THREADS` | `2` | Threads per worker |
| `GUNICORN_TIMEOUT` | `60` | Worker timeout in seconds |

No arbitrary local paths, URL credentials, non-HTTP schemes, or unlisted image hosts are accepted. Keep the allowlist narrow in production.
