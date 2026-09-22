import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from . import __version__
from .config import Settings
from .service import ImageFetchError, InputError, parse_label_config, predict_task

MODEL_VERSION = f"opencv-segmentation-{__version__}"


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes

    @application.get("/health")
    def health():
        return jsonify(status="UP", model_class="OpenCVInteractiveSegmentation", model_version=MODEL_VERSION)

    @application.post("/setup")
    def setup():
        payload = request.get_json(silent=True) or {}
        parse_label_config(payload.get("schema") or payload.get("label_config"))
        return jsonify(model_version=MODEL_VERSION)

    @application.post("/validate")
    def validate():
        payload = request.get_json(silent=True) or {}
        config = payload.get("label_config") or payload.get("schema") or payload.get("config")
        if isinstance(config, str):
            parse_label_config(config)
        return jsonify(status="ok")

    @application.post("/predict")
    def predict():
        payload = request.get_json(silent=False)
        if not isinstance(payload, dict):
            raise InputError("Request body must be a JSON object")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise InputError("tasks must be a non-empty list")
        if len(tasks) > settings.max_tasks:
            raise InputError(f"At most {settings.max_tasks} tasks are allowed per request")
        spec = parse_label_config(payload.get("label_config"))
        params = payload.get("params") or {}
        context = params.get("context", {}) if isinstance(params, dict) else {}
        predictions = [predict_task(task, spec, context, settings, MODEL_VERSION) for task in tasks]
        return jsonify(results=predictions, model_version=MODEL_VERSION)

    @application.errorhandler(InputError)
    @application.errorhandler(BadRequest)
    def bad_input(error):
        return jsonify(error=str(error)), 400

    @application.errorhandler(ImageFetchError)
    def image_error(error):
        return jsonify(error=str(error)), 422

    @application.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="Request exceeds MAX_REQUEST_BYTES"), 413

    return application


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
