"""Update task information and append external results after a mandatory preview."""

from copy import deepcopy
import hashlib
import json

from django.db import transaction
from projects.models import Project, ProjectMember
from rest_framework import serializers
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from tasks.models import Annotation, Task
from tasks.serializers import AnnotationSerializer, PredictionSerializer
from tasks.validation import TaskValidator


class ResultImportRequest(serializers.Serializer):
    tasks = serializers.ListField(
        child=serializers.DictField(), allow_empty=False, max_length=10000
    )
    match_field = serializers.CharField(default="id", max_length=200)
    commit = serializers.BooleanField(default=False)
    preview_token = serializers.CharField(required=False)

    def validate_match_field(self, value):
        if value != "id" and (
            not value.startswith("data.") or not value[5:] or "__" in value
        ):
            raise serializers.ValidationError(
                "Use id or a task data field, for example data.image."
            )
        return value


class AnnotationUpdateAPI(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request, pk):
        project = Project.objects.filter(
            pk=pk, organization=request.user.active_organization
        ).first()
        if project is None or not project.has_role(
            request.user, ProjectMember.Role.MANAGER
        ):
            raise PermissionDenied("Only a project manager can update existing tasks.")
        # Serialize imports for this project and keep matching stable during each request.
        project = Project.objects.select_for_update().get(pk=project.pk)
        serializer = ResultImportRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        options = serializer.validated_data
        match_field = options["match_field"]
        plan, rows = [], []
        task_states = []
        seen_tasks = set()
        for index, item in enumerate(options["tasks"]):
            row = {
                "row": index + 1,
                "task_id": None,
                "annotations": 0,
                "predictions": 0,
                "changes": {},
            }
            try:
                if match_field == "id":
                    value = item.get("id")
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValidationError("Task id must be an integer.")
                    matches = list(
                        Task.objects.select_for_update().filter(
                            project=project, pk=value
                        )
                    )
                else:
                    data = item.get("data")
                    value = (
                        data.get(match_field[5:]) if isinstance(data, dict) else None
                    )
                    if (
                        value is None
                        or value == ""
                        or not isinstance(value, (str, int, float, bool))
                    ):
                        raise ValidationError(
                            "Matching data field must contain a non-empty scalar value."
                        )
                    matches = list(
                        Task.objects.select_for_update().filter(
                            project=project, **{f"data__{match_field[5:]}": value}
                        )[:2]
                    )
                if not matches:
                    row.update(
                        status="unmatched",
                        message="No matching task in this project; this row will be skipped.",
                    )
                    rows.append(row)
                    continue
                if len(matches) > 1:
                    raise ValidationError(
                        "Multiple tasks match this value. Choose a unique data field."
                    )
                task = matches[0]
                row["task_id"] = task.id
                if task.id in seen_tasks:
                    raise ValidationError(
                        "This file matches the same task more than once."
                    )
                seen_tasks.add(task.id)
                task_states.append(
                    [task.id, task.data, task.meta, task.updated_at.isoformat()]
                )
                updates = {}
                for field in ("data", "meta"):
                    if field not in item:
                        continue
                    incoming = item[field]
                    current = getattr(task, field)
                    if field == "meta" and current is None:
                        current = {}
                    if not isinstance(incoming, dict) or not isinstance(current, dict):
                        raise ValidationError(
                            f"{field} must be a JSON object to update individual keys."
                        )
                    merged = {**current, **incoming}
                    if field == "data":
                        # Validate the merged data, including required fields absent from the patch.
                        TaskValidator.check_data(project, deepcopy(merged))
                    changes = {
                        key: {
                            "before_exists": key in current,
                            "before": current.get(key),
                            "after": value,
                        }
                        for key, value in incoming.items()
                        if key not in current or current[key] != value
                    }
                    if changes:
                        row["changes"][field] = changes
                        updates[field] = merged
                results = []
                for key, result_serializer, allowed in (
                    ("annotations", AnnotationSerializer, {"result", "lead_time"}),
                    (
                        "predictions",
                        PredictionSerializer,
                        {"result", "score", "model_version"},
                    ),
                ):
                    entries = item.get(key, [])
                    if not isinstance(entries, list):
                        raise ValidationError(f"{key} must be a list.")
                    for entry in entries:
                        if not isinstance(entry, dict) or not isinstance(
                            entry.get("result"), list
                        ):
                            raise ValidationError(
                                f"Each {key} entry must contain a result list."
                            )
                        if any(
                            not isinstance(region, dict) for region in entry["result"]
                        ):
                            raise ValidationError(
                                "Each result region must be an object."
                            )
                        # External IDs, authors, workflow levels and task references are not trusted.
                        payload = {k: v for k, v in entry.items() if k in allowed}
                        payload.update(task=task.id, project=project.id)
                        result = result_serializer(
                            data=payload, context={"request": request}
                        )
                        result.is_valid(raise_exception=True)
                        results.append((key, result))
                    row[key] = len(entries)
                if not results and not any(field in item for field in ("data", "meta")):
                    raise ValidationError(
                        "Task must contain data, meta, annotations, or predictions to update."
                    )
                plan.append((task, updates, results))
                row["status"] = "matched"
            except ValidationError as exc:
                row.update(status="error", message=str(exc.detail))
            rows.append(row)

        summary = {
            "matched": sum(row["status"] == "matched" for row in rows),
            "unmatched": sum(row["status"] == "unmatched" for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "tasks_updated": sum(
                bool(row["changes"]) for row in rows if row["status"] == "matched"
            ),
            "annotations": sum(
                row["annotations"] for row in rows if row["status"] == "matched"
            ),
            "predictions": sum(
                row["predictions"] for row in rows if row["status"] == "matched"
            ),
        }
        token_data = [
            project.id,
            project.label_config,
            options["tasks"],
            match_field,
            rows,
            task_states,
        ]
        token = hashlib.sha256(
            json.dumps(token_data, sort_keys=True).encode()
        ).hexdigest()
        if options["commit"]:
            if options.get("preview_token") != token:
                raise ValidationError(
                    "Import or matching changed. Preview the file again before saving."
                )
            if summary["errors"] or not summary["matched"]:
                raise ValidationError(
                    "Resolve preview errors and match at least one task before saving."
                )
            for task, updates, results in plan:
                if updates:
                    for field, value in updates.items():
                        setattr(task, field, value)
                    task.updated_by = request.user
                    task.save(update_fields=[*updates, "updated_by", "updated_at"])
                for key, result in results:
                    if key == "annotations":
                        result.save(
                            completed_by=request.user,
                            updated_by=request.user,
                            quality_level=Annotation.QualityLevel.ANNOTATOR,
                            quality_updated_by=request.user,
                        )
                    else:
                        result.save()
        return Response(
            {
                "committed": options["commit"],
                "preview_token": token,
                "summary": summary,
                "rows": rows,
            }
        )
