import pytest
from projects.models import Project, ProjectMember
from rest_framework.test import APIClient
from tasks.models import Annotation, Prediction, Task
from users.models import User

pytestmark = pytest.mark.django_db


@pytest.fixture
def manager_client(business_client, configured_project):
    user = User.objects.create_user(email="import-manager@example.com", password="test")
    business_client.organization.add_user(user)
    configured_project.add_collaborator(user, role=ProjectMember.Role.MANAGER)
    client = APIClient()
    client.force_authenticate(user=user)
    client.user = user
    return client


def payload(task):
    result = [
        {
            "id": "external-region",
            "from_name": "text_class",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": ["class_A"]},
        }
    ]
    return {
        "tasks": [
            {
                "id": task.id,
                "annotations": [{"result": result}],
                "predictions": [
                    {"result": result, "score": 0.8, "model_version": "external"}
                ],
            }
        ]
    }


def test_preview_and_append_preserve_existing_results(
    business_client, configured_project, manager_client
):
    task = configured_project.tasks.first()
    old = Annotation.objects.create(
        task=task,
        project=configured_project,
        result=[],
        completed_by=business_client.user,
    )
    prediction = Prediction.objects.create(
        task=task, project=configured_project, result=[], model_version="old"
    )
    original_data = task.data.copy()
    url = f"/api/projects/{configured_project.id}/import/results"
    body = payload(task)
    # Foreign IDs and workflow flags in exports must not overwrite existing records.
    body["tasks"][0]["annotations"][0].update(
        id=old.id, quality_level=3, completed_by=999999
    )
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    preview = response.json()
    assert preview["summary"] == {
        "matched": 1,
        "unmatched": 0,
        "errors": 0,
        "tasks_updated": 0,
        "annotations": 1,
        "predictions": 1,
    }
    assert task.annotations.count() == 1
    assert task.predictions.count() == 1
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    assert response.json()["committed"] is True
    old.refresh_from_db()
    prediction.refresh_from_db()
    task.refresh_from_db()
    assert old.result == []
    assert prediction.model_version == "old"
    assert task.data == original_data
    assert task.total_annotations == 2
    assert task.total_predictions == 2
    new = task.annotations.exclude(pk=old.pk).get()
    assert new.quality_level == Annotation.QualityLevel.ANNOTATOR
    assert new.completed_by == manager_client.user
    assert new.result == body["tasks"][0]["annotations"][0]["result"]
    assert task.predictions.exclude(pk=prediction.pk).get().result == new.result


@pytest.mark.parametrize(
    "role", [ProjectMember.Role.ANNOTATOR, ProjectMember.Role.REVIEWER]
)
def test_import_requires_project_manager(business_client, configured_project, role):
    user = User.objects.create_user(email=f"import-{role}@example.com", password="test")
    business_client.organization.add_user(user)
    configured_project.add_collaborator(user, role=role)
    client = APIClient()
    client.force_authenticate(user=user)
    response = client.post(
        f"/api/projects/{configured_project.id}/import/results",
        payload(configured_project.tasks.first()),
        format="json",
    )
    assert response.status_code == 403


def test_data_matching_and_ambiguous_rows(
    business_client, configured_project, manager_client
):
    task = configured_project.tasks.first()
    url = f"/api/projects/{configured_project.id}/import/results"
    body = {
        "match_field": "data.text",
        "tasks": [{"id": -1, "data": task.data, "predictions": [{"result": []}]}],
    }
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    assert response.json()["rows"][0]["task_id"] == task.id
    Task.objects.create(project=configured_project, data=task.data)
    body.update(commit=True, preview_token=response.json()["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 400
    body["commit"] = False
    response = manager_client.post(url, body, format="json")
    assert response.json()["summary"]["errors"] == 1


def test_cross_project_tasks_are_skipped(
    business_client, configured_project, manager_client
):
    other = Project.objects.create(
        organization=business_client.organization, created_by=business_client.user
    )
    task = Task.objects.create(project=other, data={"text": "external"})
    response = manager_client.post(
        f"/api/projects/{configured_project.id}/import/results",
        payload(task),
        format="json",
    )
    assert response.status_code == 200, response.content
    assert response.json()["summary"]["unmatched"] == 1
    assert not task.annotations.exists()


def test_errors_block_entire_import_and_preview_is_required(
    business_client, configured_project, manager_client
):
    task, second = list(configured_project.tasks.all())[:2]
    url = f"/api/projects/{configured_project.id}/import/results"
    body = payload(task)
    body["commit"] = True
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 400
    body["commit"] = False
    body["tasks"].append({"id": second.id, "annotations": [{"result": "invalid"}]})
    preview = manager_client.post(url, body, format="json").json()
    assert preview["summary"]["errors"] == 1
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 400
    assert not task.annotations.exists()


def test_data_match_commit_skips_unmatched_tasks(configured_project, manager_client):
    task = configured_project.tasks.first()
    url = f"/api/projects/{configured_project.id}/import/results"
    body = payload(task)
    body["match_field"] = "data.text"
    body["tasks"][0].update(id=-1, data=task.data)
    body["tasks"].append(
        {"data": {"text": "does not exist"}, "annotations": [{"result": []}]}
    )
    preview = manager_client.post(url, body, format="json").json()
    assert preview["summary"]["matched"] == 1
    assert preview["summary"]["unmatched"] == 1
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    assert task.annotations.count() == 1
    assert task.predictions.count() == 1
    assert configured_project.tasks.count() == 2


def test_update_information_without_results_preserves_other_keys_and_results(
    configured_project, manager_client
):
    task = configured_project.tasks.first()
    task.data.update(keep="original", nested={"old": True})
    task.meta = {"keep": "metadata", "status": "old"}
    task.save()
    old = Annotation.objects.create(task=task, project=configured_project, result=[])
    prediction = Prediction.objects.create(
        task=task, project=configured_project, result=[]
    )
    url = f"/api/projects/{configured_project.id}/import/results"
    body = {
        "tasks": [
            {
                "id": task.id,
                "project": -1,
                "overlap": 999,
                "data": {
                    "text": "updated text",
                    "nested": {"new": True},
                    "new_key": [1, 2],
                },
                "meta": {"status": "new"},
            }
        ]
    }
    original_data = task.data.copy()
    original_overlap = task.overlap
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    preview = response.json()
    assert preview["summary"]["tasks_updated"] == 1
    assert preview["summary"]["annotations"] == preview["summary"]["predictions"] == 0
    changes = preview["rows"][0]["changes"]
    assert changes["data"]["text"]["before"] == original_data["text"]
    assert changes["data"]["text"]["after"] == "updated text"
    assert changes["data"]["new_key"]["before_exists"] is False
    assert changes["meta"]["status"] == {
        "before_exists": True,
        "before": "old",
        "after": "new",
    }
    task.refresh_from_db()
    assert task.data == original_data
    assert task.meta["status"] == "old"
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 200, response.content
    task.refresh_from_db()
    assert task.data == {**original_data, **body["tasks"][0]["data"]}
    assert task.meta == {"keep": "metadata", "status": "new"}
    assert task.overlap == original_overlap
    assert task.project_id == configured_project.id
    assert task.updated_by == manager_client.user
    assert task.annotations.get().pk == old.pk
    assert task.predictions.get().pk == prediction.pk


@pytest.mark.parametrize(
    "patch", [{"data": None}, {"meta": []}, {"data": {"text": {}}}]
)
def test_invalid_information_blocks_all_updates(
    configured_project, manager_client, patch
):
    first, second = list(configured_project.tasks.all())[:2]
    original_data = first.data.copy()
    body = {
        "tasks": [
            {"id": first.id, "data": {"text": "must not save"}},
            {"id": second.id, **patch},
        ]
    }
    url = f"/api/projects/{configured_project.id}/import/results"
    preview = manager_client.post(url, body, format="json").json()
    assert preview["summary"]["errors"] == 1
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 400, response.content
    first.refresh_from_db()
    assert first.data == original_data


def test_concurrent_information_change_requires_new_preview(
    configured_project, manager_client
):
    task = configured_project.tasks.first()
    url = f"/api/projects/{configured_project.id}/import/results"
    body = {"tasks": [{"id": task.id, "meta": {"note": "from file"}}]}
    preview = manager_client.post(url, body, format="json").json()
    task.meta = {"note": "changed since preview"}
    task.save()
    body.update(commit=True, preview_token=preview["preview_token"])
    response = manager_client.post(url, body, format="json")
    assert response.status_code == 400, response.content
    task.refresh_from_db()
    assert task.meta == {"note": "changed since preview"}
