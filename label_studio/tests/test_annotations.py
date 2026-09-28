"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import json

import pytest
import requests_mock
from django.apps import apps
from django.urls import reverse
from organizations.models import OrganizationMember
from projects.models import Project, ProjectMember
from tasks.models import Annotation, Task

from .utils import invite_client_to_project


@pytest.fixture
def configured_project_min_annotations_1(configured_project):
    p = Project.objects.get(id=configured_project.id)
    p.min_annotations_to_start_training = 1
    # p.agreement_method = p.SINGLE
    p.save()
    return p


@pytest.mark.django_db
@pytest.mark.parametrize(
    'result, logtext, ml_upload_called',
    [
        (
            json.dumps(
                [
                    {
                        'from_name': 'text_class',
                        'to_name': 'text',
                        'type': 'labels',
                        'value': {'labels': ['class_A'], 'start': 0, 'end': 1},
                    }
                ]
            ),
            None,
            True,
        ),
        (json.dumps([]), None, True),
    ],
)
def test_create_annotation(
    caplog, annotator_client, configured_project_min_annotations_1, result, logtext, ml_upload_called
):
    task = Task.objects.first()
    assert invite_client_to_project(annotator_client, task.project).status_code == 200
    with requests_mock.Mocker() as m:
        m.post('http://localhost:8999/train')
        m.post('http://localhost:8999/webhook')
        annotation = {'result': result, 'task': task.id, 'lead_time': 2.54}
        r = annotator_client.post(f'/api/tasks/{task.id}/annotations/', data=annotation)
        # check that submitted VALID data for task_annotation
        # makes task labeled
        task.refresh_from_db()
        assert task.is_labeled is True
        assert r.status_code == 201
        annotation = Annotation.objects.all()
        assert annotation.count() == 1
        annotation = annotation.first()
        assert annotation.task.id == task.id
        assert annotation.completed_by.id == annotator_client.user.id
        assert annotation.updated_by.id == annotator_client.user.id
        assert annotation.quality_level == Annotation.QualityLevel.ANNOTATOR

        if apps.is_installed('businesses'):
            assert annotation.task.accuracy == 1.0

        if logtext:
            assert logtext in caplog.text


@pytest.mark.django_db
def test_annotator_cannot_set_reviewer_or_manager_quality_level(annotator_client, configured_project):
    task = Task.objects.filter(project=configured_project).first()
    payload = {'task': task.id, 'result': json.dumps([]), 'quality_level': Annotation.QualityLevel.REVIEWER}

    response = annotator_client.post(f'/api/tasks/{task.id}/annotations/', data=payload)

    assert response.status_code == 400
    assert not Annotation.objects.filter(task=task).exists()


@pytest.mark.django_db
def test_reviewer_cannot_set_manager_quality_level(annotator_client, configured_project):
    task = Task.objects.filter(project=configured_project).first()
    membership = ProjectMember.objects.get(user=annotator_client.user, project=configured_project)
    membership.role = ProjectMember.Role.REVIEWER
    membership.save(update_fields=['role'])
    payload = {'task': task.id, 'result': json.dumps([]), 'quality_level': Annotation.QualityLevel.MANAGER}

    response = annotator_client.post(f'/api/tasks/{task.id}/annotations/', data=payload)

    assert response.status_code == 400
    assert not Annotation.objects.filter(task=task).exists()


@pytest.mark.django_db
def test_reviewer_cannot_downgrade_annotation_from_level_two_to_one(annotator_client, configured_project):
    task = Task.objects.filter(project=configured_project).first()
    membership = ProjectMember.objects.get(user=annotator_client.user, project=configured_project)
    membership.role = ProjectMember.Role.REVIEWER
    membership.save(update_fields=['role'])
    annotation = Annotation.objects.create(
        task=task,
        project=configured_project,
        completed_by=annotator_client.user,
        result=[],
        quality_level=Annotation.QualityLevel.REVIEWER,
    )

    response = annotator_client.patch(
        reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation.id}),
        data={'quality_level': Annotation.QualityLevel.ANNOTATOR},
        content_type='application/json',
    )

    annotation.refresh_from_db()
    assert response.status_code == 400
    assert annotation.quality_level == Annotation.QualityLevel.REVIEWER


@pytest.mark.django_db
def test_create_annotation_with_ground_truth(caplog, annotator_client, configured_project_min_annotations_1):

    task = Task.objects.first()
    assert invite_client_to_project(annotator_client, task.project).status_code == 200

    webhook_called = True
    ground_truth = {
        'task': task.id,
        'result': json.dumps(
            [{'from_name': 'text_class', 'to_name': 'text', 'value': {'labels': ['class_A'], 'start': 0, 'end': 1}}]
        ),
        'ground_truth': True,
    }

    annotation = {
        'task': task.id,
        'result': json.dumps(
            [{'from_name': 'text_class', 'to_name': 'text', 'value': {'labels': ['class_B'], 'start': 0, 'end': 1}}]
        ),
    }

    with requests_mock.Mocker() as m:
        m.post('http://localhost:8999/webhook')
        m.post('http://localhost:8999/train')

        # Ground-truth annotation creation still emits the project webhook.
        r = annotator_client.post('/api/tasks/{}/annotations/'.format(task.id), data=ground_truth)
        assert r.status_code == 201
        assert m.called == webhook_called

        # real annotation triggers uploading to ML backend and recalculating accuracy
        r = annotator_client.post('/api/tasks/{}/annotations/'.format(task.id), data=annotation)
        assert r.status_code == 201
        assert m.called
        task = Task.objects.get(id=task.id)
        assert task.annotations.count() == 2
        annotations = Annotation.objects.filter(task=task)
        for a in annotations:
            assert a.updated_by.id == annotator_client.user.id


@pytest.mark.django_db
def test_delete_annotation(annotator_client, configured_project):
    task = Task.objects.first()
    annotation = Annotation.objects.create(
        task=task,
        project=configured_project,
        completed_by=annotator_client.user,
        result=[],
        quality_level=Annotation.QualityLevel.ANNOTATOR,
    )
    assert task.annotations.count() == 1
    r = annotator_client.delete('/api/annotations/{}/'.format(annotation.id))
    assert r.status_code == 204
    assert task.annotations.count() == 0


@pytest.fixture
def annotations():
    task = Task.objects.first()
    return {
        'class_A': {
            'task': task.id,
            'result': json.dumps(
                [
                    {
                        'from_name': 'text_class',
                        'to_name': 'text',
                        'type': 'labels',
                        'value': {'labels': ['class_A'], 'start': 0, 'end': 10},
                    }
                ]
            ),
        },
        'class_B': {
            'task': task.id,
            'result': json.dumps(
                [
                    {
                        'from_name': 'text_class',
                        'to_name': 'text',
                        'type': 'labels',
                        'value': {'labels': ['class_B'], 'start': 0, 'end': 10},
                    }
                ]
            ),
        },
        'empty': {'task': task.id, 'result': json.dumps([])},
    }


@pytest.fixture
def project_with_max_annotations_2(configured_project):
    configured_project.maximum_annotations = 2
    # configured_project.agreement_method = Project.SINGLE
    configured_project.save()


@pytest.fixture
def reviewer_client(annotator2_client, business_client, configured_project):
    OrganizationMember.objects.filter(
        user=annotator2_client.user,
        deleted_at__isnull=True,
    ).exclude(organization=business_client.organization).first().soft_delete()
    business_client.organization.add_user(annotator2_client.user)
    configured_project.add_collaborator(annotator2_client.user, role=ProjectMember.Role.REVIEWER)
    return annotator2_client


@pytest.mark.parametrize(
    'annotations_sequence, accuracy, is_labeled',
    [
        ([], None, False),
        ([('class_A', 'reviewer')], 1, False),
        ([('class_A', 'annotator')], 1, False),
        ([('class_A', 'reviewer'), ('class_A', 'reviewer')], 1, True),
        ([('class_A', 'reviewer'), ('class_A', 'annotator')], 1, True),
        ([('class_A', 'annotator'), ('class_A', 'reviewer')], 1, True),
        ([('class_A', 'reviewer'), ('class_B', 'reviewer')], 0.5, True),
        ([('class_A', 'reviewer'), ('class_B', 'annotator')], 0.5, True),
        ([('class_A', 'annotator'), ('class_B', 'reviewer')], 0.5, True),
        ([('empty', 'annotator'), ('empty', 'reviewer')], 1, True),
        ([('class_A', 'annotator'), ('empty', 'reviewer')], 0.5, True),
    ],
)
@pytest.mark.django_db
def test_accuracy(
    reviewer_client,
    annotator_client,
    project_with_max_annotations_2,
    annotations,
    annotations_sequence,
    accuracy,
    is_labeled,
):
    client = {'reviewer': reviewer_client, 'annotator': annotator_client}
    task_id = next(iter(annotations.values()))['task']
    task = Task.objects.get(id=task_id)
    invite_client_to_project(annotator_client, task.project)
    invite_client_to_project(reviewer_client, task.project)

    for annotation_key, client_key in annotations_sequence:
        r = client[client_key].post(
            reverse('tasks:api:task-annotations', kwargs={'pk': task_id}), data=annotations[annotation_key]
        )
        assert r.status_code == 201
    task = Task.objects.get(id=task_id)
    assert task.is_labeled == is_labeled


@pytest.mark.django_db
def test_accuracy_on_delete(reviewer_client, project_with_max_annotations_2, annotations):
    task_id = next(iter(annotations.values()))['task']
    for annotation in annotations.values():
        reviewer_client.post(reverse('tasks:api:task-annotations', kwargs={'pk': task_id}), data=annotation)

    task = Task.objects.get(id=task_id)
    assert task.annotations.count() == len(annotations)
    assert task.is_labeled
    annotation_ids = [c.id for c in task.annotations.all()]
    r = reviewer_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[0]}))
    assert r.status_code == 204
    task = Task.objects.get(id=task_id)
    assert task.is_labeled  # project.max_annotations = 2

    r = reviewer_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[1]}))
    assert r.status_code == 204
    task = Task.objects.get(id=task_id)
    assert not task.is_labeled

    r = reviewer_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[2]}))
    assert r.status_code == 204
    task = Task.objects.get(id=task_id)
    assert not task.is_labeled


# @pytest.mark.django_db
# def test_accuracy_on_delete(business_client, project_with_max_annotations_2, annotations):
#     task_id = next(iter(annotations.values()))['task']
#     for annotation in annotations.values():
#         business_client.post(reverse('tasks:api:task-annotations', kwargs={'pk': task_id}), data=annotation)
#
#     task = Task.objects.get(id=task_id)
#     assert task.annotations.count() == len(annotations)
#     if apps.is_installed('businesses'):
#         assert math.fabs(task.accuracy - 1 / 3) < 0.00001
#     assert task.is_labeled
#     annotation_ids = [c.id for c in task.annotations.all()]
#     r = business_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[0]}))
#     assert r.status_code == 204
#     task = Task.objects.get(id=task_id)
#     if apps.is_installed('businesses'):
#         assert task.accuracy == 0.5
#     assert task.is_labeled  # project.max_annotations = 2
#
#     r = business_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[1]}))
#     assert r.status_code == 204
#     task = Task.objects.get(id=task_id)
#     if apps.is_installed('businesses'):
#         assert task.accuracy == 1.0
#     assert not task.is_labeled
#
#     r = business_client.delete(reverse('tasks:api-annotations:annotation-detail', kwargs={'pk': annotation_ids[2]}))
#     assert r.status_code == 204
#     task = Task.objects.get(id=task_id)
#     if apps.is_installed('businesses'):
#         assert task.accuracy is None
#     assert not task.is_labeled
