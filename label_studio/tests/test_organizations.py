"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
from types import SimpleNamespace

import pytest
from organizations.models import OrganizationMember
from organizations.serializers import OrganizationMemberUserSerializer
from projects.models import Project, ProjectMember
from projects.permissions import OrganizationProjectCreatePermission, ProjectRolePermission
from tests.utils import make_project
from tasks.models import Annotation, Task
from tasks.permissions import AnnotationWorkflowPermission
from tasks.serializers import AnnotationSerializer
from users.models import User


@pytest.mark.django_db
def test_active_organization_filled(business_client):
    response = business_client.get('/api/users/')
    response_data = response.json()
    assert response_data[0]['active_organization'] == business_client.organization.id


@pytest.mark.django_db
def test_api_list_organizations(business_client):
    response = business_client.get('/api/organizations/')
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]['id'] == business_client.organization.id


@pytest.mark.django_db
def test_organization_manager_can_create_projects_and_admin_sees_all(business_client):
    organization = business_client.organization
    business_client.user.active_organization = organization
    business_client.user.save(update_fields=['active_organization'])
    manager = User.objects.create(email='organization-manager@example.com')
    organization.add_user(manager)
    manager_membership = OrganizationMember.objects.get(user=manager, organization=organization)
    manager_membership.role = OrganizationMember.Role.MANAGER
    manager_membership.save(update_fields=['role'])

    member = User.objects.create(email='organization-member@example.com')
    organization.add_user(member)
    permission = OrganizationProjectCreatePermission()

    assert permission.has_permission(SimpleNamespace(method='POST', user=manager), None)
    assert not permission.has_permission(SimpleNamespace(method='POST', user=business_client.user), None)
    assert not permission.has_permission(SimpleNamespace(method='POST', user=member), None)

    project = make_project({'title': 'Manager project'}, manager, use_ml_backend=False, org=organization)
    assert Project.objects.for_user(business_client.user).filter(pk=project.pk).exists()
    assert not Project.objects.for_user(member).filter(pk=project.pk).exists()
    assert project.has_role(manager, ProjectMember.Role.MANAGER)


@pytest.mark.django_db
def test_project_delete_permissions_are_admin_or_project_manager(business_client):
    organization = business_client.organization
    business_client.user.active_organization = organization
    business_client.user.save(update_fields=['active_organization'])

    manager = User.objects.create(email='project-manager@example.com')
    organization.add_user(manager)
    project = make_project({'title': 'Managed project'}, manager, use_ml_backend=False, org=organization)

    member = User.objects.create(email='project-member@example.com')
    organization.add_user(member)
    project.add_collaborator(member, role=ProjectMember.Role.ANNOTATOR)

    permission = ProjectRolePermission()
    view_kwargs = {'pk': project.pk}
    view = SimpleNamespace(kwargs=view_kwargs)

    assert permission.has_permission(SimpleNamespace(method='DELETE', user=business_client.user), view)
    assert permission.has_permission(SimpleNamespace(method='DELETE', user=manager), view)
    assert not permission.has_permission(SimpleNamespace(method='DELETE', user=member), view)


@pytest.mark.django_db
def test_organization_owner_cannot_be_demoted(business_client):
    owner_membership = OrganizationMember.objects.get(
        user=business_client.user,
        organization=business_client.organization,
    )
    serializer = OrganizationMemberUserSerializer(
        instance=owner_membership,
        data={'role': OrganizationMember.Role.MANAGER},
        partial=True,
    )

    assert not serializer.is_valid()
    assert 'role' in serializer.errors


@pytest.mark.django_db
def test_project_manager_can_change_review_confirmation_status(business_client):
    organization = business_client.organization
    manager = User.objects.create(email='confirming-manager@example.com')
    organization.add_user(manager)
    manager_membership = OrganizationMember.objects.get(user=manager, organization=organization)
    manager_membership.role = OrganizationMember.Role.MANAGER
    manager_membership.save(update_fields=['role'])

    reviewer = User.objects.create(email='project-reviewer@example.com')
    organization.add_user(reviewer)
    project = make_project({'title': 'Review workflow'}, manager, use_ml_backend=False, org=organization)
    project.add_collaborator(reviewer, role=ProjectMember.Role.REVIEWER)
    task = Task.objects.create(project=project, data={})
    reviewed = Annotation.objects.create(
        project=project,
        task=task,
        completed_by=reviewer,
        result=[],
        quality_level=Annotation.QualityLevel.REVIEWER,
    )
    annotator_level = Annotation.objects.create(
        project=project,
        task=task,
        completed_by=reviewer,
        result=[],
        quality_level=Annotation.QualityLevel.ANNOTATOR,
    )
    permission = AnnotationWorkflowPermission()
    view = SimpleNamespace()

    confirm_request = SimpleNamespace(user=manager, method='PATCH', data={'quality_level': 3})
    return_to_review_request = SimpleNamespace(user=manager, method='PATCH', data={'quality_level': 2})
    content_edit_request = SimpleNamespace(
        user=manager,
        method='PATCH',
        data={'quality_level': 3, 'result': [{'value': {'labels': ['changed']}}]},
    )
    confirmed = Annotation.objects.create(
        project=project,
        task=task,
        completed_by=reviewer,
        result=[],
        quality_level=Annotation.QualityLevel.MANAGER,
    )

    assert permission.has_object_permission(confirm_request, view, reviewed)
    assert permission.has_object_permission(return_to_review_request, view, confirmed)
    manager_serializer = AnnotationSerializer(
        instance=confirmed,
        context={'request': SimpleNamespace(user=manager, query_params=SimpleNamespace(getlist=lambda _key: []))},
    )
    assert manager_serializer.validate_quality_level(Annotation.QualityLevel.REVIEWER) == Annotation.QualityLevel.REVIEWER
    assert not permission.has_object_permission(content_edit_request, view, reviewed)
    assert not permission.has_object_permission(confirm_request, view, annotator_level)

    annotation_list_view = type('AnnotationsListAPI', (), {'kwargs': {'pk': task.pk}})()
    manager_request = SimpleNamespace(user=manager, method='POST')
    admin_request = SimpleNamespace(user=business_client.user, method='POST')
    assert not permission.has_permission(manager_request, annotation_list_view)
    assert not permission.has_permission(admin_request, annotation_list_view)
