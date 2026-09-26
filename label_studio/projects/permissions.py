from rest_framework.permissions import BasePermission, SAFE_METHODS

from organizations.models import OrganizationMember
from projects.models import Project, ProjectMember


class ProjectImportPermission(BasePermission):
    """Allow project import actions only for Manager-level users and above."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project_pk = view.kwargs.get('pk') or view.kwargs.get('project_pk')
        if not project_pk:
            return False

        try:
            project = Project.objects.get(pk=project_pk)
        except Project.DoesNotExist:
            return False

        return project.has_role(request.user, ProjectMember.Role.MANAGER)


class OrganizationProjectCreatePermission(BasePermission):
    """Only organization admins and managers may create projects."""

    def has_permission(self, request, view):
        if request.method != 'POST':
            return True
        organization = getattr(request.user, 'active_organization', None)
        return bool(
            organization
            and organization.get_role(request.user) == OrganizationMember.Role.MANAGER
        )


class ProjectRolePermission(BasePermission):
    """Use the project role hierarchy to gate project-level actions."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project_pk = view.kwargs.get('pk')
        if not project_pk:
            return False

        try:
            project = Project.objects.get(pk=project_pk)
        except Project.DoesNotExist:
            return False

        if request.method in SAFE_METHODS:
            return (
                project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
                or project.has_role(request.user, ProjectMember.Role.REVIEWER)
                or project.has_role(request.user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(request.user, OrganizationMember.Role.ADMIN)
            )

        if request.method == 'DELETE':
            return project.has_role(request.user, ProjectMember.Role.MANAGER) or project.organization.has_role(
                request.user, OrganizationMember.Role.ADMIN
            )

        return project.has_role(request.user, ProjectMember.Role.MANAGER)


class ProjectMemberManagePermission(BasePermission):
    """Project member listing and updates are allowed for managers and admins."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project_pk = view.kwargs.get('pk')
        if not project_pk:
            return False

        try:
            project = Project.objects.get(pk=project_pk)
        except Project.DoesNotExist:
            return False

        if request.method in SAFE_METHODS:
            return project.has_role(request.user, ProjectMember.Role.MANAGER) or project.organization.has_role(
                request.user, OrganizationMember.Role.ADMIN
            )

        return project.has_role(request.user, ProjectMember.Role.MANAGER) or project.organization.has_role(
            request.user, OrganizationMember.Role.ADMIN
        )


class ProjectTaskPermission(BasePermission):
    """Project Managers manage tasks; members annotate and organization Admins read."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project_pk = view.kwargs.get('pk')
        try:
            project = Project.objects.select_related('organization').get(pk=project_pk)
        except (Project.DoesNotExist, TypeError, ValueError):
            return False

        if request.method in SAFE_METHODS:
            return (
                project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
                or project.has_role(request.user, ProjectMember.Role.REVIEWER)
                or project.has_role(request.user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(request.user, OrganizationMember.Role.ADMIN)
            )

        return project.has_role(request.user, ProjectMember.Role.MANAGER)


class ProjectAnnotationPermission(BasePermission):
    """Only project Annotators and Reviewers may enter the labeling queue."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        try:
            project = Project.objects.select_related('organization').get(pk=view.kwargs.get('pk'))
        except (Project.DoesNotExist, TypeError, ValueError):
            return False

        return project.has_role(request.user, ProjectMember.Role.ANNOTATOR)


class ProjectActionsPermission(BasePermission):
    """Only annotators/reviewers may run Data Manager actions that can alter annotations."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project_pk = request.query_params.get('project') or request.data.get('project')
        try:
            project = Project.objects.select_related('organization').get(pk=project_pk)
        except (Project.DoesNotExist, TypeError, ValueError):
            return False

        if request.method in SAFE_METHODS:
            return (
                project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
                or project.has_role(request.user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(request.user, OrganizationMember.Role.ADMIN)
            )

        return project.has_role(request.user, ProjectMember.Role.ANNOTATOR)

    def has_object_permission(self, request, view, project):
        if request.method in SAFE_METHODS:
            return (
                project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
                or project.has_role(request.user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(request.user, OrganizationMember.Role.ADMIN)
            )
        return project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
