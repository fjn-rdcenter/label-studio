from rest_framework.permissions import SAFE_METHODS, BasePermission

from organizations.models import OrganizationMember
from projects.models import Project, ProjectMember
from ml.models import MLBackend


class MLBackendProjectPermission(BasePermission):
    """Allow labelers to list safe backend metadata, but keep configuration Manager/Admin-only."""

    @staticmethod
    def _project_for_request(request, view):
        project_id = view.kwargs.get('pk') or request.query_params.get('project') or request.data.get('project')
        if not project_id:
            return None
        try:
            if view.kwargs.get('pk'):
                backend = MLBackend.objects.select_related('project__organization').filter(pk=project_id).first()
                return backend.project if backend else None
            return Project.objects.select_related('organization').filter(pk=project_id).first()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _can_read_configuration(project, user):
        return bool(
            project
            and (
                project.has_role(user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(user, OrganizationMember.Role.ADMIN)
            )
        )

    @classmethod
    def _can_read_list(cls, project, user):
        return bool(
            cls._can_read_configuration(project, user)
            or (project and project.has_role(user, ProjectMember.Role.ANNOTATOR))
            or (project and project.has_role(user, ProjectMember.Role.REVIEWER))
        )

    @classmethod
    def _can_read(cls, project, user, view):
        if view.__class__.__name__ == 'MLBackendListAPI':
            return cls._can_read_list(project, user)
        return cls._can_read_configuration(project, user)

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        project = self._project_for_request(request, view)
        if request.method in SAFE_METHODS:
            return self._can_read(project, request.user, view)
        return bool(project and project.has_role(request.user, ProjectMember.Role.MANAGER))

    def has_object_permission(self, request, view, obj):
        project = obj.project if isinstance(obj, MLBackend) else obj
        if request.method in SAFE_METHODS:
            return self._can_read(project, request.user, view)
        return bool(project and project.has_role(request.user, ProjectMember.Role.MANAGER))


class MLBackendInteractivePermission(BasePermission):
    """Allow Annotators and Reviewers to use model suggestions while labeling."""

    def has_permission(self, request, view):
        return bool(getattr(request.user, 'is_authenticated', False))

    def has_object_permission(self, request, view, backend):
        return backend.project.has_role(request.user, ProjectMember.Role.ANNOTATOR) or backend.project.has_role(request.user, ProjectMember.Role.REVIEWER)
    
