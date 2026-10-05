from rest_framework.permissions import BasePermission, SAFE_METHODS

from projects.models import ProjectMember
from tasks.models import Annotation, Task


class AnnotationWorkflowPermission(BasePermission):
    """Enforce annotation capabilities and restrict Manager edits to status transitions."""

    def has_permission(self, request, view):
        if not getattr(request.user, 'is_authenticated', False):
            return False

        if request.method in SAFE_METHODS:
            return True

        if view.__class__.__name__ == 'AnnotationsListAPI':
            task = Task.objects.filter(pk=view.kwargs.get('pk')).select_related('project').first()
            return task is not None and task.project.has_role(request.user, ProjectMember.Role.ANNOTATOR)

        return True

    def has_object_permission(self, request, view, obj):
        if not isinstance(obj, Annotation):
            return True

        project = obj.project
        if request.method in SAFE_METHODS:
            return (
                project.has_role(request.user, ProjectMember.Role.ANNOTATOR)
                or project.has_role(request.user, ProjectMember.Role.MANAGER)
                or project.organization.has_role(request.user, 'AD')
            )

        role = project.get_role(request.user)
        if role in {ProjectMember.Role.MANAGER, ProjectMember.Role.ADMIN}:
            return (
                request.method == 'PATCH'
                and (obj.quality_level, request.data.get('quality_level'))
                in {
                    (Annotation.QualityLevel.REVIEWER, Annotation.QualityLevel.MANAGER),
                    (Annotation.QualityLevel.MANAGER, Annotation.QualityLevel.REVIEWER),
                }
                and set(request.data.keys()) == {'quality_level'}
            )

        if role == ProjectMember.Role.ANNOTATOR:
            # Annotators can always edit annotation content, even ones imported/created
            # by a manager or another annotator; quality_level change rules are
            # enforced separately in AnnotationSerializer.validate_quality_level.
            if request.method == 'DELETE':
                return obj.completed_by_id == request.user.id and obj.quality_level == Annotation.QualityLevel.ANNOTATOR

            return True

        if obj.quality_level >= Annotation.QualityLevel.MANAGER:
            return False

        return project.has_role(request.user, ProjectMember.Role.REVIEWER)
