from rest_framework.permissions import SAFE_METHODS, BasePermission

def _related_project(obj):
    project = getattr(obj, 'project', None)
    if project is not None:
        return project

    task = getattr(obj, 'task', None)
    project = getattr(task, 'project', None)
    if project is not None:
        return project

    return obj if hasattr(obj, 'organization') and hasattr(obj, 'has_role') else None


class HasObjectPermission(BasePermission):
    def has_object_permission(self, request, view, obj):
        project = _related_project(obj)
        if project is None:
            return obj.has_permission(request.user)

        from organizations.models import OrganizationMember

        organization = getattr(project, 'organization', None)
        is_org_admin = bool(organization and organization.has_role(request.user, OrganizationMember.Role.ADMIN))
        is_manager = bool(project.has_role(request.user, 'MA'))
        is_labeler = bool(project.has_role(request.user, 'AN'))

        if request.method in SAFE_METHODS:
            return is_org_admin or is_manager or is_labeler

        if is_org_admin:
            return False

        model_name = getattr(getattr(obj, '_meta', None), 'model_name', '')
        view_name = view.__class__.__name__

        if model_name == 'annotation':
            return is_manager or is_labeler

        if model_name == 'annotationdraft':
            return is_labeler and getattr(obj, 'user_id', None) == request.user.id

        if model_name == 'task' and view_name in {'AnnotationsListAPI', 'AnnotationDraftListAPI'}:
            return is_labeler

        return is_manager


class MemberHasOwnerPermission(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method not in SAFE_METHODS and not request.user.own_organization:
            return False

        return obj.has_permission(request.user)
