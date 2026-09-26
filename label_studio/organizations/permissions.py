from rest_framework.permissions import SAFE_METHODS, BasePermission

from organizations.models import Organization, OrganizationMember


class OrganizationMemberManagePermission(BasePermission):
    """Allow organization admins to update or remove organization members."""

    def has_permission(self, request, view):
        organization_id = view.kwargs.get('pk')
        if not organization_id or not getattr(request.user, 'is_authenticated', False):
            return False

        try:
            organization = Organization.objects.get(pk=organization_id)
        except Organization.DoesNotExist:
            return False

        required_role = OrganizationMember.Role.MEMBER if request.method in SAFE_METHODS else OrganizationMember.Role.ADMIN
        return organization.has_role(request.user, required_role)