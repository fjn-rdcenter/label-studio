from typing import TYPE_CHECKING, Mapping, Optional

from core.redis import start_job_async_or_sync
from django.db.models import QuerySet
from django.utils.functional import cached_property

if TYPE_CHECKING:
    from users.models import User


class ProjectMixin:
    def rearrange_overlap_cohort(self):
        """
        Async start rearrange overlap depending on annotation count in tasks
        """
        start_job_async_or_sync(self._rearrange_overlap_cohort)

    def update_tasks_counters_and_is_labeled(self, tasks_queryset, from_scratch=True):
        """
        Async start updating tasks counters and than is_labeled
        :param tasks_queryset: Tasks to update queryset
        :param from_scratch: Skip calculated tasks
        """
        # get only id from queryset to decrease data size in job
        if not (isinstance(tasks_queryset, set) or isinstance(tasks_queryset, list)):
            tasks_queryset = set(tasks_queryset.values_list('id', flat=True))
        start_job_async_or_sync(
            self._update_tasks_counters_and_is_labeled, list(tasks_queryset), from_scratch=from_scratch
        )

    def update_tasks_counters_and_task_states(
        self,
        tasks_queryset,
        maximum_annotations_changed,
        overlap_cohort_percentage_changed,
        tasks_number_changed,
        from_scratch=True,
        recalculate_stats_counts: Optional[Mapping[str, int]] = None,
    ):
        """
        Async start updating tasks counters and than rearrange
        :param tasks_queryset: Tasks to update queryset
        :param maximum_annotations_changed: If maximum_annotations param changed
        :param overlap_cohort_percentage_changed: If cohort_percentage param changed
        :param tasks_number_changed: If tasks number changed in project
        :param from_scratch: Skip calculated tasks
        """
        # get only id from queryset to decrease data size in job
        if not (isinstance(tasks_queryset, set) or isinstance(tasks_queryset, list)):
            tasks_queryset = set(tasks_queryset.values_list('id', flat=True))
        start_job_async_or_sync(
            self._update_tasks_counters_and_task_states,
            tasks_queryset,
            maximum_annotations_changed,
            overlap_cohort_percentage_changed,
            tasks_number_changed,
            from_scratch=from_scratch,
            recalculate_stats_counts=recalculate_stats_counts,
        )

    def update_tasks_states(
        self, maximum_annotations_changed, overlap_cohort_percentage_changed, tasks_number_changed
    ):
        """
        Async start updating tasks states after settings change
        :param maximum_annotations_changed: If maximum_annotations param changed
        :param overlap_cohort_percentage_changed: If cohort_percentage param changed
        :param tasks_number_changed: If tasks number changed in project
        """
        start_job_async_or_sync(
            self._update_tasks_states,
            maximum_annotations_changed,
            overlap_cohort_percentage_changed,
            tasks_number_changed,
        )

    def get_role(self, user):
        if user is None or not getattr(user, 'is_authenticated', False):
            return None

        if getattr(self, 'organization', None) is None or not self.organization.has_permission(user):
            return None
        if self.organization.has_role(user, 'AD'):
            return None
        if getattr(self, 'created_by_id', None) == getattr(user, 'id', None):
            return 'MA'

        membership = self.members.filter(user=user, enabled=True).first() if hasattr(self, 'members') else None
        if membership is not None:
            return membership.role

        return None

    def has_role(self, user, min_role='AN'):
        role = self.get_role(user)
        if role is None:
            return None

        capabilities = {
            'AN': {'AN', 'RE'},
            'RE': {'RE'},
            'MA': {'MA', 'AD'},
        }
        if min_role in capabilities:
            return role in capabilities[min_role]

        return False

    def has_permission(self, user):
        """
        Object-write permission for project members; read-only organization Admin access is handled separately.
        """
        user.project = self  # link for activity log
        return bool(self.has_role(user, 'AN') or self.has_role(user, 'MA'))

    def _can_use_overlap(self):
        """
        Returns if we can use overlap for is_labeled calculation
        :return:
        """
        return True

    @cached_property
    def all_members(self) -> QuerySet['User']:
        """
        Returns all users of project
        :return: QuerySet[User]
        """
        from users.models import User

        return User.objects.filter(id__in=self.organization.members.values_list('user__id'))
