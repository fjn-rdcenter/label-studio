from django.db import migrations


def assign_creator_manager_role(apps, schema_editor):
    Project = apps.get_model('projects', 'Project')
    ProjectMember = apps.get_model('projects', 'ProjectMember')
    OrganizationMember = apps.get_model('organizations', 'OrganizationMember')
    Organization = apps.get_model('organizations', 'Organization')

    projects = Project.objects.exclude(created_by_id__isnull=True).exclude(organization_id__isnull=True)
    for project_id, creator_id, organization_id in projects.values_list('id', 'created_by_id', 'organization_id').iterator():
        if ProjectMember.objects.filter(project_id=project_id, user_id=creator_id).exists():
            continue

        organization = Organization.objects.filter(pk=organization_id).values_list('created_by_id', flat=True).first()
        if organization == creator_id:
            continue

        membership = OrganizationMember.objects.filter(
            organization_id=organization_id,
            user_id=creator_id,
            deleted_at__isnull=True,
        ).first()
        if membership is None or membership.role == 'AD':
            continue

        ProjectMember.objects.create(
            project_id=project_id,
            user_id=creator_id,
            role='MA',
            enabled=True,
        )


class Migration(migrations.Migration):
    dependencies = [
        ('organizations', '0008_organization_manager_role'),
        ('projects', '0027_projectmember_role'),
    ]

    operations = [
        migrations.RunPython(assign_creator_manager_role, migrations.RunPython.noop),
    ]