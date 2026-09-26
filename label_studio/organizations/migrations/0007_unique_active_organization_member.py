from django.db import migrations, models


def ensure_active_membership_constraint(apps, schema_editor):
    organization_member = apps.get_model('organizations', 'OrganizationMember')
    table = organization_member._meta.db_table
    constraint_name = 'unique_active_user_organization'

    with schema_editor.connection.cursor() as cursor:
        constraints = schema_editor.connection.introspection.get_constraints(cursor, table)

    if constraint_name in constraints:
        return

    schema_editor.add_constraint(
        organization_member,
        models.UniqueConstraint(
            condition=models.Q(deleted_at__isnull=True),
            fields=('user',),
            name=constraint_name,
        ),
    )


class Migration(migrations.Migration):

    dependencies = [
        ('organizations', '0006_alter_organizationmember_deleted_at'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(ensure_active_membership_constraint, migrations.RunPython.noop)
            ],
            state_operations=[
                migrations.AddConstraint(
                    model_name='organizationmember',
                    constraint=models.UniqueConstraint(
                        condition=models.Q(deleted_at__isnull=True),
                        fields=('user',),
                        name='unique_active_user_organization',
                    ),
                )
            ],
        ),
    ]
