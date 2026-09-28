from django.db import migrations, models


def ensure_role_column(apps, schema_editor):
    organization_member = apps.get_model('organizations', 'OrganizationMember')
    table = organization_member._meta.db_table
    with schema_editor.connection.cursor() as cursor:
        columns = {
            column.name
            for column in schema_editor.connection.introspection.get_table_description(cursor, table)
        }

    if 'role' in columns:
        return

    role_field = models.CharField(
        choices=[('AD', 'Admin'), ('MA', 'Manager'), ('ME', 'Member')],
        default='ME',
        help_text='Organization role of the member',
        max_length=2,
    )
    role_field.set_attributes_from_name('role')
    role_field.model = organization_member
    schema_editor.add_field(organization_member, role_field)


class Migration(migrations.Migration):
    dependencies = [
        ('organizations', '0007_unique_active_organization_member'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[migrations.RunPython(ensure_role_column, migrations.RunPython.noop)],
            state_operations=[
                migrations.AddField(
                    model_name='organizationmember',
                    name='role',
                    field=models.CharField(
                        choices=[('AD', 'Admin'), ('MA', 'Manager'), ('ME', 'Member')],
                        default='ME',
                        help_text='Organization role of the member',
                        max_length=2,
                    ),
                )
            ],
        ),
    ]
