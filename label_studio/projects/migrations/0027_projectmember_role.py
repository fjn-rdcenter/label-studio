from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0026_auto_20231103_0020'),
    ]

    operations = [
        migrations.AddField(
            model_name='projectmember',
            name='role',
            field=models.CharField(
                choices=[('AD', 'Admin'), ('MA', 'Manager'), ('RE', 'Reviewer'), ('AN', 'Annotator')],
                default='AN',
                help_text='Project role of the member',
                max_length=2,
            ),
        ),
        migrations.AddConstraint(
            model_name='projectmember',
            constraint=models.UniqueConstraint(fields=('user', 'project'), name='unique_project_member'),
        ),
    ]
