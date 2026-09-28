from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0047_merge_20240318_2210'),
    ]

    operations = [
        migrations.AddField(
            model_name='annotation',
            name='quality_level',
            field=models.PositiveSmallIntegerField(
                choices=[(1, 'Annotator'), (2, 'Reviewer'), (3, 'Manager')],
                default=1,
                help_text='Label review level: annotator, reviewer, or manager.',
            ),
        ),
        migrations.AddField(
            model_name='annotation',
            name='quality_updated_by',
            field=models.ForeignKey(
                blank=True,
                help_text='User who last changed the label review level.',
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='quality_annotations',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]