from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0020_ctf_prompt"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="FlowConfiguration",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=200)),
                ("slug", models.SlugField(unique=True)),
                ("yaml", models.TextField()),
                (
                    "created_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="created_flow_configurations",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "use_groups",
                    models.ManyToManyField(
                        blank=True,
                        related_name="usable_flow_configurations",
                        to="auth.group",
                    ),
                ),
                (
                    "view_groups",
                    models.ManyToManyField(
                        blank=True,
                        related_name="viewable_flow_configurations",
                        to="auth.group",
                    ),
                ),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.AlterField(
            model_name="thread",
            name="agent",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="threads",
                to="ctf.agentconfiguration",
            ),
        ),
        migrations.AddField(
            model_name="thread",
            name="flow",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="threads",
                to="ctf.flowconfiguration",
            ),
        ),
    ]
