from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("ctf", "0019_recover_legacy_app_events")]

    operations = [
        migrations.AddField(
            model_name="ctf",
            name="prompt",
            field=models.TextField(blank=True, default=""),
        ),
    ]
