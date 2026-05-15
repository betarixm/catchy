import copy

from django.db import migrations


def _column_nullability(connection, table_name: str, column_name: str) -> bool | None:
    with connection.cursor() as cursor:
        description = connection.introspection.get_table_description(cursor, table_name)
    for col in description:
        if col.name == column_name:
            return bool(getattr(col, "null_ok", False))
    return None


def ensure_thread_agent_nullable(apps, schema_editor):
    connection = schema_editor.connection
    thread_model = apps.get_model("ctf", "Thread")

    # Some SQLite DBs drifted with ctf_thread.agent_id as NOT NULL even though
    # Django state expects nullable for flow runtime threads.
    is_nullable = _column_nullability(connection, "ctf_thread", "agent_id")
    if is_nullable is True:
        return

    new_field = thread_model._meta.get_field("agent")
    old_field = copy.copy(new_field)
    old_field.null = False
    old_field.blank = False
    schema_editor.alter_field(thread_model, old_field, new_field, strict=False)


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0022_reconcile_graph_schema_with_flow_models"),
    ]

    operations = [
        migrations.RunPython(ensure_thread_agent_nullable, migrations.RunPython.noop),
    ]
