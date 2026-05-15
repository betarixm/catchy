from django.db import migrations


def _table_exists(connection, cursor, table_name: str) -> bool:
    tables = set(connection.introspection.table_names(cursor))
    return table_name in tables


def _column_names(connection, cursor, table_name: str) -> set[str]:
    description = connection.introspection.get_table_description(cursor, table_name)
    return {col.name for col in description}


def reconcile_graph_to_flow(apps, schema_editor):
    connection = schema_editor.connection
    quote = connection.ops.quote_name

    flow_model = apps.get_model("ctf", "FlowConfiguration")

    with connection.cursor() as cursor:
        has_flow_table = _table_exists(connection, cursor, "ctf_flowconfiguration")
        has_graph_table = _table_exists(connection, cursor, "ctf_graphconfiguration")

        # Legacy DBs may still have the older graph table name.
        if has_graph_table and not has_flow_table:
            schema_editor.execute(
                f"ALTER TABLE {quote('ctf_graphconfiguration')} "
                f"RENAME TO {quote('ctf_flowconfiguration')}"
            )
            has_flow_table = True

        if has_flow_table:
            flow_columns = _column_names(connection, cursor, "ctf_flowconfiguration")

            if "yaml" not in flow_columns:
                schema_editor.execute(
                    f"ALTER TABLE {quote('ctf_flowconfiguration')} "
                    f"ADD COLUMN {quote('yaml')} TEXT NOT NULL DEFAULT ''"
                )
                flow_columns = _column_names(connection, cursor, "ctf_flowconfiguration")

            # Preserve legacy graph definitions in the new yaml column.
            if "definition" in flow_columns:
                schema_editor.execute(
                    "UPDATE ctf_flowconfiguration "
                    "SET yaml = COALESCE(definition, yaml, '') "
                    "WHERE COALESCE(yaml, '') = ''"
                )

        thread_columns = _column_names(connection, cursor, "ctf_thread")
        if "graph_configuration_id" in thread_columns and "flow_id" not in thread_columns:
            schema_editor.execute(
                f"ALTER TABLE {quote('ctf_thread')} "
                f"RENAME COLUMN {quote('graph_configuration_id')} TO {quote('flow_id')}"
            )

    # Legacy DBs may miss m2m tables for flow permissions.
    for m2m_field in flow_model._meta.local_many_to_many:
        through_model = m2m_field.remote_field.through
        through_table = through_model._meta.db_table
        with connection.cursor() as cursor:
            if not _table_exists(connection, cursor, through_table):
                schema_editor.create_model(through_model)


class Migration(migrations.Migration):
    dependencies = [
        ("ctf", "0021_graph_configuration_thread_graph_fk"),
    ]

    operations = [
        migrations.RunPython(reconcile_graph_to_flow, migrations.RunPython.noop),
    ]
