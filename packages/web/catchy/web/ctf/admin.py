from django.contrib import admin

from .models import (
    AgentConfiguration,
    Challenge,
    Credential,
    Ctf,
    FlowConfiguration,
    ModelConfiguration,
    ModelPricing,
    Provider,
    SteeringMessage,
    StreamEvent,
    Thread,
    ThreadCostSnapshot,
)


@admin.register(Provider)
class ProviderAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]


@admin.register(Credential)
class CredentialAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "kind", "provider", "base_url", "created_at"]
    search_fields = ["name", "slug", "kind", "base_url", "organization_id"]
    list_filter = ["provider", "kind"]
    filter_horizontal = ["allowed_groups", "allowed_users"]


@admin.register(ModelConfiguration)
class ModelConfigurationAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]
    filter_horizontal = ["view_groups", "use_groups"]


@admin.register(ModelPricing)
class ModelPricingAdmin(admin.ModelAdmin):
    list_display = [
        "model",
        "provider",
        "input_per_million",
        "cached_input_per_million",
        "output_per_million",
        "created_at",
    ]
    list_filter = ["provider"]
    search_fields = ["model__name", "provider__name", "provider__slug"]


@admin.register(AgentConfiguration)
class AgentConfigurationAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]
    filter_horizontal = ["view_groups", "use_groups"]


@admin.register(FlowConfiguration)
class FlowConfigurationAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "created_at"]
    search_fields = ["name", "slug"]
    filter_horizontal = ["view_groups", "use_groups"]


@admin.register(Ctf)
class CtfAdmin(admin.ModelAdmin):
    list_display = ["title", "slug", "created_at"]
    search_fields = ["title", "slug"]
    filter_horizontal = ["view_groups", "init_groups"]


@admin.register(Challenge)
class ChallengeAdmin(admin.ModelAdmin):
    list_display = ["challenge_id", "ctf", "created_at"]
    search_fields = ["challenge_id", "description"]


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "uuid",
        "name",
        "ctf",
        "challenge",
        "agent",
        "flow",
        "model",
        "credential",
        "status",
        "is_public",
    ]
    list_filter = ["status", "is_public", "ctf", "agent", "flow", "model", "credential"]
    search_fields = ["uuid", "name", "challenge__challenge_id", "ctf__title"]


@admin.register(StreamEvent)
class StreamEventAdmin(admin.ModelAdmin):
    list_display = ["thread", "format", "created_at"]
    list_filter = ["format"]


@admin.register(SteeringMessage)
class SteeringMessageAdmin(admin.ModelAdmin):
    list_display = ["thread", "created_by", "delivered_at", "created_at"]
    list_filter = ["delivered_at"]


@admin.register(ThreadCostSnapshot)
class ThreadCostSnapshotAdmin(admin.ModelAdmin):
    list_display = ["thread", "created_at"]
