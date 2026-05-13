from django.urls import path

from . import views

app_name = "ctf"

urlpatterns = [
    path("", views.index, name="index"),
    path("credentials/", views.credential_list, name="credential_list"),
    path("credentials/new/", views.credential_create, name="credential_create"),
    path(
        "credentials/<slug:slug>/edit/",
        views.credential_update,
        name="credential_update",
    ),
    path("providers/", views.provider_list, name="provider_list"),
    path("providers/new/", views.provider_create, name="provider_create"),
    path(
        "providers/<slug:slug>/edit/",
        views.provider_update,
        name="provider_update",
    ),
    path("models/", views.model_list, name="model_list"),
    path("models/new/", views.model_create, name="model_create"),
    path("models/<slug:slug>/edit/", views.model_update, name="model_update"),
    path("pricing/", views.pricing_list, name="pricing_list"),
    path("pricing/new/", views.pricing_create, name="pricing_create"),
    path("pricing/<int:pk>/edit/", views.pricing_update, name="pricing_update"),
    path("agents/", views.agent_list, name="agent_list"),
    path("agents/new/", views.agent_create, name="agent_create"),
    path("agents/<slug:slug>/", views.agent_detail, name="agent_detail"),
    path("agents/<slug:slug>/edit/", views.agent_update, name="agent_update"),
    path("ctfs/new/", views.ctf_create, name="ctf_create"),
    path("ctfs/<slug:slug>/", views.ctf_detail, name="ctf_detail"),
    path("ctfs/<slug:slug>/edit/", views.ctf_update, name="ctf_update"),
    path(
        "ctfs/<slug:ctf_slug>/challenges/new/",
        views.challenge_create,
        name="challenge_create",
    ),
    path(
        "ctfs/<slug:ctf_slug>/challenges/<slug:challenge_id>/",
        views.challenge_detail,
        name="challenge_detail",
    ),
    path(
        "ctfs/<slug:ctf_slug>/challenges/<slug:challenge_id>/edit/",
        views.challenge_update,
        name="challenge_update",
    ),
    path(
        "ctfs/<slug:ctf_slug>/challenges/<slug:challenge_id>/threads/new/",
        views.thread_create,
        name="thread_create",
    ),
    path("threads/<uuid:thread_uuid>/", views.thread_detail, name="thread_detail"),
    path("threads/<uuid:thread_uuid>/fork/", views.thread_fork, name="thread_fork"),
    path(
        "threads/<uuid:thread_uuid>/publish/",
        views.thread_publish,
        name="thread_publish",
    ),
    path("threads/<uuid:thread_uuid>/steer/", views.thread_steer, name="thread_steer"),
    path("threads/<uuid:thread_uuid>/stop/", views.thread_stop, name="thread_stop"),
    path(
        "threads/<uuid:thread_uuid>/stream/",
        views.thread_stream,
        name="thread_stream",
    ),
    path(
        "threads/<uuid:thread_uuid>/filetree/",
        views.thread_filetree,
        name="thread_filetree",
    ),
]
