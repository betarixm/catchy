from __future__ import annotations

from django import template
from django.forms import BoundField, Form

register = template.Library()


@register.filter(name="getfield")
def getfield(form: Form, name: str) -> BoundField | None:
    if name in form.fields:
        return form[name]
    return None


@register.filter(name="display_status")
def display_status(status: str | None) -> str:
    """Map raw Thread status → user-visible status. Folds 'stopped' into
    'completed' since the distinction is not meaningful to the operator."""
    value = str(status or "").strip()
    if value == "stopped":
        return "completed"
    return value
