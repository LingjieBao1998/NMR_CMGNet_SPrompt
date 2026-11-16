from django import template
import json

register = template.Library()

@register.filter
def div(value, arg):
    """Divide value by arg"""
    try:
        return float(value) / float(arg) if float(arg) != 0 else 0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def mul(value, arg):
    """Multiply value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def sub(value, arg):
    """Subtract arg from value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def add(value, arg):
    """Add arg to value"""
    try:
        return float(value) + float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def js_bool(value):
    """Convert Python boolean to JavaScript boolean"""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value).lower()

@register.filter
def to_json(value):
    """Convert Python object to JSON-safe JavaScript"""
    # Convert Python bool to JavaScript bool
    json_str = json.dumps(value)
    # Replace Python true/false with JavaScript true/false
    return json_str.replace('True', 'true').replace('False', 'false')

@register.filter
def range_filter(value):
    """Generate a range from 0 to value-1 for use in template loops"""
    try:
        return range(int(value))
    except (ValueError, TypeError):
        return []

@register.filter
def add_int(value, arg):
    """Add arg to value and return as integer"""
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        return 0