"""Code rendering for inferred type definitions."""

from __future__ import annotations

from typing import Any


def render_types(schema: dict[str, Any], root_name: str = "StudyItem") -> str:
    """Render dataclass code from an inferred schema."""

    classes: dict[str, list[tuple[str, str, bool, bool]] | None] = {}

    def type_name(node: dict[str, Any], context: str) -> str:
        if node.get("type") == "object":
            class_name = "".join(part[:1].upper() + part[1:] for part in context.split("_")) or root_name
            while class_name in classes:
                class_name = f"{class_name}Item"
            classes[class_name] = None
            fields = []
            for key, value in node.get("fields", {}).items():
                fields.append(
                    (
                        key,
                        type_name(value, key),
                        value.get("nullable", False),
                        value.get("is_json_string", False),
                    )
                )
            classes[class_name] = fields
            return class_name
        if node.get("type") == "list":
            return f"list[{type_name(node.get('item_type', {}), context + '_item')}]"
        if node.get("type") == "primitive":
            return str(node.get("python_type", "Any"))
        return "Any"

    type_name(schema, root_name)
    lines = [
        '"""Auto-generated type definitions for CARP data."""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "from dataclasses import dataclass",
        "from typing import Any",
        "",
        "",
        "def parse_json_field(value: Any) -> Any:",
        '    """Parse JSON-like string fields when possible."""',
        "",
        "    if not isinstance(value, str):",
        "        return value",
        "    try:",
        "        return json.loads(value)",
        "    except json.JSONDecodeError:",
        "        return value",
        "",
    ]
    for class_name, fields in classes.items():
        lines.extend(["@dataclass(slots=True)", f"class {class_name}:", f'    """Generated dataclass for `{class_name}`."""'])
        if not fields:
            lines.extend(["    pass", ""])
            continue
        for name, annotation, nullable, _ in fields:
            type_hint = f"{annotation} | None" if nullable else annotation
            safe_name = f"{name}_" if name in {"class", "from", "type"} else name
            lines.append(f"    {safe_name}: {type_hint} = None")
        lines.extend(
            [
                "",
                "    @classmethod",
                "    def from_dict(cls, obj: Any) -> Any:",
                '        """Build an instance from a dictionary."""',
                "",
                "        if not isinstance(obj, dict):",
                "            return obj",
                "        instance = cls()",
            ]
        )
        for name, annotation, _, is_json in fields:
            safe_name = f"{name}_" if name in {"class", "from", "type"} else name
            base_type = annotation.removeprefix("list[").removesuffix("]")
            lines.append(f"        value = obj.get('{name}')")
            if is_json:
                lines.append("        value = parse_json_field(value)")
            if annotation.startswith("list[") and base_type in classes:
                lines.extend(
                    [
                        "        if isinstance(value, list):",
                        f"            value = [{base_type}.from_dict(item) for item in value]",
                    ]
                )
            elif base_type in classes:
                lines.extend(["        if value is not None:", f"            value = {base_type}.from_dict(value)"])
            lines.append(f"        instance.{safe_name} = value")
        lines.extend(["        return instance", ""])
    return "\n".join(lines)
