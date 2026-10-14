from __future__ import annotations

from typing import Any, Dict, Optional

from .config import deep_get


_MISSING = object()
_FIELDS = {
    "segment": {"segments_messages", "conversation_id", "session_ids", "session_num_segments", "messages"},
    "message": {"role", "content", "time", "order"},
    "question": {"question", "question_id", "question_type", "question_date", "answer", "answer_options", "correct_answer_label", "dataset"},
}


class DatasetSchema:
    """Map source JSON fields to the pipeline's canonical names without editing data."""

    def __init__(self, dataset_config: Optional[Dict[str, Any]] = None):
        config = dataset_config or {}
        mapping = config.get("field_mapping", {})
        if not isinstance(mapping, dict):
            raise ValueError("dataset.field_mapping must be a mapping")
        unknown_sections = set(mapping) - set(_FIELDS)
        if unknown_sections:
            raise ValueError(f"Unknown field_mapping sections: {sorted(unknown_sections)}")
        self.fields: Dict[str, Dict[str, str]] = {}
        for section, allowed in _FIELDS.items():
            fields = mapping.get(section, {})
            if not isinstance(fields, dict):
                raise ValueError(f"dataset.field_mapping.{section} must be a mapping")
            unknown = set(fields) - allowed
            if unknown:
                raise ValueError(f"Unknown field_mapping.{section} fields: {sorted(unknown)}")
            for field, path in fields.items():
                if not isinstance(path, str) or not path.strip() or any(not part for part in path.split(".")):
                    raise ValueError(f"dataset.field_mapping.{section}.{field} must be a nonempty dotted field path")
            self.fields[section] = dict(fields)
        roles = config.get("role_mapping", {})
        if not isinstance(roles, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in roles.items()
        ):
            raise ValueError("dataset.role_mapping must map role names to nonempty strings")
        self.roles = {k.strip().lower(): v.strip().lower() for k, v in roles.items()}

    def read(self, obj: Dict[str, Any], section: str, field: str, default: Any = None) -> Any:
        path = self.fields[section].get(field, field)
        return deep_get(obj, path, default)

    def question_meta(self, obj: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(defaults)
        for field in self.fields["question"]:
            result[field] = self.read(obj, "question", field)
        options = result.get("answer_options")
        if options is not None and not isinstance(options, list):
            raise ValueError("The mapped answer_options field must contain a list")
        result["answer_options"] = list(options or [])
        return result

    def segment_sample(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        segments = self.read(obj, "segment", "segments_messages", _MISSING)
        path = self.fields["segment"].get("segments_messages", "segments_messages")
        if segments is _MISSING:
            raise ValueError(f"Segment field '{path}' is missing; configure dataset.field_mapping.segment.segments_messages")
        if not isinstance(segments, list):
            raise ValueError(f"Segment field '{path}' must contain a list of segments")
        normalized = []
        for segment_index, segment in enumerate(segments):
            if "messages" in self.fields["segment"]:
                segment = self.read(segment, "segment", "messages", _MISSING)
            if not isinstance(segment, list):
                raise ValueError(
                    f"Segment {segment_index} in '{path}' must be a list of messages; "
                    "for segment objects, configure dataset.field_mapping.segment.messages"
                )
            messages = []
            for message_index, message in enumerate(segment):
                if not isinstance(message, dict):
                    raise ValueError(f"Segment {segment_index}, message {message_index} must be an object")
                mapped = dict(message)
                for field in self.fields["message"]:
                    value = self.read(message, "message", field, _MISSING)
                    if value is _MISSING and field in {"role", "content"}:
                        field_path = self.fields["message"][field]
                        raise ValueError(f"Segment {segment_index}, message {message_index}: field '{field_path}' for {field} is missing")
                    mapped[field] = None if value is _MISSING else value
                if self.roles:
                    role = str(mapped.get("role") or "").strip().lower()
                    mapped["role"] = self.roles.get(role, mapped.get("role"))
                messages.append(mapped)
            normalized.append(messages)
        result = dict(obj)
        result["segments_messages"] = normalized
        for field in ("conversation_id", "session_ids", "session_num_segments"):
            result[field] = self.read(obj, "segment", field)
        return result
