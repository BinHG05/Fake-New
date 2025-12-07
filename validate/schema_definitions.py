# schema_definitions.py
import json

CORE_SCHEMA = {
    "type": "object",
    "required": ["id", "timestamp", "label", "raw_text", "media_url", "user_id"],
    "properties": {
        "id": {"type": "string"},
        "timestamp": {"type": "integer"},
        "label": {"type": "string"},
        "raw_text": {"type": "string"},
        "media_url": {"type": "string"},
        "user_id": {"type": "string", "pattern": "^[A-Za-z0-9_]{1,15}$"}, # Bổ sung: Kiểm tra format/độ dài user_id (Lỗi 11)
        "retweet_count": {"type": "integer", "minimum": 0}
    },
    "additionalProperties": False
}

EXTENDED_SCHEMA = {
    "type": "object",
    "required": ["id", "timestamp", "label", "clean_text", "image_info", "text_features"], 
    "properties": {
        "id": {"type": "string"},
        "timestamp": {"type": "integer"},
        "label": {"type": "string"},
        "clean_text": {"type": "string"},
        
        "text_features": {
            "type": "object",
            "required": ["word_count", "has_caps_lock", "sentiment_score"],
            "properties": {
                "word_count": {"type": "integer", "minimum": 0},
                "has_caps_lock": {"type": "boolean"},
                "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0}
            }
        },
        
        "image_info": {
            "type": "object",
            "required": ["processed_path", "image_size", "is_video"],
            "properties": {
                "processed_path": {"type": "string"},
                "image_size": { 
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2
                },
                "is_video": {"type": "boolean"},
                "keyframe_paths": {"type": "array", "items": {"type": "string"}}
            }
        },
        
        "user_features": {"type": "object"},
        "graph_features": {"type": "object"}
    },
    "additionalProperties": True
}