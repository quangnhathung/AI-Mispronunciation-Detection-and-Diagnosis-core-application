import uuid


def generate_request_id() -> str:
    return uuid.uuid4().hex[:16]


def generate_session_id() -> str:
    return uuid.uuid4().hex
