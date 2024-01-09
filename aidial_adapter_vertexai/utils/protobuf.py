import proto
from google.protobuf import json_format


def print_proto_message(message: proto.Message) -> str:
    return json_format.MessageToJson(message._pb)


def message_to_dict(message: proto.Message) -> dict:
    return json_format.MessageToDict(message._pb)
