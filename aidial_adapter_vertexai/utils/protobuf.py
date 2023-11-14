import proto
from google.protobuf import json_format


def print_proto_message(message: proto.Message):
    return json_format.MessageToJson(message._pb)
