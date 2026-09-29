from typing import cast

import proto
from google.protobuf import json_format


def message_to_string(message: proto.Message) -> str:
    return json_format.MessageToJson(message._pb)


def message_to_dict(message: proto.Message) -> dict:
    return cast(dict, json_format.MessageToDict(message._pb))
