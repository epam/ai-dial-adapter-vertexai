import proto
from google.protobuf import json_format
from proto.marshal.collections import maps, repeated


def message_to_string(message: proto.Message) -> str:
    return json_format.MessageToJson(message._pb)


def message_to_dict(message: proto.Message) -> dict:
    return json_format.MessageToDict(message._pb)


# Borrowed from helper functions
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/function_calling_data_structures.ipynb
def recurse_proto_repeated_composite(repeated_object):
    repeated_list = []
    for item in repeated_object:
        if isinstance(item, repeated.RepeatedComposite):
            item = recurse_proto_repeated_composite(item)
            repeated_list.append(item)
        elif isinstance(item, maps.MapComposite):
            item = recurse_proto_marshal_to_dict(item)
            repeated_list.append(item)
        else:
            repeated_list.append(item)

    return repeated_list


def recurse_proto_marshal_to_dict(marshal_object):
    new_dict = {}
    for k, v in marshal_object.items():
        if not v:
            continue
        elif isinstance(v, maps.MapComposite):
            v = recurse_proto_marshal_to_dict(v)
        elif isinstance(v, repeated.RepeatedComposite):
            v = recurse_proto_repeated_composite(v)
        new_dict[k] = v

    return new_dict
