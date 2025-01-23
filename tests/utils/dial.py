import json
from typing import Mapping


def get_extra_headers(region: str | None) -> Mapping[str, str]:
    return (
        {"x-upstream-extra-data": json.dumps({"region": region})}
        if region is not None
        else {}
    )
