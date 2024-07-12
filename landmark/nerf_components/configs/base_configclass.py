import sys

from pydantic import BaseModel, ConfigDict, PydanticUserError


class ConfigClass(BaseModel):
    """Base Config"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except PydanticUserError as e:
            print(e)
            print("Please check your config class.")
            sys.exit(1)

    def check_args(self):
        pass
