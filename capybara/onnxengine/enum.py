from enum import Enum

from ..enums import EnumCheckMixin


class Backend(EnumCheckMixin, Enum):
    cpu = 0
    cuda = 1
    coreml = 2
