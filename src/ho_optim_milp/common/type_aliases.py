"""Type aliases for commonly used types in the configuration module."""

from typing import Annotated
from pydantic import StrictInt, StrictFloat, Field

StrictNonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0)]
StrictNonNegativeInt = Annotated[StrictInt, Field(ge=0)]
StrictPositiveFloat = Annotated[StrictFloat, Field(gt=0.0)]
StrictPositiveInt = Annotated[StrictInt, Field(gt=0)]
StrictProb = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
