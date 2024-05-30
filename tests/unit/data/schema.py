"""Dummy schema module for testing."""

import pandera as pa
from pandera.typing import Series


class Schema(pa.SchemaModel):
    """Pandera schema for the project."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for the schema."""

        strict = True

    my_field: Series[int] = pa.Field(
        description="A description of the field",
        ge=1,  # the field must be greater than or equal to 1
    )
