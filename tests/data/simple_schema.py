"""Dummy schema module for testing."""

import pandera as pa
from pandera.typing import Series


class Schema(pa.SchemaModel):
    """Pandera schema for the project."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for the schema."""

        strict = True

    col1: Series[int] = pa.Field(
        description="A description of the field",
        ge=0,
    )
    col2: Series[int] = pa.Field(
        description="Another description of the field",
        ge=0,
    )
