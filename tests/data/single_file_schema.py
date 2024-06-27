"""Dummy schema for testing that single-file-specific cleaners work."""

import pandera as pa
from pandera.typing import Series


class Schema(pa.SchemaModel):
    """Pandera schema for the project."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for the schema."""

        strict = True

    col1: Series[str] = pa.Field(
        description="A description of the field",
        nullable=False,
    )
    col2: Series[str] = pa.Field(
        description="A description of the field",
        nullable=False,
    )
