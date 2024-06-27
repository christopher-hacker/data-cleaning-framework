"""Dummy schema module for testing geospatial data."""

import pandera as pa
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries


class Schema(pa.SchemaModel):
    """Pandera schema for the project."""

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration for the schema."""

        strict = True

    name: Series[str] = pa.Field(
        description="The name of the location",
    )
    geometry: GeoSeries = pa.Field(
        description="The geometry of the location",
    )
