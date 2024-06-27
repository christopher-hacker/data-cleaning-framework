"""Contains constant variables shared by two or more modules."""

SUPPORTED_OUTPUT_TYPES = {
    "csv": [".csv", ".csv.gz"],
    "pkl": [".pkl"],
    "geojson": [".geojson"],
}

APPENDABLE_OUTPUT_TYPES = SUPPORTED_OUTPUT_TYPES["csv"]
