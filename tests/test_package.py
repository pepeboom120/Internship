from stem_analytics import __version__
from stem_analytics.cli import build_parser


def test_package_exposes_version_and_cli() -> None:
    assert __version__ == "0.1.0"
    assert build_parser().prog == "stem-analytics"
