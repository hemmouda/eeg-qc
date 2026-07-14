from pathlib import Path

from ..struct.recording import Recording

from ..report_generator.csv.csv_report import generate_csv_report
from ..report_generator.html.html_report import generate_html_report


def generate_reports(recording: Recording, location: Path) -> None:
    """Generates and saves the CSV and HTML report for the given recording in the given location."""

    generate_csv_report(recording, location)
    generate_html_report(recording, location)
