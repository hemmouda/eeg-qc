import json
from datetime import datetime
from pathlib import Path

from ...struct.recording import Recording, RecordingQuality
from ..util.report_utils import get_unique_dated_filename
from .report_data_generator import build_report_data

RES_DIR = Path(__file__).parent / "res"


def generate_html_report(recording: Recording, location: Path) -> None:
    """Generates and saves the HTML report for the given recording in the given location."""

    if recording.quality is RecordingQuality.BLACK:
        html = _build_black_report_html(recording)
    else:
        html = _build_full_report_html(recording)

    out_path = get_unique_dated_filename(location, f"{recording.file_path.name}.html")
    out_path.write_text(html, encoding="utf-8")


def _build_full_report_html(recording: Recording) -> str:
    """Builds the full (RED/YELLOW/GREEN) HTML report by injecting the
    recording data, CSS, and JS into template.html."""

    data = build_report_data(recording)
    data_json = json.dumps(data)

    css = (RES_DIR / "style.css").read_text(encoding="utf-8")
    js = (RES_DIR / "script.js").read_text(encoding="utf-8")
    template = (RES_DIR / "template.html").read_text(encoding="utf-8")

    return (
        template.replace("{{STYLE}}", css)
        .replace("{{SCRIPT}}", js)
        .replace("{{DATA_JSON}}", data_json)
        .replace("{{FILENAME}}", data["filename"])
        .replace("{{QUALITY}}", data["quality"])
        .replace("{{QUALITY_UP}}", data["quality_up"])
        .replace("{{NOW}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    )


def _build_black_report_html(recording: Recording) -> str:
    """Builds the minimal BLACK-recording HTML report"""

    template = (RES_DIR / "black_template.html").read_text(encoding="utf-8")

    return (
        template.replace("{{FILENAME}}", recording.file_path.name)
        .replace("{{JUSTIFICATION}}", recording.quality_justification)
        .replace("{{NOW}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    )


# Fair "disclaimer": HTML report was also claude-ed.
