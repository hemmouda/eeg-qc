import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import TYPE_CHECKING

from .tool_tip import ToolTip
from ..utils.image_locator import get_image_path, get_traffic_light_image_path_for
from ...qc.struct.recording import Recording, RecordingQuality
from ...qc.report_generator.report_generator import generate_reports

if TYPE_CHECKING:
    from ..app import App


class ResultsFrame(tk.Frame):
    """Final frame: summary + scrollable list of processed recordings,
    each with its quality, file name, and an individual report download."""

    MIN_LIST_HEIGHT = 200

    def __init__(self, master: "App", recordings: list[Recording]):
        super().__init__(master)
        self.master: "App" = master
        self.recordings = recordings
        self.no_gc = []  # Items that shouldn't be GC-ed

        self._build_ui()

    def _build_ui(self):
        # Summary row: count + download-all button
        summary_row = tk.Frame(self)
        summary_row.pack(fill="x", padx=(10, 30), pady=10)

        tk.Label(summary_row, text=f"Processed {len(self.recordings)} files").pack(
            side="left"
        )

        download_icon = tk.PhotoImage(file=get_image_path("download")).subsample(2, 2)
        self.no_gc.append(download_icon)
        download_all_button = tk.Label(summary_row, image=download_icon, cursor="hand2")
        download_all_button.pack(side="right")
        download_all_button.bind("<Button-1>", lambda e: self._on_download_all())
        self.no_gc.append(
            ToolTip(
                download_all_button,
                f"Download reports for all {len(self.recordings)} files.",
            )
        )

        # Scrollable list of recordings
        list_container = tk.Frame(self)
        list_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(
            list_container, height=self.MIN_LIST_HEIGHT, highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            list_container, orient="vertical", command=canvas.yview
        )
        self.list_frame = tk.Frame(canvas)

        self.list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        list_frame_id = canvas.create_window(
            (0, 0), window=self.list_frame, anchor="nw"
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfig(list_frame_id, width=e.width)
        )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for recording in self.recordings:
            self._add_recording_row(recording)

    def _add_recording_row(self, recording: Recording):
        row = tk.Frame(self.list_frame)
        row.pack(fill="x", expand=True, pady=2)

        quality_icon = tk.PhotoImage(
            file=get_traffic_light_image_path_for(recording.quality)
        ).subsample(2, 2)
        self.no_gc.append(quality_icon)
        quality_label = tk.Label(row, image=quality_icon, cursor="question_arrow")
        quality_label.pack(side="left", padx=(0, 5))
        self.no_gc.append(
            ToolTip(quality_label, self._get_quality_tool_tip_text(recording))
        )

        tk.Label(row, text=recording.file_path.name).pack(side="left", padx=(0, 5))

        download_icon = tk.PhotoImage(file=get_image_path("download")).subsample(2, 2)
        self.no_gc.append(download_icon)
        download_button = tk.Label(row, image=download_icon, cursor="hand2")
        download_button.pack(side="right", padx=(0, 5))
        download_button.bind(
            "<Button-1>", lambda e, r=recording: self._on_download_single(r)
        )
        self.no_gc.append(
            ToolTip(download_button, f"Download {recording.file_path.name}'s reports.")
        )

    def _get_quality_tool_tip_text(self, recording: Recording) -> list[str]:
        if recording.quality is RecordingQuality.BLACK:
            return [
                f"File: {recording.file_path.name}",
                f"Quality: {recording.quality.name}",
                f"Quality justification: {recording.quality_justification}",
            ]
        else:
            return [
                f"File: {recording.file_path.name}",
                f"Patient ID: {recording.subject_identifier}",
                f"Patient name: {recording.last_name}, {recording.first_name}",
                f"Recording date: {recording.recording_date}",
                f"Recording duration: {recording.formatted_duration}",
                f"Quality: {recording.quality.name}",
                f"Quality justification: {recording.quality_justification}",
            ]

    def _on_download_all(self):
        directory = filedialog.askdirectory(title="Select location to save reports")
        if directory:
            self._download_reports(self.recordings, Path(directory))

    def _on_download_single(self, recording: Recording):
        directory = filedialog.askdirectory(title="Select location to save report")
        if directory:
            self._download_reports([recording], Path(directory))

    def _download_reports(self, recordings: list[Recording], directory: Path):
        self._open_in_file_explorer(directory)

        for recording in recordings:
            generate_reports(recording, directory)

    def _open_in_file_explorer(self, directory: Path):
        if sys.platform == "win32":
            subprocess.run(["explorer", str(directory)])
        elif sys.platform == "darwin":
            subprocess.run(["open", str(directory)])
        else:
            subprocess.run(["xdg-open", str(directory)])
