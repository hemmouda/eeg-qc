import tkinter as tk
from tkinter import ttk
import threading
import queue
from pathlib import Path
from typing import TYPE_CHECKING

from .results_frame import ResultsFrame

from ...qc.quality_control import determine_qualities
from ...qc.struct.config import Config
from ...qc.struct.recording import Recording

if TYPE_CHECKING:
    from ..app import App


class ProgressFrame(tk.Frame):
    """Shows progress while determine_qualities runs on a background thread."""

    def __init__(self, master: "App", files: list[Path], config: Config):
        super().__init__(master)
        self.master: "App" = master
        self.files = files
        self.config = config
        self.total = len(files)
        self.completed = 0
        self.recordings: list[Recording] = []

        self._queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._start_processing()
        self._poll_queue()

    def _build_ui(self):
        self.status_label = tk.Label(self, text=f"Processing 0 / {self.total} files...")
        self.status_label.pack(padx=10, pady=(10, 5))

        self.progress_bar = ttk.Progressbar(
            self,
            orient="horizontal",
            length=300,
            mode="determinate",
            maximum=self.total,
        )
        self.progress_bar.pack(padx=10, pady=(0, 10))

    def _start_processing(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        # Runs on the background thread. Only touches self._queue, never
        # the GUI directly, since Tkinter widgets aren't thread-safe.
        def on_progress(recording: Recording):
            self._queue.put(recording)

        try:
            determine_qualities(
                self.files,
                self.config,
                progress_callback=on_progress,
            )
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(None)  # sentinel: all done

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()

                if item is None:
                    self._on_finished()
                    return

                if isinstance(item, Exception):
                    self.master.show_critical_error(
                        "Processing Failed",
                        f"Something went wrong while processing files:\n\n{item}",
                    )
                    return

                self._on_recording_done(item)

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    def _on_recording_done(self, recording: Recording):
        self.completed += 1
        self.recordings.append(recording)
        self.progress_bar["value"] = self.completed
        self.status_label.config(
            text=f"Processing {self.completed} / {self.total} files..."
        )

    def _on_finished(self):
        self.master.show_frame(ResultsFrame, recordings=self.recordings)
