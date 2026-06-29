import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import TYPE_CHECKING

from .progress_frame import ProgressFrame
from .help_window import HelpWindow
from .tool_tip import ToolTip
from ..utils.default_config_file_locator import get_default_config_file_path
from ..utils.image_locator import get_image_path

from ...qc.config_reader import read_config_file
from ...qc.struct.config import Config

if TYPE_CHECKING:
    from ..app import App


class SelectFilesFrame(tk.Frame):
    """First frame: file selection (browse), config file indicator,
    and a help button."""

    def __init__(self, master: "App"):
        super().__init__(master)
        self.master: "App" = master
        self.no_gc = []  # Items that shouldn't be GC-ed

        self.config_file_label = None
        self.config_file_path: Path = get_default_config_file_path()
        # Check that default config file is valid
        config_file, err = read_config_file(self.config_file_path)
        if config_file is None:
            self.master.show_critical_error(
                "Invalid Configuration File",
                f"The default configuration file ({self.config_file_path.name}) is invalid:\n\n{err}",
            )
            return
        self.config_file: Config = config_file

        self._build_ui()

    def _build_ui(self):
        # Select files button
        select_button = ttk.Button(
            self,
            text="Select EDF Files",
            command=self._on_select_edf_files,
            cursor="hand2",
        )
        select_button.pack(pady=10, padx=10, ipady=4)

        # Config row: gear icon + config file name
        config_row = tk.Frame(self)
        config_row.pack(fill="x", padx=10, pady=10)

        gear_icon = tk.PhotoImage(file=get_image_path("gear")).subsample(2, 2)
        self.no_gc.append(gear_icon)
        gear_button = tk.Label(config_row, image=gear_icon, cursor="hand2")
        gear_button.pack(side="left", padx=(0, 5))
        gear_button.bind("<Button-1>", lambda e: self._on_select_config_file())
        self.no_gc.append(
            ToolTip(gear_button, "Select a different configuration file.")
        )

        self.config_file_label = tk.Label(config_row, text=self.config_file_path.name)
        self.config_file_label.pack(side="left")

        # Help button
        help_icon = tk.PhotoImage(file=get_image_path("help")).subsample(2, 2)
        self.no_gc.append(help_icon)
        help_button = tk.Label(self, image=help_icon, cursor="hand2")
        help_button.pack(side="right", padx=5, pady=5)
        help_button.bind("<Button-1>", lambda e: self._on_help())
        self.no_gc.append(ToolTip(help_button, "Get help."))

    def _on_select_edf_files(self):
        files = filedialog.askopenfilenames(
            title="Select EDF files",
            filetypes=[("EDF files", "*.edf *.EDF")],
        )

        if files:
            self.master.show_frame(
                ProgressFrame,
                files=[Path(f) for f in files],
                config=self.config_file,
            )

    def _on_select_config_file(self):
        file = filedialog.askopenfilename(
            title="Select a YAML configuration file",
            filetypes=[("YAML files", "*.yaml *.yml")],
        )

        if file:
            config_file, err = read_config_file(file)
            if config_file is None:
                self.master.show_error(
                    "Invalid Configuration File",
                    f"The selected configuration file is invalid:\n\n{err}",
                )
                return

            self.config_file = config_file
            self.config_file_path = Path(file)
            self.config_file_label.config(text=self.config_file_path.name)

    def _on_help(self):
        HelpWindow(self.master)
