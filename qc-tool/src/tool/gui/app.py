import tkinter as tk
from tkinter import messagebox

from .frames.select_files_frame import SelectFilesFrame
from .utils.image_locator import get_image_path


class App(tk.Tk):
    """Main application window. Owns the current frame and acts as the
    parent for any pop-up windows (help, error dialogs, etc.)."""

    def __init__(self):
        super().__init__()
        self.title("QC Tool")
        self.no_gc = []

        self.current_frame: tk.Frame | None = None
        self.show_frame(SelectFilesFrame)

    def show_frame(self, frame_class, **kwargs):
        """Destroys the current frame and replaces it with a new one."""

        if self.current_frame is not None:
            self.current_frame.destroy()

        self.current_frame = frame_class(self, **kwargs)
        self.current_frame.pack(fill="both", expand=True)

        self.geometry("")
        self.update_idletasks()

    def show_critical_error(self, title: str, message: str):
        """Central place to surface exceptions/errors as a pop-up. Exits afterwards."""

        messagebox.showerror(title, message, parent=self)
        self.destroy()
        exit()

    def show_error(self, title: str, message: str):
        """Central place to surface exceptions/errors as a pop-up"""

        messagebox.showerror(title, message, parent=self)


def run_app():
    app = App()

    logo = tk.PhotoImage(file=get_image_path("logo"))
    app.no_gc.append(logo)
    app.iconphoto(True, logo)

    app.mainloop()


# Fair "disclaimer": Most TK code was claude-ed.
