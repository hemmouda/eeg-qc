import sys
import traceback
import tkinter as tk
from tkinter import scrolledtext


def show_error_window(exc_text: str) -> None:
    # Close any existing Tk windows first
    try:
        default_root = tk._default_root
        if default_root is not None:
            for w in list(default_root.children.values()):
                try:
                    w.destroy()
                except Exception:
                    pass
            try:
                default_root.destroy()
            except Exception:
                pass
    except Exception:
        pass

    root = tk.Tk()
    root.title("Unexpected Error")
    root.geometry("600x400")

    label = tk.Label(
        root, text="An unexpected error occurred:", font=("Segoe UI", 10, "bold")
    )
    label.pack(anchor="w", padx=10, pady=(10, 0))

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text_area.insert(tk.END, exc_text)
    text_area.configure(state="disabled")
    text_area.pack(fill="both", expand=True, padx=10, pady=10)

    close_btn = tk.Button(root, text="Close", command=root.destroy)
    close_btn.pack(pady=(0, 10))

    root.mainloop()


def main() -> None:
    try:
        from tool.gui.app import run_app

        run_app()

    except Exception:
        exc_text = traceback.format_exc()
        show_error_window(exc_text)
        sys.exit(1)


if __name__ == "__main__":
    main()
