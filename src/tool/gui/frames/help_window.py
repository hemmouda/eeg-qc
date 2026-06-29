import tkinter as tk

from ...consts import VERSION_STR, URL, CONTACT

_HELP_TEXT = f"""This simple QC tool is meant to allow the quick evaluation of the quality of HST EEG files.

First, click the 'Select EDF Files' button to choose which HST EDF files you want to evaluate. The tool will then automatically process them and show a list with the results.

You can hover over the quality traffic-light icon in the result window to see a quick summary about the file and the justification behind the attributed quality.

The download button at the top in the result window allows you to download all the reports in the location of your choosing. Or, alternatively, the individual download buttons let you download individual reports.

In the first window, and before choosing the EDF files you want to evaluate, you can select a different configuration file that you want to use for the evaluation rules by click on the gear icon.
The currently chosen configuration file name is shown next to the gear icon. By default, this is the default configuration file that comes shipped with the tool. Said default configuration file must always exist. But feel free to edit it.
Information about the format of the configuration file and what it accepts can be found in the configuration file itself, or in the tool's website.

Please know that the "quality" of the regrouped epochs in the HTML report are based on the number of epochs within that group that have satisfied at least 1 rule. No epoch satisfies any rule -> GREEN, 1 epoch -> YELLOW, 2 or 3 -> RED. Know that this coloring just provides an easy visual indication of the epochs that have satisfied a rule, the quality of the channel is based on the percentage provided in the config file.

If you want to "uninstall"/remove the tool, it suffices to delete the folder containing the RUNME scripts. The tool does not create or modify any files outside that folder.

Do please report any bugs or unexpected behavior that you encountered during your usage of the tool. Thank you.

Website: {URL}
Version: {VERSION_STR}
Contact: {CONTACT}

"""


class HelpWindow(tk.Toplevel):
    """Pop-up help window. Not a frame swap — independent window."""

    def __init__(self, master):
        super().__init__(master)
        self.title(f"Help")
        self.geometry("400x300")

        text = tk.Text(self, wrap="word")
        text.insert(
            "1.0",
            _HELP_TEXT,
        )
        text.config(state="disabled")
        text.pack(fill="both", expand=True, padx=10, pady=10)
