Temporary README. A more sophisticated one will come later on.

# Tkinter Requirement

This application requires **Tkinter**, Python's built-in GUI library. If you see an error like:

`ModuleNotFoundError: No module named '_tkinter'`

it means your Python installation was built without Tk support. This is a separate component from Python itself and must be installed/enabled depending on your OS.

## Fixing it

### Windows
The official [python.org](https://www.python.org/downloads/) installer includes Tk by default.
- Re-run the installer → **Modify** → make sure **"tcl/tk and IDLE"** is checked.

### macOS
- **python.org installer**: Tk support is bundled in already.
- **Homebrew Python**:
```bash
  brew install python-tk
```
- **pyenv**: Tk must be available *before* building Python.
```bash
  brew install tcl-tk
  pyenv install --force <your-python-version>
```

### Linux
Install the Tk package for your distro, then restart your terminal:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora / RHEL
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

## After fixing

If a `.venv` folder already exists, **delete it** and re-run the launcher script — it will be recreated using your now-Tk-enabled Python.

## Verifying it worked

```bash
python -m tkinter
```

This should open a little GUI.
