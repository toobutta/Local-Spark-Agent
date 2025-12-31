# Installation & Execution Guide

## Option 1: Direct Execution (via PowerShell/Terminal)
SparkPlug is designed to run directly in your existing terminal environment, providing a seamless CLI experience similar to Claude Code or Factory Droid.

```powershell
# Install via pip
pip install sparkplug-dgx

# Run directly
sparkplug
```

## Option 2: Binary Installation (Warp.dev Style)
For a native application experience, download the pre-compiled binary for your system.

1. **Download:** Get the latest release from the [Releases](https://github.com/sparkplug/releases) page.
2. **Path:** Add the binary to your system PATH.
3. **Launch:** Run `sparkplug` from any terminal or use the desktop shortcut.

## Option 3: Developer Mode (Local Clone)
```bash
git clone https://github.com/replit/sparkplug-dgx.git
cd sparkplug-dgx
pip install -r requirements.txt
python tui/sparkplug_tui.py
```