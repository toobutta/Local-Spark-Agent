# SparkPlug DGX - Native TUI Client

This is the native Text User Interface (TUI) client for SparkPlug DGX, built with Python and Textual.
It provides a high-performance, keyboard-centric interface for managing your AI infrastructure.

## Requirements

- Python 3.8+
- Terminal with truecolor support (Windows Terminal, iTerm2, Alacritty)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Run the TUI directly from the source:

```bash
python sparkplug_tui.py
```

## Features

- **DGX Spark Management**: Monitor and configure NVIDIA DGX Spark with LPDDR5x unified memory (128GB @ 273GB/s).
- **Warp-like Interface**: Modern command palette and block-based output.
- **Secure Vault**: Manage API keys for Anthropic, OpenAI, and Gemini.
- **Agent Deployment**: Launch and monitor autonomous agents.

## Navigation

- **F1**: Systems Tab
- **F2**: Agents Tab
- **Q**: Quit Application
