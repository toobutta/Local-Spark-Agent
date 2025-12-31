from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static
from tui.components.header import HeaderWidget
from tui.components.sidebar import Sidebar
from tui.components.systems import SystemsContent
from tui.components.footer import FooterWidget

class SparkPlugTUI(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("f1", "switch_tab('systems')", "Systems"),
        ("f2", "switch_tab('agents')", "Agents"),
        ("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Container(
            HeaderWidget(),
            Sidebar(),
            SystemsContent(),
            # Warp-like Input Bar
            Container(
                Static("➜", id="prompt-icon"),
                Static("Type a command...", id="input-placeholder"),
                id="command-bar"
            ),
            FooterWidget(),
            id="app-grid"
        )

    def action_switch_tab(self, tab: str) -> None:
        # Tab switching logic for TUI
        self.notify(f"CHANNEL {tab.upper()} ACTIVE", title="CHANNEL SWITCH")

if __name__ == "__main__":
    app = SparkPlugTUI()
    app.run()
