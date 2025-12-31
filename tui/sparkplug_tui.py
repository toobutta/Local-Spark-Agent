from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static
from tui.components.header import HeaderWidget
from tui.components.sidebar import Sidebar
from tui.components.systems import SystemsContent
from tui.components.footer import FooterWidget

class SparkPlugTUI(App):
    CSS_PATH = "styles.tcss"
    
    def compose(self) -> ComposeResult:
        yield Container(
            HeaderWidget(),
            Sidebar(),
            SystemsContent(),
            FooterWidget(),
            id="app-grid"
        )

if __name__ == "__main__":
    app = SparkPlugTUI()
    app.run()
