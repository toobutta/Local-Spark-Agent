from textual.widget import Widget
from textual.widgets import Static
from rich.spinner import Spinner
from rich.text import Text
import random

class FooterWidget(Widget):
    def on_mount(self):
        self.set_interval(0.1, self.refresh_equalizer)

    def refresh_equalizer(self):
        self.query_one("#equalizer", Static).update(self.get_equalizer_bars())

    def get_equalizer_bars(self):
        bars = " ▂▃▄▅▆▇█"
        return "".join(random.choice(bars) for _ in range(8))

    def compose(self):
        yield Static("F1: SYSTEMS | F2: AGENTS", classes="subtitle")
        yield Static("▶ RUN #23 - AUTONOMY: HIGH - STATUS: ACTIVE", id="now-playing")
        yield Static(" ▂▃▄▅▆▇█", id="equalizer")
