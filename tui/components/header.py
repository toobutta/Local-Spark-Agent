from textual.widget import Widget
from textual.widgets import Static, Select, Label

class HeaderWidget(Widget):
    def compose(self):
        yield Label("SPARKPLUG ADMIN", id="header-title")
        yield Label("PROJECT = SparkPlug DGX", id="project-indicator")
        yield Select(
            options=[
                ("main", "Main Cluster (Default)"),
                ("research", "Research Node Alpha"),
                ("web", "Web Services Delta"),
            ],
            value="main",
            allow_blank=False
        )
        yield Label("ADMIN MODE", id="admin-badge")
