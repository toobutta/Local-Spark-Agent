from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button
from textual.containers import Vertical

class Sidebar(Widget):
    MENU_ITEMS = [
        ("UserProfile", "User Profile"),
        ("ProjectProfiles", "Project Profiles"),
        ("Systems", "Systems & Configurations"),
        ("AgentMgmt", "Agent Management"),
        ("AgentFoundry", "Agent Foundry"),
        ("Customizations", "Customizations"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("MODULES", classes="section-header")
        with Vertical():
            for id, label in self.MENU_ITEMS:
                classes = "nav-item active" if id == "Systems" else "nav-item"
                yield Static(f"│ {label}", id=id, classes=classes)
