from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Footer, Tabs

from voicepad.ui.components.app_header import AppHeader
from voicepad.ui.components.config_tab import ConfigTab
from voicepad.ui.components.main_tab import MainTab


class VoicepadUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #app-header {
        height: 3;
        dock: top;
    }

    ContentSwitcher {
        height: 1fr;
        width: 1fr;
    }
    """

    def on_mount(self) -> None:
        self.theme = "catppuccin-mocha"

    def compose(self) -> ComposeResult:
        yield AppHeader(id="app-header")

        # Content area with panes that switch based on active tab
        with ContentSwitcher(initial="main"):
            yield MainTab(id="main")
            yield ConfigTab(id="config")

        yield Footer()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Switch content when tab is activated."""
        switcher = self.query_one(ContentSwitcher)
        switcher.current = event.tab.id
