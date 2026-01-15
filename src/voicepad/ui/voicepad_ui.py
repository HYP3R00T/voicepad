from textual.app import App, ComposeResult
from textual.widgets import Footer, Header


class VoicepadUI(App):
    def on_mount(self) -> None:
        self.theme = "catppuccin-mocha"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
