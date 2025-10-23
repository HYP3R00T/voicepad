#!/bin/bash

/usr/local/bin/mise trust /workspaces/voicepad/mise.toml && /usr/local/bin/mise install
sudo apt update && sudo apt install -y python3 tmux
