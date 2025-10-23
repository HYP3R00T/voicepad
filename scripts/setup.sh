#!/bin/bash

/usr/local/bin/mise trust /workspaces/devcontainer-python-template/mise.toml && /usr/local/bin/mise install
sudo apt update && sudo apt install -y python3 tmux
