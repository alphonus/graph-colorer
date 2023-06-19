#!/bin/sh
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYTHONCMD="python $@"
nix-shell "$SCRIPTPATH"/shell.nix --run "$PYTHONCMD"
