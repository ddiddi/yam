#!/bin/zsh
# Builds tools/uvc-util (github.com/jtfrey/uvc-util, MIT) - sets C270 exposure on macOS through IOKit, no root.
# OpenCV cannot set exposure on macOS and uvcc (npm/libusb) hangs on these cameras. Used by position_series.sh:
#   tools/uvc-util -L 0x00110000 -s auto-exposure-mode=1 -s exposure-time-abs=140   (cam1)
set -e
cd "$(dirname "$0")"
src=${TMPDIR:-/tmp}/uvc-util-src
[ -d $src ] || git clone --depth 1 https://github.com/jtfrey/uvc-util.git $src
cd $src/src
clang -o "$OLDPWD/uvc-util" -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m
echo "built tools/uvc-util"; "$OLDPWD/uvc-util" -d
