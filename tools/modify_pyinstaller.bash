#!/bin/bash

# PyInstallerをカスタマイズしてから再インストールする
# 良いGPUが自動的に選択されるようにしている
# https://github.com/VOICEVOX/voicevox_engine/issues/502

# 自前ビルドすることでブートローダーのハッシュ値が変わってウイルス判定を回避する効果もあるかも

set -eux

pyinstaller_version=$(uv run pyinstaller -v)
tempdir=$(mktemp -dt modify_pyinstaller.XXXXXXXX)
trap 'rm -rf "$tempdir"' EXIT
git clone https://github.com/pyinstaller/pyinstaller.git "$tempdir" -b "v$pyinstaller_version" --depth 1
cat > "$tempdir/bootloader/src/symbols.c" << EOF
#ifdef _WIN32
#include <windows.h>

// https://docs.nvidia.com/gameworks/content/technologies/desktop/optimus.htm
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;

// https://gpuopen.com/learn/amdpowerxpressrequesthighperformance/
__declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001;
#endif
EOF
(cd "$tempdir/bootloader" && python ./waf all --msvc_targets="x64")
uv pip install -U "$tempdir" # TODO: 可能であればuv addでやる
