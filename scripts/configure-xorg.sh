#!/usr/bin/env bash
# Install only Gestur's Xorg rule; the optional root supports image staging.
set -euo pipefail
GESTUR_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
XORG_DIRECTORY="${2:-/}"
XORG_DIRECTORY="${XORG_DIRECTORY%/}/etc/X11/xorg.conf.d"
XORG_TARGET="$XORG_DIRECTORY/99-gestur-vc4.conf"

case "${1:-}" in
    install)
        install -d -m 755 "$XORG_DIRECTORY"
        XORG_TEMP=$(mktemp "$XORG_DIRECTORY/.gestur-vc4.XXXXXX")
        trap 'rm -f -- "$XORG_TEMP"' EXIT
        install -m 644 "$GESTUR_SOURCE/deployment/99-gestur-vc4.conf" "$XORG_TEMP"
        mv -f -- "$XORG_TEMP" "$XORG_TARGET"
        ;;
    uninstall)
        rm -f -- "$XORG_TARGET"
        ;;
    *) echo "Uso: bash $0 {install|uninstall} [raíz de la imagen]" >&2; exit 1 ;;
esac
