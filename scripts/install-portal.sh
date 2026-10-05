#!/bin/bash
# Run as root after the base runtime installer: scripts/install-portal.sh /opt/gestur [--hostname name]
set -euo pipefail
umask 022
export DEBIAN_FRONTEND=noninteractive
export PIP_NO_INPUT=1
exec </dev/null
INSTALL_ROOT=/opt/gestur
if [[ $# -gt 0 && "$1" != --* ]]; then
    INSTALL_ROOT=$1
    shift
fi
TASK_HOSTNAME=''
if [[ $# -gt 0 ]]; then
    if [[ $# != 2 || "$1" != --hostname || ! "$2" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
        echo 'Usa --hostname con 1–63 letras minúsculas, números o guiones, sin .local ni guiones en los extremos.' >&2
        exit 1
    fi
    TASK_HOSTNAME=$2
fi
if [[ $(id -u) != 0 ]]; then
    echo 'Ejecuta este instalador como root.' >&2
    exit 1
fi
if [[ "$INSTALL_ROOT" != /opt/gestur ]]; then
    echo 'La instalación del portal requiere /opt/gestur.' >&2
    exit 1
fi
if [[ $(uname -s) != Linux ]]; then
    echo 'El portal de dispositivo se instala en Raspberry Pi OS.' >&2
    exit 1
fi
TASK_NODE_TEMP=''
TASK_BUILD=''
TASK_SUDOERS=''
cleanup() {
    [[ -z "$TASK_NODE_TEMP" ]] || rm -rf "$TASK_NODE_TEMP"
    [[ -z "$TASK_BUILD" ]] || rm -rf "$TASK_BUILD"
    [[ -z "$TASK_SUDOERS" ]] || rm -f "$TASK_SUDOERS"
}
trap cleanup EXIT
apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update
apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends network-manager dnsmasq-base avahi-daemon python3-dbus sudo ca-certificates curl xz-utils rfkill \
    assimp-utils bubblewrap util-linux
systemctl enable --now NetworkManager
systemctl enable --now avahi-daemon
# Prefer the distribution package when sufficiently recent. Otherwise install an
# exact upstream release, checking the official SHA256 manifest (no remote shell).
TASK_EXISTING_NODE=/usr/bin/node
if [[ -x /opt/gestur-node/bin/node ]]; then
    TASK_EXISTING_NODE=/opt/gestur-node/bin/node
fi
if ! "$TASK_EXISTING_NODE" -e 'let [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||a===22&&b>=12?0:1)' >/dev/null 2>&1 || \
    { [[ "$TASK_EXISTING_NODE" == /opt/gestur-node/bin/node ]] && ! env PATH="/opt/gestur-node/bin:$PATH" /opt/gestur-node/bin/npm --version >/dev/null 2>&1; }; then
    NODE_VERSION=v22.23.3
    case $(dpkg --print-architecture) in
        arm64)
            NODE_ARCH=arm64
            ;;
        amd64)
            NODE_ARCH=x64
            ;;
        *)
            echo 'Se requiere Raspberry Pi OS de 64 bits.' >&2
            exit 1
            ;;
    esac
    TASK_NODE_TEMP=$(mktemp -d)
    NODE_ARCHIVE="node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz"
    curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "https://nodejs.org/dist/${NODE_VERSION}/${NODE_ARCHIVE}" -o "$TASK_NODE_TEMP/$NODE_ARCHIVE"
    curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "https://nodejs.org/dist/${NODE_VERSION}/SHASUMS256.txt" -o "$TASK_NODE_TEMP/SHASUMS256.txt"
    (
        cd "$TASK_NODE_TEMP"
        awk -v file="$NODE_ARCHIVE" '$2 == file { print }' SHASUMS256.txt > selected.sha256
        test -s selected.sha256
        sha256sum --check selected.sha256
    )
    mkdir -p /opt/gestur-node
    tar -xJf "$TASK_NODE_TEMP/$NODE_ARCHIVE" -C /opt/gestur-node --strip-components=1 --no-same-owner
fi
if [[ -x /opt/gestur-node/bin/node ]]; then
    NODE_BIN=/opt/gestur-node/bin/node
    BUILD_PATH=/opt/gestur-node/bin:/usr/local/bin:/usr/bin:/bin
else
    NODE_BIN=/usr/bin/node
    BUILD_PATH=/usr/local/bin:/usr/bin:/bin
    if ! command -v npm >/dev/null; then
        apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 \
            -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y npm
    fi
fi
getent group gestur >/dev/null || groupadd --system gestur
getent group gestur-portal >/dev/null || groupadd --system gestur-portal
id gestur-portal >/dev/null 2>&1 || useradd --system --gid gestur-portal --groups gestur --home-dir /var/lib/gestur --no-create-home --shell /usr/sbin/nologin gestur-portal
usermod -a -G gestur gestur-portal
# Conversion stays local and is isolated from device data/network. Verify the
# distribution permits unprivileged namespaces before accepting model uploads.
runuser -u gestur-portal -- bwrap --unshare-all --die-with-parent \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib \
    --ro-bind-try /lib64 /lib64 --proc /proc --dev /dev --tmpfs /tmp /usr/bin/true || {
    echo 'No se pudo aislar el conversor 3D. Revisa el soporte de espacios de nombres de usuario (bubblewrap) en esta imagen.' >&2
    exit 1
}
install -d -o gestur -g gestur -m 2770 /var/lib/gestur /var/lib/gestur/models
install -d -o root -g root -m 755 /etc/gestur /usr/local/libexec
if [[ ! -f /var/lib/gestur/config.json ]]; then
    TASK_CONFIG=$(mktemp /var/lib/gestur/.config.XXXXXX)
    install -o gestur -g gestur -m 660 "$INSTALL_ROOT/config/default.json" "$TASK_CONFIG"
    mv -T "$TASK_CONFIG" /var/lib/gestur/config.json
fi
# An old token identifies an existing installation, even after interrupted
# upgrades. New installs create no token, so retries stay pending onboarding.
TASK_FRESH=true
if [[ -L /etc/gestur/portal-token ]]; then
    echo 'La clave anterior no puede ser un enlace.' >&2
    exit 1
fi
if [[ -e /etc/gestur/portal-token ]]; then
    TASK_FRESH=false
    chown root:gestur-portal /etc/gestur/portal-token
    chmod 640 /etc/gestur/portal-token
fi
install -o root -g root -m 644 "$INSTALL_ROOT/config/default.json" /etc/gestur/default.json
install -o root -g root -m 755 "$INSTALL_ROOT/scripts/gestur-wifi.py" /usr/local/libexec/gestur-wifi
install -o root -g root -m 755 "$INSTALL_ROOT/scripts/gestur-device.py" /usr/local/libexec/gestur-device
TASK_SUDOERS=$(mktemp)
printf '%s\n' 'gestur-portal ALL=(root) NOPASSWD: /usr/local/libexec/gestur-wifi status, /usr/local/libexec/gestur-wifi apply' > "$TASK_SUDOERS"
visudo -cf "$TASK_SUDOERS"
install -o root -g root -m 440 "$TASK_SUDOERS" /etc/sudoers.d/gestur-wifi
printf '%s\n' 'gestur-portal ALL=(root) NOPASSWD: /usr/local/libexec/gestur-device status, /usr/local/libexec/gestur-device hostname, /usr/local/libexec/gestur-device onboarding, /usr/local/libexec/gestur-device reset' > "$TASK_SUDOERS"
visudo -cf "$TASK_SUDOERS"
install -o root -g root -m 440 "$TASK_SUDOERS" /etc/sudoers.d/gestur-device
# Build as an unprivileged account; application/helper files remain root-owned.
# Retired illustrations are archived in docs; upgrades must not republish the
# copies left by the previous installer in the public directory.
rm -f "$INSTALL_ROOT/portal/public/motion-icons/head-base.png" \
    "$INSTALL_ROOT/portal/public/motion-icons/hand-base.png"
TASK_BUILD=$(mktemp -d)
cp -a "$INSTALL_ROOT/portal/." "$TASK_BUILD/"
chown -R gestur-portal:gestur-portal "$TASK_BUILD"
runuser -u gestur-portal -- env PATH="$BUILD_PATH" npm_config_cache="$TASK_BUILD/.npm-cache" bash -c 'set -euo pipefail; cd "$1"; npm ci --ignore-scripts --no-audit --no-fund; npm run build; npm prune --omit=dev --ignore-scripts --no-audit --no-fund' _ "$TASK_BUILD"
rm -rf "$INSTALL_ROOT/portal/node_modules" "$INSTALL_ROOT/portal/dist"
cp -a "$TASK_BUILD/node_modules" "$TASK_BUILD/dist" "$INSTALL_ROOT/portal/"
chown -R root:root "$INSTALL_ROOT/portal"
# Check final dependencies early. ExecStartPre repeats this under the complete
# service sandbox; a successful runuser check alone does not prove that works.
runuser -u gestur-portal -- env PATH="$BUILD_PATH" GESTUR_PYTHON="$INSTALL_ROOT/.venv/bin/python" \
    "$NODE_BIN" "$INSTALL_ROOT/scripts/check-importer.mjs"
sed -e "s|^ExecStart=/usr/bin/node |ExecStart=$NODE_BIN |" -e "s|^ExecStartPre=/usr/bin/node |ExecStartPre=$NODE_BIN |" "$INSTALL_ROOT/deployment/gestur-portal.service" > /etc/systemd/system/gestur-portal.service
chown root:root /etc/systemd/system/gestur-portal.service
chmod 644 /etc/systemd/system/gestur-portal.service
# This creates an open GESTUR-XXXX AP only when no prior AP was configured.
# Passwords and SSID of an existing AP are retained, including on reinstall.
rfkill unblock wifi
/usr/local/libexec/gestur-wifi bootstrap
/usr/bin/python3 -c 'import json, sys; print(json.dumps({"fresh": sys.argv[1] == "true", **({"hostname": sys.argv[2]} if sys.argv[2] else {})}))' \
    "$TASK_FRESH" "$TASK_HOSTNAME" | /usr/local/libexec/gestur-device bootstrap
systemctl daemon-reload
systemctl enable --now gestur-portal
systemctl restart gestur-portal
echo 'Portal instalado: http://10.42.0.1 (o IP actual del dispositivo, sin indicar puerto).'
echo "También disponible por mDNS: http://$(hostname -s).local"
echo 'En una instalación nueva, abre el portal para completar la configuración inicial.'
