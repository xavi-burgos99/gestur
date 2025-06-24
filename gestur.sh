#!/bin/bash

CURRENT_DIR=$(pwd)

# Detectar si estamos en Raspberry Pi 5
is_rpi5() {
    if grep -q "Raspberry Pi 5" /proc/cpuinfo 2>/dev/null; then
        return 0
    fi
    return 1
}

install() {
    echo "Iniciando instalación para Raspberry Pi..."

    # Actualizar el sistema
    sudo apt-get update
    sudo apt-get upgrade -y

    # Instalar firmware y herramientas específicas de Raspberry Pi
    sudo apt-get install -y rpi-update raspi-config

    # Configurar GPU memory split para Raspberry Pi 5
    if is_rpi5; then
        echo "Configurando Raspberry Pi 5..."
        # GPU memory split de 128MB para mejor rendimiento gráfico
        if ! grep -q "gpu_mem=128" /boot/firmware/config.txt; then
            echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt >/dev/null
        fi
        # Habilitar DRM/KMS para mejor compatibilidad con X11
        if ! grep -q "dtoverlay=vc4-kms-v3d" /boot/firmware/config.txt; then
            echo "dtoverlay=vc4-kms-v3d" | sudo tee -a /boot/firmware/config.txt >/dev/null
        fi
        # Configuración adicional para Pi 5
        if ! grep -q "arm_64bit=1" /boot/firmware/config.txt; then
            echo "arm_64bit=1" | sudo tee -a /boot/firmware/config.txt >/dev/null
        fi
    fi

    # Herramientas de compilación
    sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev \
        libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev \
        libexpat1-dev liblzma-dev zlib1g-dev libffi-dev uuid-dev git cmake pkg-config

    # Instalar entorno gráfico (sin lightdm como solicitas)
    sudo apt-get install -y --no-install-recommends \
        xserver-xorg \
        xserver-xorg-video-fbdev \
        xserver-xorg-video-vesa \
        xserver-xorg-video-modesetting \
        xinit \
        openbox \
        mesa-utils \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libgl1-mesa-glx

    # Dependencias específicas para OpenCV y MediaPipe en Raspberry Pi
    sudo apt-get install -y \
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgtk-3-dev \
        libpng-dev \
        libjpeg-dev \
        libopenexr-dev \
        libtiff-dev \
        libwebp-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libatlas-base-dev \
        gfortran \
        libblas-dev \
        liblapack-dev

    # Instalar Python 3 y pip
    sudo apt-get install -y python3 python3-dev python3-pip python3-venv \
        python3-numpy python3-opencv
    python3 -m pip install --upgrade pip setuptools wheel

    # Crear usuario gestur si no existe
    if ! id -u gestur >/dev/null 2>&1; then
        sudo useradd -m -s /bin/bash gestur
        # Añadir usuario a grupos necesarios para acceso a GPU y video
        sudo usermod -a -G video,render,input gestur
    fi

    # Clonar / actualizar el repo
    if [ ! -d /opt/gestur ]; then
        sudo mkdir -p /opt/gestur
    fi

    cd /opt/gestur
    if [ ! -d .git ]; then
        sudo rm -rf /opt/gestur/*
        sudo git clone https://github.com/xavi-burgos99/gestur.git /opt/gestur
    else
        sudo git pull origin main
    fi
    cd "$CURRENT_DIR"

    # Dar propiedad del directorio al usuario gestur
    sudo chown -R gestur:gestur /opt/gestur

    # Crear entorno virtual limpio
    if [ -d /opt/gestur/.venv ]; then
        sudo rm -rf /opt/gestur/.venv
    fi
    sudo python3 -m venv /opt/gestur/.venv
    sudo chown -R gestur:gestur /opt/gestur/.venv
    sudo chmod -R 755 /opt/gestur/.venv
    sudo /opt/gestur/.venv/bin/python -m pip install --upgrade pip setuptools wheel

    # Dependencias de Python (con versiones compatibles para ARM)
    echo "Instalando dependencias de Python optimizadas para Raspberry Pi..."
    sudo /opt/gestur/.venv/bin/python -m pip install \
        numpy \
        opencv-python-headless \
        mediapipe \
        panda3d

    # Configurar autologin en tty1
    sudo mkdir -p /etc/systemd/system/getty@tty1.service.d
    sudo bash -c "cat > /etc/systemd/system/getty@tty1.service.d/override.conf" <<'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin gestur --noclear %I $TERM
EOF

    # SOLUCIÓN ESPECÍFICA para el error de framebuffer en Raspberry Pi 5
    sudo mkdir -p /etc/X11/xorg.conf.d
    sudo bash -c "cat > /etc/X11/xorg.conf.d/99-vc4.conf" <<'EOF'
Section "OutputClass"
    Identifier "vc4"
    MatchDriver "vc4"
    Driver "modesetting"
    Option "PrimaryGPU" "true"
EndSection
EOF

    # Configuración adicional para optimización de rendimiento
    sudo bash -c "cat > /etc/X11/xorg.conf.d/99-raspi.conf" <<'EOF'
Section "Device"
    Identifier "Raspberry Pi Graphics"
    Driver "modesetting"
    Option "AccelMethod" "glamor"
    Option "DRI" "3"
EndSection

Section "ServerLayout"
    Identifier "Default Layout"
    Option "BlankTime" "0"
    Option "StandbyTime" "0"
    Option "SuspendTime" "0"
    Option "OffTime" "0"
EndSection
EOF

    # Añadir arranque automático en bash_profile
    PROFILE_FILE=/home/gestur/.bash_profile
    sudo -u gestur touch "${PROFILE_FILE}"
    if ! grep -q "Gestur autostart" "${PROFILE_FILE}"; then
        sudo bash -c "cat >> ${PROFILE_FILE}" <<'EOF'
# Gestur autostart
if [ "$(tty)" = "/dev/tty1" ]; then
    # Configurar variables de entorno para hardware acelerado
    export DISPLAY=:0
    export LIBGL_ALWAYS_SOFTWARE=0
    exec startx /home/gestur/run.sh -- :0 vt1 -keeptty
fi
EOF
        sudo chmod 644 "${PROFILE_FILE}"
    fi

    # Crear script de inicio
    START_SCRIPT=/home/gestur/run.sh
    sudo -u gestur mkdir -p "$(dirname "${START_SCRIPT}")"
    sudo bash -c "cat > ${START_SCRIPT}" <<'EOF'
#!/bin/bash
# Configurar entorno para Raspberry Pi
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=0

# Arranca Openbox y el visualizador a pantalla completa
cd /opt/gestur
openbox-session &

# Esperar a que Openbox esté listo
sleep 2

# Ejecutar el visualizador
/opt/gestur/.venv/bin/python /opt/gestur/visualizer.py capitell.obj
EOF

    sudo chmod +x "${START_SCRIPT}"
    sudo chown gestur:gestur "${START_SCRIPT}"

    # Recargar systemd y reiniciar getty
    sudo systemctl daemon-reload
    sudo systemctl restart getty@tty1

    echo "=============================================="
    echo "Instalación completada para Raspberry Pi 5."
    echo "El visualizador se iniciará automáticamente."
    echo "Reinicia el sistema para aplicar todos los cambios."
    echo "=============================================="
}

uninstall() {
    # Deshacer autologin en tty1
    if [ -f /etc/systemd/system/getty@tty1.service.d/override.conf ]; then
        sudo rm -f /etc/systemd/system/getty@tty1.service.d/override.conf
        sudo systemctl daemon-reload
        sudo systemctl restart getty@tty1
    fi

    # Retirar configuraciones X11 personalizadas
    if [ -f /etc/X11/xorg.conf.d/99-vc4.conf ]; then
        sudo rm -f /etc/X11/xorg.conf.d/99-vc4.conf
    fi
    if [ -f /etc/X11/xorg.conf.d/99-raspi.conf ]; then
        sudo rm -f /etc/X11/xorg.conf.d/99-raspi.conf
    fi

    # Retirar bloque de autostart del bash_profile
    PROFILE_FILE=/home/gestur/.bash_profile
    if [ -f "${PROFILE_FILE}" ]; then
        sudo sed -i '/# Gestur autostart/,/fi/d' "${PROFILE_FILE}"
    fi

    # Borrar /opt/gestur
    if [ -d /opt/gestur ]; then
        sudo rm -rf /opt/gestur
    fi

    # Eliminar usuario gestur
    if id -u gestur >/dev/null 2>&1; then
        sudo userdel -r gestur 2>/dev/null
    fi

    echo "Gestur desinstalado por completo."
}

# Comprobaciones iniciales
if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse como root. Usa 'sudo'."
    exit 1
fi

case "$1" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Uso: $0 {install|uninstall}"
        exit 1
        ;;
esac
