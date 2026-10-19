#!/usr/bin/env bash
# Install GPU drivers so this app can see and use the card.
#
# Written to be distro-agnostic on purpose. The three vendors need different
# packages and different repositories, and the package manager differs too, so
# this detects both rather than assuming Debian.
#
# Defaults to a dry run and prints the exact commands. Nothing is installed
# unless you pass --yes, and root is only requested at that point.
#
#   ./scripts/install-gpu-drivers.sh                 # show what would happen
#   ./scripts/install-gpu-drivers.sh --vendor amd    # just the AMD path
#   ./scripts/install-gpu-drivers.sh --yes           # actually do it
#   ./scripts/install-gpu-drivers.sh --groups-only   # just add render/video
#
# Supported: Debian/Ubuntu/Mint, Fedora/RHEL/Rocky, Arch, openSUSE, Alpine.

set -euo pipefail

VENDOR="auto"
ASSUME_YES=0
GROUPS_ONLY=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --vendor) VENDOR="${2:-}"; shift 2 ;;
    --vendor=*) VENDOR="${1#*=}"; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --dry-run) ASSUME_YES=0; shift ;;
    --groups-only) GROUPS_ONLY=1; shift ;;
    -h|--help) usage ;;
    *) red "unknown option: $1"; usage ;;
  esac
done

# ---------------------------------------------------------------- detection

detect_distro() {
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ID="${ID:-unknown}"
    ID_LIKE="${ID_LIKE:-}"
    NAME="${PRETTY_NAME:-$ID}"
  else
    ID="unknown"; ID_LIKE=""; NAME="unknown"
  fi

  case "$ID" in
    debian|ubuntu|linuxmint|pop|elementary) FAMILY="debian" ;;
    fedora|rhel|centos|rocky|almalinux)      FAMILY="fedora" ;;
    arch|manjaro|endeavouros)                FAMILY="arch" ;;
    opensuse*|sles)                          FAMILY="suse" ;;
    alpine)                                  FAMILY="alpine" ;;
    *)
      case "$ID_LIKE" in
        *debian*) FAMILY="debian" ;;
        *fedora*|*rhel*) FAMILY="fedora" ;;
        *arch*) FAMILY="arch" ;;
        *suse*) FAMILY="suse" ;;
        *) FAMILY="unknown" ;;
      esac ;;
  esac
  DISTRO_ID="$ID"; DISTRO_NAME="$NAME"; DISTRO_FAMILY="$FAMILY"
}

detect_pm() {
  for c in apt-get dnf yum pacman zypper apk; do
    if command -v "$c" >/dev/null 2>&1; then PM="$c"; return
    fi
  done
  PM=""
}

# Which GPU vendor is this? Kernel sysfs is more reliable than lspci being
# installed, and works without any extra packages.
detect_gpu() {
  DETECTED="none"
  if [ -d /sys/class/drm/card0/device/vendor ] || [ -e /dev/nvidia0 ]; then
    DETECTED="nvidia"
  fi
  # 0x1002 is AMD's PCI vendor id.
  for d in /sys/class/drm/card*/device/vendor; do
    [ -r "$d" ] || continue
    if [ "$(cat "$d" 2>/dev/null)" = "0x1002" ]; then DETECTED="amd"; break; fi
  done
  [ -e /dev/kfd ] && [ "$DETECTED" = "none" ] && DETECTED="amd"
  [ -e /dev/nvidiactl ] && DETECTED="nvidia"
  # Explicit success: under `set -e` a trailing failed test would otherwise
  # make this function's return value fatal and abort the script.
  return 0
}

have() { command -v "$1" >/dev/null 2>&1; }

run() {
  if [ "$ASSUME_YES" -eq 1 ]; then
    bold "+ $*"
    "$@"
  else
    dim "+ $*"
  fi
}

as_root() {
  # Only escalate for the real command, and say so plainly.
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    if [ "$ASSUME_YES" -eq 1 ]; then
      sudo "$@"
    else
      dim "+ sudo $*"
    fi
  fi
}

# ------------------------------------------------------------------- report

detect_distro
detect_pm
detect_gpu

bold "GPU driver installer"
printf '  distro : %s (%s)\n' "$DISTRO_NAME" "$DISTRO_FAMILY"
printf '  package: %s\n' "${PM:-none found}"
printf '  gpu    : %s\n' "$DETECTED"
echo

if [ "$VENDOR" != "auto" ] && [ "$VENDOR" != "$DETECTED" ] && [ "$DETECTED" != "none" ]; then
  red "You asked for $VENDOR but this machine has $DETECTED."
  dim "Continuing with $VENDOR as requested."
  echo
fi

if [ "$PM" = "" ]; then
  red "No supported package manager found (apt/dnf/pacman/zypper/apk)."
  echo "Install the driver by hand for your distribution; the vendor docs cover it:"
  dim "  AMD    https://rocm.docs.amd.com/projects/install-on-linux/"
  dim "  NVIDIA https://docs.nvidia.com/datacenter/tesla/tesla-installation-linux-x86.html"
  exit 1
fi

# -------------------------------------------------------------- video groups

if [ "$GROUPS_ONLY" -eq 1 ]; then
  bold "Adding your user to the video/render groups"
  dim "This is the fix when ROCm says 'Unable to open /dev/kfd'."
  if [ "$(id -un)" = "root" ]; then target="root"; else target="$(id -un)"; fi
  if have usermod; then
    for g in video render; do
      as_root usermod -aG "$g" "$target"
    done
    green "done — log out and back in for it to take effect"
  else
    red "usermod not found; add yourself to video and render manually."
  fi
  exit 0
fi

# ------------------------------------------------------------- amd / rocm

install_amd() {
  bold "AMD: installing a GPU reader"
  case "$DISTRO_FAMILY" in
    debian)  as_root apt-get update; as_root apt-get install -y radeontop ;;
    fedora)  as_root dnf install -y radeontop ;;
    arch)    as_root pacman -S --needed --noconfirm radeontop ;;
    suse)    as_root zypper --non-interactive install radeontop ;;
    alpine)  red "Alpine ships no radeontop. Use the amdgpu sysfs counters instead."; return 1 ;;
    *) red "Unknown family $DISTRO_FAMILY for the AMD path."; return 1 ;;
  esac
  dim ""
  dim "radeontop reads the amdgpu counters directly, so it needs neither a ROCm"
  dim "install nor /dev/kfd access. If you want to RUN models on the GPU as"
  dim "well, you still need ROCm — install it from:"
  dim "  https://rocm.docs.amd.com/projects/install-on-linux/"
  green "GPU reader installed."
}

# ---------------------------------------------------------------- nvidia

# Which kernel module actually got installed?
#
# Ubuntu's `nvidia-driver` is a metapackage of a few kilobytes that pulls in
# everything else, and which variant it resolves to depends on what the vendor
# repo currently publishes — on Ubuntu 24.04 with the CUDA repo it is the *open*
# GPU kernel module, not the proprietary one. This banner used to call it the
# proprietary driver, which was a claim about a choice this script never made:
# it asked for a metapackage and apt picked the rest. So ask dpkg afterwards
# rather than assert before.
nvidia_module_variant() {
  if dpkg-query -W -f='${Status}' nvidia-dkms-open 2>/dev/null | grep -q "ok installed"; then
    printf 'open GPU kernel module (nvidia-dkms-open)'
  elif dpkg-query -W -f='${Status}' nvidia-dkms 2>/dev/null | grep -q "ok installed"; then
    printf 'proprietary kernel module (nvidia-dkms)'
  else
    printf 'kernel module variant this script does not recognise'
  fi
}

install_nvidia() {
  bold "NVIDIA: installing the driver"
  case "$DISTRO_FAMILY" in
    debian)
      as_root apt-get update
      # The distro driver tracks the kernel; prefer it over the CUDA repo,
      # which needs a GPG key and a static version.
      as_root apt-get install -y nvidia-driver
      # Only now is the variant knowable, and only if something was actually
      # installed — in a dry run this would describe the machine's current
      # driver as though the line above had just put it there. Worth saying
      # plainly when it is real, because the two are not interchangeable:
      # NVIDIA's open module covers Turing and newer (RTX 20xx, T4 and up) and
      # does not drive Maxwell, Pascal or Volta.
      if [ "$ASSUME_YES" -eq 1 ]; then
        printf '  installed: %s\n' "$(nvidia_module_variant)"
      fi
      ;;
    fedora)
      as_root dnf install -y xorg-x11-drv-nvidia-cuda kernel-devel
      ;;
    arch)
      as_root pacman -S --needed --noconfirm nvidia-dkms
      ;;
    suse)
      as_root zypper --non-interactive install nvidia-drivers
      ;;
    alpine)
      red "Alpine ships no NVIDIA driver; it is not supported."
      return 1
      ;;
    *) red "Unknown family $DISTRO_FAMILY for the NVIDIA path."; return 1 ;;
  esac
  green "NVIDIA driver installed."
}

# ------------------------------------------------------------------- groups

add_groups() {
  bold "Adding your user to video/render"
  if ! have usermod; then
    dim "usermod not found; skipping"
    return
  fi
  target="root"
  if [ "$(id -u)" -ne 0 ]; then target="$(id -un)"; fi
  for g in video render; do
    as_root usermod -aG "$g" "$target"
  done
  green "done — log out and back in, then re-run this script to verify."
}

# -------------------------------------------------------------------- main

case "$VENDOR" in
  auto)
    case "$DETECTED" in
      amd)    install_amd ;;
      nvidia) install_nvidia ;;
      none)   red "No GPU detected. Nothing to install."; exit 1 ;;
      *)      red "Unknown GPU vendor."; exit 1 ;;
    esac
    ;;
  amd)    install_amd ;;
  nvidia) install_nvidia ;;
  *)      red "--vendor must be auto, amd or nvidia"; exit 2 ;;
esac

add_groups

echo
if [ "$ASSUME_YES" -eq 0 ]; then
  bold "That was a dry run. Re-run with --yes to actually do it:"
  dim "  $0 --vendor $VENDOR --yes"
else
  green "Installed."
  dim "Reboot for a DKMS-built driver, then:"
  dim "  rocm-smi          # AMD"
  dim "  nvidia-smi        # NVIDIA"
  dim "Then restart this app — it reads GPU usage through those tools."
fi