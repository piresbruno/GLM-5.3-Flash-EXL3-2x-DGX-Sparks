#!/usr/bin/env bash
# install-ops-units.sh — install the R1 ops-kit user units for this checkout.
#
# Installs glm53-{xid-check,metrics-alert,check-updates}.{service,timer} into
# ~/.config/systemd/user with @REPO_DIR@ substituted for this checkout.
#
# Default: install + daemon-reload ONLY (nothing enabled). The operators
# enable explicitly (R1 Phase 4 requires operator approval):
#   local/install-ops-units.sh --enable
# and remove everything with:
#   local/install-ops-units.sh --remove
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODE="install"
[ "${1:-}" = "--enable" ] && MODE="enable"
[ "${1:-}" = "--remove" ] && MODE="remove"

UNITS=(glm53-xid-check glm53-metrics-alert glm53-check-updates)

if [ "$MODE" = "remove" ]; then
    for u in "${UNITS[@]}"; do
        systemctl --user disable --now "$u.timer" 2>/dev/null || true
        rm -f "$UNIT_DIR/$u.service" "$UNIT_DIR/$u.timer"
    done
    systemctl --user daemon-reload
    echo "ops units removed"
    exit 0
fi

mkdir -p "$UNIT_DIR"
for u in "${UNITS[@]}"; do
    sed "s|@REPO_DIR@|$REPO|g" "$SELF_DIR/systemd/$u.service" > "$UNIT_DIR/$u.service"
    cp "$SELF_DIR/systemd/$u.timer" "$UNIT_DIR/$u.timer"
done
systemctl --user daemon-reload
echo "installed ${#UNITS[@]} unit pairs into $UNIT_DIR"

if [ "$MODE" = "enable" ]; then
    for u in "${UNITS[@]}"; do
        systemctl --user enable --now "$u.timer"
        echo "enabled: $u.timer"
    done
    echo "NOTE: enablement is operator-approved (R1 Phase 4). Verify with: systemctl --user list-timers 'glm53-*'"
else
    echo "installed but NOT enabled — run with --enable after operator review, or:"
    echo "  systemctl --user enable --now glm53-xid-check.timer glm53-metrics-alert.timer glm53-check-updates.timer"
fi
