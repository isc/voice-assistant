#!/bin/bash
# Voice assistant control script
# Usage: ./ctl.sh {start|stop|restart|reload|status|logs}

SERVICE="com.voice-assistant.server"
DOMAIN="gui/$(id -u)"
LOG="/tmp/voice-assistant.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$SCRIPT_DIR/$SERVICE.plist"

generate_plist() {
    cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$SCRIPT_DIR/run.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>/tmp/voice-assistant.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/voice-assistant.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF
}

wait_for_ready() {
    local timeout=${1:-30}
    local start=$(date +%s)
    while true; do
        if grep -q "Server running" "$LOG" 2>/dev/null; then
            echo "Ready"
            return 0
        fi
        if grep -q "Fatal error" "$LOG" 2>/dev/null; then
            echo "Failed (check logs)"
            return 1
        fi
        if (( $(date +%s) - start >= timeout )); then
            echo "Timeout after ${timeout}s (check logs)"
            return 1
        fi
        sleep 0.5
    done
}

wait_for_reload() {
    local timeout=${1:-10}
    local start=$(date +%s)
    while true; do
        if grep -q "Reload complete" "$LOG" 2>/dev/null; then
            echo "Reload complete"
            return 0
        fi
        if (( $(date +%s) - start >= timeout )); then
            echo "Timeout after ${timeout}s (check logs)"
            return 1
        fi
        sleep 0.5
    done
}

case "${1:-status}" in
    start)
        > "$LOG"  # Clear log so we detect fresh "Server running"
        launchctl bootstrap "$DOMAIN" "$HOME/Library/LaunchAgents/$SERVICE.plist" 2>/dev/null \
            || launchctl kickstart "$DOMAIN/$SERVICE"
        wait_for_ready
        ;;
    stop)
        launchctl kill SIGTERM "$DOMAIN/$SERVICE" 2>/dev/null
        echo "Stopped"
        ;;
    restart)
        > "$LOG"
        launchctl kickstart -k "$DOMAIN/$SERVICE"
        wait_for_ready
        ;;
    reload)
        # Truncate log so we detect fresh "Reload complete"
        > "$LOG"
        launchctl kill SIGHUP "$DOMAIN/$SERVICE"
        wait_for_reload
        ;;
    status)
        if launchctl print "$DOMAIN/$SERVICE" 2>/dev/null | grep -q "state = running"; then
            PID=$(launchctl print "$DOMAIN/$SERVICE" 2>/dev/null | grep "pid =" | awk '{print $3}')
            echo "Running (PID $PID)"
        else
            echo "Not running"
        fi
        ;;
    logs)
        tail -f "$LOG"
        ;;
    install)
        generate_plist
        cp "$PLIST" "$HOME/Library/LaunchAgents/$SERVICE.plist"
        > "$LOG"
        launchctl bootstrap "$DOMAIN" "$HOME/Library/LaunchAgents/$SERVICE.plist"
        wait_for_ready
        ;;
    uninstall)
        launchctl bootout "$DOMAIN/$SERVICE" 2>/dev/null
        rm -f "$HOME/Library/LaunchAgents/$SERVICE.plist"
        echo "Uninstalled"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|status|logs|install|uninstall}"
        exit 1
        ;;
esac
