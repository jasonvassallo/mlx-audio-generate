#!/bin/bash
# Setup remote server access for the MLX AudioGen plugin.
#
# This script writes the Cloudflare Access service token credentials
# to ~/.mlx-audiogen/config.json so the AU/VST3 plugin can
# fall back to the remote server when local is unavailable.
#
# Prerequisites:
#   1. Create a Service Token in Cloudflare One dashboard:
#      https://one.dash.cloudflare.com → Access controls → Service credentials
#      → Service Tokens → Create Service Token
#      Name: "mlx-audiogen-plugin", Duration: Non-expiring
#
#   2. Add a Service Auth policy to the musicgen Access application:
#      https://one.dash.cloudflare.com → Access controls → Applications
#      → musicgen.djvassallo.com → Policies → Add a policy
#      Policy name: "Plugin Service Token"
#      Action: Service Auth
#      (This allows service tokens to bypass the browser login flow)
#
#   3. Run this script with the client_id and client_secret from step 1.
#
# Usage:
#   ./scripts/setup_plugin_remote.sh <client_id> <client_secret>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <cf_client_id> <cf_client_secret>"
    echo ""
    echo "Get these values from Cloudflare One → Access controls →"
    echo "Service credentials → Service Tokens → Create Service Token"
    exit 1
fi

CLIENT_ID="$1"
CLIENT_SECRET="$2"
CONFIG_DIR="$HOME/.mlx-audiogen"
CONFIG_FILE="$CONFIG_DIR/config.json"

# Validate client_id format (should end with .access)
if [[ ! "$CLIENT_ID" == *.access ]]; then
    echo "Warning: client_id usually ends with '.access'"
    echo "Got: $CLIENT_ID"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    echo "Updating existing config: $CONFIG_FILE"
    # Use python to merge into existing JSON (preserves project_path, uv_path)
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
cfg['remote_url'] = 'https://musicgen.djvassallo.com'
cfg['cf_client_id'] = sys.argv[1]
cfg['cf_client_secret'] = sys.argv[2]
with open('$CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
" "$CLIENT_ID" "$CLIENT_SECRET"
else
    echo "Creating new config: $CONFIG_FILE"
    python3 -c "
import json, sys
cfg = {
    'project_path': '$HOME/Documents/Code/mlx-audiogen',
    'uv_path': 'uv',
    'remote_url': 'https://musicgen.djvassallo.com',
    'cf_client_id': sys.argv[1],
    'cf_client_secret': sys.argv[2]
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
" "$CLIENT_ID" "$CLIENT_SECRET"
fi

chmod 600 "$CONFIG_FILE"

echo ""
echo "Config written to $CONFIG_FILE"
echo "Remote URL: https://musicgen.djvassallo.com"
echo "Client ID:  $CLIENT_ID"
echo "Secret:     (stored)"
echo ""
echo "The plugin will now try local server first, then fall back to remote."
echo "Restart your DAW to pick up the new config."
