# MLX Audio Generate — Max for Live Device

Thin Node for Max HTTP client for generating music directly inside Ableton Live.

## Setup

1. **Build the M4L device** (one-time, see [Building the .amxd](#building-the-amxd) below)

2. **Start the server** (in a terminal):
   ```bash
   uv run mlx-audiogen-server --weights-dir ./converted/musicgen-small --port 8420
   ```

3. **Load the M4L device** in Ableton:
   - Create a new MIDI track
   - Drag the built `.amxd` device onto the track
   - The device connects to `localhost:8420` by default

4. **Generate audio**:
   - Type a prompt in the text field
   - Click "Generate" or send a `generate` message
   - The WAV is auto-saved and the path is output for drag-to-track

## Max Messages

| Message | Example | Description |
|---------|---------|-------------|
| `generate <prompt>` | `generate happy rock song` | Start generation |
| `model <type>` | `model stable_audio` | Switch model |
| `seconds <n>` | `seconds 10` | Set duration |
| `temperature <n>` | `temperature 0.8` | Creativity (musicgen) |
| `guidance <n>` | `guidance 5.0` | Text adherence |
| `seed <n>` | `seed 42` | Reproducibility (-1=random) |
| `style_audio <path>` | `style_audio /path/ref.wav` | Style reference |
| `melody <path>` | `melody /path/input.wav` | Melody reference |
| `server <host:port>` | `server 127.0.0.1:8420` | Server address |

## Outputs

| Outlet | Data | Description |
|--------|------|-------------|
| `status` | text | Human-readable status |
| `progress` | 0-100 | Generation progress |
| `audio` | filepath | Path to generated WAV |
| `error` | text | Error message |
| `models` | JSON | Available models list |

## Building the .amxd

The `.amxd` device is built in Max:
1. Create a new Max for Live Audio Effect
2. Add a `node.script mlx-audiogen.js` object
3. Route the outlets to UI elements (live.text, live.dial, etc.)
4. Freeze and save as `mlx-audiogen.amxd`
