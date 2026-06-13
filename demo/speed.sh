#!/usr/bin/env bash
# Post-process the raw VHS capture into the final demo assets.
#
#   ./demo/speed.sh [speed] [trim_to_seconds]
#
# The local 30b model runs in real time (~2 min), which is too long and too
# heavy for a portfolio clip. This speeds the raw capture up (default 2.5x) and
# regenerates a scaled, low-fps gif. Pass a second arg to trim the raw to N
# seconds first (use it to cut the static tail after task_complete).
#
#   vhs demo.tape          # writes coding-agent-demo-raw.mp4
#   ./demo/speed.sh 2.5 120
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=coding-agent-demo-raw.mp4
SPEED=${1:-2.5}
TRIM=${2:-}
PTS=$(python3 -c "print(1/$SPEED)")

[ -f "$RAW" ] || { echo "missing $RAW — run 'vhs demo.tape' first" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
src="$work/src.mp4"
if [ -n "$TRIM" ]; then
  ffmpeg -y -loglevel error -i "$RAW" -t "$TRIM" -c copy "$src"
else
  cp "$RAW" "$src"
fi

# Final mp4: sped up, silent, web-friendly.
ffmpeg -y -loglevel error -i "$src" -filter:v "setpts=${PTS}*PTS" \
  -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart coding-agent-demo.mp4

# Final gif: sped up, 720px wide, 8 fps, 64-color palette for a light README embed.
ffmpeg -y -loglevel error -i "$src" -filter_complex \
  "[0:v]setpts=${PTS}*PTS,fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  coding-agent-demo.gif

echo "wrote coding-agent-demo.mp4 ($(du -h coding-agent-demo.mp4 | cut -f1)) and coding-agent-demo.gif ($(du -h coding-agent-demo.gif | cut -f1))"
