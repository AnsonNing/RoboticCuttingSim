#!/usr/bin/env bash
# Start (new) or resume the interactive Claude Code CLI session for the RoboticCuttingSim demos with /goal set.
# Called by runner.sh in pane rcs-demos:claude. Usage: launch.sh <session-id> new|resume
set -u
REPO=/home/ningan/RoboticCuttingSim
export TMPDIR=/mnt/data/ningan/tmp
cd "$REPO" || exit 1
if [ "$2" = new ]; then id=(--session-id "$1"); else id=(--resume "$1"); fi
exec claude "${id[@]}" --model claude-opus-5-5 --effort xhigh --permission-mode bypassPermissions "/goal $(cat "$REPO/tools/demo_runner/goal.txt")"
