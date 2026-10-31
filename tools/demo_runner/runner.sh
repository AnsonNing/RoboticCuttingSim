#!/usr/bin/env bash
# RoboticCuttingSim unattended runner. Interactive Claude runs in tmux window rcs-demos:claude, so you can attach and type
# into it (tmux attach -t rcs-demos; detach with Ctrl-b d). This loop runs in window rcs-demos:runner and every
# minute looks at the claude pane:
#  - claude not running: resume the session (launch.sh sets /goal from goal.txt), or start a new one if resume keeps failing;
#  - usage limit (5-hour or weekly) on screen: send "繼續" once an hour until it works;
#  - API/network error, or idle for 1 h: send "繼續".
# It never sends while claude is busy, or while you are attached and the screen changed in the last 15 min.
# Start:  tmux new-session -d -s rcs-demos -n runner -c ~/RoboticCuttingSim 'bash tools/demo_runner/runner.sh; exec bash'
# Stop:   touch $STATE/STOP (the claude window keeps running; nothing new is started or sent).
set -u
REPO=/home/ningan/RoboticCuttingSim
STATE=/mnt/data/ningan/lab_records/rcs_demo_runner
S=rcs-demos
P=$S:claude
TICK=60; LIMIT_RETRY=3600; ERR_RETRY=600; IDLE_NUDGE=3600; USER_QUIET=900
MSG='（runner 自動訊息）請接著做：看 docs/PROGRESS.md 最上方「進行中」與 tools/demo_runner/goal.txt 指定的計畫，從停下的地方繼續；記得隨時 commit 並 push。'
mkdir -p "$STATE"
cd "$REPO" || exit 1

say() { echo "$(date '+%F %T') $*" | tee -a "$STATE/runner.log"; }
online() { [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' https://api.anthropic.com/)" != 000 ]; }
launch() {  # $1 new|resume
  local sid
  if [ "$1" = new ]; then sid=$(python3 -c 'import uuid; print(uuid.uuid4())'); echo "$sid" > "$STATE/current_sid"
  else sid=$(cat "$STATE/current_sid"); fi
  tmux respawn-pane -k -t "$P" "bash $REPO/tools/demo_runner/launch.sh $sid $1"
  say "launch $1 $sid"; launched=$(date +%s); last_send=$launched
}
send() {
  tmux send-keys -t "$P" -l "$MSG" && sleep 1 && tmux send-keys -t "$P" Enter
  last_send=$(date +%s); say "sent continue ($1)"
}

tmux has-session -t $S 2>/dev/null || tmux new-session -d -s $S -n runner
tmux list-windows -t $S -F '#W' | grep -qx claude || tmux new-window -d -t $S -n claude
tmux set-option -w -t "$P" remain-on-exit on >/dev/null

say "runner start (pid $$)"
last_send=0; launched=0; fast_fails=0; last_hash=; same_since=$(date +%s); state=
while :; do
  now=$(date +%s)
  [ -e "$STATE/DONE" ] && { say "DONE present: plan finished, runner exits"; exit 0; }
  [ -e "$STATE/STOP" ] && { say "STOP present: runner exits"; exit 0; }

  if [ "$(tmux display -p -t "$P" '#{pane_dead}')" = 1 ] || [ "$(tmux display -p -t "$P" '#{pane_current_command}')" != claude ]; then
    if ! online; then [ "$state" = offline ] || say "claude not running, API unreachable: waiting"; state=offline; sleep $TICK; continue; fi
    if [ ! -s "$STATE/current_sid" ]; then launch new
    else
      if [ $((now - launched)) -lt 300 ]; then fast_fails=$((fast_fails + 1)); else fast_fails=0; fi
      if [ $fast_fails -ge 3 ]; then say "resume keeps failing: new session"; fast_fails=0; launch new; else launch resume; fi
    fi
    state=launched; sleep $TICK; continue
  fi

  screen=$(tmux capture-pane -p -J -t "$P")
  tail_=$(printf '%s\n' "$screen" | grep -v '^\s*$' | tail -25)
  hash=$(printf '%s' "$screen" | md5sum | cut -d' ' -f1)
  if [ "$hash" != "$last_hash" ]; then last_hash=$hash; same_since=$now; fi

  if printf '%s' "$tail_" | grep -q 'esc to interrupt'; then state=busy; sleep $TICK; continue; fi
  if [ -n "$(tmux list-clients -t $S 2>/dev/null)" ] && [ $((now - same_since)) -lt $USER_QUIET ]; then
    state=user; sleep $TICK; continue
  fi
  if printf '%s' "$tail_" | grep -qi 'continuing automatically'; then state=autowait; sleep $TICK; continue; fi

  if printf '%s' "$tail_" | grep -Eqi 'hit your [a-z ]*limit|usage limit|limit reached|limit · resets'; then
    [ "$state" = limit ] || { say "usage limit on screen: retry every hour"; last_send=$now; }
    state=limit
    [ $((now - last_send)) -ge $LIMIT_RETRY ] && send "usage limit, hourly retry"
    sleep $TICK; continue
  fi

  if printf '%s' "$tail_" | grep -Eqi 'API Error|Connection error|Unable to connect|fetch failed|ECONNRESET|ETIMEDOUT|Request timed out|overloaded'; then
    state=error
    online && [ $((now - last_send)) -ge $ERR_RETRY ] && send "API/network error"
    sleep $TICK; continue
  fi

  if [ $((now - same_since)) -ge $IDLE_NUDGE ] && [ $((now - last_send)) -ge $IDLE_NUDGE ]; then
    send "idle 1 h"
  fi
  state=idle; sleep $TICK
done
