import { AppStreamer, EventStatus, LogLevel, StreamStatus, StreamType, VideoCodec,
  type StreamEvent } from '@nvidia/ov-web-rtc';
import './style.css';

type ViewerConfig = { signalingServer: string; signalingPort: number; mediaServer: string;
  mediaPort: number; width: number; height: number; fps: number };
const status = document.querySelector<HTMLElement>('#status')!;
const video = document.querySelector<HTMLVideoElement>('#remote-video')!;
const events: { time: string; kind: string; value: unknown }[] = [];
const diagnostics = { sdkVersion: '6.8.2', config: null as ViewerConfig | null,
  streamStatus: StreamStatus.NONE, events, lastStatistics: null as unknown };
declare global { interface Window { __remoteViewer: typeof diagnostics } }
window.__remoteViewer = diagnostics;

function record(kind: string, value: unknown) {
  events.push({ time: new Date().toISOString(), kind, value });
  if (events.length > 500) events.shift();
  console.info('[RemoteViewer]', kind, value);
}

document.querySelector('#reconnect')!.addEventListener('click', () => location.reload());
document.querySelector('#fullscreen')!.addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen();
});
video.addEventListener('pointerdown', () => video.focus());
video.addEventListener('playing', () => {
  status.textContent = 'Live Isaac Sim — use the controls inside the streamed GUI';
  record('video-playing', { width: video.videoWidth, height: video.videoHeight });
});
// Keep browser navigation shortcuts local; ordinary input belongs to Kit.
document.querySelector('#stream-container')!.addEventListener('keydown', (event) => {
  const key = event as KeyboardEvent;
  if (key.key === 'F5' || ((key.ctrlKey || key.metaKey) && key.key.toLowerCase() === 'r')) {
    key.preventDefault(); key.stopPropagation(); location.reload();
  } else if (key.key === 'F12') key.stopPropagation();
  else if (key.key === 'F7') key.preventDefault();
}, true);

async function connect() {
  const response = await fetch('/viewer-config.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Viewer configuration HTTP ${response.status}`);
  const config: ViewerConfig = await response.json();
  diagnostics.config = config;
  video.width = config.width; video.height = config.height;
  const stream = new AppStreamer();
  const onStart = (event: StreamEvent) => {
    record('stream-start', event);
    if (event.status === EventStatus.ERROR) status.textContent = `Connection unavailable: ${event.info ?? 'unknown error'}`;
  };
  await stream.connect({ streamSource: StreamType.DIRECT, logLevel: LogLevel.INFO,
    streamConfig: { ...config, signalingPath: '', forceWSS: false,
      videoElementId: 'remote-video', audioElementId: 'remote-audio',
      codec: VideoCodec.H264, mic: false, maxReconnects: 5, reconnectDelay: 2000,
      onStart, onStop: (event) => { record('stream-stop', event); status.textContent = 'Stream stopped — reconnect when Isaac Sim is ready'; },
      onStreamStatusChange: (value) => { diagnostics.streamStatus = value; record('stream-status', value); },
      onStreamStats: (event) => { diagnostics.lastStatistics = event; }
    } });
}
void connect().catch((error: unknown) => {
  record('connection-error', String(error));
  status.textContent = 'Connection unavailable — start the live Isaac Sim session, then reconnect';
});
