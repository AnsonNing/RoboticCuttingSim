#!/usr/bin/env node
/** Actual Chromium WebRTC evidence. Actions use only streamed mouse/keyboard. */
import { chromium } from '@playwright/test';
import { parseArgs } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const { values } = parseArgs({ options: {
  url: { type: 'string', default: 'http://100.111.162.17:8210' },
  out: { type: 'string' }, timeout: { type: 'string', default: '45000' },
  chrome: { type: 'string', default: '/usr/bin/google-chrome' },
  'ui-status': { type: 'string' }, actions: { type: 'string' }, 'kit-events': { type: 'string' },
} });
if (!values.out) throw new Error('--out must name a new evidence directory');
const output = path.resolve(values.out);
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.mkdirSync(output); // Preserve every previous attempt, including failures.
const write = (name, data) => fs.writeFileSync(path.join(output, name), JSON.stringify(data, null, 2)+'\n');
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const snapshot = path.join(output, 'implementation');
fs.mkdirSync(snapshot);
fs.copyFileSync(new URL(import.meta.url), path.join(snapshot, 'probe_remote.mjs'));
write('protocol.json', { url: values.url, browser: values.chrome, timeout_ms: Number(values.timeout),
  scope: 'One local headless Chrome client using the Tailscale URL; actual RTC stats/video and optional streamed input. Not an external Mac/Tailscale-client validation.',
  required_video_evidence: ['connected RTCPeerConnection', 'received and decoded video RTP', 'advancing HTML video frames', 'actual requested dimensions'],
  ui_status: values['ui-status'] ?? null, actions_sha256: values.actions ? hash(values.actions) : null,
  probe_source_sha256: hash(new URL(import.meta.url)), probe_source_copied_before_browser_launch: true });
const consoleRows = [], networkRows = [], actionRows = [];
const readStatus = () => {
  if (!values['ui-status']) return null;
  try { return JSON.parse(fs.readFileSync(values['ui-status'], 'utf8')); }
  catch (error) { return { status_read_error: String(error) }; }
};
const readKitEvents = () => values['kit-events'] ? fs.readFileSync(values['kit-events'],'utf8')
  .trim().split('\n').filter(Boolean).map((line) => JSON.parse(line)) : null;
const at = (object, key) => key.split('.').reduce((row, item) => row?.[item], object);
let browser, page, collect;
let result = { status: 'FAIL', stage: 'browser_launch' };
try {
  browser = await chromium.launch({ executablePath: values.chrome, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--autoplay-policy=no-user-gesture-required'] });
  page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('console', (message) => consoleRows.push({ type: message.type(), text: message.text().slice(0, 2000) }));
  page.on('pageerror', (error) => consoleRows.push({ type: 'pageerror', text: String(error) }));
  page.on('websocket', (socket) => {
    networkRows.push({ event: 'websocket-created', url: socket.url() });
    socket.on('close', () => networkRows.push({ event: 'websocket-close', url: socket.url() }));
  });
  await page.addInitScript(() => {
    const peers = [], channels = [], observed = new WeakSet();
    const observe = (channel) => {
      if (observed.has(channel)) return;
      observed.add(channel);
      const row = { label: channel.label, sentMessages: 0, sentBytes: 0 };
      channels.push(row);
      const original = channel.send;
      channel.send = function(data) {
        const result = original.call(this, data);
        row.sentMessages++;
        row.sentBytes += typeof data === 'string' ? new TextEncoder().encode(data).length : data.byteLength ?? data.size ?? 0;
        return result;
      };
    };
    const Original = window.RTCPeerConnection;
    window.RTCPeerConnection = class extends Original {
      constructor(...args) { super(...args); peers.push(this); this.addEventListener('datachannel', (event) => observe(event.channel)); }
      createDataChannel(...args) { const channel = super.createDataChannel(...args); observe(channel); return channel; }
    };
    window.__rtcProbe = { peers, channels };
  });
  collect = () => page.evaluate(async () => {
    const video = document.querySelector('#remote-video');
    const peers = [];
    for (const peer of window.__rtcProbe.peers) {
      const reports = [...(await peer.getStats()).values()];
      const inbound = reports.filter((row) => row.type === 'inbound-rtp' && (row.kind === 'video' || row.mediaType === 'video'));
      const pairs = reports.filter((row) => row.type === 'candidate-pair' && row.state === 'succeeded' && row.nominated);
      peers.push({ connectionState: peer.connectionState, iceConnectionState: peer.iceConnectionState,
        inbound: inbound.map((row) => Object.fromEntries(['type','kind','framesDecoded','framesReceived','bytesReceived','packetsReceived',
          'packetsLost','framesPerSecond','frameWidth','frameHeight','jitter','freezeCount','totalFreezesDuration'].map((key) => [key,row[key]]))),
        candidatePairs: pairs.map((row) => ({ state: row.state, bytesReceived: row.bytesReceived, currentRoundTripTime: row.currentRoundTripTime,
          local: reports.find((entry) => entry.id === row.localCandidateId), remote: reports.find((entry) => entry.id === row.remoteCandidateId) })) });
    }
    return { peers, dataChannels: window.__rtcProbe.channels, video: video ? {
      readyState: video.readyState, width: video.videoWidth, height: video.videoHeight, currentTime: video.currentTime,
      paused: video.paused, playbackQuality: video.getVideoPlaybackQuality().toJSON?.() ?? {
        totalVideoFrames: video.getVideoPlaybackQuality().totalVideoFrames, droppedVideoFrames: video.getVideoPlaybackQuality().droppedVideoFrames }
    } : null, viewer: window.__remoteViewer };
  });
  result.stage = 'page_load';
  await page.goto(values.url, { waitUntil: 'domcontentloaded', timeout: Number(values.timeout) });
  result.stage = 'actual_video_wait';
  await page.waitForFunction(() => {
    const video = document.querySelector('#remote-video');
    return video && video.videoWidth > 0 && video.readyState >= 2 && video.getVideoPlaybackQuality().totalVideoFrames >= 3;
  }, null, { timeout: Number(values.timeout) });
  const first = await collect();
  await page.screenshot({ path: path.join(output, 'connected.png') });
  write('connected.json', first);
  const requested = values.actions ? JSON.parse(fs.readFileSync(values.actions,'utf8')) : [];
  result.stage = 'actual_kit_input';
  for (const [index, action] of requested.entries()) {
    const before = readStatus();
    const beforeKitEventCount = readKitEvents()?.length ?? 0;
    const beforeStats = await collect();
    const bounds = await page.locator('#remote-video').boundingBox();
    const clickRegion = async (name) => {
      const region = readStatus()?.ui_regions?.[name];
      if (!region) throw new Error('Missing actual Kit UI region: '+name);
      const rectangle = region.rect ?? region;
      if (!('x' in rectangle && 'y' in rectangle && 'width' in rectangle && 'height' in rectangle))
        throw new Error('UI region must have x,y,width,height in actual streamed-window pixels');
      const point = { x: bounds.x+(rectangle.x+rectangle.width/2)/first.video.width*bounds.width,
        y: bounds.y+(rectangle.y+rectangle.height/2)/first.video.height*bounds.height };
      await page.mouse.click(point.x, point.y);
      return point;
    };
    const preparationPoint = action.prepare_region ? await clickRegion(action.prepare_region) : null;
    if (preparationPoint) await page.waitForTimeout(200);
    const clearPoint = action.clear_region ? await clickRegion(action.clear_region) : null;
    if (clearPoint) await page.waitForTimeout(200);
    const point = action.region ? await clickRegion(action.region) : null;
    if (action.kind === 'type') {
      // The remote StringField may not have keyboard focus until Kit's next
      // UI frame. Home/Shift+End is a real-key alternative to browser Ctrl+A.
      await page.waitForTimeout(300);
      if (!clearPoint) {
        await page.keyboard.press('Home');
        await page.waitForTimeout(100);
        await page.keyboard.press('Shift+End');
        await page.waitForTimeout(100);
        await page.keyboard.press('Backspace');
        await page.waitForTimeout(100);
      }
      await page.screenshot({ path: path.join(output, `cleared_${String(index).padStart(2,'0')}.png`) });
      await page.keyboard.type(String(action.text), { delay: 100 });
      await page.keyboard.press('Enter');
      await page.waitForTimeout(200);
    } else if (action.kind === 'key') {
      await page.locator('#remote-video').focus();
      await page.keyboard.press(action.key);
    } else if (action.kind !== 'click') throw new Error('Unknown action kind: '+action.kind);
    const submitPoint = action.submit_region ? await clickRegion(action.submit_region) : null;
    const expected = action.expect ?? {};
    const deadline = Date.now()+Number(action.timeout_ms ?? 10000);
    let after;
    do {
      await page.waitForTimeout(200);
      after = readStatus();
      if (Object.keys(expected).length && Object.entries(expected).every(([key,value]) => JSON.stringify(at(after,key)) === JSON.stringify(value))) break;
    } while (Date.now() < deadline && Object.keys(expected).length);
    const expectedStateChanged = Object.entries(expected).some(([key,value]) => JSON.stringify(at(before,key)) !== JSON.stringify(value));
    const kitEvents = readKitEvents();
    const newKitEvents = kitEvents?.slice(beforeKitEventCount) ?? null;
    const acceptedTypedTarget = action.kind !== 'type' || !newKitEvents ? null : newKitEvents.some((event) => {
      if (event.command !== 'action' || event.status !== 'ACCEPTED') return false;
      try { return JSON.stringify(JSON.parse(event.target)) === JSON.stringify(JSON.parse(action.text)); }
      catch { return false; }
    });
    const acknowledged = Object.keys(expected).length > 0 && expectedStateChanged
      && Object.entries(expected).every(([key,value]) => JSON.stringify(at(after,key)) === JSON.stringify(value))
      && acceptedTypedTarget !== false;
    actionRows.push({ action, point_css: point, preparation_point_css: preparationPoint,
      clear_point_css: clearPoint, submit_point_css: submitPoint,
      typed_text: action.kind === 'type' ? String(action.text) : null, actual_kit_events: kitEvents,
      new_actual_kit_events: newKitEvents, accepted_typed_target_matches_exactly: acceptedTypedTarget,
      before_kit_status: before, after_kit_status: after,
      before_rtc: beforeStats, after_rtc: await collect(), actual_kit_expectation_matched: acknowledged });
    await page.screenshot({ path: path.join(output, `action_${String(index).padStart(2,'0')}.png`) });
    write('actions.json', actionRows);
    if (!acknowledged) throw new Error('Actual Kit did not acknowledge action '+index+'; no HTTP task command was sent');
  }
  await page.waitForTimeout(1500);
  const final = await collect();
  await page.screenshot({ path: path.join(output, 'final.png') });
  const streams = final.peers.flatMap((peer) => peer.inbound);
  const checks = { connected_peer: final.peers.some((peer) => peer.connectionState === 'connected'),
    actual_video_rtp_received_and_decoded: streams.some((row) => row.bytesReceived > 0 && row.framesDecoded > 0),
    html_video_frames_advanced: final.video.playbackQuality.totalVideoFrames > first.video.playbackQuality.totalVideoFrames,
    video_dimensions_match_requested: final.video.width === final.viewer.config.width && final.video.height === final.viewer.config.height };
  const pass = Object.values(checks).every(Boolean);
  result = { status: pass ? (requested.length ? 'PASS_WEBRTC_VIDEO_AND_KIT_INPUT' : 'PASS_WEBRTC_VIDEO') : 'FAIL_WEBRTC_VIDEO',
    stage: 'complete', checks, initial: first, final, actual_kit_input: requested.length ? 'ACKNOWLEDGED_BY_RECORDED_KIT_STATE' : 'NOT_RUN',
    actions: actionRows, actual_remote_client_over_tailnet: 'NOT_RUN; local Chrome connected through the Tailscale URL',
    browser_version: browser.version(), url: values.url, page_count: page.context().pages().length };
} catch (error) {
  result.error = String(error);
  result.actual_kit_input = 'NOT_VERIFIED';
  if (page && !page.isClosed()) {
    try {
      await page.screenshot({ path: path.join(output, 'failure.png'), timeout: 3000 });
      if (collect) result.failure_snapshot = await collect();
    } catch (snapshotError) { result.failure_snapshot_error = String(snapshotError); }
  }
} finally {
  write('result.json', result);
  write('browser_console.json', consoleRows);
  write('browser_network.json', networkRows);
  if (browser) await browser.close(); // Release the only streaming-client slot.
}
console.log(JSON.stringify({ status: result.status, stage: result.stage, output }));
if (!result.status.startsWith('PASS_')) process.exitCode = 1;
