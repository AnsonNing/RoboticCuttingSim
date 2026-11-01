// Exact-commit remote bytes, rendered GitHub GIF motion, and full MP4 playback.
// Uses the existing Node/Chrome installations; no packages or shared profiles.
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawn, execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';

const [sha, destination] = process.argv.slice(2);
if (!/^[0-9a-f]{40}$/.test(sha || '') || !destination) throw Error('usage: node script SHA NEW_OUTPUT');
const out = path.resolve(destination);
await fs.mkdir(out);
const repo = 'AnsonNing/RoboticCuttingSim';
const raw = ref => `https://raw.githubusercontent.com/${repo}/${ref}/`;
const hash = data => createHash('sha256').update(data).digest('hex');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function fetchBytes(url) {
  const response = await fetch(url, {signal: AbortSignal.timeout(60000)});
  if (!response.ok) throw Error(`${response.status} ${url}`);
  return Buffer.from(await response.arrayBuffer());
}
const pinned = await fetchBytes(raw(sha)+'README.md');
const main = await fetchBytes(raw('main')+'README.md');
if (!pinned.equals(main)) throw Error('main README bytes differ from pinned merge commit');
const readme = pinned.toString();
const paths = [...new Set([...readme.matchAll(/\]\((docs\/media\/wp(?:13|14)\/[^)#]+)\)/g)].map(m=>m[1]))];
if (!paths.some(p=>p.includes('/wp14/'))) throw Error('No WP-14 media linked from README');
const receipt = {merge_sha:sha, started:new Date().toISOString(), readme_sha256:hash(pinned),
                 page_url:`https://github.com/${repo}`, media:[], gifs:[], videos:[], documents:[]};
const documentPaths=new Set();
for(const page of ['README.md','docs/README.md','docs/WP14_PLAN.md','docs/reports/WP14_RESULTS.md','docs/WP14_SCENES.md']) {
  const bytes=await fetchBytes(raw(sha)+page);
  documentPaths.add(page);
  const source=bytes.toString().replace(/^```.*?^```\s*$/gms,'');
  for(const match of source.matchAll(/\]\(([^\s)]+)(?:\s+"[^"]*")?\)/g)) {
    const target=match[1].replace(/^<|>$/g,'');
    if(/^[a-z]+:|^\/\//i.test(target))continue;
    const relative=decodeURIComponent(target.split('#')[0].split('?')[0]);
    if(!relative)continue;
    const resolved=path.posix.normalize(path.posix.join(path.posix.dirname(page),relative));
    if(resolved.startsWith('../')||resolved.startsWith('/'))throw Error(`Document link escapes tree: ${resolved}`);
    if(/\.(md|json|py|yaml|toml|sh)$/.test(resolved))documentPaths.add(resolved);
  }
}
for(const filename of documentPaths) {
  const bytes=await fetchBytes(raw(sha)+filename);
  const expected=execFileSync('git',['show',`${sha}:${filename}`],{maxBuffer:16*1024*1024});
  if(!bytes.equals(expected))throw Error(`Remote document differs from exact commit: ${filename}`);
  receipt.documents.push({path:filename,status:'PASS_REMOTE_EXACT_BYTES',sha256:hash(bytes)});
}
for (const mediaPath of paths) {
  const data = await fetchBytes(raw(sha)+mediaPath);
  const live = await fetchBytes(raw('main')+mediaPath);
  if (!data.equals(live)) throw Error(`main media differs: ${mediaPath}`);
  const filename = path.join(out, mediaPath.replaceAll('/','_'));
  await fs.writeFile(filename,data,{flag:'wx'});
  const media = {path:mediaPath, url:raw(sha)+mediaPath, bytes:data.length, sha256:hash(data), local:filename};
  if (mediaPath.endsWith('.mp4')) {
    execFileSync('ffmpeg',['-v','error','-xerror','-i',filename,'-f','null','-'],{stdio:['ignore','ignore','pipe']});
    media.full_decode = 'PASS';
  }
  receipt.media.push(media);
}
const chromeLog = await fs.open(path.join(out,'chrome.log'),'wx');
const chrome = spawn('/usr/bin/google-chrome',[
  '--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
  '--no-first-run','--no-default-browser-check','--remote-debugging-port=0',
  `--user-data-dir=${path.join(out,'chrome-profile')}`,'about:blank'
],{stdio:['ignore',chromeLog.fd,chromeLog.fd]});
let ws;
let serial = 0;
const pending = new Map();
try {
  let port;
  for (let i=0;i<100;i++) {
    try {port = (await fs.readFile(path.join(out,'chrome-profile/DevToolsActivePort'),'utf8')).split('\n')[0]; break;}
    catch {await pause(100);}
  }
  if (!port) throw Error('Chrome did not open its local debugging endpoint');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws = new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  ws.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.id && pending.has(data.id)) {
      const [resolve,reject,timer] = pending.get(data.id); pending.delete(data.id); clearTimeout(timer);
      if (data.error) reject(Error(JSON.stringify(data.error))); else resolve(data.result);
    }
  };
  const call = (method,params={}) => new Promise((resolve,reject)=>{
    const id=++serial;
    const timer=setTimeout(()=>{pending.delete(id);reject(Error(`CDP timeout ${method}`));},60000);
    pending.set(id,[resolve,reject,timer]);ws.send(JSON.stringify({id,method,params}));
  });
  const evaluate = async expression => {
    const result=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:receipt.page_url});
  for(let i=0;i<100;i++) {
    if(await evaluate("!!document.querySelector('article.markdown-body')")) break;
    await pause(300);
  }
  if(!await evaluate("!!document.querySelector('article.markdown-body')")) throw Error('Rendered GitHub README missing');
  await fs.writeFile(path.join(out,'github_readme.png'),Buffer.from((await call('Page.captureScreenshot')).data,'base64'),{flag:'wx'});
  for(const media of receipt.media.filter(m=>m.path.endsWith('.gif'))) {
    const select = `Array.from(document.querySelectorAll('article.markdown-body img')).find(i=>
      decodeURIComponent(i.getAttribute('data-canonical-src')||i.src).includes(${JSON.stringify(media.path)}))`;
    if(!await evaluate(`!!(${select})`)) throw Error(`Rendered GIF missing: ${media.path}`);
    await evaluate(`(${select}).scrollIntoView({block:'center'})`);
    let imageState;
    for(let i=0;i<50;i++) {
      imageState = await evaluate(`(()=>{const i=${select};return {complete:i.complete,width:i.naturalWidth,height:i.naturalHeight,src:i.src}})()`);
      if(imageState.complete && imageState.width>0) break;
      await pause(200);
    }
    if(!imageState.width) throw Error(`GIF failed to decode: ${media.path}`);
    const clip=await evaluate(`(()=>{const r=(${select}).getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()`);
    const screenshots=[];
    for(let frame=0;frame<8;frame++) {
      await pause(350);
      const bytes=Buffer.from((await call('Page.captureScreenshot',{format:'png',clip,captureBeyondViewport:true})).data,'base64');
      screenshots.push(hash(bytes));
      await fs.writeFile(path.join(out,path.basename(media.path)+`_${frame}.png`),bytes,{flag:'wx'});
      if(new Set(screenshots).size>1) break;
    }
    if(new Set(screenshots).size<2) throw Error(`Rendered GIF motion not established: ${media.path}`);
    receipt.gifs.push({path:media.path,status:'PASS_RENDERED_MOTION',imageState,screenshot_sha256:screenshots});
  }
  const scenePage=`https://github.com/${repo}/blob/main/docs/WP14_SCENES.md`;
  await call('Page.navigate',{url:scenePage});
  const sceneSelector="Array.from(document.querySelectorAll('img')).find(i=>decodeURIComponent(i.getAttribute('data-canonical-src')||i.src).includes('docs/media/wp14/scene_overview.png'))";
  let sceneState;
  for(let i=0;i<100;i++) {
    sceneState=await evaluate(`(()=>{const i=${sceneSelector};return i?{complete:i.complete,width:i.naturalWidth,height:i.naturalHeight,src:i.src}:null})()`);
    if(sceneState?.complete && sceneState.width>0)break;
    await pause(300);
  }
  if(!sceneState?.width)throw Error('Rendered scene overview failed to load');
  await evaluate(`(${sceneSelector}).scrollIntoView({block:'center'})`);
  await fs.writeFile(path.join(out,'github_scene.png'),Buffer.from((await call('Page.captureScreenshot')).data,'base64'),{flag:'wx'});
  receipt.scene_page={url:scenePage,status:'PASS_RENDERED_SCENE_IMAGE',...sceneState};
  await call('Page.navigate',{url:'about:blank'});
  await pause(200);
  for(const media of receipt.media.filter(m=>m.path.endsWith('.mp4'))) {
    const playback = await evaluate(`(async()=>{
      const response=await fetch(${JSON.stringify(media.url)});if(!response.ok)throw Error('video fetch '+response.status);
      const video=document.createElement('video');video.controls=true;video.muted=true;video.width=960;
      video.src=URL.createObjectURL(await response.blob());document.body.replaceChildren(video);
      await new Promise((resolve,reject)=>{video.onloadedmetadata=resolve;video.onerror=()=>reject(Error('video load'));});
      const duration=video.duration;video.playbackRate=8;await video.play();
      await new Promise((resolve,reject)=>{video.onended=resolve;video.onerror=()=>reject(Error('video playback'));setTimeout(()=>reject(Error('playback timeout')),50000);});
      return {duration_s:duration,current_time_s:video.currentTime,ended:video.ended,
        decoded_frames:video.getVideoPlaybackQuality().totalVideoFrames,ready_state:video.readyState,
        playback_rate:video.playbackRate};})()`);
    if(!playback.ended || playback.decoded_frames<5 || Math.abs(playback.current_time_s-playback.duration_s)>.1)
      throw Error(`Full browser playback incomplete: ${media.path}`);
    receipt.videos.push({path:media.path,status:'PASS_FULL_PLAYBACK',...playback});
  }
  receipt.status='PASS';receipt.finished=new Date().toISOString();
  await fs.writeFile(path.join(out,'remote_media_verified.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({status:'PASS',merge_sha:sha,gifs:receipt.gifs.length,videos:receipt.videos.length}));
} catch(error) {
  receipt.status='FAIL';receipt.error=String(error);receipt.finished=new Date().toISOString();
  await fs.writeFile(path.join(out,'failure.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  throw error;
} finally {
  ws?.close();chrome.kill('SIGTERM');await chromeLog.close();
}
