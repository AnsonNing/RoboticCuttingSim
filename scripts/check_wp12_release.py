#!/usr/bin/env python3
"""Read-only WP-12 publication checks: tracked links/media and optional protected Git blobs.

Checks the working tree by default, or committed content with --ref. This is
an engineering publication check and does not evaluate physical cutting.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import posixpath

ROOT = Path(__file__).resolve().parents[1]
PAGES = ['README.md', 'docs/README.md', 'docs/WP12_PLAN.md',
         'docs/reports/WP12_GENERALIZATION.md', 'docs/reports/WP12_FINAL_VALIDATION.md']


def animated_gif_metadata(data):
    """Decode every frame: multiple frame headers alone do not prove motion."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        count = getattr(im, 'n_frames', 1)
        loop = im.info.get('loop')
        width, height = im.size
        frames = set()
        for frame in range(count):
            im.seek(frame)
            frames.add(hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest())
    if count < 2 or len(frames) < 2:
        raise ValueError('GIF has no decoded pixel motion')
    if loop != 0:
        raise ValueError('GIF does not loop indefinitely')
    return {'format': 'GIF', 'width': width, 'height': height, 'frames': count,
            'distinct_decoded_frames': len(frames), 'loop': loop,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def headings(text):
    used, anchors = {}, set()
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        title = re.sub(r'<[^>]+>', '', title).lower()
        slug = re.sub(r'[^\w\- ]', '', title).replace(' ', '-')
        count = used.get(slug, 0)
        anchors.add(slug if not count else f'{slug}-{count}')
        used[slug] = count+1
    return anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', help='committed Git tree, e.g. origin/main')
    parser.add_argument('--protected-inventory', type=Path, help='baseline_protected_blobs.json from takeover')
    parser.add_argument('--out', type=Path, required=True, help='new JSON receipt (never overwritten)')
    parser.add_argument('--wp13', action='store_true', help='also check WP-13 report, animated GIFs and full MP4 decode')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    tracked = set((git('ls-tree', '-r', '-z', '--name-only', args.ref) if args.ref else git('ls-files', '-z'))
                  .decode().rstrip('\0').split('\0'))

    def content(path):
        return git('show', f'{args.ref}:{path}') if args.ref else (ROOT/path).read_bytes()

    errors, links, media = [], [], {}
    sources = {}
    pages = PAGES+(['docs/WP13_PLAN.md', 'docs/reports/WP13_RESULTS.md'] if args.wp13 else [])
    gif_references = []
    for page in pages+['docs/PROGRESS.md']:
        data = content(page)
        sources[page] = hashlib.sha256(data).hexdigest()
        text = data.decode()
        if page == 'docs/PROGRESS.md':
            text = text.split('## Preserved handoff and history')[0]
        # Ignore fenced shell/Python examples when inspecting Markdown links.
        text = re.sub(r'^```.*?^```\s*$', '', text, flags=re.M | re.S)
        # Also find the outer MP4 link of [![preview](preview.gif)](full.mp4).
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
            url = urlsplit(target.strip('<>'))
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            path = posixpath.normpath(posixpath.join(posixpath.dirname(page), path)) if path else page
            edge = {'source': page, 'target': target, 'resolved': path}
            links.append(edge)
            if path.startswith('artifacts/') or path.startswith('../') or path.startswith('/'):
                errors.append({**edge, 'error': 'outside tracked publication scope'})
                continue
            directory = any(p.startswith(path.rstrip('/')+'/') for p in tracked)
            if path not in tracked and not directory:
                errors.append({**edge, 'error': 'target absent from Git'})
                continue
            if path not in tracked:
                continue
            if path.endswith('.gif') and args.wp13:
                if page == 'README.md':
                    gif_references.append(path)
                if path not in media:
                    try:
                        media[path] = animated_gif_metadata(content(path))
                    except Exception as error:
                        errors.append({**edge, 'error': f'GIF decode/motion: {error}'})
            if url.fragment and path.endswith('.md') and unquote(url.fragment) not in headings(content(path).decode()):
                errors.append({**edge, 'error': 'Markdown heading anchor absent'})
            if path.endswith('.png') and path not in media:
                try:
                    from PIL import Image
                    data = content(path)
                    with Image.open(io.BytesIO(data)) as im:
                        width, height = im.size
                        im.verify()
                    media[path] = {'format': 'PNG', 'width': width, 'height': height,
                                   'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                except Exception as error:
                    errors.append({**edge, 'error': f'PNG decode: {error}'})
            if path.endswith('.mp4') and path not in media:
                data = content(path)
                # ffprobe requires a seekable source for MP4 metadata. The local
                # file must equal this tree's blob before using it for the probe.
                if not (ROOT/path).is_file() or (ROOT/path).read_bytes() != data:
                    errors.append({**edge, 'error': 'local MP4 differs from checked tree'})
                    continue
                probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                        'stream=codec_name,width,height,nb_frames:format=duration',
                                        '-of', 'json', str(ROOT/path)], capture_output=True, text=True)
                if probe.returncode:
                    errors.append({**edge, 'error': 'MP4 probe failed: '+probe.stderr})
                else:
                    media[path] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                                   'ffprobe': json.loads(probe.stdout)}
                    if args.wp13:
                        decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(ROOT/path), '-f', 'null', '-'],
                                                capture_output=True, text=True)
                        media[path]['full_decode'] = 'PASS' if decode.returncode == 0 else 'FAIL'
                        if decode.returncode:
                            errors.append({**edge, 'error': 'MP4 full decode failed: '+decode.stderr[-500:]})
    if args.wp13:
        if len(gif_references) < 4 or len(set(gif_references)) < 3:
            errors.append({'error': 'README needs a hero and at least three visible animated gallery previews'})
        provenance_path = 'docs/evidence/wp13/media_001.json'
        try:
            provenance = json.loads(content(provenance_path))
            by_gif = {row['gif']: row for row in provenance['media']}
            for path in set(gif_references):
                row = by_gif[path]
                if row['runtime_source'] != 'live_isaac_sim':
                    raise ValueError(f'{path} is not labelled Live Native')
                if hashlib.sha256(content(row['mp4'])).hexdigest() != row['mp4_sha256']:
                    raise ValueError(f'{path} full MP4 hash differs from provenance')
                if hashlib.sha256(content(path)).hexdigest() != row['gif_sha256']:
                    raise ValueError(f'{path} GIF hash differs from provenance')
        except Exception as error:
            errors.append({'error': 'WP-13 Native media provenance: '+str(error)})
    protected = {'status': 'NOT_RUN', 'reason': 'no protected inventory supplied'}
    if args.protected_inventory:
        baseline = json.loads(args.protected_inventory.read_text())
        changed = []
        for path, expected in baseline['tracked_blobs'].items():
            got = (git('rev-parse', f'{args.ref}:{path}').decode().strip() if args.ref else
                   git('hash-object', '--', path).decode().strip())
            if got != expected:
                changed.append({'path': path, 'expected': expected, 'actual': got})
        protected = {'status': 'FAIL' if changed else 'PASS', 'baseline_sha': baseline['baseline_sha'],
                     'checked': len(baseline['tracked_blobs']), 'changed': changed}
        errors.extend({'error': 'protected blob changed', **row} for row in changed)
    report = {'schema': 'rcs_wp12_publication_check_v1', 'created': datetime.now().astimezone().isoformat(),
              'command': sys.argv, 'scope': 'ENGINEERING_PUBLICATION_ONLY', 'ref': args.ref or 'working_tree',
              'commit': git('rev-parse', args.ref or 'HEAD').decode().strip(),
              'status': 'FAIL' if errors else 'PASS', 'source_sha256': sources,
              'links_checked': len(links), 'links': links, 'media': media, 'protected': protected, 'errors': errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'links': len(links), 'media': len(media),
                      'protected': protected, 'errors': errors}))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
