#!/usr/bin/env python3
"""OSF r7s9b 温和列目录(单线程、每请求间隔、遇 429 按 Retry-After 退避),把全部 24 人的文件清单合并进 _manifest.json,
并顺手下载所有 events.tsv 到 data/mobile_bci_osf/_events_all/。用法: osf_gentle_list.py [只列这些被试,逗号分隔;默认全部]
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'mobile_bci_osf')
API = 'https://files.osf.io/v1/resources/r7s9b/providers/osfstorage/'
MANIFEST = os.path.join(OUT, '_manifest.json')
PAUSE = 0.7


def get(url, binary=False):
    for k in range(8):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
                time.sleep(PAUSE)
                return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get('Retry-After', 0) or 0) or 60 * (k + 1)
            print(f'  HTTP {e.code} on {url[-40:]}; sleeping {wait}s', flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f'  error {e}; sleeping {30 * (k + 1)}s', flush=True)
            time.sleep(30 * (k + 1))
    raise RuntimeError('giving up ' + url)


def list_folder(url, prefix):
    items = []
    for it in get(url + ('&' if '?' in url else '?') + 'meta=')['data']:
        a = it['attributes']
        if a['kind'] == 'folder':
            items += list_folder(API + a['path'].strip('/') + '/', prefix + a['name'] + '/')
        else:
            items.append({'path': prefix + a['name'], 'size': a.get('sizeInt') or a.get('size'),
                          'md5': (a.get('extra') or {}).get('hashes', {}).get('md5'), 'url': API + a['path'].strip('/')})
    return items


want = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None
old = {i['path']: i for i in json.load(open(MANIFEST))} if os.path.exists(MANIFEST) else {}
root = get(API + '?meta=')['data']
items = []
for it in root:
    a = it['attributes']
    if a['kind'] == 'folder':
        if want is not None and a['name'] not in want:
            continue
        if want is None and any(p.startswith(a['name'] + '/') for p in old):
            print('skip (already listed)', a['name'], flush=True)
            continue
        sub = list_folder(API + a['path'].strip('/') + '/', a['name'] + '/')
        items += sub
        print('listed', a['name'], len(sub), 'files', flush=True)
    else:
        items.append({'path': a['name'], 'size': a.get('sizeInt') or a.get('size'), 'md5': (a.get('extra') or {}).get('hashes', {}).get('md5'), 'url': API + a['path'].strip('/')})
old.update({i['path']: i for i in items})
json.dump(list(old.values()), open(MANIFEST, 'w'), indent=1)
subs = sorted({p.split('/')[0] for p in old if p.startswith('sub-')})
print(f'manifest: {len(old)} files; subjects {len(subs)}: {subs}', flush=True)
# 下载所有 events.tsv
ev = [i for i in old.values() if i['path'].endswith('_events.tsv')]
for i in ev:
    dst = os.path.join(OUT, '_events_all', i['path'])
    if os.path.exists(dst) and os.path.getsize(dst) == i['size']:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(get(i['url'], binary=True))
print('events.tsv downloaded:', len(ev), flush=True)
print('LIST DONE', flush=True)
