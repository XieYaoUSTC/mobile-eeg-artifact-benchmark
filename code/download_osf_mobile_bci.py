#!/usr/bin/env python3
"""从 OSF(r7s9b)下载 Mobile BCI 完整版(24 人,BrainVision 格式)。
用法: download_osf_mobile_bci.py list [sub-19 sub-20 …]  # 列清单(可只列指定受试者)到 data/mobile_bci_osf/_manifest.json
      download_osf_mobile_bci.py get sub-19 sub-20 …  # 下载指定受试者(断点续传 + md5 校验)
      download_osf_mobile_bci.py get all
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'mobile_bci_osf')
API = 'https://files.osf.io/v1/resources/r7s9b/providers/osfstorage/'
MANIFEST = os.path.join(OUT, '_manifest.json')


def get_json(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(5 * (k + 1))
            err = e
    raise err


def crawl(url, prefix='', only=None):
    """递归列目录;only=顶层要进的子目录名集合(None=全部)。子目录并行列。"""
    items, folders = [], []
    for it in get_json(url + ('&' if '?' in url else '?') + 'meta=')['data']:
        a = it['attributes']
        if a['kind'] == 'folder':
            if prefix == '' and only is not None and a['name'] not in only:
                continue
            folders.append((API + a['path'].strip('/') + '/', prefix + a['name'] + '/'))
        else:
            items.append({'path': prefix + a['name'], 'size': a.get('sizeInt') or a.get('size'),
                          'md5': (a.get('extra') or {}).get('hashes', {}).get('md5'),
                          'url': API + a['path'].strip('/')})
    with ThreadPoolExecutor(6) as ex:
        for sub in ex.map(lambda f: crawl(f[0], f[1]), folders):
            items += sub
    return items


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def fetch(it):
    dst = os.path.join(OUT, it['path'])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == it['size'] and (not it['md5'] or md5(dst) == it['md5']):
        return True
    for k in range(5):
        subprocess.run(['curl', '-sS', '-L', '--retry', '3', '--connect-timeout', '30', '-C', '-', '-o', dst, it['url']])
        if os.path.exists(dst) and os.path.getsize(dst) == it['size'] and (not it['md5'] or md5(dst) == it['md5']):
            return True
        if os.path.exists(dst) and os.path.getsize(dst) >= it['size']:
            os.remove(dst)
        time.sleep(10 * (k + 1))
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == 'list' or not os.path.exists(MANIFEST):
        only = set(sys.argv[2:]) if sys.argv[1] == 'list' and len(sys.argv) > 2 else None
        items = crawl(API, only=only)
        if os.path.exists(MANIFEST):                                # 与已有清单合并
            old = {i['path']: i for i in json.load(open(MANIFEST))}
            old.update({i['path']: i for i in items})
            items = list(old.values())
        json.dump(items, open(MANIFEST, 'w'), indent=1)
        tot = sum(i['size'] or 0 for i in items)
        subs = sorted({i['path'].split('/')[0] for i in items})
        print(f'{len(items)} files, {tot/1e9:.2f} GB; top-level: {subs}', flush=True)
        if sys.argv[1] == 'list':
            return
    items = json.load(open(MANIFEST))
    want = sys.argv[2:]
    sel = items if 'all' in want else [i for i in items if i['path'].split('/')[0] in want or '/' not in i['path']]
    print(f'downloading {len(sel)} files, {sum(i["size"] or 0 for i in sel)/1e9:.2f} GB', flush=True)
    bad = []
    for n, it in enumerate(sel, 1):
        ok = fetch(it)
        print(f'{n}/{len(sel)} {it["path"]} {(it["size"] or 0)/1e6:.1f}MB {"ok" if ok else "FAILED"}', flush=True)
        if not ok:
            bad.append(it['path'])
    print(f'DONE failed={len(bad)} {bad[:10]}', flush=True)


if __name__ == '__main__':
    main()
