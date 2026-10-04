#!/usr/bin/env python3
"""下载 figshare 数据集(可断点续传, md5 校验)。用法: download_figshare.py <article_id> <out_dir> [并发数,默认1]"""
import sys, os, json, hashlib, subprocess, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

def md5(p, buf=1 << 22):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main(aid, out, workers=1):
    os.makedirs(out, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{aid}', timeout=60))
    json.dump(meta, open(os.path.join(out, '_figshare_meta.json'), 'w'), ensure_ascii=False, indent=1)
    files = meta['files']
    print(f'[{aid}] {meta["title"]} | {len(files)} files | {sum(f["size"] for f in files)/1e9:.2f} GB', flush=True)
    def fetch(args):
        i, f = args
        dst = os.path.join(out, f['name'])
        want = f.get('computed_md5') or f.get('supplied_md5')
        if os.path.exists(dst) and os.path.getsize(dst) == f['size'] and (not want or md5(dst) == want):
            return None
        ok = False
        for attempt in range(5):
            r = subprocess.run(['curl', '-sS', '-L', '--retry', '3', '--connect-timeout', '30', '-C', '-', '-o', dst, f['download_url']])
            if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == f['size'] and (not want or md5(dst) == want):
                ok = True
                break
            if os.path.exists(dst) and os.path.getsize(dst) > f['size']:
                os.remove(dst)
            time.sleep(10 * (attempt + 1))
        print(f'[{aid}] {i}/{len(files)} {f["name"]} {f["size"]/1e6:.1f}MB {"ok" if ok else "FAILED"}', flush=True)
        return None if ok else f['name']

    with ThreadPoolExecutor(workers) as ex:
        bad = [b for b in ex.map(fetch, enumerate(files, 1)) if b]
    print(f'[{aid}] DONE failed={len(bad)} {bad[:10]}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
