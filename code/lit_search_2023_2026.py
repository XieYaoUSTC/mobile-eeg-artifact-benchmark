#!/usr/bin/env python3
"""2023–2026 文献补充检索:Europe PMC + arXiv + Crossref,主题 = 运动中脑电去伪迹/IMU 参考/因果自适应/步行 BCI/评价方案/基础模型去伪迹。
输出 results/lit_search_2023_2026.json 与 docs/文献补充检索_2023_2026.md(题名/作者/年/刊/DOI/摘要前 400 字),供人工筛选。"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {'User-Agent': 'mobile-eeg-benchmark/1.0 (mailto:xieyao@mail.ustc.edu.cn)'}


def get(url, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        d = r.read()
    return d if binary else d.decode('utf-8', 'replace')


EPMC_QUERIES = [
    '("motion artifact" OR "movement artifact" OR "gait artifact") AND (EEG OR electroencephalography) AND (walking OR gait OR locomotion OR ambulatory OR treadmill OR running)',
    '("inertial measurement unit" OR IMU OR accelerometer) AND EEG AND artifact AND (walking OR gait OR motion OR movement)',
    '"artifact subspace reconstruction" AND EEG',
    '("mobile EEG" OR "ambulatory EEG" OR "mobile brain") AND ("brain-computer interface" OR BCI) AND (walking OR running OR gait OR locomotion)',
    '(SSVEP OR "steady-state visual evoked") AND (walking OR gait OR treadmill OR locomotion OR ambulatory)',
    '(P300 OR "event-related potential") AND (walking OR gait OR treadmill OR locomotion) AND ("brain-computer interface" OR BCI)',
    '("adaptive filter" OR "adaptive filtering" OR "least mean square" OR NLMS) AND EEG AND ("motion artifact" OR "movement artifact")',
    '("canonical correlation" OR CCA) AND EEG AND artifact AND (motion OR movement OR gait OR iCanClean)',
    '("foundation model" OR "large brain model" OR "self-supervised") AND EEG AND (artifact OR denoising)',
    '("dual-layer" OR "dual electrode" OR "noise electrode" OR "reference electrode") AND EEG AND ("motion artifact" OR "movement artifact")',
    '("exoskeleton" OR "robotic gait" OR "lower-limb") AND EEG AND artifact',
    '(benchmark OR "evaluation framework" OR "evaluation protocol") AND EEG AND artifact AND removal',
]
ARXIV_QUERIES = [
    'all:"motion artifact" AND all:EEG AND (all:walking OR all:gait OR all:IMU OR all:ambulatory)',
    'all:EEG AND all:artifact AND (all:"foundation model" OR all:"large brain model")',
    'all:EEG AND all:artifact AND all:"canonical correlation"',
    'all:"mobile EEG" AND all:"brain-computer interface"',
]
out = {}


def add(rec):
    key = (rec.get('doi') or rec.get('arxiv') or rec['title'].lower())
    if key in out:
        out[key]['queries'] = sorted(set(out[key]['queries'] + rec['queries']))
    else:
        out[key] = rec


# ---- Europe PMC
for q in EPMC_QUERIES:
    full = f'({q}) AND (PUB_YEAR:[2023 TO 2026]) AND (SRC:MED OR SRC:PPR OR SRC:PMC)'
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=60&resultType=core&query=' + urllib.parse.quote(full)
    try:
        d = json.loads(get(url))
    except Exception as e:
        print('EPMC ERR', q[:40], e); continue
    hits = d.get('resultList', {}).get('result', [])
    print('EPMC', len(hits), q[:70], flush=True)
    for r in hits:
        add({'source': 'EPMC', 'title': r.get('title', ''), 'authors': r.get('authorString', ''), 'year': r.get('pubYear'), 'journal': r.get('journalTitle') or (r.get('journalInfo') or {}).get('journal', {}).get('title', ''),
             'doi': (r.get('doi') or '').lower(), 'pmid': r.get('pmid'), 'abstract': (r.get('abstractText') or '')[:1500], 'queries': [q[:50]]})
    time.sleep(0.5)

# ---- arXiv
for q in ARXIV_QUERIES:
    url = 'http://export.arxiv.org/api/query?max_results=50&sortBy=submittedDate&sortOrder=descending&search_query=' + urllib.parse.quote(q)
    try:
        x = get(url)
    except Exception as e:
        print('arXiv ERR', e); continue
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(x)
    n = 0
    for e in root.findall('a:entry', ns):
        yr = int(e.find('a:published', ns).text[:4])
        if yr < 2023:
            continue
        aid = e.find('a:id', ns).text.split('/abs/')[-1]
        add({'source': 'arXiv', 'title': re.sub(r'\s+', ' ', e.find('a:title', ns).text.strip()), 'authors': ', '.join(a.find('a:name', ns).text for a in e.findall('a:author', ns)),
             'year': yr, 'journal': 'arXiv', 'arxiv': aid, 'doi': '', 'abstract': re.sub(r'\s+', ' ', e.find('a:summary', ns).text.strip())[:1500], 'queries': [q[:50]]})
        n += 1
    print('arXiv', n, q[:60], flush=True)
    time.sleep(3)

json.dump(list(out.values()), open(os.path.join(ROOT, 'results', 'lit_search_2023_2026.json'), 'w'), indent=1, ensure_ascii=False)
L = ['# 2023–2026 文献补充检索(Europe PMC + arXiv;自动拉取,待人工筛选)', '', f'共 {len(out)} 条去重候选。', '']
for rec in sorted(out.values(), key=lambda r: (-int(r['year'] or 0), r['title'])):
    L.append(f"- **{rec['title']}** — {rec['authors'][:120]} ({rec['year']}) *{rec['journal']}* DOI:{rec.get('doi') or ''} {('arXiv:' + rec['arxiv']) if rec.get('arxiv') else ''}\n  - {rec['abstract'][:400].replace(chr(10), ' ')}")
open(os.path.join(ROOT, 'docs', '文献补充检索_2023_2026.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('total candidates', len(out))
