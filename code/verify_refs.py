#!/usr/bin/env python3
"""用 Crossref 核实候选参考文献:按题名查,取首条,核对第一作者姓与年份,输出 DOI/刊名/卷/页。
结果写 results/refs_verified.json 与 docs/refs_verified.md;不匹配的标 UNVERIFIED,不入稿。"""
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDS = [
    ('gramann2011', 'Cognition in action: imaging brain/body dynamics in mobile humans', 'Gramann', 2011),
    ('castermans2014', 'About the cortical origin of the low-delta and high-gamma rhythms observed in EEG signals during treadmill walking', 'Castermans', 2014),
    ('nathan2016', 'Negligible motion artifacts in scalp electroencephalography (EEG) during treadmill walking', 'Nathan', 2016),
    ('snyder2015', 'Independent component analysis of gait-related movement artifact recorded using EEG electrodes during treadmill walking', 'Snyder', 2015),
    ('oliveira2016', 'Proposing metrics for benchmarking novel EEG technologies towards real-world measurements', 'Oliveira', 2016),
    ('oliveira2017', 'A channel rejection method for attenuating motion-related artifacts in EEG recordings during walking', 'Oliveira', 2017),
    ('nordin2020', 'Faster gait speeds reduce alpha and beta EEG spectral power from human sensorimotor cortex', 'Nordin', 2020),
    ('richer2020', 'Motion and muscle artifact removal validation using an electrical head phantom, robotic motion platform, and dual layer mobile EEG', 'Richer', 2020),
    ('symeonidou2018', 'Effects of cable sway, electrode surface area, and electrode mass on electroencephalography signal quality during motion', 'Symeonidou', 2018),
    ('debener2012', 'How about taking a low-cost, small, and wireless EEG for a walk?', 'Debener', 2012),
    ('devos2014', 'Towards a truly mobile auditory brain-computer interface: exploring the P300 to take away', 'De Vos', 2014),
    ('lin2014', 'Assessing the quality of steady-state visual-evoked potentials for moving humans using a mobile electroencephalogram headset', 'Lin', 2014),
    ('zink2016', 'Mobile EEG on the bike: disentangling attentional and physical contributions to auditory attention tasks', 'Zink', 2016),
    ('kwak2020', 'Error correction regression framework for enhancing the decoding accuracies of ear-EEG brain-computer interfaces', 'Kwak', 2020),
    ('widrow1975', 'Adaptive noise cancelling: principles and applications', 'Widrow', 1975),
    ('mullen2015', 'Real-time neuroimaging and cognitive monitoring using wearable dry EEG', 'Mullen', 2015),
    ('chang2020', 'Evaluation of artifact subspace reconstruction for automatic artifact components removal in multi-channel EEG recordings', 'Chang', 2020),
    ('blum2019', 'A Riemannian modification of artifact subspace reconstruction for EEG artifact handling', 'Blum', 2019),
    ('sweeney2012', 'Artifact removal in physiological signals-practices and possibilities', 'Sweeney', 2012),
    ('sweeney2013', 'The use of ensemble empirical mode decomposition with canonical correlation analysis as a novel artifact removal technique', 'Sweeney', 2013),
    ('uriguen2015', 'EEG artifact removal-state-of-the-art and guidelines', 'Urigüen', 2015),
    ('blankertz2011', 'Single-trial analysis and classification of ERP components-a tutorial', 'Blankertz', 2011),
    ('lin2006', 'Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs', 'Lin', 2006),
    ('holm1979', 'A simple sequentially rejective multiple test procedure', 'Holm', 1979),
    ('hodges1963', 'Estimates of location based on rank tests', 'Hodges', 1963),
    ('wilcoxon1945', 'Individual comparisons by ranking methods', 'Wilcoxon', 1945),
    ('varoquaux2018', 'Cross-validation failure: small sample sizes lead to large error bars', 'Varoquaux', 2018),
    ('gramfort2013', 'MEG and EEG data analysis with MNE-Python', 'Gramfort', 2013),
    ('pedregosa2011', 'Scikit-learn: Machine learning in Python', 'Pedregosa', 2011),
    ('virtanen2020', 'SciPy 1.0: fundamental algorithms for scientific computing in Python', 'Virtanen', 2020),
    ('harris2020', 'Array programming with NumPy', 'Harris', 2020),
    ('decheveigne2018', 'Robust detrending, rereferencing, outlier detection, and inpainting for multichannel data', 'de Cheveigné', 2018),
    ('cervera2018', 'Brain-computer interfaces for post-stroke motor rehabilitation: a meta-analysis', 'Cervera', 2018),
    ('wagner2012', 'Level of participation in robotic-assisted treadmill walking modulates midline sensorimotor EEG rhythms in able-bodied subjects', 'Wagner', 2012),
    ('lotte2018', 'A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update', 'Lotte', 2018),
    ('jacobsen2021', 'Mobile electroencephalography captures differences of walking over even and uneven terrain but not of single and dual-task gait', 'Jacobsen', 2021),
    ('efron1993', 'An Introduction to the Bootstrap', 'Efron', 1993),
    ('artoni2017', 'Unidirectional brain to muscle connectivity reveals motor cortex control of leg muscles during stereotyped walking', 'Artoni', 2017),
    ('kothe2013', 'BCILAB: a platform for brain-computer interface development', 'Kothe', 2013),
]


def q(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'mobile-eeg-benchmark/1.0 (mailto:xieyao@mail.ustc.edu.cn)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def norm(s):
    return ''.join(c for c in s.lower() if c.isalnum())


out = {}
L = ['# 参考文献核实(Crossref,首条命中)', '', '| key | 匹配 | 第一作者 | 年 | 刊 | 卷 | 页 | DOI | 题名 |', '|---|---|---|---|---|---|---|---|---|']
for key, title, fa, yr in CANDS:
    try:
        d = q('https://api.crossref.org/works?rows=3&query.bibliographic=' + urllib.parse.quote(title))
        items = d['message']['items']
    except Exception as e:
        items = []
        print(key, 'ERR', e)
    best = None
    for it in items:
        t = (it.get('title') or [''])[0]
        auth = it.get('author') or []
        fam = auth[0].get('family', '') if auth else ''
        y = (it.get('issued', {}).get('date-parts') or [[None]])[0][0]
        tmatch = norm(title)[:40] in norm(t) or norm(t)[:40] in norm(title)
        amatch = norm(fa)[:5] in norm(fam) or norm(fam)[:5] in norm(fa)
        if tmatch and amatch and y and abs(int(y) - yr) <= 1:
            best = it
            break
    if best is None:
        out[key] = {'status': 'UNVERIFIED', 'query': title}
        L.append(f'| {key} | UNVERIFIED | | | | | | | {title} |')
        print(key, 'UNVERIFIED', [((it.get("title") or [""])[0][:60], (it.get("author") or [{}])[0].get("family"), (it.get("issued", {}).get("date-parts") or [[None]])[0][0]) for it in items[:2]])
    else:
        auth = best.get('author') or []
        rec = {'status': 'OK', 'title': (best.get('title') or [''])[0], 'authors': [(a.get('family', ''), a.get('given', '')) for a in auth],
               'year': (best.get('issued', {}).get('date-parts') or [[None]])[0][0], 'journal': (best.get('container-title') or [''])[0],
               'short_journal': (best.get('short-container-title') or [''])[0] if best.get('short-container-title') else '',
               'volume': best.get('volume', ''), 'issue': best.get('issue', ''), 'page': best.get('page', ''), 'article': best.get('article-number', ''),
               'doi': best.get('DOI', ''), 'type': best.get('type', ''), 'publisher': best.get('publisher', '')}
        out[key] = rec
        L.append(f"| {key} | OK | {rec['authors'][0][0] if rec['authors'] else ''} | {rec['year']} | {rec['journal']} | {rec['volume']} | {rec['page'] or rec['article']} | {rec['doi']} | {rec['title'][:70]} |")
        print(key, 'OK', rec['year'], rec['journal'], rec['volume'], rec['page'] or rec['article'], rec['doi'])
    time.sleep(0.6)
json.dump(out, open(os.path.join(ROOT, 'results', 'refs_verified.json'), 'w'), indent=1, ensure_ascii=False)
open(os.path.join(ROOT, 'docs', 'refs_verified.md'), 'w').write('\n'.join(L))
print('done', sum(1 for v in out.values() if v['status'] == 'OK'), '/', len(out))
