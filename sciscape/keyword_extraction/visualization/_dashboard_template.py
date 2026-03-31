"""HTML template for the keyword extraction dashboard."""

_DASHBOARD_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f0f2f5;color:#1f2933;}
.header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);color:#fff;padding:1.2rem 2rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;position:relative;overflow:hidden;}
.header::after{content:"";position:absolute;bottom:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#00d4ff,#0d6efd,#7c3aed,#0d6efd,#00d4ff);background-size:200% 100%;animation:headerShimmer 4s linear infinite;}
@keyframes headerShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.header .brand{display:flex;align-items:center;gap:.75rem;}
.header .brand h1{font-size:1.4rem;font-weight:700;letter-spacing:-.01em;}
.header .brand .subtitle{font-size:.75rem;opacity:.55;font-weight:400;letter-spacing:.03em;}
.header .stats{display:flex;gap:1.5rem;font-size:.85rem;opacity:.85;}
.header .stats .stat-item{display:flex;flex-direction:column;align-items:center;}
.header .stats .stat-value{font-weight:700;font-size:1.1rem;color:#00d4ff;}
.header .stats .stat-label{font-size:.7rem;opacity:.6;text-transform:uppercase;letter-spacing:.04em;}
.controls{background:#fff;border-bottom:1px solid #dee2e6;padding:.75rem 2rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;position:sticky;top:0;z-index:100;box-shadow:0 2px 4px rgba(0,0,0,.05);}
.controls label{font-weight:600;font-size:.9rem;}
.controls select{padding:.4rem .8rem;border:1px solid #ced4da;border-radius:4px;font-size:.9rem;min-width:20rem;}
.tabs{display:flex;gap:0;border-bottom:2px solid #dee2e6;padding:0 2rem;background:#fff;}
.tab{padding:.7rem 1.2rem;cursor:pointer;font-size:.9rem;font-weight:500;color:#6c757d;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s;}
.tab:hover{color:#212529;background:#f8f9fa;}
.tab.active{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600;}
.tab-divider{width:1px;background:#ced4da;margin:0.4rem 0.3rem;align-self:stretch;}
.tab-group-label{font-size:.7rem;font-weight:700;color:#adb5bd;text-transform:uppercase;letter-spacing:.05em;padding:.7rem .4rem .7rem .6rem;align-self:center;user-select:none;}
.controls.dimmed .cluster-controls{opacity:.35;pointer-events:none;}
.container{max-width:1400px;margin:0 auto;padding:1.5rem 2rem;}
.panel{display:none;}
.panel.active{display:block;}
.chart-card{background:#fff;border-radius:8px;padding:1rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,.08);}
.panel-desc{background:#e8f4fd;border-left:4px solid #0d6efd;border-radius:4px;padding:.75rem 1rem;margin-bottom:1rem;font-size:.9rem;line-height:1.6;color:#2c3e50;}
.cluster-meta{background:#fff;border-radius:8px;padding:1rem 1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,.08);display:flex;gap:2rem;flex-wrap:wrap;font-size:.9rem;}
.cluster-meta .meta-item{flex:1;min-width:140px;}
.cluster-meta .meta-label{font-weight:600;color:#6c757d;font-size:.8rem;text-transform:uppercase;}
.cluster-meta .meta-value{font-size:1.1rem;font-weight:600;margin-top:.2rem;}
#network-container{width:100%;height:600px;position:relative;overflow:hidden;}
#network-container svg{width:100%;height:100%;}
.node-label{font-size:11px;pointer-events:none;text-anchor:middle;}
.merge-table{width:100%;border-collapse:collapse;font-size:.9rem;}
.merge-table th{background:#343a40;color:#fff;padding:.6rem 1rem;text-align:left;font-size:.85rem;position:sticky;top:0;}
.merge-table td{padding:.5rem 1rem;border-bottom:1px solid #e9ecef;}
.merge-table tr:hover td{background:#f8f9fa;}
.search-box{padding:.4rem .8rem;border:1px solid #ced4da;border-radius:4px;font-size:.9rem;width:100%;max-width:24rem;margin-bottom:.75rem;}
.metric-select{padding:.35rem .6rem;border:1px solid #ced4da;border-radius:4px;font-size:.85rem;margin-bottom:.5rem;}
.dict-subtabs{display:flex;gap:0;border-bottom:1px solid #dee2e6;margin-bottom:.75rem;}
.dict-subtab{padding:.5rem 1rem;cursor:pointer;font-size:.85rem;font-weight:500;color:#6c757d;border-bottom:2px solid transparent;margin-bottom:-1px;}
.dict-subtab:hover{color:#212529;}
.dict-subtab.active{color:#0d6efd;border-bottom-color:#0d6efd;font-weight:600;}
.badge{display:inline-block;padding:.15rem .45rem;border-radius:10px;font-size:.75rem;font-weight:600;margin-left:.3rem;}
.badge-vocab{background:#d4edda;color:#155724;}
.badge-norm{background:#cce5ff;color:#004085;}
.footer{text-align:center;padding:1.5rem 2rem;font-size:.8rem;color:#6c757d;border-top:1px solid #e9ecef;margin-top:2rem;display:flex;align-items:center;justify-content:center;gap:.5rem;}
.footer svg{opacity:.5;transition:opacity .2s;}
.footer:hover svg{opacity:.8;}
.footer a{color:#6c757d;text-decoration:none;}
.footer a:hover{color:#0d6efd;}
.export-btn{padding:.3rem .7rem;border:1px solid #ced4da;border-radius:4px;font-size:.8rem;background:#fff;cursor:pointer;color:#495057;}
.export-btn:hover{background:#e9ecef;}
.trend-badge{display:inline-block;padding:.2rem .5rem;border-radius:12px;font-size:.8rem;font-weight:600;margin:.15rem;}
.trend-up{background:#d4edda;color:#155724;}
.trend-down{background:#f8d7da;color:#721c24;}
.search-result{padding:.5rem 1rem;cursor:pointer;border-bottom:1px solid #f0f0f0;font-size:.85rem;}
.search-result:hover{background:#f8f9fa;}
.search-result .sr-cluster{color:#6c757d;font-size:.8rem;}
.compare-tag{display:inline-block;padding:.2rem .5rem;background:#e9ecef;border-radius:12px;font-size:.8rem;margin:.15rem;cursor:pointer;}
.compare-tag:hover{background:#dee2e6;text-decoration:line-through;}
.network-info-panel{position:absolute;background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:.75rem 1rem;box-shadow:0 4px 12px rgba(0,0,0,.15);font-size:.85rem;z-index:50;max-width:300px;pointer-events:auto;}
.network-info-panel .info-close{position:absolute;top:4px;right:8px;cursor:pointer;font-size:1rem;color:#6c757d;}
.network-info-panel .info-close:hover{color:#212529;}
.note-indicator{cursor:pointer;color:#f0ad4e;font-size:.85rem;}
.note-cell{cursor:pointer;min-width:80px;max-width:200px;color:#495057;font-size:.85rem;}
.note-cell:empty::after{content:"click to add";color:#ccc;font-style:italic;}
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <img src="data:image/png;base64,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" width="44" height="44" alt="SciScape" style="border-radius:6px;">
    <div>
      <h1>{{TITLE}}</h1>
      <div class="subtitle">Scientific Landscape Explorer</div>
    </div>
  </div>
  <div class="stats">
    <div class="stat-item"><span class="stat-value">{{N_KEYWORDS}}</span><span class="stat-label">Keywords</span></div>
    <div class="stat-item"><span class="stat-value">{{N_CLUSTERS}}</span><span class="stat-label">Clusters</span></div>
  </div>
</div>

<div class="controls" id="controls-bar" style="position:relative;">
  <span class="cluster-controls">
    <label for="cluster-select">Cluster:</label>
    <select id="cluster-select"></select>
  </span>
  <input type="text" class="search-box" id="global-search" placeholder="Search all keywords..." style="min-width:16rem;margin-bottom:0;">
  <div id="global-search-results" style="position:absolute;top:100%;left:0;right:0;background:#fff;border:1px solid #dee2e6;border-radius:0 0 4px 4px;max-height:300px;overflow-y:auto;z-index:200;display:none;box-shadow:0 4px 8px rgba(0,0,0,.1);"></div>
  <button class="export-btn" onclick="exportCurrentData()">&#x2B07; Export CSV</button>
</div>

<div class="tabs" id="tabs">
  <div class="tab-group-label">Global</div>
  <div class="tab active" data-tab="overview" data-scope="global">Overview</div>
  <div class="tab" data-tab="crosscluster" data-scope="global">Cross-Cluster</div>
  <div class="tab-divider"></div>
  <div class="tab-group-label">All / Cluster</div>
  <div class="tab" data-tab="keywords">Keywords</div>
  <div class="tab" data-tab="temporal">Temporal</div>
  <div class="tab" data-tab="hierarchy">Hierarchy</div>
  <div class="tab" data-tab="network">Network</div>
  <div class="tab" data-tab="dictionary">Dictionary</div>
</div>

<div class="container">
  <div class="cluster-meta" id="cluster-meta"></div>

  <div class="panel active" id="panel-overview">
    <div class="panel-desc">
      전체 클러스터의 요약 정보를 한눈에 비교합니다.
      버블 크기는 키워드 수, 색상은 평균 점수, 위치는 emerging 비율과 평균 centrality를 나타냅니다.
    </div>
    <div class="chart-card"><div id="chart-overview" style="width:100%;min-height:500px;"></div></div>
    <div class="chart-card" id="overview-table-container"></div>
    <details style="margin-top:1rem;">
      <summary style="cursor:pointer;font-weight:600;font-size:.9rem;">Pipeline Configuration</summary>
      <div id="pipeline-config-display" style="margin-top:.5rem;"></div>
    </details>
  </div>

  <div class="panel" id="panel-keywords">
    <div class="panel-desc">
      클러스터를 선택하면 해당 클러스터의 키워드 랭킹을 표시합니다.
      선택하지 않으면 전체 클러스터의 Top 키워드를 요약합니다.
    </div>
    <div id="keywords-global" style="display:none;">
      <div class="chart-card"><div id="chart-keywords-global" style="width:100%;min-height:500px;"></div></div>
      <div class="chart-card"><div id="keywords-global-table" style="max-height:600px;overflow:auto;"></div></div>
    </div>
    <div id="keywords-cluster" style="display:none;">
      <div class="chart-card"><div id="chart-keywords" style="width:100%;min-height:500px;"></div></div>
    </div>
  </div>

  <div class="panel" id="panel-temporal">
    <div class="panel-desc">
      키워드의 연도별 추이를 보여줍니다.
      지표(Metric)를 전환하여 문서 수(Documents), PPM(Parts Per Million), Log-Lift를 비교할 수 있습니다.
      범례를 클릭하면 개별 키워드 곡선을 켜고 끌 수 있으며, 기본 상위 5개만 표시됩니다.
    </div>
    <div class="chart-card">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <label>Metric: </label>
        <select class="metric-select" id="temporal-metric">
          <option value="pub_year_series" selected>Documents per Year</option>
          <option value="ppm_series">PPM (Parts Per Million)</option>
          <option value="loglift_series">Log-Lift</option>
        </select>
        <button class="export-btn" id="temporal-compare-btn">Compare</button>
        <button class="export-btn" id="temporal-compare-clear" style="display:none;">Clear comparison</button>
      </div>
      <div id="temporal-compare-controls" style="display:none;margin-top:.5rem;">
        <input type="text" class="search-box" id="temporal-compare-search" placeholder="Search keywords to compare (any cluster)..." style="margin-bottom:.3rem;">
        <div id="temporal-compare-results" style="max-height:150px;overflow-y:auto;border:1px solid #dee2e6;border-radius:4px;display:none;background:#fff;"></div>
        <div id="temporal-compare-tags" style="margin-top:.3rem;"></div>
      </div>
      <div id="temporal-highlights" style="margin:.5rem 0;"></div>
      <div id="chart-temporal" style="width:100%;min-height:500px;"></div>
    </div>
  </div>

  <div class="panel" id="panel-hierarchy">
    <div class="panel-desc">
      키워드의 계보도(hierarchy)를 두 가지 방식으로 보여줍니다.<br>
      <b>Subphrase Tree</b>: 한 키워드가 다른 키워드를 단어 수준에서 포함하는 관계
      (예: "point cloud" ⊂ "lidar point cloud")를 트리로 표현합니다.<br>
      <b>Depth Sunburst</b>: 깊이 수준(Broad → Mid → Specific)별 계층 구조입니다.
    </div>
    <div class="chart-card">
      <div class="dict-subtabs" id="hierarchy-subtabs">
        <div class="dict-subtab active" data-view="depth">Depth Sunburst</div>
        <div class="dict-subtab" data-view="subphrase">Subphrase Tree</div>
      </div>
      <div id="chart-hierarchy" style="width:100%;min-height:500px;"></div>
    </div>
  </div>

  <div class="panel" id="panel-network">
    <div class="panel-desc" id="network-desc">
      키워드 간 <b>실제 공출현(co-occurrence) 네트워크</b>입니다.
      동일 문헌에서 함께 등장한 빈도를 기반으로 엣지가 생성됩니다.
      엣지 두께는 공출현 빈도에, 노드 크기는 점수에 비례합니다.
      색상은 깊이 수준(Broad=파랑, Mid=빨강, Specific=초록)입니다. 드래그와 줌이 가능합니다.
    </div>
    <div class="chart-card">
      <div class="dict-subtabs" id="network-subtabs" style="display:none;">
        <div class="dict-subtab active" data-view="cluster">Cluster Network</div>
        <div class="dict-subtab" data-view="keyword">Keyword Network</div>
      </div>
      <div style="margin-bottom:.5rem;display:flex;align-items:center;gap:1rem;">
        <label style="font-size:.85rem;">Min edge weight:</label>
        <input type="range" id="network-edge-slider" min="0" max="100" value="0" style="flex:1;max-width:300px;">
        <span id="network-edge-value" style="font-size:.85rem;min-width:3rem;">0</span>
      </div>
      <div id="network-container" style="position:relative;"><svg id="network-svg"></svg></div>
    </div>
  </div>

  <div class="panel" id="panel-dictionary">
    <div class="panel-desc">
      키워드 사전과 클렌징(merge) 이력을 확인합니다.<br>
      <b>Keywords</b>: 전체 키워드 목록 (점수, 빈도, 깊이, 정규화 이력 포함).<br>
      <b>Vocab Merges</b>: Stage 2에서 수행된 어휘 병합 (복수형→단수형, 하이픈 정규화 등).<br>
      <b>Norm Merges</b>: Stage 5에서 수행된 정규화 병합 (약어 확장, 철자 통합, 편집 거리 병합 등).
    </div>
    <div class="chart-card">
      <div class="dict-subtabs" id="dict-subtabs">
        <div class="dict-subtab active" data-view="keywords">Keywords</div>
        <div class="dict-subtab" data-view="vocab">Vocab Merges</div>
        <div class="dict-subtab" data-view="norm">Norm Merges</div>
      </div>
      <input type="text" class="search-box" id="dict-search" placeholder="Search...">
      <div id="dict-table-container" style="max-height:600px;overflow-y:auto;"></div>
    </div>
  </div>

  <div class="panel" id="panel-crosscluster">
    <div class="panel-desc">
      여러 클러스터에 걸쳐 등장하는 키워드를 보여줍니다. 학제간 연결고리(bridging term)를 파악할 수 있습니다.
      "Score"는 원점수, "Rank"는 클러스터 내 순위를 정규화(0~1)한 값입니다.
    </div>
    <div class="chart-card">
      <label>Color: </label>
      <select class="metric-select" id="crosscluster-mode">
        <option value="rank" selected>Rank (normalized)</option>
        <option value="score">Score (raw)</option>
        <option value="frequency">Frequency</option>
      </select>
      <div id="chart-crosscluster" style="width:100%;min-height:500px;"></div>
    </div>
  </div>
</div>

<div class="footer">
  <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAFZUlEQVR42u2X228bxxXGfzO7y11yKd5EipKoiyXHqpqoTmsHDZo2bRIgD33qcxCg/19RoA99CBK0RZreUrhG5VZOZVmxLhZpUhRpkiKXt92d6YNIQ4UaSmoKGAU6LzvYwc755nzfd86s0FprXuKQvOTxvw9AvywA48ACOK+i6wIyrxtYjaIJIah5AzQwHbNQCqSUI0AaIcR/D4AGlFKESjMMQ4QQGELw54MThBD8cDVLEGp8pQiURqDJTrlIwJDi61EwPrEebSaFxDFNdqptat6Ak1aXrVITxzJodgfU6g22vnzK7lEZbxhcSstEAFpr/FARhIq+HxAojdIKQwra/SERKWh6A/5RqtH1QzqnbYJQ06kesfnFNqfdPoFS/yqS6wBQGkKlCJTCEAIhwJSSvxzWeFBqkp+Kkk04PCrWuPe4xEw6zif3v0DHs9xYWqTldQmVhgl6mKgBQ55xbVsmfz2s4UZMIpbB1rMmS6koCdvkmVKYpsHmXpmFjMuHP/oO+8dNlmemyWWS2KYxkWLxVaU4CEcnl4J2b8hvHh5x7Pm40Qg3MjH6gUYKzaPjBlFLUqp3iGLx3us52sFT7szfQSmJZYJtmtiWeXUAWmu0PhPPbx/uc1jrEAiDmUSMrXKLjUKKhBNBodl/XiEbS7NTrmPbDXy/w62lFJ1OhzcWN1jPr6C1wjDkqGpcgQIx4vvBkyJ75QbxeIyYY7OcS+BreNbqMh13QAsy0SQp1+OtbxhEzAz1nkVBhRhLr/Kno78RtR1WMvNXo+Bspml6fT66t83aQo5WP6QzCElOuXSDNpZ0CJSk3QtYzZnEHKi0Syyn8vgyQlDeYrH5mF7hDU4Cn+qgjzPo8f3X3gF7aqRHMckFZ4tzaRdHgikF31xIYFltvGAXx6nSDxvcueHgBU36QYtX59Y49X3KB5+z2tjBzqyQN2HdjrJsL5M20nxV1xeT7gOtjseDwyqldoVsOmA9f4t6p8lu/ZD81DQLyRz9IKTdquD0n7MxqBIk5iC9QuP4iE5okb75Jtmp6PVcoEdClELgh4rnjTabxW20qbi9+AqbpW2S0TTdcIhru9weVIkHHQLL5rTZoKXiOCtvUUhHwYigtUIIef0M6HNseYMeh5UqO/V9knGbJTeK1zpmultlxpR0g4Cys4EZS7MyN40ZTV7Y49oAztty3N3q7SYH1SJOt8VaxkZoOGgMMfw202s/IBF3L3zzHxWi8xuM50oppDxLZaXRpfH4UyQa99a7FDLui1Y8trJSGnlJN5xMwaWn8FFDHxmJXTmDlxaicRX8xa9+j/B9vnt3g3tbO4RKsTyfx+v12S9WeO2VG0QiEbZ39lhcyPP6+k1++ennfPDjd/jZx58Rs20iEZOTepPVpXnevnt7lJkrNSONYUg6vYCPfnePQj7Hu29+m0/+eB+v18eQkp39IxqnbX76k/fxun1+/vFnbD7cIRV3KVZqLMxm2SuVGfoBt9dvjrLABUleoGCcrgePnvBk7ynzhTzNU49ev8/yfJ7UVJzdgyJ932dhNkexVMGNu9gRi9XCLA+fHDKbzfCttRVqjRa//sN9lhZmefvuBlqLCxmY0Iw0f3+8x+5BkYV8lkjE4qhyglaaW8sFEIIvD4u4UQelFPP5HADHteekE3EM02T3oMh0MsF737uDGtWVK98HpJTYlsWUGwUNlmGgQoXvBwyGPlrD0A/JpmzqrVM6Xhc/CBkMhiilIQxRWjPw/a9nw/1imb2jZyTdGAPfpz8Yks2k8P0Ar9cjk0rS6XhIQ2IaxguNIARLc3kW52YmuuBSG449/e9qw/gppXyxNn5//vouJ1hZ/P/n9GUD+CfcGuVIq0FOHgAAAABJRU5ErkJggg==" width="16" height="16" alt="" style="border-radius:2px;">
  <span>Generated by <b>SciScape</b> v0.1 &mdash; <a href="https://github.com/KISTI-GlobalRnD/sciscape" target="_blank">GitHub</a> &mdash; Center for Global R&amp;D, KISTI</span>
</div>

<!-- Upload overlay (shown when DATA is null — viewer mode) -->
<div id="upload-overlay" style="display:none;position:fixed;inset:0;background:rgba(26,26,46,.95);z-index:9999;align-items:center;justify-content:center;">
  <div id="drop-zone" style="border:3px dashed #00d4ff;border-radius:16px;padding:4rem 3rem;text-align:center;max-width:560px;width:90%;cursor:pointer;transition:background .2s;">
    <div style="font-size:3rem;margin-bottom:1rem;">&#128194;</div>
    <h2 style="color:#fff;font-size:1.5rem;margin-bottom:.5rem;">SciScape Viewer</h2>
    <p style="color:#adb5bd;font-size:1rem;margin-bottom:1.5rem;">Drop a file here or click to browse</p>
    <div style="display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;margin-bottom:1rem;">
      <span style="background:#2d2d44;padding:.3rem .7rem;border-radius:4px;color:#00d4ff;font-size:.85rem;">data.json</span>
      <span style="background:#2d2d44;padding:.3rem .7rem;border-radius:4px;color:#4ecdc4;font-size:.85rem;">keywords.csv</span>
      <span style="background:#2d2d44;padding:.3rem .7rem;border-radius:4px;color:#4ecdc4;font-size:.85rem;">keywords.tsv</span>
    </div>
    <p style="color:#6c757d;font-size:.85rem;">CSV/TSV: requires <b>cluster_id</b>, <b>term</b>, <b>score</b> columns (tab or comma separated)</p>
    <input type="file" id="file-input" accept=".json,.csv,.tsv,.txt" style="display:none;">
    <div id="upload-error" style="color:#ff6b6b;margin-top:1rem;display:none;"></div>
  </div>
</div>

<script>
let DATA = {{DATA_JSON}};

// ── Upload / Viewer mode ──
let VOCAB_MERGES = {}, NORM_MERGES = {}, CROSS_CLUSTER_TERMS = null;
let TREND_SCORES = {}, CENTRALITY = {}, PIPELINE_CONFIG = {};

function _extractGlobals() {
  VOCAB_MERGES = DATA["_vocab_merges"] || {};
  NORM_MERGES = DATA["_norm_merges"] || {};
  CROSS_CLUSTER_TERMS = DATA["_cross_cluster_terms"] || null;
  TREND_SCORES = DATA["_trend_scores"] || {};
  CENTRALITY = DATA["_centrality"] || {};
  PIPELINE_CONFIG = DATA["_pipeline_config"] || {};
  delete DATA["_vocab_merges"];
  delete DATA["_norm_merges"];
  delete DATA["_cross_cluster_terms"];
  delete DATA["_trend_scores"];
  delete DATA["_centrality"];
  delete DATA["_pipeline_config"];
}

const clusterSelect = document.getElementById("cluster-select");
const metaEl = document.getElementById("cluster-meta");
const tabEls = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
let currentCluster = null;
let currentClusterId = null;
let currentTab = "overview";
let hierarchyView = "depth";
let networkView = "cluster";
let dictView = "keywords";
let clusterIds = [];

function _initClusters() {
  // Clear existing options
  clusterSelect.innerHTML = '<option value="">\u2726 All Clusters</option>';
  clusterIds = Object.keys(DATA).map(Number).filter(n => !isNaN(n)).sort((a,b) => a-b);
  clusterIds.forEach(cid => {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = `C${cid}: ${DATA[cid].label}`;
    clusterSelect.appendChild(opt);
  });
  // Update header stats
  const totalKw = clusterIds.reduce((s, c) => s + (DATA[c].keywords||[]).length, 0);
  const statsEl = document.querySelector(".header .stats");
  if (statsEl) {
    statsEl.innerHTML = `<div class="stat-item"><span class="stat-value">${totalKw.toLocaleString()}</span><span class="stat-label">Keywords</span></div><div class="stat-item"><span class="stat-value">${clusterIds.length}</span><span class="stat-label">Clusters</span></div>`;
  }
}

// ── Upload handler ──
const overlay = document.getElementById("upload-overlay");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadError = document.getElementById("upload-error");

function _loadJSON(text) {
  try {
    DATA = JSON.parse(text);
    _extractGlobals();
    _initClusters();
    overlay.style.display = "none";
    if (clusterSelect.options.length > 1) {
      clusterSelect.selectedIndex = 1;
      clusterSelect.dispatchEvent(new Event("change"));
    }
  } catch (e) {
    uploadError.textContent = "Invalid JSON: " + e.message;
    uploadError.style.display = "block";
  }
}

function _parseCSV(text, sep) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV has no data rows");
  const header = lines[0].split(sep).map(h => h.trim().toLowerCase().replace(/^"|"$/g, ""));
  const ciIdx = header.indexOf("cluster_id");
  const tIdx = header.indexOf("term");
  const sIdx = header.indexOf("score");
  if (ciIdx < 0 || tIdx < 0) throw new Error("CSV must have 'cluster_id' and 'term' columns. Found: " + header.join(", "));

  const freqIdx = header.indexOf("frequency");
  const dcIdx = header.indexOf("doc_coverage");
  const depthIdx = header.indexOf("depth_level");

  const clusters = {};
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(sep).map(c => c.trim().replace(/^"|"$/g, ""));
    if (cols.length < header.length) continue;
    const cid = parseInt(cols[ciIdx], 10);
    if (isNaN(cid)) continue;
    if (!clusters[cid]) clusters[cid] = { label: "", keywords: [], network_edges: [], subphrase_tree: [], norm_merges: {} };
    const kw = { term: cols[tIdx], score: sIdx >= 0 ? parseFloat(cols[sIdx]) || 0 : 0 };
    if (freqIdx >= 0) kw.frequency = parseInt(cols[freqIdx], 10) || 0;
    if (dcIdx >= 0) kw.doc_coverage = parseInt(cols[dcIdx], 10) || 0;
    if (depthIdx >= 0) kw.depth_level = parseInt(cols[depthIdx], 10) || 0;
    clusters[cid].keywords.push(kw);
  }
  // Sort keywords by score descending and create labels
  for (const cid of Object.keys(clusters)) {
    clusters[cid].keywords.sort((a, b) => b.score - a.score);
    clusters[cid].label = clusters[cid].keywords.slice(0, 3).map(k => k.term).join(", ");
  }
  return clusters;
}

function _loadCSV(text, filename) {
  try {
    const sep = (filename.endsWith(".tsv") || text.split("\n")[0].split("\t").length > 2) ? "\t" : ",";
    DATA = _parseCSV(text, sep);
    VOCAB_MERGES = {}; NORM_MERGES = {}; CROSS_CLUSTER_TERMS = null;
    TREND_SCORES = {}; CENTRALITY = {}; PIPELINE_CONFIG = {};
    _initClusters();
    overlay.style.display = "none";
    if (clusterSelect.options.length > 1) {
      clusterSelect.selectedIndex = 1;
      clusterSelect.dispatchEvent(new Event("change"));
    }
  } catch (e) {
    uploadError.textContent = "CSV error: " + e.message;
    uploadError.style.display = "block";
  }
}

function _handleFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  const valid = name.endsWith(".json") || name.endsWith(".csv") || name.endsWith(".tsv") || name.endsWith(".txt");
  if (!valid) {
    uploadError.textContent = "Supported formats: .json, .csv, .tsv";
    uploadError.style.display = "block";
    return;
  }
  uploadError.style.display = "none";
  const reader = new FileReader();
  reader.onload = (e) => {
    if (name.endsWith(".json")) { _loadJSON(e.target.result); }
    else { _loadCSV(e.target.result, name); }
  };
  reader.readAsText(file);
}

if (dropZone) {
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => _handleFile(e.target.files[0]));
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.style.background = "rgba(0,212,255,.1)"; });
  dropZone.addEventListener("dragleave", () => { dropZone.style.background = ""; });
  dropZone.addEventListener("drop", (e) => { e.preventDefault(); dropZone.style.background = ""; _handleFile(e.dataTransfer.files[0]); });
}

// ── Init: embedded data or viewer mode ──
if (DATA) {
  _extractGlobals();
  _initClusters();
} else {
  overlay.style.display = "flex";
}

// Tab switching
const globalOnlyTabs = new Set(["overview", "crosscluster"]);
function updateScopeUI() {
  const bar = document.getElementById("controls-bar");
  if (globalOnlyTabs.has(currentTab)) {
    bar.classList.add("dimmed");
  } else {
    bar.classList.remove("dimmed");
  }
  metaEl.style.display = "flex";
}

tabEls.forEach(tab => {
  tab.addEventListener("click", () => {
    tabEls.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const tabName = tab.dataset.tab;
    panels.forEach(p => p.classList.remove("active"));
    document.getElementById("panel-" + tabName).classList.add("active");
    currentTab = tabName;
    updateScopeUI();
    updateHash();
    renderCurrentTab();
  });
});

// Hierarchy subtabs
document.querySelectorAll("#hierarchy-subtabs .dict-subtab").forEach(st => {
  st.addEventListener("click", () => {
    document.querySelectorAll("#hierarchy-subtabs .dict-subtab").forEach(s => s.classList.remove("active"));
    st.classList.add("active");
    hierarchyView = st.dataset.view;
    renderHierarchy();
  });
});

// Network subtabs
document.querySelectorAll("#network-subtabs .dict-subtab").forEach(st => {
  st.addEventListener("click", () => {
    document.querySelectorAll("#network-subtabs .dict-subtab").forEach(s => s.classList.remove("active"));
    st.classList.add("active");
    networkView = st.dataset.view;
    renderNetwork();
  });
});

// Dictionary subtabs
document.querySelectorAll("#dict-subtabs .dict-subtab").forEach(st => {
  st.addEventListener("click", () => {
    document.querySelectorAll("#dict-subtabs .dict-subtab").forEach(s => s.classList.remove("active"));
    st.classList.add("active");
    dictView = st.dataset.view;
    renderDictionary();
  });
});

// Temporal metric switch
document.getElementById("temporal-metric").addEventListener("change", () => renderTemporal());
document.getElementById("crosscluster-mode").addEventListener("change", () => renderCrossCluster());

// Cluster change
clusterSelect.addEventListener("change", () => {
  const val = clusterSelect.value;
  if (val === "") {
    currentClusterId = null;
    currentCluster = null;
  } else {
    currentClusterId = Number(val);
    currentCluster = DATA[currentClusterId];
  }
  updateMeta();
  updateScopeUI();
  updateHash();
  renderCurrentTab();
});

function updateMeta() {
  if (!currentCluster) {
    // Global summary
    const totalKw = clusterIds.reduce((s, c) => s + (DATA[c].keywords||[]).length, 0);
    const allKws = clusterIds.flatMap(c => DATA[c].keywords || []);
    const avgScore = allKws.length ? (allKws.reduce((s, k) => s + k.score, 0) / allKws.length).toFixed(4) : "N/A";
    const depthCounts = {};
    allKws.forEach(k => { if (k.depth_level != null) depthCounts[k.depth_level] = (depthCounts[k.depth_level]||0)+1; });
    const depthStr = Object.entries(depthCounts).sort((a,b)=>a[0]-b[0]).map(([l,c]) => `L${l}: ${c}`).join(", ") || "N/A";
    metaEl.innerHTML = `
      <div class="meta-item"><div class="meta-label">Clusters</div><div class="meta-value">${clusterIds.length}</div></div>
      <div class="meta-item"><div class="meta-label">Total Keywords</div><div class="meta-value">${totalKw}</div></div>
      <div class="meta-item"><div class="meta-label">Avg Score</div><div class="meta-value">${avgScore}</div></div>
      <div class="meta-item"><div class="meta-label">Depth</div><div class="meta-value">${depthStr}</div></div>
    `;
    metaEl.style.display = "flex";
    return;
  }
  const kws = currentCluster.keywords;
  const avgScore = (kws.reduce((s, k) => s + k.score, 0) / kws.length).toFixed(4);
  const avgFreq = Math.round(kws.reduce((s, k) => s + k.frequency, 0) / kws.length).toLocaleString();
  const depthCounts = {};
  kws.forEach(k => { if (k.depth_level != null) depthCounts[k.depth_level] = (depthCounts[k.depth_level]||0)+1; });
  const depthStr = Object.entries(depthCounts).sort((a,b)=>a[0]-b[0])
    .map(([l,c]) => `L${l}: ${c}`).join(", ") || "N/A";
  const nEdges = currentCluster.network_edges ? currentCluster.network_edges.length : 0;
  const nSub = currentCluster.subphrase_tree ? currentCluster.subphrase_tree.length : 0;
  const nNorm = Object.keys(currentCluster.norm_merges || {}).length;

  metaEl.innerHTML = `
    <div class="meta-item"><div class="meta-label">Keywords</div><div class="meta-value">${kws.length}</div></div>
    <div class="meta-item"><div class="meta-label">Avg Score</div><div class="meta-value">${avgScore}</div></div>
    <div class="meta-item"><div class="meta-label">Avg Frequency</div><div class="meta-value">${avgFreq}</div></div>
    <div class="meta-item"><div class="meta-label">Depth</div><div class="meta-value">${depthStr}</div></div>
    <div class="meta-item"><div class="meta-label">Cooc Edges</div><div class="meta-value">${nEdges}</div></div>
    <div class="meta-item"><div class="meta-label">Subphrases</div><div class="meta-value">${nSub}</div></div>
    <div class="meta-item"><div class="meta-label">Norm Merges</div><div class="meta-value">${nNorm}</div></div>
  `;
}

function renderCurrentTab() {
  if (currentTab === "crosscluster") { renderCrossCluster(); return; }
  if (currentTab === "overview") { renderOverview(); return; }
  if (currentTab === "keywords") { renderKeywords(); return; }
  if (currentTab === "temporal") { renderTemporal(); return; }
  if (currentTab === "hierarchy") { renderHierarchy(); return; }
  if (currentTab === "network") { renderNetwork(); return; }
  if (currentTab === "dictionary") { renderDictionary(); return; }
}

const depthNames = { 0: "Broad", 1: "Mid", 2: "Specific" };
const depthColors = { 0: "#636EFA", 1: "#EF553B", 2: "#00CC96" };

// ---- Tab 1: Keywords ----
function renderKeywords() {
  const globalEl = document.getElementById("keywords-global");
  const clusterEl = document.getElementById("keywords-cluster");
  if (!currentCluster) {
    // Global view: top keywords per cluster
    globalEl.style.display = "block";
    clusterEl.style.display = "none";
    renderKeywordsGlobal();
    return;
  }
  globalEl.style.display = "none";
  clusterEl.style.display = "block";

  const kws = currentCluster.keywords.slice(0, 30);
  const terms = kws.map(k => k.term).reverse();
  const scores = kws.map(k => k.score).reverse();
  const colors = kws.map(k => depthColors[k.depth_level] || "#adb5bd").reverse();
  const hoverText = kws.map(k => {
    let h = `<b>${k.term}</b><br>Score: ${k.score.toFixed(4)}<br>Freq: ${k.frequency.toLocaleString()}<br>Doc coverage: ${k.doc_coverage.toLocaleString()}`;
    if (k.depth_level != null) h += `<br>Depth: ${depthNames[k.depth_level] || "L"+k.depth_level} (${k.depth_score.toFixed(3)})`;
    if (k.cross_cluster_count > 1) h += `<br>Cross-cluster: ${k.cross_cluster_count}`;
    const ts = TREND_SCORES[k.term];
    if (ts != null) h += `<br>Trend: ${ts > 0 ? "▲" : "▼"} ${ts.toFixed(3)}`;
    const ct = CENTRALITY[k.term];
    if (ct) h += `<br>Centrality: ${ct.degree.toFixed(2)} (weighted: ${ct.weighted_degree.toFixed(2)})`;
    return h;
  }).reverse();

  Plotly.react("chart-keywords", [{
    type: "bar", orientation: "h",
    y: terms, x: scores,
    hovertext: hoverText, hoverinfo: "text",
    marker: { color: colors }
  }], {
    title: "Keyword Ranking (colored by depth)",
    xaxis: { title: "Score" },
    margin: { l: 200, r: 30, t: 50, b: 40 },
    height: Math.max(400, kws.length * 22 + 80),
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });
}

function renderKeywordsGlobal() {
  const topN = 5;
  // Collect top keywords per cluster
  const rows = [];
  const traces = [];
  const palette = Plotly.d3 ? Plotly.d3.scale.category20().range() :
    ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
     "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"];
  clusterIds.forEach((cid, ci) => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    const kws = cl.keywords.slice(0, topN);
    const label = cl.label || `C${cid}`;
    kws.forEach((k, ki) => {
      rows.push({ cluster: cid, cluster_label: label, rank: ki + 1, term: k.term, score: k.score, frequency: k.frequency });
    });
    traces.push({
      name: `C${cid}`,
      type: "bar",
      x: kws.map(k => k.term),
      y: kws.map(k => k.score),
      text: kws.map(k => k.score.toFixed(3)),
      textposition: "outside",
      hovertext: kws.map(k => `<b>${k.term}</b><br>Cluster: ${label}<br>Score: ${k.score.toFixed(4)}<br>Freq: ${k.frequency.toLocaleString()}`),
      hoverinfo: "text",
      marker: { color: palette[ci % palette.length] }
    });
  });

  // Grouped bar chart
  Plotly.react("chart-keywords-global", traces, {
    title: `Top ${topN} Keywords per Cluster`,
    barmode: "group",
    xaxis: { title: "Keyword", tickangle: -45 },
    yaxis: { title: "Score" },
    margin: { l: 60, r: 30, t: 50, b: 120 },
    height: 500,
    legend: { orientation: "h", y: -0.35 },
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });

  // Summary table
  let html = '<table class="merge-table"><thead><tr><th>Cluster</th><th>Rank</th><th>Keyword</th><th>Score</th><th>Frequency</th></tr></thead><tbody>';
  rows.forEach(r => {
    html += `<tr style="cursor:pointer" onclick="document.getElementById('cluster-select').value='${r.cluster}';document.getElementById('cluster-select').dispatchEvent(new Event('change'))">`;
    html += `<td>${r.cluster_label}</td><td>${r.rank}</td><td><b>${r.term}</b></td><td>${r.score.toFixed(4)}</td><td>${r.frequency.toLocaleString()}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById("keywords-global-table").innerHTML = html;
}

// ---- Tab 2: Temporal ----
function renderTemporal() {
  if (!currentCluster) { renderTemporalGlobal(); return; }
  renderTemporalHighlights();
  const metric = document.getElementById("temporal-metric").value;
  const metricLabels = {
    "pub_year_series": "Documents per Year",
    "ppm_series": "PPM (Parts Per Million)",
    "loglift_series": "Log-Lift"
  };
  const yLabel = metricLabels[metric] || metric;

  // Filter keywords that have this metric
  const kws = currentCluster.keywords.filter(k => {
    const d = k[metric] || k.temporal;
    return d && Object.keys(d).length > 0;
  });
  const topN = kws.slice(0, 15);
  const defaultVisible = 5;

  const traces = topN.map((k, i) => {
    const data = k[metric] || k.temporal || {};
    const years = Object.keys(data).map(Number).sort((a,b)=>a-b);
    const vals = years.map(y => data[String(y)] || 0);
    return {
      name: k.term,
      x: years, y: vals,
      mode: "lines+markers",
      visible: i < defaultVisible ? true : "legendonly",
      line: { width: 2 }, marker: { size: 5 },
      hovertemplate: `<b>${k.term}</b><br>Year: %{x}<br>${yLabel}: %{y:,.2f}<br>Score: ${k.score.toFixed(4)}<extra></extra>`
    };
  });

  if (!traces.length) {
    Plotly.react("chart-temporal", [], {
      title: `No ${yLabel} data available`, height: 300,
      annotations: [{ text: "No temporal data for selected metric", xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false }]
    });
    return;
  }

  Plotly.react("chart-temporal", traces, {
    title: `Keyword Temporal Trends (${yLabel})`,
    xaxis: { title: "Year", dtick: 2, tickformat: "d" },
    yaxis: { title: yLabel, rangemode: "tozero" },
    legend: { orientation: "h", yanchor: "top", y: -0.2, font: { size: 11 } },
    margin: { l: 70, r: 30, t: 50, b: 120 },
    height: 550,
    template: "plotly_white",
    hoverlabel: { bgcolor: "#343a40", font: { color: "#fff" } }
  }, { responsive: true, displaylogo: false });
}

function renderTemporalGlobal() {
  // Heatmap: clusters × years, value = total docs
  const metric = document.getElementById("temporal-metric").value;
  const metricLabels = { "pub_year_series": "Documents", "ppm_series": "PPM", "loglift_series": "Log-Lift" };
  const yLabel = metricLabels[metric] || metric;
  const yearSet = new Set();
  const clusterData = [];
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    const agg = {};
    cl.keywords.forEach(k => {
      const series = k[metric] || k.temporal || {};
      Object.entries(series).forEach(([y, v]) => { yearSet.add(y); agg[y] = (agg[y] || 0) + v; });
    });
    clusterData.push({ cid, label: cl.label, agg });
  });
  const years = [...yearSet].sort();
  if (!years.length) {
    Plotly.react("chart-temporal", [], {
      title: "No temporal data available", height: 300,
      annotations: [{ text: "Upload data with temporal series", xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false }]
    });
    return;
  }
  const z = clusterData.map(cd => years.map(y => cd.agg[y] || 0));
  const labels = clusterData.map(cd => `C${cd.cid}: ${cd.label.split(",")[0]}`);
  Plotly.react("chart-temporal", [{
    type: "heatmap", z, x: years, y: labels,
    colorscale: "YlOrRd", hovertemplate: "<b>%{y}</b><br>Year: %{x}<br>" + yLabel + ": %{z:,.1f}<extra></extra>"
  }], {
    title: `Temporal Heatmap (${yLabel}) — All Clusters`,
    xaxis: { title: "Year", dtick: 2, tickformat: "d" },
    yaxis: { autorange: "reversed" },
    margin: { l: 250, r: 30, t: 50, b: 60 },
    height: Math.max(400, clusterData.length * 20 + 100),
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });
  // Hide per-cluster highlights
  const hlEl = document.getElementById("temporal-highlights");
  if (hlEl) hlEl.innerHTML = "";
}

// ---- Tab 3: Hierarchy ----
function renderHierarchy() {
  if (!currentCluster) { renderHierarchyGlobal(); return; }
  if (hierarchyView === "subphrase") {
    renderSubphraseTree();
  } else {
    renderDepthSunburst();
  }
}

function renderHierarchyGlobal() {
  // Stacked bar: depth distribution per cluster
  const depthData = {};  // depth_level -> [count per cluster]
  const labels = [];
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    labels.push(`C${cid}`);
    const counts = {};
    cl.keywords.forEach(k => { if (k.depth_level != null) counts[k.depth_level] = (counts[k.depth_level]||0)+1; });
    Object.entries(counts).forEach(([d, c]) => {
      if (!depthData[d]) depthData[d] = [];
    });
  });
  const idx = [];
  clusterIds.forEach((cid, i) => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    idx.push(i);
  });
  const traces = Object.keys(depthData).sort().map(d => {
    const vals = [];
    let ci = 0;
    clusterIds.forEach(cid => {
      const cl = DATA[cid];
      if (!cl || !cl.keywords) return;
      const cnt = cl.keywords.filter(k => k.depth_level == d).length;
      vals.push(cnt);
    });
    return {
      name: depthNames[d] || `L${d}`,
      type: "bar", x: labels, y: vals,
      marker: { color: depthColors[d] || "#adb5bd" },
      hovertemplate: `<b>%{x}</b><br>${depthNames[d] || "L"+d}: %{y}<extra></extra>`
    };
  });
  Plotly.react("chart-hierarchy", traces, {
    title: "Depth Distribution — All Clusters",
    barmode: "stack",
    xaxis: { title: "Cluster", tickangle: -45 },
    yaxis: { title: "Keywords" },
    margin: { l: 60, r: 30, t: 50, b: 80 },
    height: 450,
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });
}

function renderSubphraseTree() {
  if (!currentCluster) return;
  const tree = currentCluster.subphrase_tree || [];
  const kws = currentCluster.keywords;
  const kwMap = {};
  kws.forEach(k => kwMap[k.term] = k);

  if (!tree.length) {
    Plotly.react("chart-hierarchy", [], {
      title: "No subphrase containment relationships found",
      height: 300,
      annotations: [{ text: "No keyword contains another keyword as a subphrase in this cluster", xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false }]
    });
    return;
  }

  // Build adjacency: parent → [children]
  const children = {};
  const hasParent = new Set();
  tree.forEach(e => {
    if (!children[e.parent]) children[e.parent] = [];
    children[e.parent].push(e.child);
    hasParent.add(e.child);
  });

  // Root terms: appear as parents but not as children
  const allParents = new Set(Object.keys(children));
  const roots = [...allParents].filter(p => !hasParent.has(p));

  // Build sunburst: root → parent terms → children
  const clusterLabel = "C" + currentClusterId;
  const ids = [clusterLabel];
  const labels = [clusterLabel];
  const parents = [""];
  const values = [];
  const colors = [];
  const hoverTexts = [];

  // We need to compute values bottom-up
  function getFreq(term) { return kwMap[term] ? kwMap[term].frequency : 1; }
  function getDepth(term) { return kwMap[term] != null ? kwMap[term].depth_level : null; }

  function addNode(term, parentId) {
    const id = parentId + "/" + term;
    const ch = children[term] || [];
    let childSum = 0;

    ch.forEach(c => {
      childSum += addNode(c, id);
    });

    const freq = getFreq(term);
    // For branchvalues="total", parent value must >= sum of children
    const nodeVal = Math.max(freq, childSum);
    const dl = getDepth(term);

    ids.push(id);
    labels.push(term);
    parents.push(parentId);
    values.push(nodeVal);
    colors.push(dl != null ? (depthColors[dl] || "#AB63FA") : "#adb5bd");
    const kw = kwMap[term];
    hoverTexts.push(kw
      ? `<b>${term}</b><br>Score: ${kw.score.toFixed(4)}<br>Freq: ${kw.frequency.toLocaleString()}` +
        (dl != null ? `<br>Depth: ${depthNames[dl]}` : "") +
        (ch.length ? `<br>Contains: ${ch.length} subphrases` : "")
      : term
    );
    return nodeVal;
  }

  let rootTotal = 0;
  const inTree = new Set([...hasParent, ...allParents]);

  roots.forEach(r => {
    rootTotal += addNode(r, clusterLabel);
  });

  // Add orphan terms (not in any subphrase relationship) under "Other"
  const orphans = kws.filter(k => !inTree.has(k.term));
  if (orphans.length > 0) {
    const otherId = clusterLabel + "/Other";
    let otherTotal = 0;
    orphans.forEach(k => {
      const nodeId = otherId + "/" + k.term;
      ids.push(nodeId);
      labels.push(k.term);
      parents.push(otherId);
      values.push(k.frequency);
      const dl = k.depth_level;
      colors.push(dl != null ? (depthColors[dl] || "#AB63FA") : "#adb5bd");
      hoverTexts.push(`<b>${k.term}</b><br>Score: ${k.score.toFixed(4)}<br>Freq: ${k.frequency.toLocaleString()}` +
        (dl != null ? `<br>Depth: ${depthNames[dl]}` : "") + `<br>(no subphrase relation)`);
      otherTotal += k.frequency;
    });
    ids.push(otherId);
    labels.push(`Other (${orphans.length})`);
    parents.push(clusterLabel);
    values.push(otherTotal);
    colors.push("#e0e0e0");
    hoverTexts.push(`${orphans.length} keywords with no subphrase relationships`);
    rootTotal += otherTotal;
  }

  values.unshift(rootTotal);
  colors.unshift("#dee2e6");
  hoverTexts.unshift(`${roots.length} root terms, ${tree.length} containment pairs, ${orphans.length} independent`);

  Plotly.react("chart-hierarchy", [{
    type: "sunburst",
    ids, labels, parents, values,
    branchvalues: "total",
    hovertext: hoverTexts, hoverinfo: "text",
    marker: { colors },
    textinfo: "label",
    insidetextorientation: "radial",
    maxdepth: 4,
  }], {
    title: "Subphrase Containment Hierarchy",
    height: 600,
    margin: { l: 10, r: 10, t: 50, b: 10 },
  }, { responsive: true, displaylogo: false });
}

function renderDepthSunburst() {
  if (!currentCluster) return;
  const kws = currentCluster.keywords.filter(k => k.depth_level != null);
  if (!kws.length) {
    Plotly.react("chart-hierarchy", [], { title: "No depth data available", height: 300 });
    return;
  }

  const clusterLabel = "C" + currentClusterId;
  const ids = [];
  const labelsArr = [];
  const parents = [];
  const values = [];
  const colors = [];
  const hoverTexts = [];

  const levels = [...new Set(kws.map(k => k.depth_level))].sort();
  let rootTotal = 0;

  levels.forEach(lvl => {
    const levelKws = kws.filter(k => k.depth_level === lvl);
    const id = `${clusterLabel}-L${lvl}`;
    const levelTotal = levelKws.reduce((s, k) => s + k.frequency, 0);
    rootTotal += levelTotal;

    ids.push(id);
    labelsArr.push(`${depthNames[lvl] || "L"+lvl} (${levelKws.length})`);
    parents.push(clusterLabel);
    values.push(levelTotal);
    colors.push(depthColors[lvl] || "#AB63FA");
    hoverTexts.push(`${levelKws.length} keywords<br>Total freq: ${levelTotal.toLocaleString()}`);

    levelKws.forEach(k => {
      ids.push(`${id}-${k.term}`);
      labelsArr.push(k.term);
      parents.push(id);
      values.push(k.frequency);
      colors.push(depthColors[lvl] || "#AB63FA");
      hoverTexts.push(
        `<b>${k.term}</b><br>Score: ${k.score.toFixed(4)}<br>Freq: ${k.frequency.toLocaleString()}` +
        (k.depth_score != null ? `<br>Depth score: ${k.depth_score.toFixed(3)}` : "")
      );
    });
  });

  ids.unshift(clusterLabel);
  labelsArr.unshift(clusterLabel);
  parents.unshift("");
  values.unshift(rootTotal);
  colors.unshift("#dee2e6");
  hoverTexts.unshift(`${kws.length} keywords`);

  Plotly.react("chart-hierarchy", [{
    type: "sunburst",
    ids, labels: labelsArr, parents, values,
    branchvalues: "total",
    hovertext: hoverTexts, hoverinfo: "text",
    marker: { colors },
    textinfo: "label",
    insidetextorientation: "radial",
    maxdepth: 3,
  }], {
    title: "Keyword Depth Hierarchy",
    height: 600,
    margin: { l: 10, r: 10, t: 50, b: 10 },
  }, { responsive: true, displaylogo: false });
}

// ---- Tab 0: Overview ----
function renderOverview() {
  const clusterStats = [];
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    const kws = cl.keywords;
    const n = kws.length;
    const avgScore = kws.reduce((s,k) => s + k.score, 0) / (n || 1);
    const avgFreq = kws.reduce((s,k) => s + k.frequency, 0) / (n || 1);
    let emergingCount = 0;
    kws.forEach(k => {
      const ts = TREND_SCORES[k.term];
      if (ts != null && ts > 0.3) emergingCount++;
    });
    const emergingRatio = n > 0 ? emergingCount / n : 0;
    let totalCentrality = 0, centralityCount = 0;
    kws.forEach(k => {
      const c = CENTRALITY[k.term];
      if (c && c.degree != null) { totalCentrality += c.degree; centralityCount++; }
    });
    const avgCentrality = centralityCount > 0 ? totalCentrality / centralityCount : 0;
    clusterStats.push({ cid, label: cl.label, n, avgScore, avgFreq, emergingRatio, avgCentrality });
  });

  if (!clusterStats.length) {
    Plotly.react("chart-overview", [], { title: "No cluster data", height: 300 });
    return;
  }

  // Bubble chart
  Plotly.react("chart-overview", [{
    type: "scatter", mode: "markers+text",
    x: clusterStats.map(c => c.emergingRatio),
    y: clusterStats.map(c => c.avgCentrality),
    text: clusterStats.map(c => "C" + c.cid),
    textposition: "top center",
    marker: {
      size: clusterStats.map(c => Math.max(15, Math.sqrt(c.n) * 8)),
      color: clusterStats.map(c => c.avgScore),
      colorscale: "YlOrRd", showscale: true,
      colorbar: { title: "Avg Score" },
      line: { width: 1, color: "#fff" }
    },
    hovertemplate: clusterStats.map(c =>
      "<b>C" + c.cid + ": " + c.label.split(",")[0] + "</b><br>" +
      "Keywords: " + c.n + "<br>Avg Score: " + c.avgScore.toFixed(4) + "<br>" +
      "Emerging ratio: " + (c.emergingRatio * 100).toFixed(1) + "%<br>" +
      "Avg Centrality: " + c.avgCentrality.toFixed(3) + "<extra></extra>"
    ),
    customdata: clusterStats.map(c => c.cid)
  }], {
    title: "Cluster Overview",
    xaxis: { title: "Emerging Ratio (trend > 0.3)", tickformat: ".0%" },
    yaxis: { title: "Avg Centrality" },
    height: 500,
    margin: { l: 70, r: 30, t: 50, b: 60 },
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });

  // Click bubble to switch cluster
  const chartEl = document.getElementById("chart-overview");
  chartEl.removeAllListeners && chartEl.removeAllListeners("plotly_click");
  chartEl.on("plotly_click", function(data) {
    if (data.points && data.points[0]) {
      const cid = data.points[0].customdata;
      if (cid != null) {
        clusterSelect.value = cid;
        currentClusterId = Number(cid);
        currentCluster = DATA[currentClusterId];
        updateMeta();
        // Switch to keywords tab
        tabEls.forEach(t => t.classList.remove("active"));
        const kwTab = document.querySelector('.tab[data-tab="keywords"]');
        if (kwTab) kwTab.classList.add("active");
        panels.forEach(p => p.classList.remove("active"));
        document.getElementById("panel-keywords").classList.add("active");
        currentTab = "keywords";
        updateHash();
        renderKeywords();
      }
    }
  });

  // Summary table
  const tableContainer = document.getElementById("overview-table-container");
  if (tableContainer) {
    let html = '<table class="merge-table"><thead><tr><th>Cluster</th><th>Label</th><th>Keywords</th><th>Avg Score</th><th>Avg Freq</th><th>Emerging %</th><th>Avg Centrality</th></tr></thead><tbody>';
    clusterStats.forEach(c => {
      html += "<tr><td>C" + c.cid + "</td><td>" + c.label + "</td><td>" + c.n + "</td><td>" +
        c.avgScore.toFixed(4) + "</td><td>" + Math.round(c.avgFreq).toLocaleString() + "</td><td>" +
        (c.emergingRatio * 100).toFixed(1) + "%</td><td>" + c.avgCentrality.toFixed(3) + "</td></tr>";
    });
    html += "</tbody></table>";
    tableContainer.innerHTML = html;
  }

  // Pipeline config
  const configEl = document.getElementById("pipeline-config-display");
  if (configEl && PIPELINE_CONFIG && Object.keys(PIPELINE_CONFIG).length > 0) {
    let html = '<table class="merge-table"><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>';
    Object.entries(PIPELINE_CONFIG).forEach(([key, val]) => {
      let display;
      if (key === "stages_enabled" && typeof val === "object") {
        display = Object.entries(val).map(([stage, enabled]) =>
          (enabled ? "\u2713" : "\u2717") + " " + stage
        ).join("<br>");
      } else if (typeof val === "object") {
        display = JSON.stringify(val, null, 2).replace(/\n/g, "<br>").replace(/ /g, "&nbsp;");
      } else {
        display = String(val);
      }
      html += "<tr><td><strong>" + key + "</strong></td><td>" + display + "</td></tr>";
    });
    html += "</tbody></table>";
    configEl.innerHTML = html;
  } else if (configEl) {
    configEl.innerHTML = '<div style="color:#6c757d;font-size:.85rem;">No pipeline configuration data available.</div>';
  }
}

// ---- Tab 4: Network (D3 Force) ----
function renderNetworkClusterGlobal() {
  // Build cluster-to-cluster connections from cross-cluster terms
  const svg = d3.select("#network-svg");
  svg.selectAll("*").remove();
  d3.select("#network-container").selectAll(".network-info-panel").remove();
  const container = document.getElementById("network-container");
  const width = container.clientWidth || 800;
  const height = 600;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  if (!CROSS_CLUSTER_TERMS || !CROSS_CLUSTER_TERMS.length) {
    svg.append("text").attr("x", width/2).attr("y", height/2)
      .attr("text-anchor", "middle").attr("font-size", "14px").attr("fill", "#6c757d")
      .text("No cross-cluster data available for global network");
    return;
  }

  // Aggregate edges between cluster pairs
  const edgeMap = {};
  const sharedTerms = {};
  CROSS_CLUSTER_TERMS.forEach(ct => {
    const cls = ct.clusters || [];
    for (let i = 0; i < cls.length; i++) {
      for (let j = i + 1; j < cls.length; j++) {
        const key = `${cls[i]}-${cls[j]}`;
        edgeMap[key] = (edgeMap[key] || 0) + 1;
        if (!sharedTerms[key]) sharedTerms[key] = [];
        if (sharedTerms[key].length < 5) sharedTerms[key].push(ct.term);
      }
    }
  });

  const nodes = clusterIds.map(cid => ({
    id: cid,
    label: `C${cid}`,
    fullLabel: DATA[cid] ? DATA[cid].label : `C${cid}`,
    nKw: DATA[cid] ? (DATA[cid].keywords || []).length : 0
  }));
  const nodeIndex = {};
  nodes.forEach((n, i) => nodeIndex[n.id] = i);

  const edges = [];
  Object.entries(edgeMap).forEach(([key, weight]) => {
    const [s, t] = key.split("-").map(Number);
    if (nodeIndex[s] != null && nodeIndex[t] != null) {
      edges.push({ source: nodeIndex[s], target: nodeIndex[t], weight, terms: sharedTerms[key] || [] });
    }
  });

  if (!edges.length) {
    svg.append("text").attr("x", width/2).attr("y", height/2)
      .attr("text-anchor", "middle").attr("font-size", "14px").attr("fill", "#6c757d")
      .text("No inter-cluster connections found");
    return;
  }

  const maxNkw = d3.max(nodes, d => d.nKw) || 1;
  const maxW = d3.max(edges, d => d.weight) || 1;
  const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).distance(d => 120 / (1 + Math.log1p(d.weight))))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.nKw / maxNkw) * 25 + 15));

  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (event) => g.attr("transform", event.transform)));

  const link = g.selectAll(".link").data(edges).join("line")
    .attr("stroke", "#adb5bd")
    .attr("stroke-opacity", d => 0.3 + 0.7 * (d.weight / maxW))
    .attr("stroke-width", d => 1 + 4 * (d.weight / maxW));
  link.append("title").text(d => {
    const s = typeof d.source === "object" ? d.source : nodes[d.source];
    const t = typeof d.target === "object" ? d.target : nodes[d.target];
    return `${s.label} \u2014 ${t.label}\nShared terms: ${d.weight}\n${d.terms.join(", ")}`;
  });

  const node = g.selectAll(".node").data(nodes).join("circle")
    .attr("r", d => Math.sqrt(d.nKw / maxNkw) * 20 + 8)
    .attr("fill", (d, i) => palette[i % palette.length])
    .attr("stroke", "#fff").attr("stroke-width", 1.5)
    .style("cursor", "pointer")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );
  node.append("title").text(d => `${d.label}: ${d.fullLabel}\nKeywords: ${d.nKw}`);
  node.on("click", (event, d) => {
    clusterSelect.value = d.id;
    clusterSelect.dispatchEvent(new Event("change"));
  });

  const label = g.selectAll(".node-label").data(nodes).join("text")
    .attr("class", "node-label").attr("dy", "0.35em")
    .text(d => d.label).attr("font-size", "11px");

  simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  });
}

function renderNetworkKeywordGlobal() {
  const svg = d3.select("#network-svg");
  svg.selectAll("*").remove();
  d3.select("#network-container").selectAll(".network-info-panel").remove();
  const container = document.getElementById("network-container");
  const width = container.clientWidth || 800;
  const height = 600;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  // Aggregate keyword co-occurrence edges across all clusters
  const edgeMap = {};
  const kwInfo = {};  // term -> {score, freq, bestCluster, clusterCount}
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl) return;
    (cl.keywords || []).forEach(k => {
      if (!kwInfo[k.term]) {
        kwInfo[k.term] = { score: k.score, freq: k.frequency || 0, bestCluster: cid, bestScore: k.score, clusters: new Set() };
      } else {
        kwInfo[k.term].score = Math.max(kwInfo[k.term].score, k.score);
        kwInfo[k.term].freq += (k.frequency || 0);
        if (k.score > kwInfo[k.term].bestScore) { kwInfo[k.term].bestCluster = cid; kwInfo[k.term].bestScore = k.score; }
      }
      kwInfo[k.term].clusters.add(cid);
    });
    (cl.network_edges || []).forEach(e => {
      const key = e.source < e.target ? `${e.source}\t${e.target}` : `${e.target}\t${e.source}`;
      edgeMap[key] = (edgeMap[key] || 0) + e.weight;
    });
  });

  // Take top edges by weight
  const allEdges = Object.entries(edgeMap).map(([key, w]) => {
    const [s, t] = key.split("\t");
    return { source: s, target: t, weight: w };
  }).sort((a, b) => b.weight - a.weight).slice(0, 120);

  if (!allEdges.length) {
    svg.append("text").attr("x", width/2).attr("y", height/2)
      .attr("text-anchor", "middle").attr("font-size", "14px").attr("fill", "#6c757d")
      .text("No keyword co-occurrence edges available");
    return;
  }

  // Build node set from edges
  const nodeSet = new Set();
  allEdges.forEach(e => { nodeSet.add(e.source); nodeSet.add(e.target); });
  const nodes = [...nodeSet].map(id => ({
    id,
    score: kwInfo[id] ? kwInfo[id].score : 0,
    freq: kwInfo[id] ? kwInfo[id].freq : 0,
    bestCluster: kwInfo[id] ? kwInfo[id].bestCluster : 0,
    nClusters: kwInfo[id] ? kwInfo[id].clusters.size : 1,
  }));
  const nodeIndex = {};
  nodes.forEach((n, i) => nodeIndex[n.id] = i);

  const edges = allEdges.filter(e => nodeIndex[e.source] != null && nodeIndex[e.target] != null)
    .map(e => ({ source: nodeIndex[e.source], target: nodeIndex[e.target], weight: e.weight }));

  const maxScore = d3.max(nodes, d => d.score) || 1;
  const maxW = d3.max(edges, d => d.weight) || 1;
  const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

  // Setup slider
  const edgeSlider = document.getElementById("network-edge-slider");
  const edgeValueSpan = document.getElementById("network-edge-value");
  if (edgeSlider) { edgeSlider.max = Math.ceil(maxW); edgeSlider.value = 0; edgeValueSpan.textContent = "0"; }

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id((d, i) => i).distance(d => 100 / (1 + Math.log1p(d.weight))))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.score / maxScore) * 20 + 10));

  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (event) => g.attr("transform", event.transform)));

  const link = g.selectAll(".link").data(edges).join("line")
    .attr("stroke", "#adb5bd")
    .attr("stroke-opacity", d => 0.3 + 0.7 * (d.weight / maxW))
    .attr("stroke-width", d => 1 + 4 * (d.weight / maxW));

  const node = g.selectAll(".node").data(nodes).join("circle")
    .attr("r", d => Math.sqrt(d.score / maxScore) * 18 + 5)
    .attr("fill", d => palette[d.bestCluster % palette.length])
    .attr("stroke", d => d.nClusters > 1 ? "#ffd700" : "#fff")
    .attr("stroke-width", d => d.nClusters > 1 ? 2.5 : 1.5)
    .style("cursor", "pointer")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );
  // Permanent labels for top nodes only
  const sortedByScore = [...nodes].sort((a, b) => b.score - a.score);
  const labelThreshold = sortedByScore[Math.min(20, sortedByScore.length - 1)].score;
  const label = g.selectAll(".node-label").data(nodes).join("text")
    .attr("class", "node-label").attr("dy", "0.35em")
    .text(d => d.score >= labelThreshold ? d.id : "").attr("font-size", "10px");

  // Hover label for all nodes
  let hoverNode = null;
  const hoverLabel = g.append("text").attr("class", "node-label")
    .attr("font-size", "11px").attr("font-weight", "600")
    .attr("paint-order", "stroke").attr("stroke", "#fff").attr("stroke-width", "3px")
    .style("pointer-events", "none").style("display", "none");

  // Edge slider filter
  if (edgeSlider) {
    edgeSlider.oninput = () => {
      const minW = Number(edgeSlider.value);
      edgeValueSpan.textContent = minW;
      link.attr("display", d => d.weight >= minW ? null : "none");
    };
  }

  // Hover highlight + dynamic label
  const adjacency = {};
  nodes.forEach(n => adjacency[n.id] = new Set());
  node.on("mouseover", (event, d) => {
    const neighbors = adjacency[d.id] || new Set();
    node.attr("opacity", n => n.id === d.id || neighbors.has(n.id) ? 1 : 0.15);
    link.attr("opacity", e => {
      const sid = typeof e.source === "object" ? e.source.id : nodes[e.source].id;
      const tid = typeof e.target === "object" ? e.target.id : nodes[e.target].id;
      return sid === d.id || tid === d.id ? 1 : 0.05;
    });
    label.attr("opacity", n => n.id === d.id || neighbors.has(n.id) ? 1 : 0.1);
    // Show hover label with detail
    hoverNode = d;
    hoverLabel.style("display", null)
      .attr("x", d.x).attr("y", d.y - Math.sqrt(d.score / maxScore) * 18 - 10)
      .text(`${d.id}  (${d.score.toFixed(3)}, C${d.bestCluster})`);
  }).on("mouseout", () => {
    hoverNode = null;
    node.attr("opacity", 1); link.attr("opacity", 1); label.attr("opacity", 1);
    hoverLabel.style("display", "none");
  });

  // Build adjacency after simulation starts (edges get mutated)
  edges.forEach(e => {
    const sid = typeof e.source === "object" ? e.source.id : nodes[e.source].id;
    const tid = typeof e.target === "object" ? e.target.id : nodes[e.target].id;
    if (adjacency[sid]) adjacency[sid].add(tid);
    if (adjacency[tid]) adjacency[tid].add(sid);
  });

  simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
    if (hoverNode) {
      hoverLabel.attr("x", hoverNode.x).attr("y", hoverNode.y - Math.sqrt(hoverNode.score / maxScore) * 18 - 10);
    }
  });
}

function renderNetwork() {
  const subtabsEl = document.getElementById("network-subtabs");
  const descEl = document.getElementById("network-desc");
  if (!currentCluster) {
    subtabsEl.style.display = "flex";
    if (networkView === "keyword") {
      descEl.innerHTML = '전체 클러스터의 키워드를 합산한 <b>키워드 공출현 네트워크</b>입니다. 노드 색상은 가장 빈번한 클러스터를 나타냅니다.';
      renderNetworkKeywordGlobal();
    } else {
      descEl.innerHTML = '<b>Cross-cluster terms</b> 기반 클러스터 연결 네트워크입니다. 공유 키워드가 많을수록 엣지가 두껍습니다. 노드 클릭으로 해당 클러스터로 이동합니다.';
      renderNetworkClusterGlobal();
    }
    return;
  }
  subtabsEl.style.display = "none";
  descEl.innerHTML = '키워드 간 <b>실제 공출현(co-occurrence) 네트워크</b>입니다. 동일 문헌에서 함께 등장한 빈도를 기반으로 엣지가 생성됩니다. 엣지 두께는 공출현 빈도에, 노드 크기는 점수에 비례합니다. 색상은 깊이 수준(Broad=파랑, Mid=빨강, Specific=초록)입니다.';
  const svg = d3.select("#network-svg");
  svg.selectAll("*").remove();
  // Remove any existing info panels
  d3.select("#network-container").selectAll(".network-info-panel").remove();

  const container = document.getElementById("network-container");
  const width = container.clientWidth || 800;
  const height = 600;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  const kws = currentCluster.keywords;
  const rawEdges = currentCluster.network_edges || [];

  if (!rawEdges.length) {
    svg.append("text").attr("x", width/2).attr("y", height/2)
      .attr("text-anchor", "middle").attr("font-size", "14px").attr("fill", "#6c757d")
      .text("No co-occurrence edges available for this cluster");
    return;
  }

  // Take top edges to avoid overcrowding
  const edges = rawEdges.slice(0, 80).map(e => ({...e}));

  // Build node set: all keywords (including isolated ones without edges)
  const kwMap = {};
  kws.forEach(k => kwMap[k.term] = k);
  const connectedSet = new Set();
  edges.forEach(e => { connectedSet.add(e.source); connectedSet.add(e.target); });

  const nodes = kws.map(k => ({
    id: k.term,
    score: k.score || 0,
    frequency: k.frequency || 0,
    depth_level: k.depth_level,
    isolated: !connectedSet.has(k.term),
  }));

  const maxScore = d3.max(nodes, d => d.score) || 1;
  const maxWeight = d3.max(edges, d => d.weight) || 1;

  // Set up edge slider dynamically
  const edgeSlider = document.getElementById("network-edge-slider");
  const edgeValueSpan = document.getElementById("network-edge-value");
  if (edgeSlider) {
    edgeSlider.max = Math.ceil(maxWeight * 100);
    edgeSlider.value = 0;
    edgeValueSpan.textContent = "0";
  }

  // Build adjacency for hover highlight
  const adjacency = {};
  nodes.forEach(n => adjacency[n.id] = new Set());
  edges.forEach(e => {
    const sid = typeof e.source === "object" ? e.source.id : e.source;
    const tid = typeof e.target === "object" ? e.target.id : e.target;
    adjacency[sid] && adjacency[sid].add(tid);
    adjacency[tid] && adjacency[tid].add(sid);
  });

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(d => 150 / (1 + Math.log1p(d.weight))))
    .force("charge", d3.forceManyBody().strength(d => d.isolated ? -30 : -250))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => d.isolated ? 8 : Math.sqrt(d.score / maxScore) * 25 + 15))
    .force("radial", d3.forceRadial(d => d.isolated ? Math.min(width, height) * 0.42 : 0, width/2, height/2).strength(d => d.isolated ? 0.15 : 0));

  const g = svg.append("g");

  // Zoom
  svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (event) => {
    g.attr("transform", event.transform);
  }));

  const link = g.selectAll(".link")
    .data(edges).join("line")
    .attr("class", "link")
    .attr("stroke", "#adb5bd")
    .attr("stroke-opacity", d => 0.3 + 0.7 * Math.min(1, d.weight / maxWeight))
    .attr("stroke-width", d => 1 + 4 * (d.weight / maxWeight))
    .style("transition", "opacity 0.2s");

  // Edge tooltips
  link.append("title").text(d => `${d.source.id || d.source} \u2014 ${d.target.id || d.target}\nCo-occurrence: ${d.weight}`);

  const node = g.selectAll(".node")
    .data(nodes).join("circle")
    .attr("class", "node")
    .attr("r", d => d.isolated ? 4 : Math.sqrt(d.score / maxScore) * 20 + 5)
    .attr("fill", d => d.depth_level != null ? (depthColors[d.depth_level] || "#AB63FA") : "#adb5bd")
    .attr("fill-opacity", d => d.isolated ? 0.5 : 1)
    .attr("stroke", d => d.isolated ? "#adb5bd" : "#fff")
    .attr("stroke-width", d => d.isolated ? 0.5 : 1.5)
    .attr("stroke-dasharray", d => d.isolated ? "2,2" : "none")
    .style("cursor", "pointer")
    .style("transition", "opacity 0.2s")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  // Hover highlight
  node.on("mouseenter", (event, d) => {
    const neighbors = adjacency[d.id] || new Set();
    node.style("opacity", n => (n.id === d.id || neighbors.has(n.id)) ? 1 : 0.1);
    link.style("opacity", l => {
      const sid = typeof l.source === "object" ? l.source.id : l.source;
      const tid = typeof l.target === "object" ? l.target.id : l.target;
      return (sid === d.id || tid === d.id) ? 1 : 0.1;
    }).attr("stroke", l => {
      const sid = typeof l.source === "object" ? l.source.id : l.source;
      const tid = typeof l.target === "object" ? l.target.id : l.target;
      return (sid === d.id || tid === d.id) ? "#0d6efd" : "#adb5bd";
    });
    label.style("opacity", n => (n.id === d.id || neighbors.has(n.id)) ? 1 : 0.1);
  }).on("mouseleave", () => {
    node.style("opacity", d => d.isolated ? 0.5 : 1);
    link.style("opacity", 1).attr("stroke", "#adb5bd");
    label.style("opacity", 1);
  });

  // Click info panel
  node.on("click", (event, d) => {
    event.stopPropagation();
    d3.select("#network-container").selectAll(".network-info-panel").remove();
    const neighbors = adjacency[d.id] || new Set();
    const kw = kwMap[d.id] || {};
    const ts = TREND_SCORES[d.id];
    const ct = CENTRALITY[d.id];
    let infoHtml = '<span class="info-close">&times;</span>';
    infoHtml += "<div><strong>" + d.id + "</strong></div>";
    infoHtml += "<div>Score: " + (kw.score || d.score).toFixed(4) + "</div>";
    infoHtml += "<div>Frequency: " + (kw.frequency || d.frequency).toLocaleString() + "</div>";
    if (kw.depth_level != null) infoHtml += "<div>Depth: " + (depthNames[kw.depth_level] || "L" + kw.depth_level) + "</div>";
    if (ts != null) infoHtml += "<div>Trend: " + (ts > 0 ? "\u25b2" : "\u25bc") + " " + ts.toFixed(3) + "</div>";
    if (ct) infoHtml += "<div>Centrality: " + ct.degree.toFixed(2) + "</div>";
    if (neighbors.size) infoHtml += "<div style='margin-top:.3rem;font-size:.8rem;'>Neighbors: " + [...neighbors].slice(0, 10).join(", ") + (neighbors.size > 10 ? "..." : "") + "</div>";

    const panel = document.createElement("div");
    panel.className = "network-info-panel";
    panel.innerHTML = infoHtml;
    // Position near the click
    const rect = container.getBoundingClientRect();
    let left = event.clientX - rect.left + 10;
    let top = event.clientY - rect.top + 10;
    if (left + 300 > rect.width) left = Math.max(0, left - 320);
    if (top + 200 > rect.height) top = Math.max(0, top - 210);
    panel.style.left = left + "px";
    panel.style.top = top + "px";
    container.appendChild(panel);
    panel.querySelector(".info-close").addEventListener("click", () => panel.remove());
  });

  // Click elsewhere to dismiss info panel
  svg.on("click", () => {
    d3.select("#network-container").selectAll(".network-info-panel").remove();
  });

  const label = g.selectAll(".node-label")
    .data(nodes).join("text")
    .attr("class", "node-label")
    .text(d => d.isolated ? (d.id.length > 12 ? d.id.slice(0, 10) + "\u2026" : d.id) : (d.id.length > 22 ? d.id.slice(0, 20) + "\u2026" : d.id))
    .attr("dy", d => d.isolated ? -8 : -(Math.sqrt(d.score / maxScore) * 20 + 8))
    .attr("fill", d => d.isolated ? "#adb5bd" : "#1f2933")
    .attr("font-size", d => d.isolated ? 8 : Math.max(9, Math.min(13, d.score / maxScore * 20 + 8)))
    .style("transition", "opacity 0.2s");

  simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  });

  // Edge weight filter slider
  if (edgeSlider) {
    edgeSlider.addEventListener("input", function() {
      const threshold = Number(this.value) / 100;
      edgeValueSpan.textContent = threshold.toFixed(2);
      // Filter edges
      link.style("opacity", d => d.weight >= threshold ? (0.3 + 0.7 * Math.min(1, d.weight / maxWeight)) : 0)
          .style("pointer-events", d => d.weight >= threshold ? "auto" : "none");
      // Find connected nodes at this threshold
      const visibleConnected = new Set();
      edges.forEach(e => {
        if (e.weight >= threshold) {
          const sid = typeof e.source === "object" ? e.source.id : e.source;
          const tid = typeof e.target === "object" ? e.target.id : e.target;
          visibleConnected.add(sid);
          visibleConnected.add(tid);
        }
      });
      // If threshold is 0, show all; otherwise hide disconnected
      if (threshold <= 0) {
        node.style("opacity", d => d.isolated ? 0.5 : 1);
        label.style("opacity", 1);
      } else {
        node.style("opacity", d => visibleConnected.has(d.id) ? 1 : 0.05);
        label.style("opacity", d => visibleConnected.has(d.id) ? 1 : 0.05);
      }
    });
  }

  // Legend (fixed position, not affected by zoom)
  const legend = svg.append("g").attr("transform", `translate(15, ${height - 95})`);
  legend.append("rect").attr("x", -8).attr("y", -8).attr("width", 145).attr("height", 90)
    .attr("fill", "#fff").attr("stroke", "#dee2e6").attr("rx", 4).attr("opacity", 0.9);
  const legendItems = [
    { label: "Broad", color: depthColors[0] },
    { label: "Mid", color: depthColors[1] },
    { label: "Specific", color: depthColors[2] },
  ];
  legendItems.forEach((item, i) => {
    legend.append("circle").attr("cx", 8).attr("cy", 10 + i * 22).attr("r", 6).attr("fill", item.color);
    legend.append("text").attr("x", 22).attr("y", 14 + i * 22).text(item.label)
      .attr("font-size", "11px").attr("fill", "#1f2933");
  });
  const nIsolated = nodes.filter(d => d.isolated).length;
  legend.append("text").attr("x", 0).attr("y", 80).text(`${nodes.length} nodes (${nIsolated} isolated), ${edges.length} edges`)
    .attr("font-size", "10px").attr("fill", "#6c757d");
}

// ---- Tab 5: Dictionary ----
function renderDictionary() {
  const container = document.getElementById("dict-table-container");
  const searchBox = document.getElementById("dict-search");

  if (!currentCluster && dictView === "keywords") { renderDictKeywordsGlobal(container, searchBox); return; }
  if (!currentCluster) {
    // Vocab/Norm merges are already global — render them
    if (dictView === "vocab") renderDictVocabMerges(container, searchBox);
    else if (dictView === "norm") renderDictNormMerges(container, searchBox);
    else renderDictKeywordsGlobal(container, searchBox);
    return;
  }
  if (dictView === "keywords") renderDictKeywords(container, searchBox);
  else if (dictView === "vocab") renderDictVocabMerges(container, searchBox);
  else if (dictView === "norm") renderDictNormMerges(container, searchBox);
}

function renderDictKeywordsGlobal(container, searchBox) {
  // Gather all keywords from all clusters with cluster labels
  const allKws = [];
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    cl.keywords.forEach(k => allKws.push({ ...k, cluster: cid, clusterLabel: cl.label }));
  });
  allKws.sort((a, b) => b.score - a.score);

  function buildTable(filter) {
    const fl = (filter || "").toLowerCase();
    const filtered = fl ? allKws.filter(k => k.term.toLowerCase().includes(fl) || (`C${k.cluster}`).includes(fl) || k.clusterLabel.toLowerCase().includes(fl)) : allKws;
    const shown = filtered.slice(0, 200);
    let html = '<table class="merge-table"><thead><tr><th>#</th><th>Cluster</th><th>Term</th><th>Score</th><th>Freq</th><th>Doc Cov</th><th>Depth</th><th>Cross-CL</th></tr></thead><tbody>';
    shown.forEach((k, i) => {
      const depth = k.depth_level != null ? `${depthNames[k.depth_level]||"L"+k.depth_level} (${(k.depth_score||0).toFixed(2)})` : "";
      html += `<tr style="cursor:pointer" onclick="document.getElementById('cluster-select').value='${k.cluster}';document.getElementById('cluster-select').dispatchEvent(new Event('change'))">`;
      html += `<td>${i+1}</td><td>C${k.cluster}</td><td><b>${k.term}</b></td><td>${k.score.toFixed(4)}</td>`;
      html += `<td>${(k.frequency||0).toLocaleString()}</td><td>${(k.doc_coverage||0).toLocaleString()}</td>`;
      html += `<td>${depth}</td><td>${k.cross_cluster_count > 1 ? k.cross_cluster_count : ""}</td></tr>`;
    });
    if (filtered.length > 200) html += `<tr><td colspan="8" style="text-align:center;color:#6c757d;font-style:italic;">Showing 200 of ${filtered.length} keywords. Use search to filter.</td></tr>`;
    html += '</tbody></table>';
    container.innerHTML = html;
  }
  buildTable("");
  if (searchBox) {
    searchBox.oninput = () => buildTable(searchBox.value);
  }
}

function renderDictKeywords(container, searchBox) {
  if (!currentCluster) return;
  const kws = currentCluster.keywords;
  const normMerges = currentCluster.norm_merges || {};

  function buildTable(filter) {
    const fl = (filter || "").toLowerCase();
    const filtered = fl
      ? kws.filter(k => k.term.toLowerCase().includes(fl) ||
                        (k.expanded_from || "").toLowerCase().includes(fl) ||
                        (k.source_terms || []).join(" ").toLowerCase().includes(fl) ||
                        (normMerges[k.term] || []).join(" ").toLowerCase().includes(fl))
      : kws;

    let html = `<table class="merge-table">
      <thead><tr>
        <th>#</th><th>Term</th><th>Score</th><th>Freq</th><th>Doc Cov</th>
        <th>Depth</th><th>Cross-CL</th><th>Expanded From</th><th>Norm Merged From</th><th>Notes</th>
      </tr></thead><tbody>`;

    filtered.forEach((k, i) => {
      const dl = k.depth_level != null ? `${depthNames[k.depth_level] || "L"+k.depth_level} (${k.depth_score.toFixed(3)})` : "-";
      const cc = k.cross_cluster_count > 1 ? k.cross_cluster_count : "-";
      const ef = k.expanded_from || "-";
      const nm = normMerges[k.term];
      const nmStr = nm ? nm.map(t => `<span class="badge badge-norm">${t}</span>`).join(" ") : "-";
      const noteKey = "sciscape_notes_" + k.term;
      const existingNote = localStorage.getItem(noteKey) || "";
      const noteIndicator = existingNote ? ' <span class="note-indicator">\ud83d\udcdd</span>' : "";
      html += `<tr>
        <td>${i + 1}</td><td><strong>${k.term}</strong>${noteIndicator}</td>
        <td>${k.score.toFixed(4)}</td><td>${k.frequency.toLocaleString()}</td>
        <td>${k.doc_coverage.toLocaleString()}</td>
        <td>${dl}</td><td>${cc}</td><td>${ef}</td><td>${nmStr}</td>
        <td class="note-cell" contenteditable="true" data-term="${k.term.replace(/"/g, '&quot;')}">${existingNote}</td>
      </tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  buildTable(searchBox.value);
  searchBox.oninput = () => buildTable(searchBox.value);

  // Note save handler (event delegation)
  container.addEventListener("blur", function(e) {
    if (e.target && e.target.classList.contains("note-cell")) {
      const term = e.target.dataset.term;
      const note = e.target.textContent.trim();
      const noteKey = "sciscape_notes_" + term;
      if (note) {
        localStorage.setItem(noteKey, note);
      } else {
        localStorage.removeItem(noteKey);
      }
      // Refresh to update indicators
      buildTable(searchBox.value);
    }
  }, true);
}

function renderDictVocabMerges(container, searchBox) {
  const entries = Object.entries(VOCAB_MERGES);
  function buildTable(filter) {
    const fl = (filter || "").toLowerCase();
    const filtered = fl
      ? entries.filter(([src, tgt]) => src.toLowerCase().includes(fl) || tgt.toLowerCase().includes(fl))
      : entries;

    let html = `<table class="merge-table">
      <thead><tr><th>#</th><th>Source (merged away)</th><th>→</th><th>Target (canonical)</th></tr></thead><tbody>`;
    filtered.forEach(([src, tgt], i) => {
      html += `<tr><td>${i+1}</td><td>${src}</td><td>→</td><td><strong>${tgt}</strong></td></tr>`;
    });
    if (!filtered.length) html += `<tr><td colspan="4" style="text-align:center;color:#6c757d;">No vocab merges found</td></tr>`;
    html += "</tbody></table>";
    html += `<div style="margin-top:.5rem;font-size:.85rem;color:#6c757d;">Total: ${filtered.length} / ${entries.length} merges (Stage 2: plural→singular, hyphen normalization)</div>`;
    container.innerHTML = html;
  }
  buildTable(searchBox.value);
  searchBox.oninput = () => buildTable(searchBox.value);
}

function renderDictNormMerges(container, searchBox) {
  const entries = Object.entries(NORM_MERGES);
  function buildTable(filter) {
    const fl = (filter || "").toLowerCase();
    const filtered = fl
      ? entries.filter(([tgt, srcs]) => tgt.toLowerCase().includes(fl) || srcs.some(s => s.toLowerCase().includes(fl)))
      : entries;

    let html = `<table class="merge-table">
      <thead><tr><th>#</th><th>Canonical Term</th><th>Merged From</th></tr></thead><tbody>`;
    filtered.forEach(([tgt, srcs], i) => {
      const srcBadges = srcs.map(s => `<span class="badge badge-norm">${s}</span>`).join(" ");
      html += `<tr><td>${i+1}</td><td><strong>${tgt}</strong></td><td>${srcBadges}</td></tr>`;
    });
    if (!filtered.length) html += `<tr><td colspan="3" style="text-align:center;color:#6c757d;">No normalization merges found</td></tr>`;
    html += "</tbody></table>";
    html += `<div style="margin-top:.5rem;font-size:.85rem;color:#6c757d;">Total: ${filtered.length} / ${entries.length} merges (Stage 5: abbreviation, spelling, edit-distance, plural)</div>`;
    container.innerHTML = html;
  }
  buildTable(searchBox.value);
  searchBox.oninput = () => buildTable(searchBox.value);
}

// ---- Tab 6: Cross-Cluster ----
function renderCrossCluster() {
  const mode = document.getElementById("crosscluster-mode").value;
  const modeLabels = { rank: "Rank (0=lowest, 1=highest)", score: "Score", frequency: "Frequency" };

  // Build per-cluster data: term→{cid: score}, term→{cid: freq}, term→{cid: rank}
  const scoreMap = {}, freqMap = {}, rankMap = {};
  clusterIds.forEach(cid => {
    const cl = DATA[cid];
    if (!cl || !cl.keywords) return;
    // Sort keywords to compute rank within cluster
    const sorted = [...cl.keywords].sort((a, b) => a.score - b.score);
    const n = sorted.length;
    sorted.forEach((k, i) => {
      if (!scoreMap[k.term]) { scoreMap[k.term] = {}; freqMap[k.term] = {}; rankMap[k.term] = {}; }
      scoreMap[k.term][cid] = k.score;
      freqMap[k.term][cid] = k.frequency;
      rankMap[k.term][cid] = n > 1 ? i / (n - 1) : 0.5;  // normalized 0-1
    });
  });

  // Find shared terms (2+ clusters)
  let sharedTerms;
  if (CROSS_CLUSTER_TERMS && Array.isArray(CROSS_CLUSTER_TERMS)) {
    sharedTerms = CROSS_CLUSTER_TERMS.map(t => typeof t === "object" ? t.term : t)
      .filter(t => scoreMap[t] && Object.keys(scoreMap[t]).length >= 2);
  } else {
    sharedTerms = Object.keys(scoreMap).filter(t => Object.keys(scoreMap[t]).length >= 2);
  }

  if (!sharedTerms.length) {
    Plotly.react("chart-crosscluster", [], {
      title: "No cross-cluster terms found", height: 300,
      annotations: [{ text: "No keyword appears in 2 or more clusters", xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false }]
    });
    return;
  }

  // Sort by number of clusters desc, then max score desc
  sharedTerms.sort((a, b) => {
    const na = Object.keys(scoreMap[a]).length, nb = Object.keys(scoreMap[b]).length;
    if (nb !== na) return nb - na;
    const ma = Math.max(...Object.values(scoreMap[a])), mb = Math.max(...Object.values(scoreMap[b]));
    return mb - ma;
  });

  // Choose value map based on mode
  const valMap = mode === "frequency" ? freqMap : mode === "rank" ? rankMap : scoreMap;
  const clusterLabels = clusterIds.map(cid => "C" + cid + ": " + (DATA[cid].label || "").split(",")[0].trim());
  const z = sharedTerms.map(term =>
    clusterIds.map(cid => valMap[term] ? (valMap[term][cid] || 0) : 0)
  );

  // Hover always shows score + freq + rank
  const customdata = sharedTerms.map(term =>
    clusterIds.map(cid => ({
      score: scoreMap[term] ? (scoreMap[term][cid] || 0) : 0,
      freq: freqMap[term] ? (freqMap[term][cid] || 0) : 0,
      rank: rankMap[term] ? (rankMap[term][cid] || 0) : 0,
    }))
  );

  Plotly.react("chart-crosscluster", [{
    type: "heatmap",
    z: z, x: clusterLabels, y: sharedTerms,
    customdata: customdata,
    colorscale: mode === "rank" ? "Viridis" : "YlOrRd",
    hovertemplate: "<b>%{y}</b><br>Cluster: %{x}<br>Score: %{customdata.score:.4f}<br>Freq: %{customdata.freq:,}<br>Rank: %{customdata.rank:.2f}<extra></extra>",
    colorbar: { title: modeLabels[mode] || mode }
  }], {
    title: `Cross-Cluster Shared Keywords (${modeLabels[mode] || mode})`,
    yaxis: { autorange: "reversed", dtick: 1 },
    margin: { l: 200, r: 30, t: 50, b: 100 },
    height: Math.max(500, sharedTerms.length * 30 + 150),
    template: "plotly_white"
  }, { responsive: true, displaylogo: false });
}

// ---- Temporal highlights ----
function renderTemporalHighlights() {
  if (!currentCluster) return;
  const hlEl = document.getElementById("temporal-highlights");
  if (!hlEl) return;
  const kws = currentCluster ? currentCluster.keywords : [];
  if (!kws.length) { hlEl.innerHTML = ""; return; }

  let trendScores = {};

  if (TREND_SCORES && typeof TREND_SCORES === "object" && Object.keys(TREND_SCORES).length > 0) {
    // Use pipeline-provided trend scores, filtered to current cluster
    const clusterTerms = new Set(kws.map(k => k.term));
    Object.entries(TREND_SCORES).forEach(([term, score]) => {
      if (clusterTerms.has(term)) trendScores[term] = score;
    });
  } else {
    // Compute simple trend: mean last 3 years vs mean first 3 years of pub_year_series
    kws.forEach(k => {
      const series = k.pub_year_series || k.temporal;
      if (!series || !Object.keys(series).length) return;
      const years = Object.keys(series).map(Number).sort((a, b) => a - b);
      if (years.length < 2) return;
      const first3 = years.slice(0, 3);
      const last3 = years.slice(-3);
      const meanFirst = first3.reduce((s, y) => s + (series[String(y)] || 0), 0) / first3.length;
      const meanLast = last3.reduce((s, y) => s + (series[String(y)] || 0), 0) / last3.length;
      const denom = meanFirst || 1;
      trendScores[k.term] = (meanLast - meanFirst) / denom;
    });
  }

  const entries = Object.entries(trendScores).filter(([, v]) => v !== 0);
  if (!entries.length) { hlEl.innerHTML = ""; return; }

  entries.sort((a, b) => b[1] - a[1]);
  const emerging = entries.filter(([, v]) => v > 0).slice(0, 3);
  const declining = entries.filter(([, v]) => v < 0).slice(-3).reverse();

  let html = '<div style="font-size:.85rem;margin-bottom:.5rem;">';
  if (emerging.length) {
    html += '<strong>Emerging:</strong> ';
    emerging.forEach(([term]) => {
      html += '<span class="trend-badge trend-up">&#9650; ' + term + '</span>';
    });
  }
  if (declining.length) {
    html += ' <strong style="margin-left:.5rem;">Declining:</strong> ';
    declining.forEach(([term]) => {
      html += '<span class="trend-badge trend-down">&#9660; ' + term + '</span>';
    });
  }
  html += '</div>';
  hlEl.innerHTML = html;
}

// ---- Global Search ----
(function() {
  const searchInput = document.getElementById("global-search");
  const resultsDiv = document.getElementById("global-search-results");
  if (!searchInput || !resultsDiv) return;

  searchInput.addEventListener("input", function() {
    const q = this.value.trim().toLowerCase();
    if (!q) { resultsDiv.style.display = "none"; resultsDiv.innerHTML = ""; return; }

    const results = [];
    clusterIds.forEach(cid => {
      const cl = DATA[cid];
      if (!cl || !cl.keywords) return;
      cl.keywords.forEach(k => {
        if (k.term.toLowerCase().includes(q)) {
          results.push({ term: k.term, score: k.score, cid: cid, label: cl.label });
        }
      });
    });
    results.sort((a, b) => b.score - a.score);
    const limited = results.slice(0, 30);

    if (!limited.length) {
      resultsDiv.innerHTML = '<div class="search-result" style="color:#6c757d;">No results found</div>';
      resultsDiv.style.display = "block";
      return;
    }

    resultsDiv.innerHTML = limited.map(r =>
      '<div class="search-result" data-cid="' + r.cid + '" data-term="' + r.term.replace(/"/g, '&quot;') + '">' +
        '<strong>' + r.term + '</strong> <span style="color:#0d6efd;font-size:.8rem;">(' + r.score.toFixed(4) + ')</span>' +
        '<div class="sr-cluster">C' + r.cid + ': ' + r.label + '</div>' +
      '</div>'
    ).join("");
    resultsDiv.style.display = "block";
  });

  resultsDiv.addEventListener("click", function(e) {
    const item = e.target.closest(".search-result");
    if (!item) return;
    const cid = Number(item.dataset.cid);
    const term = item.dataset.term;

    // Switch to that cluster
    clusterSelect.value = cid;
    currentClusterId = cid;
    currentCluster = DATA[cid];
    updateMeta();

    // Switch to Keywords tab
    tabEls.forEach(t => t.classList.remove("active"));
    const kwTab = document.querySelector('.tab[data-tab="keywords"]');
    if (kwTab) kwTab.classList.add("active");
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    document.getElementById("panel-keywords").classList.add("active");
    currentTab = "keywords";
    renderKeywords();

    // Try to scroll to the term in the chart
    resultsDiv.style.display = "none";
    searchInput.value = "";
  });

  // Hide results when clicking outside
  document.addEventListener("click", function(e) {
    if (!searchInput.contains(e.target) && !resultsDiv.contains(e.target)) {
      resultsDiv.style.display = "none";
    }
  });
})();

// ---- Temporal Compare Mode ----
let compareMode = false;
let compareKeywords = []; // [{term, cid}]

(function() {
  const compareBtn = document.getElementById("temporal-compare-btn");
  const clearBtn = document.getElementById("temporal-compare-clear");
  const controls = document.getElementById("temporal-compare-controls");
  const searchInput = document.getElementById("temporal-compare-search");
  const resultsDiv = document.getElementById("temporal-compare-results");
  const tagsDiv = document.getElementById("temporal-compare-tags");
  if (!compareBtn) return;

  compareBtn.addEventListener("click", () => {
    compareMode = !compareMode;
    controls.style.display = compareMode ? "block" : "none";
    compareBtn.textContent = compareMode ? "Cancel" : "Compare";
    clearBtn.style.display = compareMode && compareKeywords.length ? "inline-block" : "none";
    if (!compareMode && compareKeywords.length) renderTemporal();
  });

  clearBtn.addEventListener("click", () => {
    compareKeywords = [];
    tagsDiv.innerHTML = "";
    clearBtn.style.display = "none";
    compareMode = false;
    controls.style.display = "none";
    compareBtn.textContent = "Compare";
    renderTemporal();
  });

  if (searchInput) {
    searchInput.addEventListener("input", function() {
      const q = this.value.trim().toLowerCase();
      if (!q) { resultsDiv.style.display = "none"; return; }
      const results = [];
      clusterIds.forEach(cid => {
        const cl = DATA[cid];
        if (!cl || !cl.keywords) return;
        cl.keywords.forEach(k => {
          if (k.term.toLowerCase().includes(q)) {
            results.push({ term: k.term, cid, score: k.score });
          }
        });
      });
      results.sort((a, b) => b.score - a.score);
      const limited = results.slice(0, 20);
      if (!limited.length) { resultsDiv.innerHTML = '<div class="search-result" style="color:#6c757d;">No results</div>'; resultsDiv.style.display = "block"; return; }
      resultsDiv.innerHTML = limited.map(r =>
        '<div class="search-result" data-cid="' + r.cid + '" data-term="' + r.term.replace(/"/g, '&quot;') + '">' +
        '<strong>' + r.term + '</strong> <span style="color:#6c757d;font-size:.8rem;">(C' + r.cid + ')</span></div>'
      ).join("");
      resultsDiv.style.display = "block";
    });

    resultsDiv.addEventListener("click", function(e) {
      const item = e.target.closest(".search-result");
      if (!item) return;
      const term = item.dataset.term;
      const cid = Number(item.dataset.cid);
      if (compareKeywords.length >= 10) return;
      if (compareKeywords.some(c => c.term === term && c.cid === cid)) return;
      compareKeywords.push({ term, cid });
      renderCompareTags();
      resultsDiv.style.display = "none";
      searchInput.value = "";
      clearBtn.style.display = "inline-block";
      renderTemporal();
    });
  }

  function renderCompareTags() {
    if (!tagsDiv) return;
    tagsDiv.innerHTML = compareKeywords.map((c, i) =>
      '<span class="compare-tag" data-idx="' + i + '">' + c.term + ' (C' + c.cid + ') &times;</span>'
    ).join("");
    tagsDiv.querySelectorAll(".compare-tag").forEach(tag => {
      tag.addEventListener("click", () => {
        compareKeywords.splice(Number(tag.dataset.idx), 1);
        renderCompareTags();
        if (!compareKeywords.length) clearBtn.style.display = "none";
        renderTemporal();
      });
    });
  }
})();

// Patch renderTemporal to handle compare mode
const _origRenderTemporal = renderTemporal;
renderTemporal = function() {
  if (compareMode && compareKeywords.length > 0) {
    renderTemporalHighlights();
    const metric = document.getElementById("temporal-metric").value;
    const metricLabels = { "pub_year_series": "Documents per Year", "ppm_series": "PPM", "loglift_series": "Log-Lift" };
    const yLabel = metricLabels[metric] || metric;
    const traces = [];
    compareKeywords.forEach((ck, i) => {
      const cl = DATA[ck.cid];
      if (!cl) return;
      const kw = cl.keywords.find(k => k.term === ck.term);
      if (!kw) return;
      const data = kw[metric] || kw.temporal || {};
      if (!Object.keys(data).length) return;
      const years = Object.keys(data).map(Number).sort((a,b) => a - b);
      const vals = years.map(y => data[String(y)] || 0);
      traces.push({
        name: ck.term + " (C" + ck.cid + ")",
        x: years, y: vals, mode: "lines+markers",
        line: { width: 2 }, marker: { size: 5 },
        hovertemplate: "<b>" + ck.term + " (C" + ck.cid + ")</b><br>Year: %{x}<br>" + yLabel + ": %{y:,.2f}<extra></extra>"
      });
    });
    if (!traces.length) {
      Plotly.react("chart-temporal", [], { title: "No temporal data for selected keywords", height: 300 });
      return;
    }
    Plotly.react("chart-temporal", traces, {
      title: "Temporal Comparison (" + yLabel + ")",
      xaxis: { title: "Year", dtick: 2, tickformat: "d" },
      yaxis: { title: yLabel, rangemode: "tozero" },
      legend: { orientation: "h", yanchor: "top", y: -0.2, font: { size: 11 } },
      margin: { l: 70, r: 30, t: 50, b: 120 }, height: 550,
      template: "plotly_white"
    }, { responsive: true, displaylogo: false });
  } else {
    _origRenderTemporal();
  }
};

// ---- URL Hash State ----
function updateHash() {
  const parts = [];
  if (currentClusterId != null) parts.push("cluster=" + currentClusterId);
  parts.push("tab=" + currentTab);
  window.location.hash = parts.join("&");
}

function restoreFromHash() {
  const hash = window.location.hash.replace(/^#/, "");
  if (!hash) return false;
  const params = {};
  hash.split("&").forEach(p => {
    const [k, v] = p.split("=");
    if (k && v) params[k] = v;
  });
  let restored = false;
  if (params.cluster != null && DATA[Number(params.cluster)]) {
    const cid = Number(params.cluster);
    clusterSelect.value = cid;
    currentClusterId = cid;
    currentCluster = DATA[cid];
    updateMeta();
    restored = true;
  }
  if (params.tab) {
    const tabEl = document.querySelector('.tab[data-tab="' + params.tab + '"]');
    if (tabEl) {
      tabEls.forEach(t => t.classList.remove("active"));
      tabEl.classList.add("active");
      panels.forEach(p => p.classList.remove("active"));
      const panelEl = document.getElementById("panel-" + params.tab);
      if (panelEl) panelEl.classList.add("active");
      currentTab = params.tab;
      restored = true;
    }
  }
  return restored;
}

// ---- Export CSV ----
function exportCurrentData() {
  let csv = "";
  let filename = "sciscape_export.csv";

  if (currentTab === "crosscluster") {
    // Export shared terms matrix
    const termMap = {};
    clusterIds.forEach(cid => {
      const cl = DATA[cid];
      if (!cl || !cl.keywords) return;
      cl.keywords.forEach(k => {
        if (!termMap[k.term]) termMap[k.term] = {};
        termMap[k.term][cid] = k.score;
      });
    });
    const shared = Object.keys(termMap).filter(t => Object.keys(termMap[t]).length >= 2);
    const headers = ["term", ...clusterIds.map(c => "C" + c)];
    csv = headers.join(",") + "\n";
    shared.forEach(t => {
      csv += '"' + t.replace(/"/g, '""') + '",' + clusterIds.map(c => (termMap[t][c] || 0).toFixed(6)).join(",") + "\n";
    });
    filename = "crosscluster_terms.csv";
  } else if (currentTab === "temporal" && currentCluster) {
    const metric = document.getElementById("temporal-metric").value;
    const kws = currentCluster.keywords.filter(k => k[metric] || k.temporal);
    if (kws.length) {
      const allYears = new Set();
      kws.forEach(k => { const d = k[metric] || k.temporal || {}; Object.keys(d).forEach(y => allYears.add(y)); });
      const years = [...allYears].sort();
      csv = "term," + years.join(",") + "\n";
      kws.forEach(k => {
        const d = k[metric] || k.temporal || {};
        csv += '"' + k.term.replace(/"/g, '""') + '",' + years.map(y => d[y] || 0).join(",") + "\n";
      });
    }
    filename = "temporal_C" + currentClusterId + ".csv";
  } else if (currentTab === "network" && currentCluster) {
    const edges = currentCluster.network_edges || [];
    csv = "source,target,weight\n";
    edges.forEach(e => {
      csv += '"' + e.source.replace(/"/g, '""') + '","' + e.target.replace(/"/g, '""') + '",' + e.weight + "\n";
    });
    filename = "network_C" + currentClusterId + ".csv";
  } else if (currentCluster) {
    // Keywords / dictionary export
    const kws = currentCluster.keywords;
    csv = "term,score,frequency,doc_coverage,depth_level,cross_cluster_count\n";
    kws.forEach(k => {
      csv += '"' + k.term.replace(/"/g, '""') + '",' + k.score.toFixed(6) + "," + k.frequency + "," +
        k.doc_coverage + "," + (k.depth_level != null ? k.depth_level : "") + "," +
        (k.cross_cluster_count || "") + "\n";
    });
    filename = "keywords_C" + currentClusterId + ".csv";
  }

  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.style.display = "none";
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Init: default to Overview (global) tab
if (clusterIds.length) {
  const restored = restoreFromHash();
  if (!restored) {
    // Start on overview — no cluster pre-selected
    currentTab = "overview";
    tabEls.forEach(t => t.classList.remove("active"));
    const ovTab = document.querySelector('.tab[data-tab="overview"]');
    if (ovTab) ovTab.classList.add("active");
    panels.forEach(p => p.classList.remove("active"));
    const ovPanel = document.getElementById("panel-overview");
    if (ovPanel) ovPanel.classList.add("active");
  }
  updateScopeUI();
  renderCurrentTab();
}
</script>
</body>
</html>"""
