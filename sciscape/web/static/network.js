/**
 * SciScape Cluster Network Visualization
 * D3.js force-directed layout with:
 * - Layer toggling (DC/BC/CC/Combined)
 * - Hierarchy level switching
 * - Overlay visualization (year/citations gradient)
 * - Interactive sliders (labels, edge filter, node sizing)
 * - Click detail panel (paper list per cluster)
 * - PNG/SVG export
 * - Density heatmap overlay
 */

const PALETTE = [
  '#2563eb','#dc2626','#059669','#d97706','#7c3aed',
  '#db2777','#0891b2','#65a30d','#ea580c','#4f46e5',
  '#0d9488','#c026d3','#ca8a04','#e11d48','#0284c7',
  '#16a34a','#9333ea','#f97316','#06b6d4','#84cc16',
];

class ClusterNetwork {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.data = null;
    this.jobId = null;
    this.currentLevel = null;
    this.activeLayers = new Set(['combined']);
    this.simulation = null;
    this.svg = null;
    this.g = null;
    this.width = 0;
    this.height = 0;
    this.showHierarchy = false;
    // Overlay
    this.colorMode = 'cluster'; // 'cluster' | 'year' | 'citations'
    // Sliders
    this.labelThreshold = 0.5; // 0..1 (0=all, 1=none)
    this.edgeMinWeight = 0;
    this.sizeMetric = 'size'; // 'size' | 'citations' | 'link_strength'
    // Density
    this.showDensity = false;
    // Stored refs for updates
    this._nodes = [];
    this._edges = [];
    this._nodeEls = null;
    this._linkEls = null;
    this._labelEls = null;
    this._rScale = null;
    this._maxSize = 1;
  }

  async load(jobId) {
    this.jobId = jobId;
    const resp = await fetch(`/api/jobs/${jobId}/network`);
    this.data = await resp.json();
    if (this.data.error) {
      this.container.innerHTML = `<div style="padding:2rem;color:#64748b;text-align:center;">${this.data.error}</div>`;
      return;
    }
    // Fetch labels
    try {
      const lr = await fetch(`/api/jobs/${jobId}/labels?strategy=tfidf_distinct&top_k=2`);
      const ld = await lr.json();
      if (ld.labels) this.data._labels = ld.labels;
    } catch(e) {}
    this.currentLevel = this.data.levels?.[0] || 'default';
    this.render();
  }

  render() {
    this.container.innerHTML = '';
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width || 700;
    this.height = rect.height || 540;

    // Controls
    const controls = document.createElement('div');
    controls.className = 'net-controls';
    controls.innerHTML = this._controlsHTML();
    this.container.appendChild(controls);

    // Main area: SVG + detail panel side by side
    const main = document.createElement('div');
    main.style.cssText = 'display:flex;height:' + (this.height - 72) + 'px;';
    this.container.appendChild(main);

    // SVG wrapper
    const svgWrap = document.createElement('div');
    svgWrap.style.cssText = 'flex:1;position:relative;overflow:hidden;';
    main.appendChild(svgWrap);

    // Detail panel (hidden initially)
    const detail = document.createElement('div');
    detail.id = 'net-detail';
    detail.className = 'net-detail';
    detail.style.display = 'none';
    main.appendChild(detail);

    const svgH = this.height - 72;
    const svgW = svgWrap.offsetWidth || (this.width - 10);
    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', svgW);
    svgEl.setAttribute('height', svgH);
    svgEl.id = 'net-svg';
    svgWrap.appendChild(svgEl);

    this.svg = d3.select(svgEl);
    this._svgW = svgW;
    this._svgH = svgH;
    this._setupDefs();
    this._draw();
    this._bindControls(controls);
  }

  _controlsHTML() {
    const levels = this.data.levels || [];
    const layers = this.data.layers || ['combined'];

    let h = '<div class="net-ctrl-row">';

    // Levels
    if (levels.length > 1) {
      h += '<div class="net-ctrl-group"><span class="net-ctrl-label">Level</span>';
      levels.forEach(l => {
        h += `<button class="net-chip${l===this.currentLevel?' active':''}" data-level="${l}">${l}</button>`;
      });
      h += '</div>';
    }

    // Layers
    h += '<div class="net-ctrl-group"><span class="net-ctrl-label">Layer</span>';
    layers.forEach(l => {
      h += `<button class="net-chip layer-chip${this.activeLayers.has(l)?' active':''}" data-layer="${l}">${l.toUpperCase()}</button>`;
    });
    h += '</div>';

    // Overlay color
    h += '<div class="net-ctrl-group"><span class="net-ctrl-label">Color</span>';
    ['cluster','year','citations'].forEach(m => {
      h += `<button class="net-chip${this.colorMode===m?' active':''}" data-color="${m}">${m}</button>`;
    });
    h += '</div>';

    // Size metric
    h += '<div class="net-ctrl-group"><span class="net-ctrl-label">Size</span>';
    ['size','citations','link_strength'].forEach(m => {
      const label = m === 'link_strength' ? 'links' : m;
      h += `<button class="net-chip${this.sizeMetric===m?' active':''}" data-sizem="${m}">${label}</button>`;
    });
    h += '</div>';

    // Density + Hierarchy + Export
    h += '<div class="net-ctrl-group">';
    if (this.data.hierarchy?.length) {
      h += `<button class="net-chip${this.showHierarchy?' active':''}" id="btn-hierarchy">Hier</button>`;
    }
    h += `<button class="net-chip${this.showDensity?' active':''}" id="btn-density">Density</button>`;
    h += `<button class="net-chip" id="btn-export-png">PNG</button>`;
    h += `<button class="net-chip" id="btn-export-svg">SVG</button>`;
    h += '</div>';

    h += '</div>';

    // Sliders row
    h += '<div class="net-ctrl-row" style="margin-top:4px;">';
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Labels</span>
      <input type="range" min="0" max="100" value="${Math.round((1-this.labelThreshold)*100)}" id="sl-labels" style="width:80px;"></div>`;
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Min edge</span>
      <input type="range" min="0" max="100" value="0" id="sl-edge" style="width:80px;"></div>`;
    h += '</div>';

    return h;
  }

  _bindControls(controls) {
    controls.querySelectorAll('[data-level]').forEach(b => b.onclick = () => { this.currentLevel = b.dataset.level; this.render(); });
    controls.querySelectorAll('[data-layer]').forEach(b => b.onclick = () => {
      const l = b.dataset.layer;
      if (this.activeLayers.has(l)) { if (this.activeLayers.size > 1) this.activeLayers.delete(l); }
      else this.activeLayers.add(l);
      this.render();
    });
    controls.querySelectorAll('[data-color]').forEach(b => b.onclick = () => { this.colorMode = b.dataset.color; this._updateColors(); });
    controls.querySelectorAll('[data-sizem]').forEach(b => b.onclick = () => { this.sizeMetric = b.dataset.sizem; this.render(); });

    const hBtn = controls.querySelector('#btn-hierarchy');
    if (hBtn) hBtn.onclick = () => { this.showHierarchy = !this.showHierarchy; this.render(); };

    const dBtn = controls.querySelector('#btn-density');
    if (dBtn) dBtn.onclick = () => { this.showDensity = !this.showDensity; this._toggleDensity(); dBtn.classList.toggle('active'); };

    // Sliders
    const slLabels = controls.querySelector('#sl-labels');
    if (slLabels) slLabels.oninput = () => { this.labelThreshold = 1 - slLabels.value / 100; this._updateLabels(); };
    const slEdge = controls.querySelector('#sl-edge');
    if (slEdge) slEdge.oninput = () => { this.edgeMinWeight = slEdge.value / 100; this._updateEdgeFilter(); };

    // Export
    const pngBtn = controls.querySelector('#btn-export-png');
    if (pngBtn) pngBtn.onclick = () => this._exportPNG();
    const svgBtn = controls.querySelector('#btn-export-svg');
    if (svgBtn) svgBtn.onclick = () => this._exportSVG();
  }

  _setupDefs() {
    const defs = this.svg.append('defs');
    const glow = defs.append('filter').attr('id', 'glow');
    glow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'blur');
    const merge = glow.append('feMerge');
    merge.append('feMergeNode').attr('in', 'blur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');
  }

  _draw() {
    this.g = this.svg.append('g').attr('class', 'network-root');
    const W = this._svgW, H = this._svgH;

    // Zoom
    this.svg.call(d3.zoom().scaleExtent([0.2, 8]).on('zoom', e => this.g.attr('transform', e.transform)));

    // Data
    const levelNodes = this.data.nodes?.find(n => n.level === this.currentLevel);
    const levelEdges = this.data.edges?.find(e => e.level === this.currentLevel);
    if (!levelNodes || !levelEdges) return;

    const labels = this.data._labels || {};
    const nodes = levelNodes.data.map(d => ({
      ...d,
      label: labels[String(d.id)] || d.label,
      citations: d.citations || d.size, // fallback
    }));
    this._nodes = nodes;

    // Merge edges
    const edgeMap = new Map();
    for (const layer of this.activeLayers) {
      for (const e of (levelEdges.data[layer] || [])) {
        const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
        if (edgeMap.has(key)) { edgeMap.get(key).weight += e.weight; edgeMap.get(key).layers.push(layer); }
        else edgeMap.set(key, {...e, layers: [layer]});
      }
    }
    const edges = [...edgeMap.values()];
    this._edges = edges;

    // Compute link strength per node
    const linkStrength = {};
    edges.forEach(e => {
      linkStrength[e.source] = (linkStrength[e.source] || 0) + e.weight;
      linkStrength[e.target] = (linkStrength[e.target] || 0) + e.weight;
    });
    nodes.forEach(n => n.link_strength = linkStrength[n.id] || 0);

    // Scales
    const sizeKey = this.sizeMetric;
    const maxS = Math.max(...nodes.map(n => n[sizeKey] || 1), 1);
    this._maxSize = maxS;
    const rScale = d3.scaleSqrt().domain([0, maxS]).range([4, 40]);
    this._rScale = rScale;
    const maxW = Math.max(...edges.map(e => e.weight), 1);
    const wScale = d3.scaleLinear().domain([0, maxW]).range([0.5, 5]);

    const layerColors = {combined:'#4a5976', dc:'#2563eb', bc:'#dc2626', cc:'#059669'};

    // Density layer (canvas behind SVG)
    this._densityCanvas = null;

    // Edges
    const linkG = this.g.append('g');
    this._linkEls = linkG.selectAll('line').data(edges).join('line')
      .attr('stroke', d => d.layers.length === 1 ? (layerColors[d.layers[0]] || '#4a5976') : '#4a5976')
      .attr('stroke-width', d => wScale(d.weight))
      .attr('stroke-opacity', d => 0.12 + 0.4 * d.weight / maxW);

    // Nodes
    const nodeG = this.g.append('g');
    const self = this;
    this._nodeEls = nodeG.selectAll('g').data(nodes).join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (ev, d) => { if (!ev.active) self.simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => { if (!ev.active) self.simulation.alphaTarget(0); d.fx = null; d.fy = null; })
      )
      .on('click', (ev, d) => self._showDetail(d));

    this._nodeEls.append('circle')
      .attr('r', d => rScale(d[sizeKey] || d.size))
      .attr('fill', d => this._nodeColor(d))
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .attr('filter', 'url(#glow)');

    // Labels
    this._labelEls = this._nodeEls.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', d => rScale(d[sizeKey] || d.size) + 12)
      .attr('font-size', '9.5px')
      .attr('font-family', 'Source Sans 3, sans-serif')
      .attr('fill', '#2d3a52').attr('font-weight', '500')
      .attr('pointer-events', 'none');
    this._updateLabels();

    // Tooltips
    this._nodeEls.append('title').text(d =>
      `${d.label}\nSize: ${d.size} papers (${d.pct}%)\nLinks: ${(d.link_strength||0).toFixed(1)}`
    );

    // Simulation
    this.simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(70).strength(d => 0.08 + 0.25 * (d.weight / maxW)))
      .force('charge', d3.forceManyBody().strength(d => -rScale(d[sizeKey] || d.size) * 7))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide().radius(d => rScale(d[sizeKey] || d.size) + 3))
      .on('tick', () => {
        this._linkEls.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        this._nodeEls.attr('transform', d => `translate(${d.x},${d.y})`);
      });
  }

  // ── OVERLAY COLORS ──────────────────────────────────────
  _nodeColor(d) {
    if (this.colorMode === 'cluster') return PALETTE[d.id % PALETTE.length];
    if (this.colorMode === 'year') {
      const years = this._nodes.map(n => n.year || 2020).filter(y => y > 0);
      const minY = Math.min(...years), maxY = Math.max(...years);
      const t = maxY > minY ? ((d.year || minY) - minY) / (maxY - minY) : 0.5;
      return d3.interpolateViridis(t);
    }
    if (this.colorMode === 'citations') {
      const vals = this._nodes.map(n => n.citations || 0);
      const mx = Math.max(...vals, 1);
      return d3.interpolateYlOrRd((d.citations || 0) / mx);
    }
    return PALETTE[d.id % PALETTE.length];
  }

  _updateColors() {
    if (!this._nodeEls) return;
    this._nodeEls.select('circle').attr('fill', d => this._nodeColor(d));
    // Re-render chips
    this.container.querySelectorAll('[data-color]').forEach(b => b.classList.toggle('active', b.dataset.color === this.colorMode));
  }

  // ── LABEL DENSITY ───────────────────────────────────────
  _updateLabels() {
    if (!this._labelEls) return;
    const threshold = this.labelThreshold;
    this._labelEls.attr('display', d => {
      const rank = (d[this.sizeMetric] || d.size) / this._maxSize;
      return rank >= threshold ? null : 'none';
    });
  }

  // ── EDGE FILTER ─────────────────────────────────────────
  _updateEdgeFilter() {
    if (!this._linkEls) return;
    const maxW = Math.max(...this._edges.map(e => e.weight), 1);
    const minW = this.edgeMinWeight * maxW;
    this._linkEls.attr('display', d => d.weight >= minW ? null : 'none');
  }

  // ── DENSITY HEATMAP ─────────────────────────────────────
  _toggleDensity() {
    const existing = this.container.querySelector('.density-overlay');
    if (existing) { existing.remove(); this.showDensity = false; return; }
    if (!this._nodes.length || !this._nodes[0].x) return;

    const canvas = document.createElement('canvas');
    canvas.className = 'density-overlay';
    canvas.width = this._svgW;
    canvas.height = this._svgH;
    canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;opacity:0.35;';
    this.container.querySelector('div[style*="flex"]>div:first-child').appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const r = 60;
    this._nodes.forEach(n => {
      if (!n.x || !n.y) return;
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r);
      const intensity = (n.size / this._maxSize);
      grad.addColorStop(0, `rgba(37, 99, 235, ${0.4 * intensity})`);
      grad.addColorStop(1, 'rgba(37, 99, 235, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(n.x - r, n.y - r, r * 2, r * 2);
    });
    this.showDensity = true;
  }

  // ── CLICK DETAIL PANEL ──────────────────────────────────
  async _showDetail(d) {
    const panel = document.getElementById('net-detail');
    if (!panel) return;
    panel.style.display = 'block';
    panel.innerHTML = `
      <div class="net-detail-header">
        <strong style="color:${PALETTE[d.id % PALETTE.length]}">${this._esc(d.label)}</strong>
        <button onclick="document.getElementById('net-detail').style.display='none'" style="float:right;background:none;border:none;cursor:pointer;color:#64748b;font-size:1.1rem;">&times;</button>
      </div>
      <div class="net-detail-stats">
        <span class="stat-pill">${d.size} papers</span>
        <span class="stat-pill">${d.pct}%</span>
      </div>
      <div class="net-detail-body" id="net-detail-body">Loading papers...</div>
    `;
    // Fetch papers for this cluster
    try {
      const resp = await fetch(`/api/jobs/${this.jobId}/cluster/${d.id}`);
      const data = await resp.json();
      const body = document.getElementById('net-detail-body');
      if (data.papers?.length) {
        body.innerHTML = data.papers.slice(0, 20).map(p =>
          `<div class="net-paper"><div class="net-paper-title">${this._esc(p.title)}</div>
           <div class="net-paper-meta">${p.year || ''} &middot; cited ${p.cited_by_count || 0}x</div></div>`
        ).join('');
      } else {
        body.innerHTML = '<div style="color:#64748b;">No paper data available.</div>';
      }
    } catch(e) {
      document.getElementById('net-detail-body').innerHTML = '<div style="color:#64748b;">Could not load papers.</div>';
    }
  }

  // ── EXPORT ──────────────────────────────────────────────
  _exportSVG() {
    const svgEl = document.getElementById('net-svg');
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgEl);
    const blob = new Blob([svgStr], {type: 'image/svg+xml'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sciscape_network.svg';
    a.click();
  }

  _exportPNG() {
    const svgEl = document.getElementById('net-svg');
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgEl);
    const canvas = document.createElement('canvas');
    canvas.width = this._svgW * 2;
    canvas.height = this._svgH * 2;
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);
    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = '#f8f9fc';
      ctx.fillRect(0, 0, this._svgW, this._svgH);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(blob => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'sciscape_network.png';
        a.click();
      });
    };
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));
  }

  _esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
}

window.ClusterNetwork = ClusterNetwork;
