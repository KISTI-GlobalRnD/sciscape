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
    this.labelFontSize = 9.5;  // px
    this.labelRotate = false;
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

    // Tools
    h += '<div class="net-ctrl-group">';
    if (this.data.hierarchy?.length) {
      h += `<button class="net-chip${this.showHierarchy?' active':''}" id="btn-hierarchy">Hier</button>`;
    }
    h += `<button class="net-chip${this.showDensity?' active':''}" id="btn-density">Density</button>`;
    h += `<button class="net-chip" id="btn-reset-layout" title="Reset node positions">Reset</button>`;
    h += `<button class="net-chip" id="btn-temporal">Timeline</button>`;
    h += `<button class="net-chip" id="btn-bridge">Bridge</button>`;
    h += `<button class="net-chip" id="btn-split">Split</button>`;
    h += `<button class="net-chip" id="btn-export-png">PNG</button>`;
    h += `<button class="net-chip" id="btn-export-svg">SVG</button>`;
    h += '</div>';

    h += '</div>';

    // Sliders + Search row
    h += '<div class="net-ctrl-row" style="margin-top:4px;">';
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Labels</span>
      <input type="range" min="0" max="100" value="${Math.round((1-this.labelThreshold)*100)}" id="sl-labels" style="width:70px;"></div>`;
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Font</span>
      <input type="range" min="6" max="18" value="10" step="1" id="sl-fontsize" style="width:60px;"></div>`;
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Rotate</span>
      <input type="checkbox" id="cb-rotate"></div>`;
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Min edge</span>
      <input type="range" min="0" max="100" value="0" id="sl-edge" style="width:70px;"></div>`;
    h += `<div class="net-ctrl-group"><span class="net-ctrl-label">Search</span>
      <input type="text" id="net-search" placeholder="keyword..." style="width:120px;font-family:var(--font-mono,monospace);font-size:0.72rem;padding:2px 6px;border:1px solid #c5cfe0;border-radius:3px;"></div>`;
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
    const rlBtn = controls.querySelector('#btn-reset-layout');
    if (rlBtn) rlBtn.onclick = () => { this.resetLayout(); };

    // Sliders
    const slLabels = controls.querySelector('#sl-labels');
    if (slLabels) slLabels.oninput = () => { this.labelThreshold = 1 - slLabels.value / 100; this._updateLabels(); };
    const slFont = controls.querySelector('#sl-fontsize');
    if (slFont) slFont.oninput = () => { this.setLabelFontSize(parseFloat(slFont.value)); };
    const cbRotate = controls.querySelector('#cb-rotate');
    if (cbRotate) cbRotate.onchange = () => { this.setLabelRotate(cbRotate.checked); };
    const slEdge = controls.querySelector('#sl-edge');
    if (slEdge) slEdge.oninput = () => { this.edgeMinWeight = slEdge.value / 100; this._updateEdgeFilter(); };

    // Export
    const pngBtn = controls.querySelector('#btn-export-png');
    if (pngBtn) pngBtn.onclick = () => this._exportPNG();
    const svgBtn = controls.querySelector('#btn-export-svg');
    if (svgBtn) svgBtn.onclick = () => this._exportSVG();

    // ── Feature 1: Temporal playback ──
    const tempBtn = controls.querySelector('#btn-temporal');
    if (tempBtn) tempBtn.onclick = () => this._openTimeline();

    // ── Feature 4: Bridge analysis ──
    const bridgeBtn = controls.querySelector('#btn-bridge');
    if (bridgeBtn) bridgeBtn.onclick = () => this._enableBridgeMode();

    // ── Feature 3: Split view ──
    const splitBtn = controls.querySelector('#btn-split');
    if (splitBtn) splitBtn.onclick = () => this._toggleSplitView();

    // ── Feature 5: Search highlight ──
    const searchInput = controls.querySelector('#net-search');
    if (searchInput) {
      searchInput.oninput = () => this._searchHighlight(searchInput.value);
      searchInput.onkeydown = (e) => { if (e.key === 'Escape') { searchInput.value = ''; this._searchHighlight(''); } };
    }
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
        .on('end', (ev, d) => {
          if (!ev.active) self.simulation.alphaTarget(0);
          // Keep pinned position and save layout
          d.fx = d.x; d.fy = d.y;
          self._saveLayout();
        })
      )
      .on('click', (ev, d) => self._showDetail(d))
      .on('dblclick', (ev, d) => self._semanticZoom(d))
      .on('contextmenu', (ev, d) => { ev.preventDefault(); self._renameCluster(d); });

    this._nodeEls.append('circle')
      .attr('r', d => rScale(d[sizeKey] || d.size))
      .attr('fill', d => this._nodeColor(d))
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .attr('filter', 'url(#glow)');

    // Labels
    const fs = this.labelFontSize;
    const rot = this.labelRotate;
    this._labelEls = this._nodeEls.append('text')
      .text(d => d.label)
      .attr('text-anchor', rot ? 'start' : 'middle')
      .attr('dy', d => rot ? 4 : rScale(d[sizeKey] || d.size) + 12)
      .attr('dx', d => rot ? rScale(d[sizeKey] || d.size) + 4 : 0)
      .attr('transform', rot ? 'rotate(-30)' : null)
      .attr('font-size', `${fs}px`)
      .attr('font-family', 'Source Sans 3, sans-serif')
      .attr('fill', '#2d3a52').attr('font-weight', '500')
      .attr('pointer-events', 'none');
    this._updateLabels();

    // Tooltips (with year range and citations)
    this._nodeEls.append('title').text(d => {
      let tip = `${d.label}\nSize: ${d.size} papers (${d.pct}%)`;
      if (d.avg_year) tip += `\nAvg year: ${Math.round(d.avg_year)}`;
      if (d.year_range) tip += ` (${d.year_range[0]}–${d.year_range[1]})`;
      if (d.citations) tip += `\nCitations: ${d.citations}`;
      tip += `\nLinks: ${(d.link_strength||0).toFixed(1)}`;
      return tip;
    });

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

    // Restore saved layout (if any)
    if (this._restoreLayout()) {
      // Force one tick to apply positions
      this._linkEls.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      this._nodeEls.attr('transform', d => `translate(${d.x},${d.y})`);
    }
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

  // ── LABEL DENSITY + FONT SIZE ────────────────────────────
  _updateLabels() {
    if (!this._labelEls) return;
    const threshold = this.labelThreshold;
    const fs = this.labelFontSize;
    this._labelEls
      .attr('display', d => {
        const rank = (d[this.sizeMetric] || d.size) / this._maxSize;
        return rank >= threshold ? null : 'none';
      })
      .attr('font-size', `${fs}px`);
  }

  // ── LAYOUT PERSISTENCE ──────────────────────────────────
  _layoutKey() {
    return `sciscape_layout_${this.jobId}_${this.currentLevel}`;
  }

  _saveLayout() {
    if (!this._nodes.length) return;
    const positions = {};
    this._nodes.forEach(n => {
      if (n.fx != null && n.fy != null) {
        positions[n.id] = { x: Math.round(n.fx), y: Math.round(n.fy) };
      }
    });
    if (Object.keys(positions).length > 0) {
      try { localStorage.setItem(this._layoutKey(), JSON.stringify(positions)); } catch(e) {}
    }
  }

  _restoreLayout() {
    try {
      const saved = localStorage.getItem(this._layoutKey());
      if (!saved) return false;
      const positions = JSON.parse(saved);
      let restored = 0;
      this._nodes.forEach(n => {
        const p = positions[n.id];
        if (p) { n.x = n.fx = p.x; n.y = n.fy = p.y; restored++; }
      });
      if (restored > 0) {
        // Stop simulation since we have fixed positions
        if (this.simulation) this.simulation.alpha(0).stop();
        return true;
      }
    } catch(e) {}
    return false;
  }

  resetLayout() {
    try { localStorage.removeItem(this._layoutKey()); } catch(e) {}
    this._nodes.forEach(n => { n.fx = null; n.fy = null; });
    if (this.simulation) this.simulation.alpha(1).restart();
  }

  setLabelFontSize(px) {
    this.labelFontSize = px;
    this._updateLabels();
  }

  setLabelRotate(on) {
    this.labelRotate = on;
    if (this._labelEls) {
      this._labelEls
        .attr('text-anchor', on ? 'start' : 'middle')
        .attr('transform', on ? 'rotate(-30)' : null);
    }
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

    const W = this._svgW, H = this._svgH;
    const canvas = document.createElement('canvas');
    canvas.className = 'density-overlay';
    canvas.width = W; canvas.height = H;
    canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;opacity:0.55;';
    this.container.querySelector('div[style*="flex"]>div:first-child').appendChild(canvas);

    // 2D Kernel Density Estimation on a grid
    const gridSize = 2; // pixel resolution
    const cols = Math.ceil(W / gridSize), rows = Math.ceil(H / gridSize);
    const grid = new Float32Array(cols * rows);
    const bandwidth = 50; // kernel bandwidth in pixels
    const bw2 = bandwidth * bandwidth;

    // Accumulate density from each node (weighted by size)
    this._nodes.forEach(n => {
      if (!n.x || !n.y) return;
      const weight = (n.size || 1);
      const cx = Math.floor(n.x / gridSize), cy = Math.floor(n.y / gridSize);
      const r = Math.ceil(bandwidth / gridSize);
      for (let dy = -r; dy <= r; dy++) {
        const gy = cy + dy;
        if (gy < 0 || gy >= rows) continue;
        for (let dx = -r; dx <= r; dx++) {
          const gx = cx + dx;
          if (gx < 0 || gx >= cols) continue;
          const dist2 = (dx * gridSize) ** 2 + (dy * gridSize) ** 2;
          if (dist2 > bw2) continue;
          // Gaussian kernel
          grid[gy * cols + gx] += weight * Math.exp(-dist2 / (2 * bw2 * 0.15));
        }
      }
    });

    // Normalize to [0, 1]
    let maxVal = 0;
    for (let i = 0; i < grid.length; i++) if (grid[i] > maxVal) maxVal = grid[i];
    if (maxVal === 0) maxVal = 1;

    // Render with viridis-like colormap
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < grid.length; i++) {
      const t = grid[i] / maxVal;
      if (t < 0.01) { img.data[i*4+3] = 0; continue; } // transparent below threshold
      // Viridis approximation: dark purple → blue → teal → yellow
      const r_ = Math.round(t < 0.5 ? 68 + t * 200 : 50 + t * 400);
      const g_ = Math.round(t < 0.5 ? 1 + t * 300 : t * 255);
      const b_ = Math.round(t < 0.5 ? 84 + t * 200 : 255 - t * 200);
      img.data[i*4]   = Math.min(255, r_);
      img.data[i*4+1] = Math.min(255, g_);
      img.data[i*4+2] = Math.min(255, Math.max(0, b_));
      img.data[i*4+3] = Math.round(180 * Math.sqrt(t)); // alpha
    }

    // Scale up to canvas size
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = cols; tmpCanvas.height = rows;
    tmpCanvas.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmpCanvas, 0, 0, W, H);
    this.showDensity = true;
  }

  // ── CLICK DETAIL PANEL ──────────────────────────────────
  _renameCluster(d) {
    const newLabel = prompt(`Rename cluster "${d.label}":`, d.label);
    if (newLabel && newLabel !== d.label) {
      d.label = newLabel;
      // Update label in SVG
      if (this._labelEls) {
        this._labelEls.filter(n => n.id === d.id).text(newLabel);
      }
      // Update tooltip
      if (this._nodeEls) {
        this._nodeEls.filter(n => n.id === d.id).select('title').text(
          `${newLabel}\nSize: ${d.size} papers (${d.pct}%)\nLinks: ${(d.link_strength||0).toFixed(1)}`
        );
      }
    }
  }

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

  // ══════════════════════════════════════════════════════════
  // FEATURE 1: Temporal Playback
  // ══════════════════════════════════════════════════════════
  async _openTimeline() {
    if (this._timelineActive) { this._closeTimeline(); return; }
    const resp = await fetch(`/api/jobs/${this.jobId}/temporal`);
    const data = await resp.json();
    if (data.error || !data.years?.length) { alert('No temporal data available.'); return; }
    this._timelineData = data;
    this._timelineActive = true;

    // Add timeline bar
    const bar = document.createElement('div');
    bar.id = 'timeline-bar';
    bar.style.cssText = 'position:absolute;bottom:0;left:0;right:0;background:rgba(10,15,26,0.92);padding:8px 16px;display:flex;align-items:center;gap:10px;z-index:10;';
    bar.innerHTML = `
      <button id="tl-play" style="background:#e8a838;border:none;color:#0a0f1a;padding:3px 12px;border-radius:3px;cursor:pointer;font-weight:600;font-size:0.75rem;">Play</button>
      <input type="range" id="tl-slider" min="0" max="${data.years.length-1}" value="${data.years.length-1}" style="flex:1;">
      <span id="tl-year" style="color:#e8a838;font-family:var(--font-mono,monospace);font-size:0.85rem;min-width:40px;">${data.years[data.years.length-1]}</span>
      <button id="tl-close" style="background:none;border:none;color:#8899b3;cursor:pointer;font-size:1rem;">&times;</button>
    `;
    this.container.querySelector('div[style*="flex"]>div:first-child').appendChild(bar);

    const slider = bar.querySelector('#tl-slider');
    const yearLabel = bar.querySelector('#tl-year');
    slider.oninput = () => { yearLabel.textContent = data.years[slider.value]; this._applySnapshot(data.years[slider.value]); };
    bar.querySelector('#tl-close').onclick = () => this._closeTimeline();

    let playing = false, interval;
    bar.querySelector('#tl-play').onclick = () => {
      if (playing) { clearInterval(interval); playing = false; bar.querySelector('#tl-play').textContent = 'Play'; return; }
      playing = true; bar.querySelector('#tl-play').textContent = 'Pause';
      let idx = parseInt(slider.value);
      interval = setInterval(() => {
        idx++; if (idx >= data.years.length) { clearInterval(interval); playing = false; bar.querySelector('#tl-play').textContent = 'Play'; return; }
        slider.value = idx; yearLabel.textContent = data.years[idx]; this._applySnapshot(data.years[idx]);
      }, 800);
    };
  }

  _applySnapshot(year) {
    const snap = this._timelineData?.snapshots?.[year];
    if (!snap || !this._nodeEls) return;
    const activeIds = new Set(snap.nodes.map(n => n.id));
    const sizeMap = new Map(snap.nodes.map(n => [n.id, n.size]));
    const edgePairs = new Set(snap.edges.map(e => `${Math.min(e.source,e.target)}-${Math.max(e.source,e.target)}`));

    // Fade/show nodes
    this._nodeEls.transition().duration(300)
      .attr('opacity', d => activeIds.has(d.id) ? 1 : 0.08);
    this._nodeEls.select('circle').transition().duration(300)
      .attr('r', d => activeIds.has(d.id) ? this._rScale(sizeMap.get(d.id) || 0) : 2);

    // Fade/show edges
    this._linkEls.transition().duration(300)
      .attr('opacity', d => {
        const s = typeof d.source === 'object' ? d.source.id : d.source;
        const t = typeof d.target === 'object' ? d.target.id : d.target;
        return edgePairs.has(`${Math.min(s,t)}-${Math.max(s,t)}`) ? 0.5 : 0.02;
      });
  }

  _closeTimeline() {
    this._timelineActive = false;
    const bar = document.getElementById('timeline-bar');
    if (bar) bar.remove();
    // Restore all
    if (this._nodeEls) this._nodeEls.transition().duration(200).attr('opacity', 1);
    if (this._nodeEls) this._nodeEls.select('circle').transition().duration(200)
      .attr('r', d => this._rScale(d[this.sizeMetric] || d.size));
    if (this._linkEls) this._linkEls.transition().duration(200).attr('opacity', d => 0.12 + 0.4 * d.weight / Math.max(...this._edges.map(e=>e.weight),1));
  }

  // ══════════════════════════════════════════════════════════
  // FEATURE 2: Semantic Zoom (Drill-down on double-click)
  // ══════════════════════════════════════════════════════════
  _semanticZoom(d) {
    const levels = this.data.levels || [];
    const curIdx = levels.indexOf(this.currentLevel);
    if (curIdx < 0 || curIdx >= levels.length - 1) return; // no finer level

    // Find child clusters in the next finer level via hierarchy
    const finerLevel = levels[curIdx + 1]; // wait — finer = earlier index (nano < micro < meso)
    // Actually levels are ordered finest → coarsest, so curIdx+1 is coarser.
    // We want to drill DOWN: from coarser to finer.
    // If levels = [nano, micro, meso] and we're at micro (idx=1), drill to nano (idx=0)
    if (curIdx === 0) return; // already at finest
    const finerLevelName = levels[curIdx - 1];

    // Find which finer-level clusters belong to this coarser cluster
    const hierarchy = this.data.hierarchy || [];
    const childIds = hierarchy
      .filter(h => h.parent_level === this.currentLevel && h.child_level === finerLevelName && h.parent === d.id)
      .map(h => h.child);

    if (childIds.length === 0) {
      // Fallback: just switch level
      this.currentLevel = finerLevelName;
      this.render();
      return;
    }

    // Switch to finer level and highlight only children
    this.currentLevel = finerLevelName;
    this._drillFilter = new Set(childIds);
    this.render();
    // After render, fade non-children
    if (this._nodeEls && this._drillFilter) {
      this._nodeEls.select('circle').transition().duration(300)
        .attr('opacity', n => this._drillFilter.has(n.id) ? 1 : 0.12);
      this._labelEls?.transition().duration(300)
        .attr('opacity', n => this._drillFilter.has(n.id) ? 1 : 0);
      // Clear drill filter after 5 seconds
      setTimeout(() => {
        this._drillFilter = null;
        this._nodeEls?.select('circle').transition().duration(500).attr('opacity', 1);
        this._labelEls?.transition().duration(500).attr('opacity', 1);
        this._updateLabels();
      }, 5000);
    }
  }

  // ══════════════════════════════════════════════════════════
  // FEATURE 3: Split View (DC vs BC side by side)
  // ══════════════════════════════════════════════════════════
  _toggleSplitView() {
    if (this._splitActive) { this.render(); this._splitActive = false; return; }
    this._splitActive = true;
    const layers = (this.data.layers || []).filter(l => l !== 'combined');
    if (layers.length < 2) { alert('Need at least 2 layer types for split view.'); return; }

    this.container.innerHTML = '';
    const rect = this.container.getBoundingClientRect();
    const W = rect.width || 700;
    const H = rect.height || 500;

    // Header
    const hdr = document.createElement('div');
    hdr.className = 'net-controls';
    hdr.innerHTML = `<div class="net-ctrl-row">
      <span class="net-ctrl-label">Split View</span>
      ${layers.map(l => `<span class="net-chip active layer-chip" data-layer="${l}" style="pointer-events:none;">${l.toUpperCase()}</span>`).join('')}
      <button class="net-chip" id="btn-split-close">Close</button>
    </div>`;
    this.container.appendChild(hdr);
    hdr.querySelector('#btn-split-close').onclick = () => { this._splitActive = false; this.render(); };

    // Side-by-side SVGs
    const wrap = document.createElement('div');
    wrap.style.cssText = `display:flex;gap:4px;height:${H-40}px;`;
    this.container.appendChild(wrap);

    const levelEdges = this.data.edges?.find(e => e.level === this.currentLevel);
    const levelNodes = this.data.nodes?.find(n => n.level === this.currentLevel);
    if (!levelEdges || !levelNodes) return;

    const labels = this.data._labels || {};

    layers.forEach((layer, idx) => {
      const div = document.createElement('div');
      div.style.cssText = 'flex:1;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;background:#f8f9fc;';
      const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      const sw = (W - 8) / layers.length;
      svgEl.setAttribute('width', sw);
      svgEl.setAttribute('height', H - 44);
      div.appendChild(svgEl);
      wrap.appendChild(div);

      const svg = d3.select(svgEl);
      const g = svg.append('g');
      svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => g.attr('transform', e.transform)));

      const nodes = levelNodes.data.map(d => ({...d, label: labels[String(d.id)] || d.label}));
      const edges = (levelEdges.data[layer] || []).map(d => ({...d}));
      const maxS = Math.max(...nodes.map(n => n.size), 1);
      const rS = d3.scaleSqrt().domain([0, maxS]).range([3, 25]);
      const maxW = Math.max(...edges.map(e => e.weight), 1);
      const layerColors = {dc:'#2563eb', bc:'#dc2626', cc:'#059669'};
      const color = layerColors[layer] || '#4a5976';

      // Title
      g.append('text').text(layer.toUpperCase()).attr('x', 10).attr('y', 18)
        .attr('font-size', '11px').attr('font-weight', '700').attr('fill', color);

      const linkEls = g.append('g').selectAll('line').data(edges).join('line')
        .attr('stroke', color).attr('stroke-width', d => 0.5 + 3 * d.weight / maxW)
        .attr('stroke-opacity', d => 0.15 + 0.4 * d.weight / maxW);

      const nodeEls = g.append('g').selectAll('circle').data(nodes).join('circle')
        .attr('r', d => rS(d.size)).attr('fill', d => PALETTE[d.id % PALETTE.length])
        .attr('stroke', '#fff').attr('stroke-width', 1);

      nodeEls.append('title').text(d => `${d.label} (${d.size})`);

      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(50))
        .force('charge', d3.forceManyBody().strength(-40))
        .force('center', d3.forceCenter(sw / 2, (H - 44) / 2))
        .on('tick', () => {
          linkEls.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
          nodeEls.attr('cx', d => d.x).attr('cy', d => d.y);
        });
    });
  }

  // ══════════════════════════════════════════════════════════
  // FEATURE 4: Bridge Analysis
  // ══════════════════════════════════════════════════════════
  _enableBridgeMode() {
    if (this._bridgeMode) { this._bridgeMode = false; this._bridgeFirst = null; this._restoreAll(); return; }
    this._bridgeMode = true;
    this._bridgeFirst = null;
    // Override click handler
    if (this._nodeEls) {
      this._nodeEls.on('click', (ev, d) => this._bridgeClick(d));
    }
    // Show hint
    const panel = document.getElementById('net-detail');
    if (panel) {
      panel.style.display = 'block';
      panel.innerHTML = '<div style="padding:1rem;color:#4a5976;"><strong>Bridge Mode</strong><br>Click two clusters to find bridging papers between them.</div>';
    }
  }

  async _bridgeClick(d) {
    if (!this._bridgeFirst) {
      this._bridgeFirst = d;
      // Highlight selected
      this._nodeEls.select('circle').attr('stroke', n => n.id === d.id ? '#e8a838' : '#fff')
        .attr('stroke-width', n => n.id === d.id ? 3 : 1.5);
      const panel = document.getElementById('net-detail');
      if (panel) panel.innerHTML = `<div style="padding:1rem;color:#4a5976;"><strong>Bridge Mode</strong><br>Selected: <strong style="color:${PALETTE[d.id%PALETTE.length]}">${this._esc(d.label)}</strong><br>Now click a second cluster.</div>`;
    } else {
      const a = this._bridgeFirst, b = d;
      this._bridgeFirst = null;
      this._bridgeMode = false;

      // Highlight edge between a and b
      if (this._linkEls) {
        this._linkEls.attr('stroke', e => {
          const s = typeof e.source === 'object' ? e.source.id : e.source;
          const t = typeof e.target === 'object' ? e.target.id : e.target;
          if ((s === a.id && t === b.id) || (s === b.id && t === a.id)) return '#e8a838';
          return '#4a5976';
        }).attr('stroke-width', e => {
          const s = typeof e.source === 'object' ? e.source.id : e.source;
          const t = typeof e.target === 'object' ? e.target.id : e.target;
          if ((s === a.id && t === b.id) || (s === b.id && t === a.id)) return 4;
          return 1;
        });
      }
      this._nodeEls.select('circle')
        .attr('stroke', n => (n.id === a.id || n.id === b.id) ? '#e8a838' : '#fff')
        .attr('stroke-width', n => (n.id === a.id || n.id === b.id) ? 3 : 1.5);

      // Fetch bridge papers
      const panel = document.getElementById('net-detail');
      if (panel) {
        panel.style.display = 'block';
        panel.innerHTML = '<div style="padding:1rem;">Loading bridge papers...</div>';
        try {
          const resp = await fetch(`/api/jobs/${this.jobId}/bridge?cluster_a=${a.id}&cluster_b=${b.id}`);
          const data = await resp.json();
          let html = `<div class="net-detail-header"><strong>Bridge: <span style="color:${PALETTE[a.id%PALETTE.length]}">${this._esc(a.label)}</span> ↔ <span style="color:${PALETTE[b.id%PALETTE.length]}">${this._esc(b.label)}</span></strong>
            <button onclick="document.getElementById('net-detail').style.display='none'" style="float:right;background:none;border:none;cursor:pointer;color:#64748b;">&times;</button></div>`;
          if (data.papers?.length) {
            html += data.papers.map(p =>
              `<div class="net-paper"><div class="net-paper-title">${this._esc(p.title)}</div>
               <div class="net-paper-meta">${p.year||''} &middot; C${p.cluster} &middot; bridge: ${p.bridge_score}</div></div>`
            ).join('');
          } else {
            html += '<div style="color:#64748b;padding:0.5rem;">No bridging papers found.</div>';
          }
          panel.innerHTML = html;
        } catch(e) {
          panel.innerHTML = '<div style="padding:1rem;color:#d85a4a;">Failed to load bridge data.</div>';
        }
      }
      // Restore click handler
      this._nodeEls.on('click', (ev, nd) => this._showDetail(nd));
    }
  }

  _restoreAll() {
    if (this._nodeEls) {
      this._nodeEls.select('circle').attr('stroke', '#fff').attr('stroke-width', 1.5);
      this._nodeEls.on('click', (ev, d) => this._showDetail(d));
    }
    if (this._linkEls) {
      const maxW = Math.max(...this._edges.map(e => e.weight), 1);
      this._linkEls.attr('stroke', '#4a5976').attr('stroke-width', d => 0.5 + 4.5 * d.weight / maxW);
    }
  }

  // ══════════════════════════════════════════════════════════
  // FEATURE 5: Search & Highlight
  // ══════════════════════════════════════════════════════════
  _searchHighlight(query) {
    if (!this._nodeEls) return;
    const q = query.toLowerCase().trim();
    if (!q) {
      // Restore all
      this._nodeEls.select('circle').transition().duration(200).attr('opacity', 1);
      this._labelEls?.transition().duration(200).attr('opacity', 1);
      this._linkEls?.transition().duration(200).attr('opacity', d => 0.12 + 0.4 * d.weight / Math.max(...this._edges.map(e=>e.weight),1));
      return;
    }
    // Match nodes whose label contains query
    const matchIds = new Set();
    this._nodes.forEach(n => {
      if (n.label.toLowerCase().includes(q)) matchIds.add(n.id);
    });
    // Fade non-matches
    this._nodeEls.select('circle').transition().duration(200)
      .attr('opacity', d => matchIds.has(d.id) ? 1 : 0.1);
    this._labelEls?.transition().duration(200)
      .attr('opacity', d => matchIds.has(d.id) ? 1 : 0.1);
    // Show edges only between matched nodes
    this._linkEls?.transition().duration(200)
      .attr('opacity', d => {
        const s = typeof d.source === 'object' ? d.source.id : d.source;
        const t = typeof d.target === 'object' ? d.target.id : d.target;
        return (matchIds.has(s) || matchIds.has(t)) ? 0.4 : 0.02;
      });
  }

  _esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
}

window.ClusterNetwork = ClusterNetwork;


/**
 * Term Co-occurrence Network
 * Nodes = keywords, edges = co-occurrence in same cluster.
 * Node color = dominant cluster, size = frequency/score.
 */
class TermCooccurrenceNetwork {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.data = null;
    this.jobId = null;
    this.labelThreshold = 0.3;
    this.edgeMinWeight = 0;
    this.colorMode = 'cluster'; // 'cluster' | 'score' | 'breadth'
  }

  async load(jobId) {
    this.jobId = jobId;
    this.container.innerHTML = '';
    const resp = await fetch(`/api/jobs/${jobId}/term-network`);
    this.data = await resp.json();
    if (this.data.error) {
      this.container.innerHTML = `<div style="padding:2rem;color:#64748b;text-align:center;">${this.data.error}</div>`;
      return;
    }
    this._render();
  }

  _render() {
    const rect = this.container.getBoundingClientRect();
    const W = rect.width || 700;
    const H = rect.height || 500;

    // Controls
    const ctrl = document.createElement('div');
    ctrl.className = 'net-controls';
    ctrl.innerHTML = `<div class="net-ctrl-row">
      <div class="net-ctrl-group"><span class="net-ctrl-label">Color</span>
        <button class="net-chip active" data-tc="cluster">Cluster</button>
        <button class="net-chip" data-tc="score">Score</button>
        <button class="net-chip" data-tc="breadth">Breadth</button>
      </div>
      <div class="net-ctrl-group"><span class="net-ctrl-label">Labels</span>
        <input type="range" min="0" max="100" value="${Math.round((1-this.labelThreshold)*100)}" id="tsl-labels" style="width:80px;">
      </div>
      <div class="net-ctrl-group"><span class="net-ctrl-label">Min edge</span>
        <input type="range" min="0" max="100" value="0" id="tsl-edge" style="width:80px;">
      </div>
      <div class="net-ctrl-group">
        <button class="net-chip" id="tbtn-png">PNG</button>
        <button class="net-chip" id="tbtn-svg">SVG</button>
      </div>
    </div>`;
    this.container.appendChild(ctrl);

    const svgH = H - 50;
    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', W);
    svgEl.setAttribute('height', svgH);
    svgEl.id = 'net-svg';
    this.container.appendChild(svgEl);

    const svg = d3.select(svgEl);
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.2, 8]).on('zoom', e => g.attr('transform', e.transform)));

    const defs = svg.append('defs');
    const glow = defs.append('filter').attr('id', 'tglow');
    glow.append('feGaussianBlur').attr('stdDeviation', '1.5').attr('result', 'b');
    const m = glow.append('feMerge'); m.append('feMergeNode').attr('in', 'b'); m.append('feMergeNode').attr('in', 'SourceGraphic');

    const nodes = this.data.nodes.map(d => ({...d}));
    const edges = this.data.edges.map(d => ({...d}));

    const maxFreq = Math.max(...nodes.map(n => n.frequency), 1);
    const maxScore = Math.max(...nodes.map(n => n.score), 0.01);
    const maxBreadth = Math.max(...nodes.map(n => n.n_clusters), 1);
    const rScale = d3.scaleSqrt().domain([0, maxScore]).range([3, 22]);
    const maxW = Math.max(...edges.map(e => e.weight), 1);
    const wScale = d3.scaleLinear().domain([0, maxW]).range([0.3, 3]);

    const self = this;

    const colorFn = (d) => {
      if (self.colorMode === 'cluster') return PALETTE[d.cluster % PALETTE.length];
      if (self.colorMode === 'score') return d3.interpolateYlOrRd(d.score / maxScore);
      if (self.colorMode === 'breadth') return d3.interpolatePurples(d.n_clusters / maxBreadth);
      return '#4a5976';
    };

    // Edges
    const linkEls = g.append('g').selectAll('line').data(edges).join('line')
      .attr('stroke', '#8899b3')
      .attr('stroke-width', d => wScale(d.weight))
      .attr('stroke-opacity', d => 0.1 + 0.3 * d.weight / maxW);
    this._linkEls = linkEls;

    // Nodes
    const nodeEls = g.append('g').selectAll('g').data(nodes).join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeEls.append('circle')
      .attr('r', d => rScale(d.score))
      .attr('fill', colorFn)
      .attr('stroke', '#fff').attr('stroke-width', 1)
      .attr('filter', 'url(#tglow)');
    this._nodeCircles = nodeEls.selectAll('circle');

    // Labels
    const labelEls = nodeEls.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', d => rScale(d.score) + 10)
      .attr('font-size', '8.5px')
      .attr('font-family', 'Source Sans 3, sans-serif')
      .attr('fill', '#2d3a52').attr('font-weight', '500')
      .attr('pointer-events', 'none');
    this._labelEls = labelEls;
    this._maxScore = maxScore;

    nodeEls.append('title').text(d =>
      `${d.label}\nScore: ${d.score.toFixed(3)}\nClusters: ${d.n_clusters}\nDominant: C${d.cluster}`
    );

    // Simulation
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(50).strength(d => 0.05 + 0.2 * d.norm))
      .force('charge', d3.forceManyBody().strength(d => -rScale(d.score) * 5))
      .force('center', d3.forceCenter(W / 2, svgH / 2))
      .force('collision', d3.forceCollide().radius(d => rScale(d.score) + 2))
      .on('tick', () => {
        linkEls.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        nodeEls.attr('transform', d => `translate(${d.x},${d.y})`);
      });

    this._updateLabels();

    // Bind controls
    ctrl.querySelectorAll('[data-tc]').forEach(b => b.onclick = () => {
      self.colorMode = b.dataset.tc;
      ctrl.querySelectorAll('[data-tc]').forEach(x => x.classList.toggle('active', x.dataset.tc === self.colorMode));
      self._nodeCircles.attr('fill', colorFn);
    });
    const slL = ctrl.querySelector('#tsl-labels');
    if (slL) slL.oninput = () => { self.labelThreshold = 1 - slL.value / 100; self._updateLabels(); };
    const slE = ctrl.querySelector('#tsl-edge');
    if (slE) slE.oninput = () => {
      const minW = (slE.value / 100) * maxW;
      linkEls.attr('display', d => d.weight >= minW ? null : 'none');
    };

    // Export
    ctrl.querySelector('#tbtn-svg')?.addEventListener('click', () => {
      const s = new XMLSerializer().serializeToString(svgEl);
      const blob = new Blob([s], {type: 'image/svg+xml'});
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = 'sciscape_terms.svg'; a.click();
    });
    ctrl.querySelector('#tbtn-png')?.addEventListener('click', () => {
      const s = new XMLSerializer().serializeToString(svgEl);
      const c = document.createElement('canvas'); c.width = W*2; c.height = svgH*2;
      const ctx = c.getContext('2d'); ctx.scale(2,2);
      const img = new Image();
      img.onload = () => { ctx.fillStyle='#f8f9fc'; ctx.fillRect(0,0,W,svgH); ctx.drawImage(img,0,0);
        c.toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'sciscape_terms.png'; a.click(); });
      };
      img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(s)));
    });
  }

  _updateLabels() {
    if (!this._labelEls) return;
    const t = this.labelThreshold;
    this._labelEls.attr('display', d => (d.score / this._maxScore) >= t ? null : 'none');
  }
}

window.TermCooccurrenceNetwork = TermCooccurrenceNetwork;
