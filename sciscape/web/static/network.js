/**
 * SciScape Cluster Network Visualization
 * D3.js force-directed layout with layer toggling and hierarchy levels.
 *
 * VOSviewer-inspired: nodes = clusters (sized by paper count),
 * edges = inter-cluster connections (width by weight),
 * colors = cluster identity.
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
    this.currentLevel = null;
    this.activeLayers = new Set(['combined']);
    this.simulation = null;
    this.svg = null;
    this.width = 0;
    this.height = 0;
    this.showHierarchy = false;
  }

  async load(jobId) {
    const resp = await fetch(`/api/jobs/${jobId}/network`);
    this.data = await resp.json();
    if (this.data.error) {
      this.container.innerHTML = `<div style="padding:2rem;color:#64748b;text-align:center;">${this.data.error}</div>`;
      return;
    }
    this.currentLevel = this.data.levels?.[0] || 'default';
    this.render();
  }

  render() {
    this.container.innerHTML = '';
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width || 700;
    this.height = rect.height || 500;

    // Controls bar
    const controls = document.createElement('div');
    controls.className = 'net-controls';
    controls.innerHTML = this._controlsHTML();
    this.container.appendChild(controls);

    // SVG
    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', this.width);
    svgEl.setAttribute('height', this.height - 40);
    svgEl.style.display = 'block';
    this.container.appendChild(svgEl);

    this.svg = d3.select(svgEl);
    this._setupDefs();
    this._draw();
    this._bindControls(controls);
  }

  _controlsHTML() {
    const levels = this.data.levels || [];
    const layers = this.data.layers || ['combined'];

    let html = '<div class="net-ctrl-row">';

    // Level selector
    if (levels.length > 1) {
      html += '<div class="net-ctrl-group"><span class="net-ctrl-label">Level</span>';
      levels.forEach(l => {
        const active = l === this.currentLevel ? ' active' : '';
        html += `<button class="net-chip${active}" data-level="${l}">${l}</button>`;
      });
      html += '</div>';
    }

    // Layer toggles
    html += '<div class="net-ctrl-group"><span class="net-ctrl-label">Layers</span>';
    layers.forEach(l => {
      const active = this.activeLayers.has(l) ? ' active' : '';
      html += `<button class="net-chip layer-chip${active}" data-layer="${l}">${l.toUpperCase()}</button>`;
    });
    html += '</div>';

    // Hierarchy toggle
    if (this.data.hierarchy && this.data.hierarchy.length > 0) {
      const hActive = this.showHierarchy ? ' active' : '';
      html += `<div class="net-ctrl-group"><button class="net-chip${hActive}" id="btn-hierarchy">Hierarchy</button></div>`;
    }

    html += '</div>';
    return html;
  }

  _bindControls(controls) {
    // Level buttons
    controls.querySelectorAll('[data-level]').forEach(btn => {
      btn.addEventListener('click', () => {
        this.currentLevel = btn.dataset.level;
        this.render();
      });
    });

    // Layer buttons
    controls.querySelectorAll('[data-layer]').forEach(btn => {
      btn.addEventListener('click', () => {
        const layer = btn.dataset.layer;
        if (this.activeLayers.has(layer)) {
          if (this.activeLayers.size > 1) this.activeLayers.delete(layer);
        } else {
          this.activeLayers.add(layer);
        }
        this.render();
      });
    });

    // Hierarchy
    const hBtn = controls.querySelector('#btn-hierarchy');
    if (hBtn) {
      hBtn.addEventListener('click', () => {
        this.showHierarchy = !this.showHierarchy;
        this.render();
      });
    }
  }

  _setupDefs() {
    const defs = this.svg.append('defs');
    // Glow filter
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'blur');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'blur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');
  }

  _draw() {
    const g = this.svg.append('g').attr('class', 'network-root');
    const h = this.height - 40;

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.3, 5])
      .on('zoom', (event) => g.attr('transform', event.transform));
    this.svg.call(zoom);

    // Get data for current level
    const levelNodes = this.data.nodes?.find(n => n.level === this.currentLevel);
    const levelEdges = this.data.edges?.find(e => e.level === this.currentLevel);
    if (!levelNodes || !levelEdges) return;

    const nodes = levelNodes.data.map(d => ({...d}));
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    // Merge active layer edges
    const edgeMap = new Map();
    for (const layer of this.activeLayers) {
      const layerEdges = levelEdges.data[layer] || [];
      for (const e of layerEdges) {
        const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
        if (edgeMap.has(key)) {
          const existing = edgeMap.get(key);
          existing.weight += e.weight;
          existing.layers.push(layer);
        } else {
          edgeMap.set(key, {...e, layers: [layer]});
        }
      }
    }
    const edges = [...edgeMap.values()].map(e => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
      norm: e.norm,
      layers: e.layers,
    }));

    // Scale
    const maxSize = Math.max(...nodes.map(n => n.size), 1);
    const rScale = d3.scaleSqrt().domain([0, maxSize]).range([4, 35]);
    const maxW = Math.max(...edges.map(e => e.weight), 1);
    const wScale = d3.scaleLinear().domain([0, maxW]).range([0.5, 4]);

    // Layer color map
    const layerColors = {combined: '#4a5976', dc: '#2563eb', bc: '#dc2626', cc: '#059669'};

    // Edges
    const linkG = g.append('g').attr('class', 'links');
    const link = linkG.selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', d => {
        if (d.layers.length === 1) return layerColors[d.layers[0]] || '#4a5976';
        return '#4a5976';
      })
      .attr('stroke-width', d => wScale(d.weight))
      .attr('stroke-opacity', d => 0.15 + 0.35 * (d.weight / maxW));

    // Nodes
    const nodeG = g.append('g').attr('class', 'nodes');
    const node = nodeG.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'grab')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
      );

    node.append('circle')
      .attr('r', d => rScale(d.size))
      .attr('fill', d => PALETTE[d.id % PALETTE.length])
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('filter', 'url(#glow)');

    // Labels (only for larger clusters)
    const labelThreshold = maxSize * 0.03;
    node.filter(d => d.size >= labelThreshold)
      .append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', d => rScale(d.size) + 12)
      .attr('font-size', '10px')
      .attr('font-family', 'Source Sans 3, sans-serif')
      .attr('fill', '#4a5976')
      .attr('font-weight', '500');

    // Tooltip
    node.append('title')
      .text(d => `${d.label}\nSize: ${d.size} (${d.pct}%)`);

    // Hierarchy links
    if (this.showHierarchy && this.data.hierarchy) {
      const hierLinks = this.data.hierarchy.filter(
        h => h.child_level === this.currentLevel
      );
      // Draw dashed lines to parent level clusters
      const hierG = g.append('g').attr('class', 'hierarchy');
      // For now, just highlight parent relationships with node border
      nodes.forEach(n => {
        const parentLink = hierLinks.find(h => h.child === n.id);
        if (parentLink) {
          n._parent = parentLink.parent;
          n._parentLevel = parentLink.parent_level;
        }
      });
    }

    // Force simulation
    this.simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(80).strength(d => 0.1 + 0.3 * d.norm))
      .force('charge', d3.forceManyBody().strength(d => -rScale(d.size) * 8))
      .force('center', d3.forceCenter(this.width / 2, h / 2))
      .force('collision', d3.forceCollide().radius(d => rScale(d.size) + 3))
      .on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });
  }
}

// Export
window.ClusterNetwork = ClusterNetwork;
