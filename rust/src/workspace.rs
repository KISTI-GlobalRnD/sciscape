//! Pre-allocated workspace for Leiden hot paths.
//!
//! Avoids repeated allocation/deallocation of large working arrays
//! across iterations and recursion levels.

/// Reusable working arrays for Leiden algorithm.
/// Sized for the largest graph encountered (original n_nodes).
/// Smaller graphs (contracted) reuse a prefix of the same arrays.
pub struct Workspace {
    /// Max capacity (original n_nodes).
    pub capacity: usize,
    /// Edge weight per cluster (fast_local_move + contraction)
    pub ewpc: Vec<f64>,
    /// Cluster weights
    pub cw: Vec<f64>,
    /// Nodes per cluster count (u32 for cache efficiency)
    pub npc: Vec<u32>,
    /// Stable node flags for fast local move.
    ///
    /// Stored as bytes instead of `Vec<bool>` because this flag is read in hot
    /// neighbor loops; byte indexing is faster than bit-packed proxy access.
    pub stable: Vec<u8>,
    /// Current epoch value for `stable`.
    ///
    /// A node is stable when `stable[node] == stable_epoch`. Advancing the
    /// epoch avoids clearing the whole stable array on every local-move call.
    pub stable_epoch: u8,
    /// Neighbor cluster buffer
    pub nc_buf: Vec<u32>,
    /// Random node order for fast local move
    pub order: Vec<u32>,
    /// Empty cluster IDs available for reuse
    pub unused: Vec<u32>,
    /// Temporary weight array (contraction scatter)
    pub temp_w: Vec<f64>,
    /// Temporary seen marker (contraction scatter)
    ///
    /// Kept initialized to `u32::MAX`; contraction marks touched clusters and
    /// restores only those entries after each source cluster.
    pub temp_seen: Vec<u32>,
    /// Temporary used-indices (contraction scatter)
    pub temp_used: Vec<u32>,
    /// Flat node-per-cluster storage
    pub npc_nodes: Vec<u32>,
    /// Node-per-cluster prefix sums
    pub npc_starts: Vec<u32>,
    /// Node-per-cluster offsets (temporary)
    pub npc_off: Vec<u32>,
    /// Reusable global-to-local marker for streaming refinement subgraphs.
    ///
    /// Kept initialized to `u32::MAX`; subgraph builders mark only touched
    /// nodes and unmark them before returning.
    pub local_index: Vec<u32>,
}

impl Workspace {
    /// Create workspace for a graph with `n` nodes.
    pub fn new(n: usize) -> Self {
        Workspace {
            capacity: n,
            ewpc: vec![0.0; n],
            cw: vec![0.0; n],
            npc: vec![0; n],
            stable: vec![0; n],
            stable_epoch: 0,
            nc_buf: Vec::with_capacity(256),
            order: Vec::with_capacity(n),
            unused: Vec::with_capacity(n.min(1024)),
            temp_w: vec![0.0; n],
            temp_seen: vec![u32::MAX; n],
            temp_used: Vec::with_capacity(256),
            npc_nodes: vec![0; n],
            npc_starts: vec![0; n + 1],
            npc_off: vec![0; n],
            local_index: vec![u32::MAX; n],
        }
    }

    /// Ensure workspace is large enough for `n` nodes.
    /// Only grows, never shrinks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if n > self.capacity {
            self.capacity = n;
            self.ewpc.resize(n, 0.0);
            self.cw.resize(n, 0.0);
            self.npc.resize(n, 0);
            self.stable.resize(n, 0);
            if self.order.capacity() < n {
                self.order.reserve(n.saturating_sub(self.order.len()));
            }
            self.temp_w.resize(n, 0.0);
            self.temp_seen.resize(n, u32::MAX);
            self.npc_nodes.resize(n, 0);
            self.npc_starts.resize(n + 1, 0);
            self.npc_off.resize(n, 0);
            self.local_index.resize(n, u32::MAX);
        }
    }

    /// Advance and return the marker used for stable nodes.
    ///
    /// The `u8` epoch gives 255 clear-free calls before wrapping. On wrap,
    /// clear the active prefix once and restart from epoch 1.
    #[inline]
    pub fn next_stable_epoch(&mut self, n: usize) -> u8 {
        let next = self.stable_epoch.wrapping_add(1);
        if next == 0 {
            self.stable[..n].fill(0);
            self.stable_epoch = 1;
        } else {
            self.stable_epoch = next;
        }
        self.stable_epoch
    }

    /// Zero out arrays for `n` elements (fast memset).
    /// Call before each use to reset state.
    #[inline]
    pub fn reset(&mut self, n: usize) {
        // Only zero the prefix we'll actually use
        self.ewpc[..n].fill(0.0);
        self.cw[..n].fill(0.0);
        self.npc[..n].fill(0);
        self.stable[..n].fill(0);
        self.stable_epoch = 0;
        self.nc_buf.clear();
        self.order.clear();
        self.unused.clear();
        self.temp_w[..n].fill(0.0);
        self.temp_used.clear();
    }
}
