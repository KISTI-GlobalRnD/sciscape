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
    /// Stable node flags
    pub stable: Vec<bool>,
    /// Reusable node processing order buffer
    pub order_u32: Vec<u32>,
    /// Reusable empty-cluster stack
    pub unused_u32: Vec<u32>,
    /// Neighbor cluster buffer
    pub nc_buf: Vec<u32>,
    /// Temporary weight array (contraction scatter)
    pub temp_w: Vec<f64>,
    /// Temporary used-indices (contraction scatter)
    pub temp_used: Vec<u32>,
    /// Flat node-per-cluster storage
    pub npc_nodes: Vec<u32>,
    /// Node-per-cluster prefix sums
    pub npc_starts: Vec<u32>,
    /// Node-per-cluster offsets (temporary)
    pub npc_off: Vec<u32>,
}

impl Workspace {
    /// Create workspace for a graph with `n` nodes.
    pub fn new(n: usize) -> Self {
        Workspace {
            capacity: n,
            ewpc: vec![0.0; n],
            cw: vec![0.0; n],
            npc: vec![0; n],
            stable: vec![false; n],
            order_u32: Vec::with_capacity(n),
            unused_u32: Vec::with_capacity(n.min(1024)),
            nc_buf: Vec::with_capacity(256),
            temp_w: vec![0.0; n],
            temp_used: Vec::with_capacity(256),
            npc_nodes: vec![0; n],
            npc_starts: vec![0; n + 1],
            npc_off: vec![0; n],
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
            self.stable.resize(n, false);
            self.order_u32
                .reserve(n.saturating_sub(self.order_u32.capacity()));
            self.unused_u32
                .reserve(n.saturating_sub(self.unused_u32.capacity()));
            self.temp_w.resize(n, 0.0);
            self.npc_nodes.resize(n, 0);
            self.npc_starts.resize(n + 1, 0);
            self.npc_off.resize(n, 0);
        }
    }

    /// Zero out arrays for `n` elements (fast memset).
    /// Call before each use to reset state.
    #[inline]
    pub fn reset(&mut self, n: usize) {
        // Only zero the prefix we'll actually use
        self.ewpc[..n].fill(0.0);
        self.cw[..n].fill(0.0);
        self.npc[..n].fill(0);
        self.stable[..n].fill(false);
        self.order_u32.clear();
        self.unused_u32.clear();
        self.nc_buf.clear();
        self.temp_w[..n].fill(0.0);
        self.temp_used.clear();
    }
}
