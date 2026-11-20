//! 3D Bitmask Wave Function Collapse (`20 x 20 x 20`) with AC-3 propagation,
//! Barnes Priority-Flood elevated mountain ponds, and directional-leaf Flora WFC.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::rng::{hash_seed, Prng, TerrainField};

/// Side length of the cubic voxel diorama (`20 x 20 x 20 = 8_000` cells).
pub const GRID_SIZE: usize = 20;
/// Total number of voxel cells in the diorama.
pub const CELL_COUNT: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE;
/// Number of internal WFC tile states (fits in a single `u16` bitmask).
pub const TILE_COUNT: usize = 15;

/// High-level user-facing voxel category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelCategory {
    Air,
    Water,
    Land,
    Foliage,
}

/// 6 cardinal 3D adjacency directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Dir {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Dir {
    /// All 6 cardinal directions in enum order.
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    /// The 4 horizontal cardinal directions.
    pub const HORIZONTAL: [Self; 4] = [Self::PosX, Self::NegX, Self::PosZ, Self::NegZ];

    /// Returns the opposite cardinal direction.
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::PosX => Self::NegX,
            Self::NegX => Self::PosX,
            Self::PosY => Self::NegY,
            Self::NegY => Self::PosY,
            Self::PosZ => Self::NegZ,
            Self::NegZ => Self::PosZ,
        }
    }

    /// Returns the signed `(dx, dy, dz)` unit step for this direction.
    #[inline]
    pub const fn offset(self) -> (isize, isize, isize) {
        match self {
            Self::PosX => (1, 0, 0),
            Self::NegX => (-1, 0, 0),
            Self::PosY => (0, 1, 0),
            Self::NegY => (0, -1, 0),
            Self::PosZ => (0, 0, 1),
            Self::NegZ => (0, 0, -1),
        }
    }
}

/// Internal WFC tile states (`0..15`), each represented by one bit of a `u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tile {
    Air = 0,
    Water = 1,
    DeepRock = 2,
    Subsoil = 3,
    Grass = 4,
    CliffRock = 5,
    Bush = 6,
    TrunkBase = 7,
    TrunkMid = 8,
    CanopyCore = 9,
    LeafPosX = 10,
    LeafNegX = 11,
    LeafPosZ = 12,
    LeafNegZ = 13,
    LeafTop = 14,
}

impl Tile {
    /// All 15 tile states in bit-index order.
    pub const ALL: [Self; TILE_COUNT] = [
        Self::Air,
        Self::Water,
        Self::DeepRock,
        Self::Subsoil,
        Self::Grass,
        Self::CliffRock,
        Self::Bush,
        Self::TrunkBase,
        Self::TrunkMid,
        Self::CanopyCore,
        Self::LeafPosX,
        Self::LeafNegX,
        Self::LeafPosZ,
        Self::LeafNegZ,
        Self::LeafTop,
    ];

    /// Directional leaf states that bind strictly to [`Tile::CanopyCore`].
    pub const LEAVES: [Self; 5] = [
        Self::LeafPosX,
        Self::LeafNegX,
        Self::LeafPosZ,
        Self::LeafNegZ,
        Self::LeafTop,
    ];

    /// Returns the single-bit `u16` mask for this tile.
    #[inline]
    pub const fn bit(self) -> u16 {
        1u16 << (self as u8)
    }

    /// Converts a bit index (`0..15`) back to a [`Tile`].
    #[inline]
    pub const fn from_index(index: usize) -> Self {
        Self::ALL[index]
    }

    /// Maps this internal state to its high-level user category.
    #[inline]
    pub const fn category(self) -> VoxelCategory {
        match self {
            Self::Air => VoxelCategory::Air,
            Self::Water => VoxelCategory::Water,
            Self::DeepRock | Self::Subsoil | Self::Grass | Self::CliffRock => VoxelCategory::Land,
            Self::Bush
            | Self::TrunkBase
            | Self::TrunkMid
            | Self::CanopyCore
            | Self::LeafPosX
            | Self::LeafNegX
            | Self::LeafPosZ
            | Self::LeafNegZ
            | Self::LeafTop => VoxelCategory::Foliage,
        }
    }

    /// Returns `true` for solid terrain tiles (`DeepRock`, `Subsoil`, `Grass`, `CliffRock`).
    #[inline]
    pub const fn is_land(self) -> bool {
        matches!(self.category(), VoxelCategory::Land)
    }

    /// Returns `true` for opaque geometry (all `Land` and `Foliage` tiles).
    #[inline]
    pub const fn is_opaque(self) -> bool {
        matches!(self.category(), VoxelCategory::Land | VoxelCategory::Foliage)
    }

    /// For directional leaf tiles, returns the direction pointing back to [`Tile::CanopyCore`].
    #[inline]
    pub const fn leaf_core_dir(self) -> Option<Dir> {
        match self {
            Self::LeafPosX => Some(Dir::NegX),
            Self::LeafNegX => Some(Dir::PosX),
            Self::LeafPosZ => Some(Dir::NegZ),
            Self::LeafNegZ => Some(Dir::PosZ),
            Self::LeafTop => Some(Dir::NegY),
            _ => None,
        }
    }
}

/// Bitmask containing all 6 Pass 1 (Geology & Hydrology) tiles.
const PASS1_MASK: u16 = Tile::Air.bit()
    | Tile::Water.bit()
    | Tile::DeepRock.bit()
    | Tile::Subsoil.bit()
    | Tile::Grass.bit()
    | Tile::CliffRock.bit();

/// Symmetric 6-direction WFC adjacency lookup table (`allowed[tile][dir]` is a `u16` bitmask).
#[derive(Debug, Clone)]
pub struct AdjacencyTable {
    allowed: [[u16; 6]; TILE_COUNT],
}

impl AdjacencyTable {
    /// Builds the mathematically symmetric 15-tile 6-neighbor adjacency table.
    pub fn build() -> Self {
        let mut table = Self {
            allowed: [[0u16; 6]; TILE_COUNT],
        };

        // --- Vertical pairs (below, above) ---
        // DeepRock base layer
        table.allow_vertical(Tile::DeepRock, Tile::DeepRock);
        table.allow_vertical(Tile::DeepRock, Tile::Subsoil);
        table.allow_vertical(Tile::DeepRock, Tile::CliffRock);
        table.allow_vertical(Tile::DeepRock, Tile::Water);

        // Subsoil earth layer
        table.allow_vertical(Tile::Subsoil, Tile::Subsoil);
        table.allow_vertical(Tile::Subsoil, Tile::Grass);
        table.allow_vertical(Tile::Subsoil, Tile::CliffRock);
        table.allow_vertical(Tile::Subsoil, Tile::Water);

        // CliffRock steep drops & alpine peaks (forbids Bush & TrunkBase above)
        table.allow_vertical(Tile::CliffRock, Tile::CliffRock);
        table.allow_vertical(Tile::CliffRock, Tile::Water);
        table.allow_vertical(Tile::CliffRock, Tile::Air);

        // Arable Grass (forbids Water and all solid Land above; hosts Bush & TrunkBase)
        table.allow_vertical(Tile::Grass, Tile::Air);
        table.allow_vertical(Tile::Grass, Tile::Bush);
        table.allow_vertical(Tile::Grass, Tile::TrunkBase);

        // Water (forbids all Land above; Air sits above surface)
        table.allow_vertical(Tile::Water, Tile::Water);
        table.allow_vertical(Tile::Water, Tile::Air);

        // Air (forbids Water and all Land above -> zero floating land or water)
        table.allow_vertical(Tile::Air, Tile::Air);

        // Side canopy leaves may overhang Air, Grass, CliffRock, or Water below, with Air above
        let side_leaves = [
            Tile::LeafPosX,
            Tile::LeafNegX,
            Tile::LeafPosZ,
            Tile::LeafNegZ,
        ];
        for leaf in side_leaves {
            table.allow_vertical(Tile::Air, leaf);
            table.allow_vertical(Tile::Grass, leaf);
            table.allow_vertical(Tile::CliffRock, leaf);
            table.allow_vertical(Tile::Water, leaf);
            table.allow_vertical(leaf, Tile::Air);
        }

        // Bush strictly sits on Grass with Air above
        table.allow_vertical(Tile::Bush, Tile::Air);

        // Trees: TrunkBase -> (optional TrunkMid) -> CanopyCore -> LeafTop -> Air
        table.allow_vertical(Tile::TrunkBase, Tile::TrunkMid);
        table.allow_vertical(Tile::TrunkBase, Tile::CanopyCore);
        table.allow_vertical(Tile::TrunkMid, Tile::CanopyCore);
        table.allow_vertical(Tile::CanopyCore, Tile::LeafTop);
        table.allow_vertical(Tile::LeafTop, Tile::Air);

        // --- Horizontal pairs (undirected across +/-X, +/-Z) ---
        // DeepRock and Subsoil forbid Air horizontally -> steep exposed drops force CliffRock!
        table.allow_horizontal(Tile::DeepRock, Tile::DeepRock);
        table.allow_horizontal(Tile::DeepRock, Tile::Subsoil);
        table.allow_horizontal(Tile::DeepRock, Tile::CliffRock);
        table.allow_horizontal(Tile::DeepRock, Tile::Grass);
        table.allow_horizontal(Tile::DeepRock, Tile::Water);

        table.allow_horizontal(Tile::Subsoil, Tile::Subsoil);
        table.allow_horizontal(Tile::Subsoil, Tile::CliffRock);
        table.allow_horizontal(Tile::Subsoil, Tile::Grass);
        table.allow_horizontal(Tile::Subsoil, Tile::Water);

        table.allow_horizontal(Tile::CliffRock, Tile::CliffRock);
        table.allow_horizontal(Tile::CliffRock, Tile::Grass);
        table.allow_horizontal(Tile::CliffRock, Tile::Water);
        table.allow_horizontal(Tile::CliffRock, Tile::Air);

        table.allow_horizontal(Tile::Grass, Tile::Grass);
        table.allow_horizontal(Tile::Grass, Tile::Water);
        table.allow_horizontal(Tile::Grass, Tile::Air);

        // Water never touches Air horizontally
        table.allow_horizontal(Tile::Water, Tile::Water);
        table.allow_horizontal(Tile::Air, Tile::Air);

        // Bush horizontal adjacency (never touches Bush or TrunkBase to stay cleanly spaced)
        table.allow_horizontal(Tile::Bush, Tile::Air);
        table.allow_horizontal(Tile::Bush, Tile::Grass);
        table.allow_horizontal(Tile::Bush, Tile::CliffRock);

        // TrunkBase forbids horizontal adjacency to CliffRock, Bush, or another TrunkBase
        table.allow_horizontal(Tile::TrunkBase, Tile::Air);
        table.allow_horizontal(Tile::TrunkBase, Tile::Grass);

        // TrunkMid is strictly surrounded horizontally by Air
        table.allow_horizontal(Tile::TrunkMid, Tile::Air);

        // --- Directed CanopyCore <-> Leaf socket locks ---
        table.allow_directed(Tile::CanopyCore, Dir::PosX, Tile::LeafPosX);
        table.allow_directed(Tile::CanopyCore, Dir::NegX, Tile::LeafNegX);
        table.allow_directed(Tile::CanopyCore, Dir::PosZ, Tile::LeafPosZ);
        table.allow_directed(Tile::CanopyCore, Dir::NegZ, Tile::LeafNegZ);

        // Outer horizontal faces of directional leaves
        for &leaf in &Tile::LEAVES {
            let core_dir = leaf.leaf_core_dir();
            for dir in Dir::HORIZONTAL {
                if Some(dir) == core_dir {
                    continue;
                }
                table.allow_directed(leaf, dir, Tile::Air);
                table.allow_directed(leaf, dir, Tile::Grass);
                table.allow_directed(leaf, dir, Tile::CliffRock);
                table.allow_directed(leaf, dir, Tile::Bush);

                // Neighboring tree outer leaves may touch as long as neither face is a core socket.
                for &other_leaf in &Tile::LEAVES {
                    if Some(dir.opposite()) != other_leaf.leaf_core_dir() {
                        table.allow_directed(leaf, dir, other_leaf);
                    }
                }
            }
        }

        table
    }

    fn allow_directed(&mut self, a: Tile, dir: Dir, b: Tile) {
        self.allowed[a as usize][dir as usize] |= b.bit();
        self.allowed[b as usize][dir.opposite() as usize] |= a.bit();
    }

    fn allow_vertical(&mut self, below: Tile, above: Tile) {
        self.allow_directed(below, Dir::PosY, above);
    }

    fn allow_horizontal(&mut self, a: Tile, b: Tile) {
        for dir in Dir::HORIZONTAL {
            self.allow_directed(a, dir, b);
        }
    }

    /// Returns the bitmask of tiles allowed in direction `dir` from any tile present in `mask`.
    #[inline]
    pub fn support_mask(&self, mut mask: u16, dir: Dir) -> u16 {
        let mut union = 0u16;
        let dir_idx = dir as usize;
        while mask != 0 {
            let tz = mask.trailing_zeros() as usize;
            union |= self.allowed[tz][dir_idx];
            mask &= mask - 1;
        }
        union
    }

    /// Returns the raw allowed bitmask for single tile `tile` in direction `dir`.
    #[cfg(test)]
    #[inline]
    pub fn allowed_for(&self, tile: Tile, dir: Dir) -> u16 {
        self.allowed[tile as usize][dir as usize]
    }
}

/// Converts 3D grid coordinates `(x, y, z)` into a linear cell index.
#[inline]
pub const fn cell_index(x: usize, y: usize, z: usize) -> usize {
    (y * GRID_SIZE + z) * GRID_SIZE + x
}

/// Converts a linear cell index back into `(x, y, z)`.
#[inline]
pub const fn index_coords(idx: usize) -> (usize, usize, usize) {
    let x = idx % GRID_SIZE;
    let rem = idx / GRID_SIZE;
    let z = rem % GRID_SIZE;
    let y = rem / GRID_SIZE;
    (x, y, z)
}

/// Returns the linear neighbor index in direction `dir`, or `None` if outside `0..GRID_SIZE`.
#[inline]
pub fn neighbor_index(idx: usize, dir: Dir) -> Option<usize> {
    let (x, y, z) = index_coords(idx);
    let (dx, dy, dz) = dir.offset();
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    let nz = z as isize + dz;
    if (0..GRID_SIZE as isize).contains(&nx)
        && (0..GRID_SIZE as isize).contains(&ny)
        && (0..GRID_SIZE as isize).contains(&nz)
    {
        Some(cell_index(nx as usize, ny as usize, nz as usize))
    } else {
        None
    }
}

/// Execution phase of the multi-pass WFC diorama generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WfcPhase {
    Pass1Geology,
    Pass2Ponds,
    Pass3Flora,
    Complete,
}

impl WfcPhase {
    /// Human-readable phase label for HUD output.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pass1Geology => "Pass 1: Geology",
            Self::Pass2Ponds => "Pass 2: Alpine Ponds",
            Self::Pass3Flora => "Pass 3: Flora",
            Self::Complete => "Complete",
        }
    }
}

/// Snapshot used to recover cleanly from deep WFC contradictions.
#[derive(Clone)]
struct Checkpoint {
    cells: Vec<u16>,
    rng: Prng,
    collapsed_count: usize,
}

/// Summary statistics for the HUD and diagnostics.
#[derive(Debug, Clone, Copy, Default)]
pub struct DioramaStats {
    pub land_voxels: usize,
    pub water_voxels: usize,
    pub pond_count: usize,
    pub pond_voxels: usize,
    pub tree_count: usize,
    pub bush_count: usize,
}

/// Complete 3D WFC diorama solver supporting incremental animation and instant solves.
pub struct WfcSolver {
    seed_label: String,
    seed: u64,
    table: AdjacencyTable,
    field: TerrainField,
    rng: Prng,
    cells: Vec<u16>,
    pass1_entropy: Vec<f64>,
    cell_tiebreak: Vec<f64>,
    phase: WfcPhase,
    pass1_initial_collapsed: usize,
    pass1_collapsed: usize,
    pass1_total: usize,
    last_checkpoint_collapsed: usize,
    pass3_collapsed: usize,
    pass3_total: usize,
    pass3_retries: usize,
    pass_retries: usize,
    checkpoints: Vec<Checkpoint>,
    post_pass2_cells: Vec<u16>,
    pond_queue: VecDeque<(usize, usize, usize)>,
    pond_total: usize,
    pond_filled: usize,
    pond_count: usize,
    queue: VecDeque<usize>,
    in_queue: Vec<bool>,
    undo_trail: Vec<(usize, u16)>,
}

impl WfcSolver {
    /// Creates a new solver initialized for Pass 1 from a seed string.
    pub fn new(seed_str: &str) -> Self {
        let cleaned = seed_str.trim();
        let seed_label = if cleaned.is_empty() {
            "default".to_owned()
        } else {
            cleaned.to_owned()
        };
        let seed = hash_seed(&seed_label);
        let table = AdjacencyTable::build();
        let field = TerrainField::from_seed(seed);
        let rng = Prng::from_seed(seed ^ 0xa3c5_910b_47d2_6f81);

        let mut solver = Self {
            seed_label,
            seed,
            table,
            field,
            rng,
            cells: vec![PASS1_MASK; CELL_COUNT],
            pass1_entropy: vec![0.0; CELL_COUNT],
            cell_tiebreak: vec![0.0; CELL_COUNT],
            phase: WfcPhase::Pass1Geology,
            pass1_initial_collapsed: 0,
            pass1_collapsed: 0,
            pass1_total: CELL_COUNT,
            last_checkpoint_collapsed: 0,
            pass3_collapsed: 0,
            pass3_total: 0,
            pass3_retries: 0,
            pass_retries: 0,
            checkpoints: Vec::new(),
            post_pass2_cells: Vec::new(),
            pond_queue: VecDeque::new(),
            pond_total: 0,
            pond_filled: 0,
            pond_count: 0,
            queue: VecDeque::with_capacity(512),
            in_queue: vec![false; CELL_COUNT],
            undo_trail: Vec::with_capacity(256),
        };
        solver.init_pass1(0);
        solver
    }

    /// Returns the active seed label.
    pub fn seed_label(&self) -> &str {
        &self.seed_label
    }

    /// Returns the numeric 64-bit seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the deterministic sea level height (`4..=6`).
    pub const fn sea_level(&self) -> usize {
        self.field.sea_level
    }

    /// Returns the current solver phase.
    pub const fn phase(&self) -> WfcPhase {
        self.phase
    }

    /// Returns `true` when all three passes have finished.
    pub const fn is_complete(&self) -> bool {
        matches!(self.phase, WfcPhase::Complete)
    }

    /// Returns the collapsed [`Tile`] at `(x, y, z)`, or `None` if still uncollapsed.
    #[inline]
    pub fn tile_at(&self, x: usize, y: usize, z: usize) -> Option<Tile> {
        let mask = self.cells[cell_index(x, y, z)];
        if mask != 0 && (mask & (mask - 1)) == 0 {
            Some(Tile::from_index(mask.trailing_zeros() as usize))
        } else {
            None
        }
    }

    /// Returns the raw `u16` domain bitmask at `(x, y, z)`.
    #[cfg(test)]
    #[inline]
    pub fn mask_at(&self, x: usize, y: usize, z: usize) -> u16 {
        self.cells[cell_index(x, y, z)]
    }

    /// Returns overall progress percentage in `[0.0, 100.0]`.
    pub fn progress_percent(&self) -> f32 {
        match self.phase {
            WfcPhase::Pass1Geology => {
                let done = self
                    .pass1_collapsed
                    .saturating_sub(self.pass1_initial_collapsed);
                let total = self
                    .pass1_total
                    .saturating_sub(self.pass1_initial_collapsed)
                    .max(1);
                (done as f32 / total as f32) * 70.0
            }
            WfcPhase::Pass2Ponds => {
                let p = if self.pond_total == 0 {
                    1.0
                } else {
                    self.pond_filled as f32 / self.pond_total as f32
                };
                70.0 + p * 10.0
            }
            WfcPhase::Pass3Flora => {
                let p = if self.pass3_total == 0 {
                    1.0
                } else {
                    self.pass3_collapsed as f32 / self.pass3_total as f32
                };
                80.0 + p * 20.0
            }
            WfcPhase::Complete => 100.0,
        }
    }

    /// Computes live voxel and feature counts across collapsed cells.
    pub fn stats(&self) -> DioramaStats {
        let mut stats = DioramaStats {
            pond_count: self.pond_count,
            pond_voxels: self.pond_filled,
            ..DioramaStats::default()
        };
        for &mask in &self.cells {
            if mask == 0 || (mask & (mask - 1)) != 0 {
                continue;
            }
            let tile = Tile::from_index(mask.trailing_zeros() as usize);
            match tile {
                Tile::DeepRock | Tile::Subsoil | Tile::Grass | Tile::CliffRock => {
                    stats.land_voxels += 1;
                }
                Tile::Water => stats.water_voxels += 1,
                Tile::TrunkBase => stats.tree_count += 1,
                Tile::Bush => stats.bush_count += 1,
                _ => {}
            }
        }
        stats
    }

    /// Runs the solver to completion immediately.
    pub fn solve_all(&mut self) {
        while !self.is_complete() {
            self.step(512);
        }
    }

    /// Advances the active pass by up to `budget` atomic collapses / water fills.
    ///
    /// Returns `true` if any cell changed (signaling the mesher to rebuild).
    pub fn step(&mut self, budget: usize) -> bool {
        if self.is_complete() || budget == 0 {
            return false;
        }
        let mut changed = false;
        match self.phase {
            WfcPhase::Pass1Geology => {
                for _ in 0..budget {
                    if self.phase != WfcPhase::Pass1Geology {
                        break;
                    }
                    if self.step_pass1_once() {
                        changed = true;
                    }
                }
            }
            WfcPhase::Pass2Ponds => {
                for _ in 0..budget {
                    if let Some((x, y, z)) = self.pond_queue.pop_front() {
                        self.cells[cell_index(x, y, z)] = Tile::Water.bit();
                        if y > 0 && self.tile_at(x, y - 1, z) == Some(Tile::Grass) {
                            self.cells[cell_index(x, y - 1, z)] = Tile::Subsoil.bit();
                        }
                        self.pond_filled += 1;
                        changed = true;
                    } else {
                        self.init_pass3();
                        changed = true;
                        break;
                    }
                }
            }
            WfcPhase::Pass3Flora => {
                for _ in 0..budget {
                    if self.phase != WfcPhase::Pass3Flora {
                        break;
                    }
                    if self.step_pass3_once() {
                        changed = true;
                    }
                }
                if self.phase == WfcPhase::Pass3Flora {
                    let remaining = self
                        .cells
                        .iter()
                        .filter(|&&m| m != 0 && (m & (m - 1)) != 0)
                        .count();
                    self.pass3_collapsed = self.pass3_total.saturating_sub(remaining);
                }
            }
            WfcPhase::Complete => {}
        }
        changed
    }

    /// Initializes Pass 1 domains and runs initial AC-3 propagation.
    fn init_pass1(&mut self, attempt: usize) {
        self.phase = WfcPhase::Pass1Geology;
        self.pass_retries = attempt;
        self.checkpoints.clear();
        self.rng = Prng::from_seed(
            self.seed
                ^ 0xa3c5_910b_47d2_6f81
                ^ (attempt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );

        let y_sea = self.field.sea_level;
        for idx in 0..CELL_COUNT {
            let (x, y, z) = index_coords(idx);
            let target_h = self.field.height_at(x, z);
            let mut mask = PASS1_MASK;
            if y == 0 {
                mask = Tile::DeepRock.bit();
            } else if y == GRID_SIZE - 1 {
                mask = Tile::Air.bit();
            } else {
                if y <= y_sea {
                    mask &= !Tile::Air.bit();
                } else {
                    mask &= !Tile::Water.bit();
                }
                if y >= 14 {
                    mask &= !Tile::Grass.bit();
                }
                // Upper-sky envelope: prevent high open cells from rolling CliffRock
                // and cascading into 1x1 vertical spires via AC-3 gravity constraints.
                if (y as f32) > target_h + 1.5 {
                    mask = if y <= y_sea {
                        Tile::Water.bit()
                    } else {
                        Tile::Air.bit()
                    };
                }
            }
            self.cells[idx] = mask;
        }

        // Seed AC-3 from all pre-constrained cells.
        self.clear_queue();
        for idx in 0..CELL_COUNT {
            self.enqueue(idx);
        }
        let ok = self.propagate_committed();
        debug_assert!(ok, "Pass 1 initial domain constraints must be consistent");
        for slot in &mut self.cell_tiebreak {
            *slot = self.rng.next_f64() * 1e-4;
        }
        self.recompute_all_pass1_entropies();
        self.recount_pass1_collapsed();
        self.pass1_initial_collapsed = self.pass1_collapsed;
        self.last_checkpoint_collapsed = self.pass1_collapsed;
        self.save_checkpoint();
    }

    /// Advances Pass 1 by one cell collapse (with undo-trail + checkpoint recovery).
    fn step_pass1_once(&mut self) -> bool {
        let Some(target) = self.select_pass1_cell() else {
            self.init_pass2_ponds();
            return true;
        };

        let mut domain = self.cells[target];
        let (x, y, z) = index_coords(target);

        while domain != 0 {
            let field = &self.field;
            let chosen = sample_tile(&mut self.rng, domain, |tile| {
                pass1_weight(field, x, y, z, tile)
            });
            if self.try_collapse(target, chosen) {
                self.recount_pass1_collapsed();
                if self.pass1_collapsed >= self.last_checkpoint_collapsed + 128 {
                    self.last_checkpoint_collapsed = self.pass1_collapsed;
                    self.save_checkpoint();
                }
                return true;
            }
            // Speculative choice contradicted: ban this tile from the local trial.
            domain &= !chosen.bit();
        }

        // All tiles in `target`'s domain contradicted: restore checkpoint or retry Pass 1.
        if let Some(cp) = self.checkpoints.pop() {
            self.cells = cp.cells;
            self.rng = cp.rng;
            // Perturb RNG so we explore a different branch from the checkpoint.
            let _ = self.rng.next_u64();
            self.recompute_all_pass1_entropies();
            self.pass1_collapsed = cp.collapsed_count;
            self.last_checkpoint_collapsed = cp.collapsed_count;
        } else {
            self.init_pass1(self.pass_retries + 1);
        }
        true
    }

    /// Runs Barnes Priority-Flood on the Pass 1 solid surface heightmap to queue
    /// enclosed elevated mountain ponds.
    fn init_pass2_ponds(&mut self) {
        self.phase = WfcPhase::Pass2Ponds;
        self.pond_queue.clear();
        self.pond_filled = 0;
        self.pond_count = 0;

        let y_sea = self.field.sea_level;
        let mut surface = [[0usize; GRID_SIZE]; GRID_SIZE];
        for (x, col) in surface.iter_mut().enumerate() {
            for (z, slot) in col.iter_mut().enumerate() {
                let mut top = 0usize;
                for y in (0..GRID_SIZE).rev() {
                    if let Some(tile) = self.tile_at(x, y, z) {
                        if tile.is_land() {
                            top = y;
                            break;
                        }
                    }
                }
                *slot = top;
            }
        }

        // Barnes Priority-Flood seeded from all 4 outer perimeter edges.
        let mut spill = [[usize::MAX; GRID_SIZE]; GRID_SIZE];
        let mut visited = [[false; GRID_SIZE]; GRID_SIZE];
        let mut heap: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();

        for x in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                if x == 0 || x == GRID_SIZE - 1 || z == 0 || z == GRID_SIZE - 1 {
                    let h = surface[x][z].max(y_sea);
                    spill[x][z] = h;
                    visited[x][z] = true;
                    heap.push(Reverse((h, x, z)));
                }
            }
        }

        while let Some(Reverse((h, x, z))) = heap.pop() {
            let neighbors = [
                (x.wrapping_sub(1), z),
                (x + 1, z),
                (x, z.wrapping_sub(1)),
                (x, z + 1),
            ];
            for (nx, nz) in neighbors {
                if nx < GRID_SIZE && nz < GRID_SIZE && !visited[nx][nz] {
                    visited[nx][nz] = true;
                    let nh = surface[nx][nz].max(h);
                    spill[nx][nz] = nh;
                    heap.push(Reverse((nh, nx, nz)));
                }
            }
        }

        // Collect all interior elevated pond voxels and count connected pond basins.
        let mut is_pond_col = [[false; GRID_SIZE]; GRID_SIZE];
        let mut pending: Vec<(usize, usize, usize)> = Vec::new();
        for x in 1..(GRID_SIZE - 1) {
            for z in 1..(GRID_SIZE - 1) {
                let s = surface[x][z];
                let sp = spill[x][z];
                if sp > s && sp > y_sea + 1 {
                    let y_start = (s + 1).max(y_sea + 1);
                    let mut added_any = false;
                    for y in y_start..=sp {
                        if self.tile_at(x, y, z) == Some(Tile::Air) {
                            pending.push((x, y, z));
                            added_any = true;
                        }
                    }
                    if added_any {
                        is_pond_col[x][z] = true;
                    }
                }
            }
        }

        // Count 4-connected alpine pond components.
        let mut seen = [[false; GRID_SIZE]; GRID_SIZE];
        for x in 1..(GRID_SIZE - 1) {
            for z in 1..(GRID_SIZE - 1) {
                if is_pond_col[x][z] && !seen[x][z] {
                    self.pond_count += 1;
                    let mut bfs = VecDeque::new();
                    seen[x][z] = true;
                    bfs.push_back((x, z));
                    while let Some((cx, cz)) = bfs.pop_front() {
                        for (nx, nz) in [
                            (cx.wrapping_sub(1), cz),
                            (cx + 1, cz),
                            (cx, cz.wrapping_sub(1)),
                            (cx, cz + 1),
                        ] {
                            if nx < GRID_SIZE
                                && nz < GRID_SIZE
                                && is_pond_col[nx][nz]
                                && !seen[nx][nz]
                            {
                                seen[nx][nz] = true;
                                bfs.push_back((nx, nz));
                            }
                        }
                    }
                }
            }
        }

        // Sort bottom-up so animated basins visibly fill layer by layer.
        pending.sort_by_key(|&(x, y, z)| (y, x.abs_diff(10) + z.abs_diff(10), x, z));
        self.pond_total = pending.len();
        self.pond_queue = pending.into();

        if self.pond_queue.is_empty() {
            self.init_pass3();
        }
    }

    /// Locks all Pass 1-2 solid/water cells and opens Pass 3 Flora domains in Air space.
    fn init_pass3(&mut self) {
        self.post_pass2_cells = self.cells.clone();
        self.pass3_retries = 0;
        self.setup_pass3_domains();
    }

    /// Initializes Pass 3 Flora domains from `post_pass2_cells` (supports clean retry on deadlock).
    fn setup_pass3_domains(&mut self) {
        self.phase = WfcPhase::Pass3Flora;
        self.checkpoints.clear();
        self.cells = self.post_pass2_cells.clone();

        for idx in 0..CELL_COUNT {
            let (x, y, z) = index_coords(idx);
            if self.post_pass2_cells[idx] != Tile::Air.bit() || y == 0 || y == GRID_SIZE - 1 {
                continue;
            }

            let below_is_grass =
                self.post_pass2_cells[cell_index(x, y - 1, z)] == Tile::Grass.bit();
            let inside_margin = (1..(GRID_SIZE - 1)).contains(&x)
                && (1..(GRID_SIZE - 1)).contains(&z);

            let mut mask = Tile::Air.bit();
            if below_is_grass {
                mask |= Tile::Bush.bit();
                if inside_margin && y <= 15 {
                    mask |= Tile::TrunkBase.bit();
                }
            }
            if inside_margin && (2..=16).contains(&y) {
                mask |= Tile::TrunkMid.bit();
            }
            if inside_margin && (2..=17).contains(&y) {
                mask |= Tile::CanopyCore.bit();
            }
            if (2..=17).contains(&y) {
                if x >= 2 && (1..(GRID_SIZE - 1)).contains(&z) {
                    mask |= Tile::LeafPosX.bit();
                }
                if x <= GRID_SIZE - 3 && (1..(GRID_SIZE - 1)).contains(&z) {
                    mask |= Tile::LeafNegX.bit();
                }
                if z >= 2 && (1..(GRID_SIZE - 1)).contains(&x) {
                    mask |= Tile::LeafPosZ.bit();
                }
                if z <= GRID_SIZE - 3 && (1..(GRID_SIZE - 1)).contains(&x) {
                    mask |= Tile::LeafNegZ.bit();
                }
            }
            if inside_margin && (3..=18).contains(&y) {
                mask |= Tile::LeafTop.bit();
            }

            self.cells[idx] = mask;
        }

        // Propagate locked solid/water walls & boundary constraints immediately.
        self.clear_queue();
        for idx in 0..CELL_COUNT {
            self.enqueue(idx);
        }
        let ok = self.propagate_committed();
        debug_assert!(ok, "Pass 3 initial domain constraints must be consistent");

        for slot in &mut self.cell_tiebreak {
            *slot = self.rng.next_f64() * 1e-3;
        }
        self.recompute_all_pass3_scores();

        let uncollapsed = self
            .cells
            .iter()
            .filter(|&&m| m != 0 && (m & (m - 1)) != 0)
            .count();
        self.pass3_total = uncollapsed;
        self.pass3_collapsed = 0;

        if uncollapsed == 0 {
            self.phase = WfcPhase::Complete;
        }
    }

    /// Advances Pass 3 Flora WFC by one cell collapse.
    fn step_pass3_once(&mut self) -> bool {
        let Some(target) = self.select_pass3_cell() else {
            self.phase = WfcPhase::Complete;
            return true;
        };

        let mut domain = self.cells[target];
        let (_x, y, _z) = index_coords(target);

        while domain != 0 {
            let chosen = sample_tile(&mut self.rng, domain, |tile| pass3_weight(y, tile));
            if self.try_collapse(target, chosen) {
                return true;
            }
            domain &= !chosen.bit();
        }

        // Rare Pass 3 deadlock: cleanly restore post-Pass-2 terrain and retry with a fresh branch.
        self.pass3_retries += 1;
        if self.pass3_retries <= 8 {
            let _ = self.rng.next_u64();
            self.setup_pass3_domains();
        } else {
            // Bounded fallback: restore clean terrain/water with zero invalid or uncollapsed cells.
            self.cells = self.post_pass2_cells.clone();
            self.pass3_collapsed = self.pass3_total;
            self.phase = WfcPhase::Complete;
        }
        true
    }

    /// Selects the lowest-entropy uncollapsed cell in Pass 1 using cached entropies.
    fn select_pass1_cell(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_entropy = f64::INFINITY;

        for idx in 0..CELL_COUNT {
            let mask = self.cells[idx];
            if mask == 0 || (mask & (mask - 1)) == 0 {
                continue;
            }
            let h = self.pass1_entropy[idx] + self.cell_tiebreak[idx];
            if h < best_entropy {
                best_entropy = h;
                best_idx = Some(idx);
            }
        }
        best_idx
    }

    /// Selects the lowest-entropy uncollapsed cell in Pass 3 using cached scores.
    fn select_pass3_cell(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_score = f64::INFINITY;

        for idx in 0..CELL_COUNT {
            let mask = self.cells[idx];
            if mask == 0 || (mask & (mask - 1)) == 0 {
                continue;
            }
            let score = self.pass1_entropy[idx] + self.cell_tiebreak[idx];
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        best_idx
    }

    /// Speculatively collapses `idx` to `tile` and propagates via AC-3.
    /// Rolls back all mutations cleanly if a contradiction occurs.
    fn try_collapse(&mut self, idx: usize, tile: Tile) -> bool {
        self.undo_trail.clear();
        let old = self.cells[idx];
        let target_mask = tile.bit();
        if (old & target_mask) == 0 {
            return false;
        }
        if old != target_mask {
            self.undo_trail.push((idx, old));
            self.cells[idx] = target_mask;
        }

        self.clear_queue();
        self.enqueue(idx);

        if self.propagate_with_undo() {
            if self.phase == WfcPhase::Pass1Geology {
                for i in 0..self.undo_trail.len() {
                    let cell_idx = self.undo_trail[i].0;
                    self.update_pass1_entropy(cell_idx);
                }
            } else if self.phase == WfcPhase::Pass3Flora {
                for i in 0..self.undo_trail.len() {
                    let cell_idx = self.undo_trail[i].0;
                    self.update_pass3_score(cell_idx);
                }
            }
            self.undo_trail.clear();
            true
        } else {
            for &(cell_idx, prev_mask) in self.undo_trail.iter().rev() {
                self.cells[cell_idx] = prev_mask;
            }
            self.undo_trail.clear();
            self.clear_queue();
            false
        }
    }

    /// AC-3 constraint propagation recording every changed cell into `undo_trail`.
    fn propagate_with_undo(&mut self) -> bool {
        while let Some(curr) = self.dequeue() {
            let curr_mask = self.cells[curr];
            if curr_mask == 0 {
                return false;
            }
            for dir in Dir::ALL {
                let Some(next) = neighbor_index(curr, dir) else {
                    continue;
                };
                let allowed = self.table.support_mask(curr_mask, dir);
                let old_next = self.cells[next];
                let new_next = old_next & allowed;
                if new_next == 0 {
                    return false;
                }
                if new_next != old_next {
                    self.undo_trail.push((next, old_next));
                    self.cells[next] = new_next;
                    self.enqueue(next);
                }
            }
        }
        true
    }

    /// AC-3 constraint propagation without undo recording (used for initial domain setup).
    fn propagate_committed(&mut self) -> bool {
        while let Some(curr) = self.dequeue() {
            let curr_mask = self.cells[curr];
            if curr_mask == 0 {
                return false;
            }
            for dir in Dir::ALL {
                let Some(next) = neighbor_index(curr, dir) else {
                    continue;
                };
                let allowed = self.table.support_mask(curr_mask, dir);
                let old_next = self.cells[next];
                let new_next = old_next & allowed;
                if new_next == 0 {
                    return false;
                }
                if new_next != old_next {
                    self.cells[next] = new_next;
                    self.enqueue(next);
                }
            }
        }
        true
    }

    fn clear_queue(&mut self) {
        while let Some(idx) = self.queue.pop_front() {
            self.in_queue[idx] = false;
        }
    }

    #[inline]
    fn enqueue(&mut self, idx: usize) {
        if !self.in_queue[idx] {
            self.in_queue[idx] = true;
            self.queue.push_back(idx);
        }
    }

    #[inline]
    fn dequeue(&mut self) -> Option<usize> {
        let idx = self.queue.pop_front()?;
        self.in_queue[idx] = false;
        Some(idx)
    }

    fn recompute_all_pass1_entropies(&mut self) {
        for idx in 0..CELL_COUNT {
            self.update_pass1_entropy(idx);
        }
    }

    #[inline]
    fn update_pass1_entropy(&mut self, idx: usize) {
        let mask = self.cells[idx];
        if mask == 0 || (mask & (mask - 1)) == 0 {
            self.pass1_entropy[idx] = 0.0;
        } else {
            let (x, y, z) = index_coords(idx);
            let field = &self.field;
            self.pass1_entropy[idx] =
                shannon_entropy(mask, |tile| pass1_weight(field, x, y, z, tile));
        }
    }

    fn recompute_all_pass3_scores(&mut self) {
        for idx in 0..CELL_COUNT {
            self.update_pass3_score(idx);
        }
    }

    #[inline]
    fn update_pass3_score(&mut self, idx: usize) {
        let mask = self.cells[idx];
        if mask == 0 || (mask & (mask - 1)) == 0 {
            self.pass1_entropy[idx] = 0.0;
        } else {
            let (_x, y, _z) = index_coords(idx);
            let h = shannon_entropy(mask, |tile| pass3_weight(y, tile));
            let committed_bonus = if (mask & Tile::Air.bit()) == 0 {
                -10.0
            } else {
                0.0
            };
            self.pass1_entropy[idx] = h + (y as f64) * 0.14 + committed_bonus;
        }
    }

    fn recount_pass1_collapsed(&mut self) {
        self.pass1_collapsed = self
            .cells
            .iter()
            .filter(|&&m| m != 0 && (m & (m - 1)) == 0)
            .count();
    }

    fn save_checkpoint(&mut self) {
        if self.checkpoints.len() >= 8 {
            self.checkpoints.remove(0);
        }
        self.checkpoints.push(Checkpoint {
            cells: self.cells.clone(),
            rng: self.rng,
            collapsed_count: self.pass1_collapsed,
        });
    }
}

/// Guided Pass 1 geology weights derived from the seeded macro [`TerrainField`].
fn pass1_weight(field: &TerrainField, x: usize, y: usize, z: usize, tile: Tile) -> f64 {
    let target_h = field.height_at(x, z);
    let slope = field.slope_at(x, z);
    let dy = y as f32 - target_h;
    let y_sea = field.sea_level;

    match tile {
        Tile::Air => {
            if dy > 0.45 {
                18.0
            } else if dy > -0.35 {
                1.6
            } else {
                0.02
            }
        }
        Tile::Water => {
            if dy > 0.40 {
                20.0
            } else if dy > -0.35 {
                1.8
            } else {
                0.02
            }
        }
        Tile::DeepRock => {
            if dy <= -2.6 {
                16.0
            } else if dy <= -1.4 {
                3.2
            } else if y <= 2 {
                2.0
            } else {
                0.08
            }
        }
        Tile::Subsoil => {
            if dy <= -0.55 && dy > -2.8 {
                14.0
            } else if dy <= 0.25 && y <= y_sea {
                10.0
            } else if dy <= 0.35 {
                2.0
            } else {
                0.05
            }
        }
        Tile::Grass => {
            if y >= 14 {
                0.0
            } else {
                let alt_fade = if y == 13 {
                    0.20
                } else if y == 12 {
                    0.55
                } else {
                    1.0
                };
                let slope_fade = if slope >= 2.2 {
                    0.25
                } else if slope >= 1.5 {
                    0.65
                } else {
                    1.0
                };
                let surface_fit = if (-0.65..=0.55).contains(&dy) {
                    18.0
                } else if (-1.2..=0.95).contains(&dy) {
                    1.4
                } else {
                    0.02
                };
                surface_fit * alt_fade * slope_fade
            }
        }
        Tile::CliffRock => {
            if y >= 13 && dy <= 0.55 {
                14.0
            } else if slope >= 1.8 && (-1.4..=0.55).contains(&dy) {
                8.5
            } else if dy <= 0.35 {
                0.9
            } else {
                0.04
            }
        }
        _ => 0.0,
    }
}

/// Pass 3 flora weights yielding natural meadow spacing of 3-4 voxel trees and shrubs.
fn pass3_weight(y: usize, tile: Tile) -> f64 {
    let alt_factor = if y >= 11 { 0.55 } else { 1.0 };
    match tile {
        Tile::Air => 1.0,
        Tile::Bush => 0.13 * alt_factor,
        Tile::TrunkBase => 0.12 * alt_factor,
        Tile::TrunkMid => 0.48,
        Tile::CanopyCore => 0.52,
        Tile::LeafPosX
        | Tile::LeafNegX
        | Tile::LeafPosZ
        | Tile::LeafNegZ
        | Tile::LeafTop => 0.02,
        _ => 0.0,
    }
}

fn shannon_entropy<F>(mut mask: u16, weight_fn: F) -> f64
where
    F: Fn(Tile) -> f64,
{
    let mut sum_w = 0.0;
    let mut sum_w_log_w = 0.0;
    while mask != 0 {
        let tz = mask.trailing_zeros() as usize;
        let w = weight_fn(Tile::from_index(tz)).max(1e-6);
        sum_w += w;
        sum_w_log_w += w * w.ln();
        mask &= mask - 1;
    }
    sum_w.ln() - sum_w_log_w / sum_w
}

fn sample_tile<F>(rng: &mut Prng, mask: u16, weight_fn: F) -> Tile
where
    F: Fn(Tile) -> f64,
{
    let mut total = 0.0;
    let mut m = mask;
    while m != 0 {
        let tz = m.trailing_zeros() as usize;
        total += weight_fn(Tile::from_index(tz)).max(1e-6);
        m &= m - 1;
    }
    let mut pick = rng.next_f64() * total;
    let mut m = mask;
    let mut last = Tile::Air;
    while m != 0 {
        let tz = m.trailing_zeros() as usize;
        let tile = Tile::from_index(tz);
        last = tile;
        pick -= weight_fn(tile).max(1e-6);
        if pick <= 0.0 {
            return tile;
        }
        m &= m - 1;
    }
    last
}

#[cfg(test)]
mod tests {
    use super::{cell_index, neighbor_index, AdjacencyTable, Dir, Tile, WfcSolver, CELL_COUNT, GRID_SIZE};

    const DIVERSE_SEEDS: [&str; 16] = [
        "alpine-42",
        "0",
        "1",
        "1337",
        "tarn-101",
        "fjord-777",
        "caldera-99",
        "meadow-2026",
        "basalt-88",
        "spire-12",
        "glade-55",
        "cove-314",
        "crag-404",
        "ridge-808",
        "voxel-wfc",
        "mountain-lake",
    ];

    #[test]
    fn adjacency_table_is_strictly_symmetric() {
        let table = AdjacencyTable::build();
        for a in Tile::ALL {
            for dir in Dir::ALL {
                let allowed_a = table.allowed_for(a, dir);
                for b in Tile::ALL {
                    let b_in_a = (allowed_a & b.bit()) != 0;
                    let a_in_b = (table.allowed_for(b, dir.opposite()) & a.bit()) != 0;
                    assert_eq!(
                        b_in_a, a_in_b,
                        "asymmetry between {a:?} --{dir:?}--> {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_diverse_seeds_solve_completely_and_satisfy_every_invariant() {
        let mut seeds_with_alpine_ponds = 0usize;
        let mut total_trees = 0usize;
        let mut total_bushes = 0usize;

        for seed in DIVERSE_SEEDS {
            let mut solver = WfcSolver::new(seed);
            solver.solve_all();
            assert!(solver.is_complete(), "seed '{seed}' did not complete");

            let y_sea = solver.sea_level();
            let stats = solver.stats();
            if stats.pond_count > 0 && stats.pond_voxels > 0 {
                seeds_with_alpine_ponds += 1;
            }
            total_trees += stats.tree_count;
            total_bushes += stats.bush_count;

            for y in 0..GRID_SIZE {
                for z in 0..GRID_SIZE {
                    for x in 0..GRID_SIZE {
                        let tile = solver.tile_at(x, y, z).unwrap_or_else(|| {
                            panic!(
                                "seed '{seed}': uncollapsed cell at ({x}, {y}, {z}), mask={:#06x}",
                                solver.mask_at(x, y, z)
                            )
                        });

                        // 1. Zero floating land: every land tile rests on land or y == 0.
                        if tile.is_land() && y > 0 {
                            let below = solver
                                .tile_at(x, y - 1, z)
                                .expect("below cell must be collapsed");
                            assert!(
                                below.is_land(),
                                "seed '{seed}': floating land {tile:?} at ({x},{y},{z}) over {below:?}"
                            );
                        }

                        // 2. Zero steep Subsoil or DeepRock exposed horizontally to Air.
                        if matches!(tile, Tile::Subsoil | Tile::DeepRock) {
                            for dir in Dir::HORIZONTAL {
                                if let Some(nidx) = neighbor_index(cell_index(x, y, z), dir) {
                                    let (nx, ny, nz) = super::index_coords(nidx);
                                    let ntile = solver
                                        .tile_at(nx, ny, nz)
                                        .expect("neighbor must be collapsed");
                                    assert_ne!(
                                        ntile,
                                        Tile::Air,
                                        "seed '{seed}': exposed {tile:?} at ({x},{y},{z}) next to Air at ({nx},{ny},{nz})"
                                    );
                                }
                            }
                        }

                        // 3. Zero floating or spilled water.
                        if tile == Tile::Water {
                            assert!(y > 0, "seed '{seed}': water at bedrock y=0");
                            let below = solver
                                .tile_at(x, y - 1, z)
                                .expect("below water must be collapsed");
                            assert!(
                                below.is_land() || below == Tile::Water,
                                "seed '{seed}': floating water at ({x},{y},{z}) over {below:?}"
                            );
                            for dir in Dir::HORIZONTAL {
                                if let Some(nidx) = neighbor_index(cell_index(x, y, z), dir) {
                                    let (nx, ny, nz) = super::index_coords(nidx);
                                    let ntile = solver
                                        .tile_at(nx, ny, nz)
                                        .expect("neighbor must be collapsed");
                                    assert_ne!(
                                        ntile,
                                        Tile::Air,
                                        "seed '{seed}': spilled water at ({x},{y},{z}) touching Air at ({nx},{ny},{nz})"
                                    );
                                } else {
                                    // Water on outer diorama boundary is only allowed at/below sea level.
                                    assert!(
                                        y <= y_sea,
                                        "seed '{seed}': elevated pond water leaked to diorama edge at ({x},{y},{z}) > y_sea={y_sea}"
                                    );
                                }
                            }
                        }

                        // 4. Zero floating foliage, zero foliage on rock/water, strict tree structure.
                        match tile {
                            Tile::Bush => {
                                assert!(y > 0 && y + 1 < GRID_SIZE);
                                assert_eq!(solver.tile_at(x, y - 1, z), Some(Tile::Grass));
                                assert_eq!(solver.tile_at(x, y + 1, z), Some(Tile::Air));
                            }
                            Tile::TrunkBase => {
                                assert!(y > 0 && y + 1 < GRID_SIZE);
                                assert_eq!(solver.tile_at(x, y - 1, z), Some(Tile::Grass));
                                let above = solver.tile_at(x, y + 1, z);
                                assert!(
                                    matches!(above, Some(Tile::TrunkMid | Tile::CanopyCore)),
                                    "seed '{seed}': bare TrunkBase at ({x},{y},{z}) with {above:?} above"
                                );
                            }
                            Tile::TrunkMid => {
                                assert_eq!(solver.tile_at(x, y - 1, z), Some(Tile::TrunkBase));
                                assert_eq!(solver.tile_at(x, y + 1, z), Some(Tile::CanopyCore));
                            }
                            Tile::CanopyCore => {
                                let below = solver.tile_at(x, y - 1, z);
                                assert!(matches!(below, Some(Tile::TrunkBase | Tile::TrunkMid)));
                                assert_eq!(solver.tile_at(x + 1, y, z), Some(Tile::LeafPosX));
                                assert_eq!(solver.tile_at(x - 1, y, z), Some(Tile::LeafNegX));
                                assert_eq!(solver.tile_at(x, y, z + 1), Some(Tile::LeafPosZ));
                                assert_eq!(solver.tile_at(x, y, z - 1), Some(Tile::LeafNegZ));
                                assert_eq!(solver.tile_at(x, y + 1, z), Some(Tile::LeafTop));
                            }
                            Tile::LeafPosX
                            | Tile::LeafNegX
                            | Tile::LeafPosZ
                            | Tile::LeafNegZ
                            | Tile::LeafTop => {
                                let core_dir =
                                    tile.leaf_core_dir().expect("leaf must have core dir");
                                let core_idx = neighbor_index(cell_index(x, y, z), core_dir)
                                    .expect("leaf core must stay inside bounds");
                                let (cx, cy, cz) = super::index_coords(core_idx);
                                assert_eq!(
                                    solver.tile_at(cx, cy, cz),
                                    Some(Tile::CanopyCore),
                                    "seed '{seed}': orphan {tile:?} at ({x},{y},{z})"
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Ensure no runaway 1x1 rock spires stick >= 3 voxels above all 4 cardinal neighbors.
            let mut top_land = [[0usize; GRID_SIZE]; GRID_SIZE];
            for (x, col) in top_land.iter_mut().enumerate() {
                for (z, slot) in col.iter_mut().enumerate() {
                    for y in (0..GRID_SIZE).rev() {
                        if solver.tile_at(x, y, z).is_some_and(Tile::is_land) {
                            *slot = y;
                            break;
                        }
                    }
                }
            }
            for x in 1..(GRID_SIZE - 1) {
                for z in 1..(GRID_SIZE - 1) {
                    let h = top_land[x][z];
                    let max_nb = top_land[x - 1][z]
                        .max(top_land[x + 1][z])
                        .max(top_land[x][z - 1])
                        .max(top_land[x][z + 1]);
                    assert!(
                        h <= max_nb + 2,
                        "seed '{seed}': isolated 1x1 spire at ({x},{z}) height={h}, max_neighbor={max_nb}"
                    );
                }
            }
        }

        assert!(
            seeds_with_alpine_ponds >= 3,
            "expected elevated mountain ponds on multiple seeds, got {seeds_with_alpine_ponds}"
        );
        assert!(total_trees >= 20, "expected healthy tree generation, got {total_trees}");
        assert!(total_bushes >= 20, "expected healthy bush generation, got {total_bushes}");
    }

    #[test]
    fn explicit_enclosed_bowl_fills_elevated_pond_cleanly() {
        let mut solver = WfcSolver::new("explicit-bowl-test");
        let y_sea = solver.sea_level();
        // Construct a valid mountain plateau (CliffRock walls on steep drop) with a sunken 3x3 bowl.
        for x in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let in_rim = (6..=12).contains(&x) && (6..=12).contains(&z);
                let in_bowl = (8..=10).contains(&x) && (8..=10).contains(&z);
                let on_steep_rim = in_rim && !in_bowl;
                let top_y = if in_bowl {
                    y_sea + 3
                } else if in_rim {
                    y_sea + 5
                } else {
                    y_sea + 1
                };
                for y in 0..GRID_SIZE {
                    let tile = if y == 0 {
                        Tile::DeepRock
                    } else if on_steep_rim && y > y_sea + 1 && y <= top_y {
                        Tile::CliffRock
                    } else if y < top_y {
                        Tile::Subsoil
                    } else if y == top_y {
                        Tile::Grass
                    } else {
                        Tile::Air
                    };
                    solver.cells[cell_index(x, y, z)] = tile.bit();
                }
            }
        }

        solver.init_pass2_ponds();
        assert_eq!(solver.pond_count, 1, "expected exactly 1 enclosed mountain pond");
        // 3x3 bowl, 2 voxels deep ((y_sea + 4) and (y_sea + 5)) = 18 water voxels.
        assert_eq!(solver.pond_total, 18);

        solver.solve_all();
        assert!(solver.is_complete());
        for x in 8..=10 {
            for z in 8..=10 {
                assert_eq!(solver.tile_at(x, y_sea + 3, z), Some(Tile::Subsoil));
                assert_eq!(solver.tile_at(x, y_sea + 4, z), Some(Tile::Water));
                assert_eq!(solver.tile_at(x, y_sea + 5, z), Some(Tile::Water));
            }
        }
    }

    #[test]
    fn generation_is_bit_for_bit_deterministic() {
        let mut a = WfcSolver::new("alpine-42");
        let mut b = WfcSolver::new("alpine-42");
        a.solve_all();
        b.solve_all();

        assert_eq!(a.sea_level(), b.sea_level());
        for idx in 0..CELL_COUNT {
            assert_eq!(a.cells[idx], b.cells[idx], "mismatch at cell {idx}");
        }

        // Stepping incrementally must match solve_all() identically!
        let mut c = WfcSolver::new("alpine-42");
        while !c.is_complete() {
            c.step(17);
        }
        for idx in 0..CELL_COUNT {
            assert_eq!(a.cells[idx], c.cells[idx], "step-size mismatch at cell {idx}");
        }
    }
}
