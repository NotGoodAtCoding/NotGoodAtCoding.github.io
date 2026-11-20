//! Deterministic seed hashing, pure-Rust PRNG (`SplitMix64` + `Xoshiro256StarStar`),
//! and 2D smooth value noise with alpine bowl modulation.

use std::f32::consts::TAU;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

/// Deterministically maps an arbitrary seed string to a `u64`.
///
/// Raw unsigned integer strings parse cleanly to their numeric value; any other
/// string is hashed with 64-bit FNV-1a after trimming surrounding whitespace.
pub fn hash_seed(seed_str: &str) -> u64 {
    let trimmed = seed_str.trim();
    if let Ok(value) = trimmed.parse::<u64>() {
        return value;
    }
    let bytes = if trimmed.is_empty() {
        b"default".as_slice()
    } else {
        trimmed.as_bytes()
    };
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministic `SplitMix64` state mixer used to expand a 64-bit seed and hash lattice points.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` generator from a 64-bit seed.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next well-mixed 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Pure-Rust `Xoshiro256StarStar` PRNG, bit-identical across native and `wasm32`.
#[derive(Debug, Clone, Copy)]
pub struct Prng {
    state: [u64; 4],
}

impl Prng {
    /// Seeds a new `Xoshiro256StarStar` PRNG by expanding `seed` through `SplitMix64`.
    pub fn from_seed(seed: u64) -> Self {
        let mut mixer = SplitMix64::new(seed);
        let mut state = [
            mixer.next_u64(),
            mixer.next_u64(),
            mixer.next_u64(),
            mixer.next_u64(),
        ];
        if state == [0, 0, 0, 0] {
            state[0] = 0x9e3779b97f4a7c15;
        }
        Self { state }
    }

    /// Returns the next pseudorandom `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Returns a uniform `f32` in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Returns a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Returns a uniform `usize` in the inclusive range `[low, high]`.
    pub fn gen_inclusive_usize(&mut self, low: usize, high: usize) -> usize {
        debug_assert!(low <= high);
        let span = (high - low + 1) as u64;
        low + (self.next_u64() % span) as usize
    }

    /// Generates a short human-readable random seed label (e.g. `"tarn-742"`).
    pub fn next_seed_label(&mut self) -> String {
        const PREFIXES: [&str; 12] = [
            "alpine", "tarn", "ridge", "fjord", "cove", "crag", "meadow", "basalt", "glade",
            "spire", "caldera", "glacier",
        ];
        let idx = (self.next_u64() as usize) % PREFIXES.len();
        let num = (self.next_u64() % 900) + 100;
        format!("{}-{num}", PREFIXES[idx])
    }
}

/// Deterministic per-voxel hash returning a uniform `f32` in `[0.0, 1.0)`.
pub fn voxel_hash(x: usize, y: usize, z: usize, seed: u64) -> f32 {
    let mut k = seed;
    k ^= (x as u64).wrapping_mul(0x9e3779b185ebca87);
    k = k.rotate_left(27);
    k ^= (y as u64).wrapping_mul(0xc2b2ae3d27d4eb4f);
    k = k.rotate_left(31);
    k ^= (z as u64).wrapping_mul(0x165667b19e3779f9);
    let mut mixer = SplitMix64::new(k);
    let bits = (mixer.next_u64() >> 40) as u32;
    (bits as f32) * (1.0 / ((1u32 << 24) as f32))
}

/// 2D lattice hash returning a deterministic value in `[0.0, 1.0)`.
fn lattice_value(ix: i32, iz: i32, seed: u64) -> f32 {
    let mut h = seed;
    h ^= (ix as u64).wrapping_mul(0x9e3779b97f4a7c15);
    h ^= (iz as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    let mut mixer = SplitMix64::new(h);
    let bits = (mixer.next_u64() >> 40) as u32;
    (bits as f32) * (1.0 / ((1u32 << 24) as f32))
}

#[inline]
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Smooth 2D value noise in `[0.0, 1.0]`.
fn value_noise_2d(x: f32, z: f32, seed: u64) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = quintic(x - x0 as f32);
    let fz = quintic(z - z0 as f32);

    let v00 = lattice_value(x0, z0, seed);
    let v10 = lattice_value(x0 + 1, z0, seed);
    let v01 = lattice_value(x0, z0 + 1, seed);
    let v11 = lattice_value(x0 + 1, z0 + 1, seed);

    let vx0 = v00 + (v10 - v00) * fx;
    let vx1 = v01 + (v11 - v01) * fx;
    vx0 + (vx1 - vx0) * fz
}

/// Seeded 2D macro terrain height field guiding WFC Pass 1 geology weights.
///
/// Combines 3-octave smooth value noise, a coastal slope for ocean coves, and a
/// seeded alpine caldera/bowl modulation so enclosed highland ponds appear naturally.
#[derive(Debug, Clone)]
pub struct TerrainField {
    pub sea_level: usize,
    heights: [[f32; 20]; 20],
}

impl TerrainField {
    /// Generates a deterministic `20 x 20` macro terrain guide from `seed`.
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Prng::from_seed(seed ^ 0x51a7_3c91_b02d_8e4f);
        let sea_level = rng.gen_inclusive_usize(4, 6);

        let noise_seed_0 = rng.next_u64();
        let noise_seed_1 = rng.next_u64();
        let noise_seed_2 = rng.next_u64();
        let offset_x = rng.next_f32() * 64.0;
        let offset_z = rng.next_f32() * 64.0;

        // Coastal slope orientation: tilts one edge down below sea level while raising
        // the opposite highlands into hills and alpine crags.
        let coast_angle = rng.next_f32() * TAU;
        let (coast_sin, coast_cos) = coast_angle.sin_cos();
        let slope_strength = 3.2 + rng.next_f32() * 2.0;

        // Seeded alpine bowl / highland crater (~80% of seeds include a prominent enclosed rim).
        let has_bowl = rng.next_f32() < 0.80;
        // Position the bowl center in the higher interior quadrant, away from outer edges
        // so its surrounding rim stays strictly inside the 20 x 20 footprint.
        let bowl_cx = (9.5 + coast_cos * 2.6 + (rng.next_f32() - 0.5) * 3.0).clamp(5.5, 13.5);
        let bowl_cz = (9.5 + coast_sin * 2.6 + (rng.next_f32() - 0.5) * 3.0).clamp(5.5, 13.5);
        let rim_radius = 3.1 + rng.next_f32() * 1.1;
        let rim_amp = 4.2 + rng.next_f32() * 2.2;
        let dip_amp = 3.2 + rng.next_f32() * 1.6;

        // Optional secondary peak for crisp rocky alpine silhouettes.
        let peak_cx = (9.5 - coast_sin * 4.5 + (rng.next_f32() - 0.5) * 3.0).clamp(3.0, 16.0);
        let peak_cz = (9.5 + coast_cos * 4.5 + (rng.next_f32() - 0.5) * 3.0).clamp(3.0, 16.0);
        let peak_amp = 2.8 + rng.next_f32() * 3.0;
        let peak_sigma = 2.6 + rng.next_f32() * 1.4;

        let mut heights = [[0.0_f32; 20]; 20];
        for (x, col) in heights.iter_mut().enumerate() {
            for (z, slot) in col.iter_mut().enumerate() {
                let fx = x as f32;
                let fz = z as f32;

                // 3-octave smooth value noise in [-1.0, 1.0].
                let n0 = value_noise_2d(fx * 0.13 + offset_x, fz * 0.13 + offset_z, noise_seed_0);
                let n1 = value_noise_2d(fx * 0.27 + offset_x, fz * 0.27 + offset_z, noise_seed_1);
                let n2 = value_noise_2d(fx * 0.52 + offset_x, fz * 0.52 + offset_z, noise_seed_2);
                let fbm = (n0 * 0.60 + n1 * 0.28 + n2 * 0.12) * 2.0 - 1.0;

                // Normalized coordinate along coastal slope in [-1.0, 1.0].
                let nx = (fx - 9.5) / 9.5;
                let nz = (fz - 9.5) / 9.5;
                let coastal = (nx * coast_cos + nz * coast_sin) * slope_strength;

                let mut h = sea_level as f32 + 1.8 + coastal + fbm * 3.4;

                // Secondary highland ridge / peak.
                let pdx = fx - peak_cx;
                let pdz = fz - peak_cz;
                let pd2 = pdx * pdx + pdz * pdz;
                h += peak_amp * (-pd2 / (2.0 * peak_sigma * peak_sigma)).exp();

                // Enclosed alpine bowl: raised ring + sunken interior basin well above sea level.
                if has_bowl {
                    let bdx = fx - bowl_cx;
                    let bdz = fz - bowl_cz;
                    let r = (bdx * bdx + bdz * bdz).sqrt();
                    let rim_d = (r - rim_radius) / 1.45;
                    let rim = rim_amp * (-(rim_d * rim_d)).exp();
                    let dip = dip_amp * (-((r / (rim_radius * 0.62)).powi(2))).exp();
                    // Keep the interior basin comfortably elevated above sea_level + 1.
                    let bowl_floor_lift = (sea_level as f32 + 3.2 - h).max(0.0)
                        * (-((r / (rim_radius + 1.4)).powi(2))).exp();
                    h += bowl_floor_lift + rim - dip;
                }

                *slot = h.clamp(1.5, 16.5);
            }
        }

        Self { sea_level, heights }
    }

    /// Returns the continuous macro guide height at column `(x, z)`.
    #[inline]
    pub fn height_at(&self, x: usize, z: usize) -> f32 {
        self.heights[x][z]
    }

    /// Returns the maximum absolute height difference to the 4 cardinal neighbors.
    pub fn slope_at(&self, x: usize, z: usize) -> f32 {
        let h = self.heights[x][z];
        let mut max_diff = 0.0_f32;
        if x > 0 {
            max_diff = max_diff.max((h - self.heights[x - 1][z]).abs());
        }
        if x + 1 < 20 {
            max_diff = max_diff.max((h - self.heights[x + 1][z]).abs());
        }
        if z > 0 {
            max_diff = max_diff.max((h - self.heights[x][z - 1]).abs());
        }
        if z + 1 < 20 {
            max_diff = max_diff.max((h - self.heights[x][z + 1]).abs());
        }
        max_diff
    }
}
