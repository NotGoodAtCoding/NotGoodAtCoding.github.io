//! Smooth analytic undulating terrain with rolling hills and stepping mounds.
//!
//! All height queries, surface normals, raycasts, and mesh vertices share the
//! exact same pure-Rust heightfield so foot-placement IK matches the rendered
//! surface without any external physics dependency.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Half-width of the square terrain patch in meters (total size = 36m × 36m).
pub const TERRAIN_HALF_EXTENT: f32 = 18.0;

/// Radius within which the fox stays during manual steering and auto-roam.
pub const PLAY_AREA_RADIUS: f32 = 13.0;

/// Smooth Gaussian stepping mounds `(x, z, amplitude, sigma)` placed along the
/// roam area so individual legs visibly adapt to local height variations.
const MOUNDS: [(f32, f32, f32, f32); 7] = [
    (2.6, 1.8, 0.48, 1.25),
    (-3.2, 2.4, 0.52, 1.35),
    (-1.8, -3.4, 0.44, 1.20),
    (3.8, -2.6, 0.56, 1.40),
    (0.0, 5.4, 0.46, 1.30),
    (-5.5, -1.0, 0.50, 1.45),
    (5.8, 1.5, 0.42, 1.25),
];

/// Evaluates the terrain elevation `y` (in meters) at world coordinates `(x, z)`.
#[inline]
pub fn terrain_height(x: f32, z: f32) -> f32 {
    // Broad rolling hills
    let hills = 0.52 * (x * 0.22).sin() * (z * 0.20).cos()
        + 0.26 * (x * 0.42 + z * 0.34 + 0.7).sin()
        + 0.07 * (x * 0.85 - z * 0.78).cos();

    // Local smooth stepping mounds for pronounced quadruped leg IK
    let mut mounds = 0.0;
    for &(mx, mz, amp, sigma) in &MOUNDS {
        let dx = x - mx;
        let dz = z - mz;
        let denom = 2.0 * sigma * sigma;
        mounds += amp * (-(dx * dx + dz * dz) / denom).exp();
    }

    hills + mounds
}

/// Computes the unit surface normal at world coordinates `(x, z)` via central
/// finite differences of [`terrain_height`].
#[inline]
pub fn terrain_normal(x: f32, z: f32) -> Vec3 {
    const EPS: f32 = 0.02;
    let dh_dx = (terrain_height(x + EPS, z) - terrain_height(x - EPS, z)) / (2.0 * EPS);
    let dh_dz = (terrain_height(x, z + EPS) - terrain_height(x, z - EPS)) / (2.0 * EPS);
    Vec3::new(-dh_dx, 1.0, -dh_dz).normalize()
}

/// Intersects a world-space [`Ray3d`] against the analytic terrain surface using
/// coarse raymarching followed by binary-search refinement.
pub fn raycast_terrain(ray: Ray3d, max_distance: f32) -> Option<Vec3> {
    const STEP: f32 = 0.3;
    const REFINE_ITERS: usize = 8;

    let origin = ray.origin;
    let dir = *ray.direction;

    let mut prev_t = 0.0_f32;
    let start_diff = origin.y - terrain_height(origin.x, origin.z);
    if start_diff <= 0.0 {
        return Some(Vec3::new(origin.x, terrain_height(origin.x, origin.z), origin.z));
    }

    let mut t = STEP;
    while t <= max_distance {
        let p = origin + dir * t;
        let diff = p.y - terrain_height(p.x, p.z);
        if diff <= 0.0 {
            let mut lo = prev_t;
            let mut hi = t;
            for _ in 0..REFINE_ITERS {
                let mid = 0.5 * (lo + hi);
                let mp = origin + dir * mid;
                if mp.y - terrain_height(mp.x, mp.z) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let hit = origin + dir * (0.5 * (lo + hi));
            return Some(Vec3::new(hit.x, terrain_height(hit.x, hit.z), hit.z));
        }
        prev_t = t;
        t += STEP;
    }

    None
}

/// Builds a subdivided vertex-colored terrain [`Mesh`] whose vertices and
/// normals match [`terrain_height`] and [`terrain_normal`].
pub fn build_terrain_mesh(half_extent: f32, subdivisions: u32) -> Mesh {
    let verts_per_side = subdivisions as usize + 1;
    let total_verts = verts_per_side * verts_per_side;
    let total_indices = (subdivisions as usize) * (subdivisions as usize) * 6;

    let mut positions = Vec::with_capacity(total_verts);
    let mut normals = Vec::with_capacity(total_verts);
    let mut uvs = Vec::with_capacity(total_verts);
    let mut colors = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);

    let size = half_extent * 2.0;

    for iz in 0..verts_per_side {
        let v = iz as f32 / subdivisions as f32;
        let z = -half_extent + v * size;
        for ix in 0..verts_per_side {
            let u = ix as f32 / subdivisions as f32;
            let x = -half_extent + u * size;

            let y = terrain_height(x, z);
            let n = terrain_normal(x, z);

            positions.push([x, y, z]);
            normals.push([n.x, n.y, n.z]);
            uvs.push([u, v]);
            colors.push(terrain_vertex_color(x, y, z, n));
        }
    }

    for iz in 0..subdivisions {
        for ix in 0..subdivisions {
            let i0 = iz * (subdivisions + 1) + ix;
            let i1 = i0 + 1;
            let i2 = i0 + (subdivisions + 1);
            let i3 = i2 + 1;

            // Counter-clockwise winding viewed from +Y
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Computes a natural linear RGBA vertex tint from elevation, slope, and subtle
/// contour lines so undulating slopes read clearly without external textures.
fn terrain_vertex_color(x: f32, y: f32, z: f32, normal: Vec3) -> [f32; 4] {
    let height_t = ((y + 0.8) / 2.0).clamp(0.0, 1.0);
    let slope_t = (1.0 - normal.y).clamp(0.0, 0.35) / 0.35;

    // Linear-space meadow palette: cool mossy hollows -> warm sunlit grass -> earthy rocky slopes
    let hollow = Vec3::new(0.11, 0.22, 0.12);
    let crest = Vec3::new(0.28, 0.42, 0.16);
    let rocky = Vec3::new(0.32, 0.27, 0.20);

    let grass = hollow.lerp(crest, height_t);
    let mut rgb = grass.lerp(rocky, slope_t * 0.55);

    // Subtle elevation contour bands + soft spatial patchiness for motion readability
    let contour = ((y * 5.0).sin().abs()).powf(8.0);
    let patch = 0.5 + 0.5 * (x * 0.7).sin() * (z * 0.7).cos();
    rgb *= 0.92 + 0.08 * patch + 0.05 * contour;

    [rgb.x, rgb.y, rgb.z, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_height_is_finite_and_bounded() {
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for ix in -18..=18 {
            for iz in -18..=18 {
                let y = terrain_height(ix as f32, iz as f32);
                assert!(y.is_finite(), "non-finite height at ({ix}, {iz})");
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        assert!(min_y > -1.5 && max_y < 2.0, "unexpected range [{min_y}, {max_y}]");
        assert!(max_y - min_y > 0.8, "terrain should have noticeable relief");
    }

    #[test]
    fn terrain_normal_is_unit_and_orthogonal_to_tangents() {
        let samples = [(0.0, 0.0), (2.6, 1.8), (-3.2, 2.4), (5.0, -4.0)];
        let h = 1e-3;
        for (x, z) in samples {
            let n = terrain_normal(x, z);
            assert!((n.length() - 1.0).abs() < 1e-4, "normal must be unit length");
            assert!(n.y > 0.65, "normal should point generally upward, got {n:?}");

            let tx = Vec3::new(2.0 * h, terrain_height(x + h, z) - terrain_height(x - h, z), 0.0)
                .normalize();
            let tz = Vec3::new(0.0, terrain_height(x, z + h) - terrain_height(x, z - h), 2.0 * h)
                .normalize();
            assert!(n.dot(tx).abs() < 1e-3, "normal not orthogonal to X tangent");
            assert!(n.dot(tz).abs() < 1e-3, "normal not orthogonal to Z tangent");
        }
    }

    #[test]
    fn raycast_terrain_hits_surface_accurately() {
        let ray = Ray3d::new(Vec3::new(2.0, 10.0, -1.5), Dir3::from_xyz(0.15, -1.0, 0.1).unwrap());
        let hit = raycast_terrain(ray, 30.0).expect("downward ray must intersect terrain");
        let expected_y = terrain_height(hit.x, hit.z);
        assert!(
            (hit.y - expected_y).abs() < 1e-3,
            "hit y={} differs from terrain_height={}",
            hit.y,
            expected_y
        );
    }

    #[test]
    fn build_terrain_mesh_has_valid_topology() {
        let subs = 16;
        let mesh = build_terrain_mesh(10.0, subs);
        let expected_verts = ((subs + 1) * (subs + 1)) as usize;
        let expected_indices = (subs * subs * 6) as usize;

        assert_eq!(mesh.count_vertices(), expected_verts);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        assert_eq!(indices.len(), expected_indices);
        assert!(indices.iter().all(|&i| (i as usize) < expected_verts));
    }
}
