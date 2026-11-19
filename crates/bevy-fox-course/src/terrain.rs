//! Smooth 56m x 56m analytic undulating meadow with rolling hills, stepping
//! mounds, and smooth Gaussian grading around the 8 show-jump corridors.
//!
//! All height queries, normals, raycasts, obstacle foundations, jump landing
//! predictions, and foot-placement IK share the exact same pure-Rust function.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::obstacles::COURSE_SPECS;

/// Half-width of the square terrain patch in meters (total size = 56m x 56m).
pub const TERRAIN_HALF_EXTENT: f32 = 28.0;

/// Radius within which the fox stays during manual steering and auto-course demo.
pub const PLAY_AREA_RADIUS: f32 = 23.0;

/// Smooth Gaussian stepping mounds `(x, z, amplitude, sigma)` placed in the
/// open meadow zones between jump lanes so individual legs visibly adapt to
/// local relief during roaming and approach turns.
const MOUNDS: [(f32, f32, f32, f32); 8] = [
    (-2.5, 6.2, 0.44, 1.85),
    (6.8, 3.2, 0.48, 1.95),
    (5.2, -5.8, 0.46, 1.80),
    (-7.2, -4.2, 0.42, 1.90),
    (0.0, 17.2, 0.54, 2.30),
    (17.5, -1.5, 0.56, 2.40),
    (-17.8, -2.5, 0.52, 2.35),
    (-6.0, -17.5, 0.50, 2.20),
];

/// Broad rolling meadow hills without local mounds or jump-pad grading.
#[inline]
pub fn broad_hill_height(x: f32, z: f32) -> f32 {
    0.56 * (x * 0.14).sin() * (z * 0.13).cos()
        + 0.28 * (x * 0.28 + z * 0.24 + 0.55).sin()
        + 0.08 * (x * 0.58 - z * 0.52).cos()
}

/// Evaluates the terrain elevation `y` (in meters) at world coordinates `(x, z)`.
#[inline]
pub fn terrain_height(x: f32, z: f32) -> f32 {
    let mut h = broad_hill_height(x, z);

    // Local stepping mounds for expressive quadruped leg articulation.
    for &(mx, mz, amp, sigma) in &MOUNDS {
        let dx = x - mx;
        let dz = z - mz;
        let d2 = dx * dx + dz * dz;
        let denom = 2.0 * sigma * sigma;
        if d2 < denom * 9.0 {
            h += amp * (-d2 / denom).exp();
        }
    }

    // Smooth Gaussian grading around each show-jump takeoff/landing corridor so
    // uprights seat cleanly and takeoff/landing turf stays gently cambered while
    // retaining the natural hill elevation of each fence's location.
    let mut pad_weight_sum = 0.0_f32;
    let mut pad_target_sum = 0.0_f32;
    for spec in &COURSE_SPECS {
        let dx = x - spec.x;
        let dz = z - spec.z;
        if dx * dx + dz * dz > 169.0 {
            continue;
        }
        let (sin_y, cos_y) = spec.yaw.sin_cos();
        let dn = dx * sin_y + dz * cos_y;
        let dt = dx * cos_y - dz * sin_y;

        let sigma_n = 3.1 + spec.spread * 0.5;
        let sigma_t = spec.wing_half_width + 1.5;
        let exponent = -((dn / sigma_n).powi(2) + (dt / sigma_t).powi(2));
        let w = exponent.exp();
        if w > 1e-4 {
            let pad_y = broad_hill_height(spec.x, spec.z);
            pad_weight_sum += w;
            pad_target_sum += w * pad_y;
        }
    }

    if pad_weight_sum > 1e-4 {
        let blended_pad_y = pad_target_sum / pad_weight_sum;
        let alpha = pad_weight_sum.clamp(0.0, 0.94);
        h = h + (blended_pad_y - h) * alpha;
    }

    h
}

/// Computes the unit surface normal at world coordinates `(x, z)` via central
/// finite differences of [`terrain_height`].
#[inline]
pub fn terrain_normal(x: f32, z: f32) -> Vec3 {
    const EPS: f32 = 0.025;
    let dh_dx = (terrain_height(x + EPS, z) - terrain_height(x - EPS, z)) / (2.0 * EPS);
    let dh_dz = (terrain_height(x, z + EPS) - terrain_height(x, z - EPS)) / (2.0 * EPS);
    Vec3::new(-dh_dx, 1.0, -dh_dz).normalize()
}

/// Intersects a world-space [`Ray3d`] against the analytic terrain surface using
/// coarse raymarching followed by binary-search refinement.
pub fn raycast_terrain(ray: Ray3d, max_distance: f32) -> Option<Vec3> {
    const STEP: f32 = 0.35;
    const REFINE_ITERS: usize = 8;

    let origin = ray.origin;
    let dir = *ray.direction;

    let mut prev_t = 0.0_f32;
    let start_diff = origin.y - terrain_height(origin.x, origin.z);
    if start_diff <= 0.0 {
        return Some(Vec3::new(
            origin.x,
            terrain_height(origin.x, origin.z),
            origin.z,
        ));
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

            // Counter-clockwise winding viewed from +Y.
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

/// Computes a natural linear-space meadow vertex tint with subtle mown-turf
/// jump-lane highlights and soft arena perimeter shading.
fn terrain_vertex_color(x: f32, y: f32, z: f32, normal: Vec3) -> [f32; 4] {
    let height_t = ((y + 0.9) / 2.2).clamp(0.0, 1.0);
    let slope_t = (1.0 - normal.y).clamp(0.0, 0.30) / 0.30;

    let hollow = Vec3::new(0.11, 0.23, 0.12);
    let crest = Vec3::new(0.28, 0.43, 0.16);
    let earthy = Vec3::new(0.32, 0.27, 0.20);
    let lane_turf = Vec3::new(0.24, 0.39, 0.16);

    let grass = hollow.lerp(crest, height_t);
    let mut rgb = grass.lerp(earthy, slope_t * 0.50);

    // Subtle groomed-turf tint under each jump approach corridor.
    let mut lane_blend = 0.0_f32;
    for spec in &COURSE_SPECS {
        let dx = x - spec.x;
        let dz = z - spec.z;
        if dx * dx + dz * dz > 36.0 {
            continue;
        }
        let (sin_y, cos_y) = spec.yaw.sin_cos();
        let dn = dx * sin_y + dz * cos_y;
        let dt = dx * cos_y - dz * sin_y;
        let w = (-((dn / 2.6).powi(2) + (dt / (spec.rail_half_width + 0.4)).powi(2))).exp();
        lane_blend = lane_blend.max(w);
    }
    rgb = rgb.lerp(lane_turf, lane_blend * 0.45);

    // Subtle contour bands + spatial grass variation for motion readability.
    let contour = (y * 4.5).sin().abs().powf(8.0);
    let patch = 0.5 + 0.5 * (x * 0.55).sin() * (z * 0.55).cos();
    rgb *= 0.92 + 0.08 * patch + 0.04 * contour;

    // Gentle darker border outside the circular equestrian arena.
    let r = (x * x + z * z).sqrt();
    if r > PLAY_AREA_RADIUS {
        let rim = ((r - PLAY_AREA_RADIUS) / 4.5).clamp(0.0, 1.0);
        rgb *= 1.0 - 0.26 * rim;
    }

    [rgb.x, rgb.y, rgb.z, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_height_is_finite_and_bounded() {
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for ix in -28..=28 {
            for iz in -28..=28 {
                let y = terrain_height(ix as f32, iz as f32);
                assert!(y.is_finite(), "non-finite height at ({ix}, {iz})");
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        assert!(
            min_y > -1.5 && max_y < 2.2,
            "unexpected range [{min_y}, {max_y}]"
        );
        assert!(max_y - min_y > 0.9, "terrain should have noticeable relief");
    }

    #[test]
    fn terrain_normal_is_unit_and_orthogonal_to_tangents() {
        let samples = [(0.0, 0.0), (-2.5, 6.2), (6.8, 3.2), (-8.5, 11.0), (12.0, -8.0)];
        let h = 1e-3;
        for (x, z) in samples {
            let n = terrain_normal(x, z);
            assert!((n.length() - 1.0).abs() < 1e-4, "normal must be unit length");
            assert!(n.y > 0.70, "normal should point upward, got {n:?}");

            let tx = Vec3::new(
                2.0 * h,
                terrain_height(x + h, z) - terrain_height(x - h, z),
                0.0,
            )
            .normalize();
            let tz = Vec3::new(
                0.0,
                terrain_height(x, z + h) - terrain_height(x, z - h),
                2.0 * h,
            )
            .normalize();
            assert!(n.dot(tx).abs() < 1e-3, "normal not orthogonal to X tangent");
            assert!(n.dot(tz).abs() < 1e-3, "normal not orthogonal to Z tangent");
        }
    }

    #[test]
    fn jump_corridors_are_smoothly_graded() {
        for spec in &COURSE_SPECS {
            let n = terrain_normal(spec.x, spec.z);
            assert!(
                n.y > 0.95,
                "jump '{}' pad at ({}, {}) should be smoothly graded, got normal {:?}",
                spec.name,
                spec.x,
                spec.z,
                n
            );
        }
    }

    #[test]
    fn raycast_terrain_hits_surface_accurately() {
        let ray = Ray3d::new(
            Vec3::new(3.0, 12.0, -2.5),
            Dir3::from_xyz(0.12, -1.0, 0.14).unwrap(),
        );
        let hit = raycast_terrain(ray, 40.0).expect("downward ray must intersect terrain");
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
        let mesh = build_terrain_mesh(TERRAIN_HALF_EXTENT, subs);
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
