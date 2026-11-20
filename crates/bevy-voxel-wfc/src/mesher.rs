//! Custom Bevy voxel mesher producing:
//! 1. An opaque terrain & foliage mesh with strict 6-face neighbor culling,
//!    baked 4-neighbor per-vertex ambient occlusion with diagonal flip, and
//!    subtle per-voxel linear color variation.
//! 2. A translucent recessed water mesh with depth-tinted glacial/coastal hues
//!    and clean diorama cross-section side faces.

use bevy::asset::RenderAssetUsages;
use bevy::color::{Color, ColorToComponents};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::rng::voxel_hash;
use crate::wfc::{Dir, Tile, WfcSolver, GRID_SIZE};

/// Half-extent used to center the `20 x 20` diorama horizontally at `(0, y, 0)`.
pub const DIORAMA_HALF_EXTENT: f32 = GRID_SIZE as f32 * 0.5;

/// Water surface height inside a top-surface water voxel (leaves a crisp bank lip).
const WATER_SURFACE_HEIGHT: f32 = 0.84;

/// Ambient occlusion brightness table for `ao_level` in `0..=3`.
const AO_CURVE: [f32; 4] = [0.48, 0.68, 0.85, 1.0];

struct FaceSpec {
    dir: Dir,
    normal: [f32; 3],
    /// Four corners in counter-clockwise winding order when viewed from outside.
    corners: [[u8; 3]; 4],
    /// Directional shading multiplier applied alongside baked AO.
    shade: f32,
}

const FACES: [FaceSpec; 6] = [
    FaceSpec {
        dir: Dir::PosX,
        normal: [1.0, 0.0, 0.0],
        corners: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        shade: 0.94,
    },
    FaceSpec {
        dir: Dir::NegX,
        normal: [-1.0, 0.0, 0.0],
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        shade: 0.89,
    },
    FaceSpec {
        dir: Dir::PosY,
        normal: [0.0, 1.0, 0.0],
        corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
        shade: 1.00,
    },
    FaceSpec {
        dir: Dir::NegY,
        normal: [0.0, -1.0, 0.0],
        corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        shade: 0.72,
    },
    FaceSpec {
        dir: Dir::PosZ,
        normal: [0.0, 0.0, 1.0],
        corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        shade: 0.92,
    },
    FaceSpec {
        dir: Dir::NegZ,
        normal: [0.0, 0.0, -1.0],
        corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        shade: 0.87,
    },
];

/// Pair of custom meshes generated from a [`WfcSolver`] snapshot.
pub struct VoxelMeshes {
    pub opaque: Mesh,
    pub water: Mesh,
}

/// Builds both the opaque terrain/foliage mesh and translucent water mesh.
pub fn build_diorama_meshes(solver: &WfcSolver) -> VoxelMeshes {
    VoxelMeshes {
        opaque: build_opaque_mesh(solver),
        water: build_water_mesh(solver),
    }
}

/// Builds the culled, AO-baked opaque voxel mesh for all collapsed Land & Foliage cells.
pub fn build_opaque_mesh(solver: &WfcSolver) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(16_384);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(16_384);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(16_384);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(16_384);
    let mut indices: Vec<u32> = Vec::with_capacity(24_576);

    let seed = solver.seed();

    for y in 0..GRID_SIZE {
        for z in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let Some(tile) = solver.tile_at(x, y, z) else {
                    continue;
                };
                if !tile.is_opaque() {
                    continue;
                }

                let jitter = voxel_hash(x, y, z, seed) - 0.5;

                for face in &FACES {
                    let (dx, dy, dz) = face.dir.offset();
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;

                    let is_outer_wall = nx < 0
                        || nx >= GRID_SIZE as isize
                        || ny < 0
                        || ny >= GRID_SIZE as isize
                        || nz < 0
                        || nz >= GRID_SIZE as isize;

                    if !is_outer_wall {
                        if let Some(ntile) = solver.tile_at(nx as usize, ny as usize, nz as usize) {
                            if ntile.is_opaque() {
                                continue;
                            }
                        }
                    }

                    let base_rgb = tile_linear_rgb(tile, face.dir, y, jitter);
                    let wall_factor = if is_outer_wall && face.dir != Dir::PosY {
                        0.93
                    } else {
                        1.0
                    };

                    let base_vertex = positions.len() as u32;
                    let mut ao_levels = [3u8; 4];

                    for (ci, corner) in face.corners.iter().enumerate() {
                        let ao_level = vertex_ao_level(solver, x, y, z, face.dir, *corner);
                        ao_levels[ci] = ao_level;
                        let shade = AO_CURVE[ao_level as usize] * face.shade * wall_factor;

                        let wx = (x + usize::from(corner[0])) as f32 - DIORAMA_HALF_EXTENT;
                        let wy = (y + usize::from(corner[1])) as f32;
                        let wz = (z + usize::from(corner[2])) as f32 - DIORAMA_HALF_EXTENT;

                        positions.push([wx, wy, wz]);
                        normals.push(face.normal);
                        uvs.push(match ci {
                            0 => [0.0, 0.0],
                            1 => [1.0, 0.0],
                            2 => [1.0, 1.0],
                            _ => [0.0, 1.0],
                        });
                        colors.push([
                            base_rgb[0] * shade,
                            base_rgb[1] * shade,
                            base_rgb[2] * shade,
                            1.0,
                        ]);
                    }

                    // Flip quad diagonal when opposing AO corners are brighter to remove
                    // triangle interpolation anisotropy artifacts.
                    if ao_levels[0] + ao_levels[2] > ao_levels[1] + ao_levels[3] {
                        indices.extend_from_slice(&[
                            base_vertex,
                            base_vertex + 1,
                            base_vertex + 2,
                            base_vertex,
                            base_vertex + 2,
                            base_vertex + 3,
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            base_vertex + 1,
                            base_vertex + 2,
                            base_vertex + 3,
                            base_vertex + 1,
                            base_vertex + 3,
                            base_vertex,
                        ]);
                    }
                }
            }
        }
    }

    finalize_mesh(positions, normals, uvs, colors, indices)
}

/// Builds the translucent water mesh with recessed top surfaces and outer glass-tank edges.
pub fn build_water_mesh(solver: &WfcSolver) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(2_048);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(2_048);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(2_048);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(2_048);
    let mut indices: Vec<u32> = Vec::with_capacity(3_072);

    let y_sea = solver.sea_level();
    let seed = solver.seed();

    for y in 0..GRID_SIZE {
        for z in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                if solver.tile_at(x, y, z) != Some(Tile::Water) {
                    continue;
                }

                let above_is_water =
                    y + 1 < GRID_SIZE && solver.tile_at(x, y + 1, z) == Some(Tile::Water);
                let top_h = if above_is_water {
                    1.0
                } else {
                    WATER_SURFACE_HEIGHT
                };

                // Top faces use total water depth below surface (deep water looks darker);
                // outer side walls use depth-from-surface above (seabed looks darker than top).
                let mut depth_below = 1usize;
                let mut cy = y;
                while cy > 0 && solver.tile_at(x, cy - 1, z) == Some(Tile::Water) {
                    depth_below += 1;
                    cy -= 1;
                }
                let mut depth_above = 1usize;
                let mut uy = y;
                while uy + 1 < GRID_SIZE && solver.tile_at(x, uy + 1, z) == Some(Tile::Water) {
                    depth_above += 1;
                    uy += 1;
                }
                let is_alpine_pond = y > y_sea;
                let jitter = (voxel_hash(x, y, z, seed ^ 0x44aa_1234) - 0.5) * 0.04;
                let top_rgba = water_linear_rgba(depth_below, is_alpine_pond, jitter);
                let side_rgba = water_linear_rgba(depth_above, is_alpine_pond, jitter);

                for face in &FACES {
                    let water_rgba = if face.dir == Dir::PosY {
                        top_rgba
                    } else {
                        side_rgba
                    };
                    let (dx, dy, dz) = face.dir.offset();
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;

                    let is_outer_wall = nx < 0
                        || nx >= GRID_SIZE as isize
                        || ny < 0
                        || ny >= GRID_SIZE as isize
                        || nz < 0
                        || nz >= GRID_SIZE as isize;

                    let emit = match face.dir {
                        Dir::PosY => !above_is_water,
                        Dir::NegY => {
                            is_outer_wall
                                || solver.tile_at(nx as usize, ny as usize, nz as usize).is_none()
                        }
                        Dir::PosX | Dir::NegX | Dir::PosZ | Dir::NegZ => {
                            if is_outer_wall {
                                true
                            } else {
                                // Only emit internal side face if bordering uncollapsed air-in-progress.
                                solver.tile_at(nx as usize, ny as usize, nz as usize).is_none()
                            }
                        }
                    };
                    if !emit {
                        continue;
                    }

                    let base_vertex = positions.len() as u32;
                    let shade = if is_outer_wall && face.dir != Dir::PosY {
                        face.shade * 0.90
                    } else {
                        face.shade
                    };

                    for (ci, corner) in face.corners.iter().enumerate() {
                        let wx = (x + usize::from(corner[0])) as f32 - DIORAMA_HALF_EXTENT;
                        let wy = y as f32 + if corner[1] == 1 { top_h } else { 0.0 };
                        let wz = (z + usize::from(corner[2])) as f32 - DIORAMA_HALF_EXTENT;

                        positions.push([wx, wy, wz]);
                        normals.push(face.normal);
                        uvs.push(match ci {
                            0 => [0.0, 0.0],
                            1 => [1.0, 0.0],
                            2 => [1.0, 1.0],
                            _ => [0.0, 1.0],
                        });
                        colors.push([
                            water_rgba[0] * shade,
                            water_rgba[1] * shade,
                            water_rgba[2] * shade,
                            water_rgba[3],
                        ]);
                    }

                    indices.extend_from_slice(&[
                        base_vertex,
                        base_vertex + 1,
                        base_vertex + 2,
                        base_vertex,
                        base_vertex + 2,
                        base_vertex + 3,
                    ]);
                }
            }
        }
    }

    finalize_mesh(positions, normals, uvs, colors, indices)
}

fn finalize_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
) -> Mesh {
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

/// Computes the 4-neighbor Minecraft/0fps ambient occlusion level (`0..=3`) for a
/// quad corner on `(x, y, z)` facing `dir`.
fn vertex_ao_level(
    solver: &WfcSolver,
    x: usize,
    y: usize,
    z: usize,
    dir: Dir,
    corner: [u8; 3],
) -> u8 {
    let (nx, ny, nz) = dir.offset();
    let ox = if corner[0] == 0 { -1isize } else { 1isize };
    let oy = if corner[1] == 0 { -1isize } else { 1isize };
    let oz = if corner[2] == 0 { -1isize } else { 1isize };

    let bx = x as isize + nx;
    let by = y as isize + ny;
    let bz = z as isize + nz;

    let (s1, s2, c) = match dir {
        Dir::PosX | Dir::NegX => (
            is_occluder(solver, bx, by + oy, bz),
            is_occluder(solver, bx, by, bz + oz),
            is_occluder(solver, bx, by + oy, bz + oz),
        ),
        Dir::PosY | Dir::NegY => (
            is_occluder(solver, bx + ox, by, bz),
            is_occluder(solver, bx, by, bz + oz),
            is_occluder(solver, bx + ox, by, bz + oz),
        ),
        Dir::PosZ | Dir::NegZ => (
            is_occluder(solver, bx + ox, by, bz),
            is_occluder(solver, bx, by + oy, bz),
            is_occluder(solver, bx + ox, by + oy, bz),
        ),
    };

    if s1 && s2 {
        0
    } else {
        3 - (u8::from(s1) + u8::from(s2) + u8::from(c))
    }
}

#[inline]
fn is_occluder(solver: &WfcSolver, x: isize, y: isize, z: isize) -> bool {
    if !(0..GRID_SIZE as isize).contains(&x)
        || !(0..GRID_SIZE as isize).contains(&y)
        || !(0..GRID_SIZE as isize).contains(&z)
    {
        return false;
    }
    solver
        .tile_at(x as usize, y as usize, z as usize)
        .is_some_and(Tile::is_opaque)
}

/// Converts an sRGB triple with slight per-voxel brightness jitter into linear RGB.
fn srgb_jittered(r: f32, g: f32, b: f32, jitter: f32, scale: f32) -> [f32; 3] {
    let v = jitter * scale;
    let linear = Color::srgb(
        (r + v).clamp(0.0, 1.0),
        (g + v * 0.9).clamp(0.0, 1.0),
        (b + v * 0.7).clamp(0.0, 1.0),
    )
    .to_linear();
    let [lr, lg, lb, _] = linear.to_f32_array();
    [lr, lg, lb]
}

fn tile_linear_rgb(tile: Tile, face: Dir, y: usize, jitter: f32) -> [f32; 3] {
    match tile {
        Tile::DeepRock => {
            let depth_dim = if y == 0 { -0.04 } else { 0.0 };
            srgb_jittered(0.28 + depth_dim, 0.30 + depth_dim, 0.34 + depth_dim, jitter, 0.05)
        }
        Tile::Subsoil => srgb_jittered(0.49, 0.34, 0.22, jitter, 0.06),
        Tile::Grass => {
            if face == Dir::PosY {
                // Sunlit alpine meadow green with gentle altitude coolness.
                let alt = ((y as f32 - 5.0) / 10.0).clamp(0.0, 1.0);
                srgb_jittered(0.35 - alt * 0.04, 0.66 - alt * 0.06, 0.24 + alt * 0.03, jitter, 0.07)
            } else {
                srgb_jittered(0.33, 0.54, 0.23, jitter, 0.06)
            }
        }
        Tile::CliffRock => {
            if face == Dir::PosY && y >= 15 {
                // High alpine crags pick up a crisp frost/snow dusting on top faces.
                srgb_jittered(0.84, 0.87, 0.91, jitter, 0.04)
            } else {
                let alt = ((y as f32 - 6.0) / 12.0).clamp(0.0, 1.0);
                srgb_jittered(0.52 + alt * 0.08, 0.54 + alt * 0.08, 0.58 + alt * 0.08, jitter, 0.07)
            }
        }
        Tile::Bush => srgb_jittered(0.20, 0.49, 0.20, jitter, 0.08),
        Tile::TrunkBase | Tile::TrunkMid => srgb_jittered(0.42, 0.27, 0.16, jitter, 0.05),
        Tile::CanopyCore => srgb_jittered(0.21, 0.47, 0.19, jitter, 0.06),
        Tile::LeafTop => srgb_jittered(0.32, 0.63, 0.24, jitter, 0.08),
        Tile::LeafPosX | Tile::LeafNegX | Tile::LeafPosZ | Tile::LeafNegZ => {
            srgb_jittered(0.25, 0.55, 0.21, jitter, 0.08)
        }
        Tile::Air | Tile::Water => [1.0, 1.0, 1.0],
    }
}

fn water_linear_rgba(depth: usize, is_alpine_pond: bool, jitter: f32) -> [f32; 4] {
    let t = ((depth.saturating_sub(1)) as f32 / 4.0).clamp(0.0, 1.0);
    let (r, g, b, a) = if is_alpine_pond {
        // Crystal-clear glacial turquoise for elevated mountain ponds.
        (
            0.18 - t * 0.05 + jitter,
            0.74 - t * 0.12 + jitter,
            0.84 - t * 0.06,
            0.76 + t * 0.08,
        )
    } else {
        // Shallow coastal aqua transitioning into deeper oceanic teal.
        (
            0.16 - t * 0.09 + jitter,
            0.62 - t * 0.24 + jitter,
            0.78 - t * 0.16,
            0.72 + t * 0.15,
        )
    };
    let linear = Color::srgba(
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        a.clamp(0.0, 1.0),
    )
    .to_linear();
    linear.to_f32_array()
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec3;
    use bevy::mesh::{Indices, Mesh, VertexAttributeValues};

    use super::{build_diorama_meshes, FACES};
    use crate::wfc::WfcSolver;

    #[test]
    fn face_corner_windings_match_outward_normals() {
        for face in &FACES {
            let v0 = Vec3::new(
                f32::from(face.corners[0][0]),
                f32::from(face.corners[0][1]),
                f32::from(face.corners[0][2]),
            );
            let v1 = Vec3::new(
                f32::from(face.corners[1][0]),
                f32::from(face.corners[1][1]),
                f32::from(face.corners[1][2]),
            );
            let v2 = Vec3::new(
                f32::from(face.corners[2][0]),
                f32::from(face.corners[2][1]),
                f32::from(face.corners[2][2]),
            );
            let computed = (v1 - v0).cross(v2 - v0).normalize();
            let expected = Vec3::from_array(face.normal);
            assert!(
                (computed - expected).length() < 1e-5,
                "CCW winding mismatch for {:?}: got {computed:?}, expected {expected:?}",
                face.dir
            );
        }
    }

    #[test]
    fn mesher_produces_valid_non_empty_indices_and_unit_normals() {
        let mut solver = WfcSolver::new("alpine-42");
        solver.solve_all();
        let meshes = build_diorama_meshes(&solver);

        validate_mesh(&meshes.opaque, "opaque", true);
        validate_mesh(&meshes.water, "water", false);
    }

    fn validate_mesh(mesh: &Mesh, label: &str, must_be_non_empty: bool) {
        let vert_count = mesh.count_vertices();
        if must_be_non_empty {
            assert!(vert_count > 0, "{label} mesh must have vertices");
        }
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("{label} mesh missing U32 indices");
        };
        assert_eq!(indices.len() % 3, 0, "{label} indices must be triangle triples");
        if must_be_non_empty {
            assert!(!indices.is_empty(), "{label} mesh must have triangles");
        }
        for &idx in indices {
            assert!(
                (idx as usize) < vert_count,
                "{label} index {idx} out of bounds ({vert_count})"
            );
        }

        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("{label} mesh missing Float32x3 normals");
        };
        assert_eq!(normals.len(), vert_count);
        for &n in normals {
            let len = Vec3::from_array(n).length();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "{label} normal {n:?} is not unit length (len={len})"
            );
        }
    }
}
