//! Procedural 8-jump equestrian show-jumping course, bidirectional OBB helpers,
//! and grounded 2D safety-capsule collision pushout.
//!
//! All fences are constructed purely from shared Bevy primitives (striped poles,
//! winged uprights grounded to `terrain_height`, lattice wings, top flags, and
//! ground line poles) with zero extra glTF or texture assets.

use std::f32::consts::{FRAC_PI_2, TAU};

use bevy::prelude::*;

use crate::terrain::{terrain_height, PLAY_AREA_RADIUS};

/// Visual construction style of each show-jumping obstacle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpStyle {
    CrossRail,
    Vertical,
    BrushBox,
    Oxer,
    TripleBar,
}

/// Static compile-time specification for one show-jump fence on the course.
#[derive(Debug, Clone, Copy)]
pub struct ObstacleSpec {
    pub name: &'static str,
    pub x: f32,
    pub z: f32,
    /// Orientation yaw (radians): `normal = (sin(yaw), cos(yaw))`.
    pub yaw: f32,
    pub rail_height: f32,
    pub rail_half_width: f32,
    pub wing_half_width: f32,
    /// Front-to-back rail spread (0.0 for verticals and cross-rails).
    pub spread: f32,
    pub min_jump_speed: f32,
    pub max_approach_deg: f32,
    pub style: JumpStyle,
    /// Accent color `[r, g, b]` in sRGB for alternating pole stripes and wings.
    pub accent_rgb: [f32; 3],
}

/// Eight bidirectional horse-show jumps arranged in a flowing arena circuit
/// inside `PLAY_AREA_RADIUS = 23.0 m`.
pub const COURSE_SPECS: [ObstacleSpec; 8] = [
    ObstacleSpec {
        name: "Starter Cross-Rail",
        x: -8.5,
        z: 11.0,
        yaw: 1.57,
        rail_height: 0.48,
        rail_half_width: 1.55,
        wing_half_width: 2.05,
        spread: 0.0,
        min_jump_speed: 2.6,
        max_approach_deg: 36.0,
        style: JumpStyle::CrossRail,
        accent_rgb: [0.16, 0.45, 0.82],
    },
    ObstacleSpec {
        name: "Meadow Vertical",
        x: 3.5,
        z: 11.5,
        yaw: 1.85,
        rail_height: 0.64,
        rail_half_width: 1.65,
        wing_half_width: 2.15,
        spread: 0.0,
        min_jump_speed: 3.1,
        max_approach_deg: 34.0,
        style: JumpStyle::Vertical,
        accent_rgb: [0.80, 0.18, 0.18],
    },
    ObstacleSpec {
        name: "Hunter Brush-Box",
        x: 13.5,
        z: 4.5,
        yaw: 2.95,
        rail_height: 0.68,
        rail_half_width: 1.70,
        wing_half_width: 2.20,
        spread: 0.35,
        min_jump_speed: 3.2,
        max_approach_deg: 34.0,
        style: JumpStyle::BrushBox,
        accent_rgb: [0.16, 0.48, 0.24],
    },
    ObstacleSpec {
        name: "Ascending Oxer",
        x: 12.0,
        z: -8.0,
        yaw: -2.35,
        rail_height: 0.76,
        rail_half_width: 1.75,
        wing_half_width: 2.25,
        spread: 0.62,
        min_jump_speed: 3.6,
        max_approach_deg: 32.0,
        style: JumpStyle::Oxer,
        accent_rgb: [0.90, 0.68, 0.12],
    },
    ObstacleSpec {
        name: "Hillcrest Vertical",
        x: 0.5,
        z: -13.5,
        yaw: -1.45,
        rail_height: 0.72,
        rail_half_width: 1.65,
        wing_half_width: 2.15,
        spread: 0.0,
        min_jump_speed: 3.4,
        max_approach_deg: 32.0,
        style: JumpStyle::Vertical,
        accent_rgb: [0.14, 0.56, 0.44],
    },
    ObstacleSpec {
        name: "Grand Prix Oxer",
        x: -12.0,
        z: -8.5,
        yaw: -0.25,
        rail_height: 0.86,
        rail_half_width: 1.80,
        wing_half_width: 2.30,
        spread: 0.72,
        min_jump_speed: 4.3,
        max_approach_deg: 30.0,
        style: JumpStyle::Oxer,
        accent_rgb: [0.15, 0.24, 0.62],
    },
    ObstacleSpec {
        name: "Narrow Skinny Gate",
        x: -2.0,
        z: -1.5,
        yaw: 0.78,
        rail_height: 0.66,
        rail_half_width: 1.00,
        wing_half_width: 1.50,
        spread: 0.0,
        min_jump_speed: 3.2,
        max_approach_deg: 24.0,
        style: JumpStyle::Vertical,
        accent_rgb: [0.78, 0.28, 0.52],
    },
    ObstacleSpec {
        name: "Triple Bar Finale",
        x: -12.5,
        z: 3.0,
        yaw: 0.20,
        rail_height: 0.90,
        rail_half_width: 1.85,
        wing_half_width: 2.35,
        spread: 0.85,
        min_jump_speed: 4.6,
        max_approach_deg: 30.0,
        style: JumpStyle::TripleBar,
        accent_rgb: [0.84, 0.34, 0.12],
    },
];

/// Runtime geometric representation of one show-jumping obstacle.
#[derive(Debug, Clone)]
pub struct CourseObstacle {
    pub id: usize,
    pub name: &'static str,
    pub center_xz: Vec2,
    pub base_y: f32,
    pub yaw: f32,
    /// Unit vector perpendicular to the rail (`(sin(yaw), cos(yaw))`).
    pub normal_xz: Vec2,
    /// Unit vector parallel to the rail (`(cos(yaw), -sin(yaw))`, pointing left when facing `+normal`).
    pub tangent_xz: Vec2,
    pub rail_height: f32,
    pub rail_half_width: f32,
    pub wing_half_width: f32,
    pub spread: f32,
    pub min_jump_speed: f32,
    pub max_approach_cos: f32,
    pub style: JumpStyle,
}

/// Resource containing all runtime course obstacles for jump, avoidance, and
/// grounded collision queries.
#[derive(Resource, Debug, Clone)]
pub struct CourseObstacles {
    pub list: Vec<CourseObstacle>,
}

impl Default for CourseObstacles {
    fn default() -> Self {
        Self {
            list: build_course_obstacles(),
        }
    }
}

/// Constructs the runtime [`CourseObstacle`] list from [`COURSE_SPECS`].
pub fn build_course_obstacles() -> Vec<CourseObstacle> {
    COURSE_SPECS
        .iter()
        .enumerate()
        .map(|(id, spec)| {
            let (sin_y, cos_y) = spec.yaw.sin_cos();
            let normal_xz = Vec2::new(sin_y, cos_y).normalize();
            let tangent_xz = Vec2::new(cos_y, -sin_y).normalize();
            CourseObstacle {
                id,
                name: spec.name,
                center_xz: Vec2::new(spec.x, spec.z),
                base_y: terrain_height(spec.x, spec.z),
                yaw: spec.yaw,
                normal_xz,
                tangent_xz,
                rail_height: spec.rail_height,
                rail_half_width: spec.rail_half_width,
                wing_half_width: spec.wing_half_width,
                spread: spec.spread,
                min_jump_speed: spec.min_jump_speed,
                max_approach_cos: spec.max_approach_deg.to_radians().cos(),
                style: spec.style,
            }
        })
        .collect()
}

/// Transforms a world XZ position into a fence's local `(along_normal, along_tangent)`
/// coordinates centered at `obs.center_xz`.
#[inline]
pub fn local_fence_coords(obs: &CourseObstacle, pos_xz: Vec2) -> (f32, f32) {
    let rel = pos_xz - obs.center_xz;
    (rel.dot(obs.normal_xz), rel.dot(obs.tangent_xz))
}

/// Returns the world-space position of a fence wing upright (`side = +1.0` for
/// `+tangent_xz`, `-1.0` for `-tangent_xz`) at roughly fox eye height.
#[inline]
pub fn wing_world_position(obs: &CourseObstacle, side: f32) -> Vec3 {
    let wing_xz = obs.center_xz + obs.tangent_xz * (side * (obs.rail_half_width + 0.25));
    let ground_y = terrain_height(wing_xz.x, wing_xz.y);
    Vec3::new(wing_xz.x, ground_y + obs.rail_height * 0.85, wing_xz.y)
}

/// Pushes a grounded 2D circle (`pos_xz`, `radius`) out of every fence's solid
/// physical footprint (rails + winged uprights) so slow creep/walk input can
/// never clip through a jump. Disabled while airborne.
pub fn grounded_capsule_pushout(
    mut pos_xz: Vec2,
    radius: f32,
    obstacles: &[CourseObstacle],
) -> Vec2 {
    for _ in 0..2 {
        for obs in obstacles {
            let (along_n, along_t) = local_fence_coords(obs, pos_xz);
            let half_depth = 0.16 + obs.spread * 0.5;
            let half_w = obs.wing_half_width + 0.06;

            // Quick broadphase reject.
            if along_n.abs() > half_depth + radius + 0.25 || along_t.abs() > half_w + radius + 0.25
            {
                continue;
            }

            let clamped_n = along_n.clamp(-half_depth, half_depth);
            let clamped_t = along_t.clamp(-half_w, half_w);
            let dn = along_n - clamped_n;
            let dt = along_t - clamped_t;
            let dist_sq = dn * dn + dt * dt;

            if dist_sq < 1e-8 {
                // Center is strictly inside the fence OBB: push out along nearest face.
                let pen_n = half_depth - along_n.abs();
                let pen_t = half_w - along_t.abs();
                if pen_n <= pen_t {
                    let sign = if along_n >= 0.0 { 1.0 } else { -1.0 };
                    let push = pen_n + radius;
                    pos_xz += obs.normal_xz * (sign * push);
                } else {
                    let sign = if along_t >= 0.0 { 1.0 } else { -1.0 };
                    let push = pen_t + radius;
                    pos_xz += obs.tangent_xz * (sign * push);
                }
            } else if dist_sq < radius * radius {
                let dist = dist_sq.sqrt();
                let push = radius - dist;
                let push_n = dn / dist;
                let push_t = dt / dist;
                pos_xz += obs.normal_xz * (push_n * push) + obs.tangent_xz * (push_t * push);
            }
        }
    }
    pos_xz
}

/// Spawns all 8 procedural show-jump obstacles plus subtle arena boundary
/// markers using shared Bevy primitive meshes and lightweight materials.
pub fn spawn_course_obstacles(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    obstacles: &[CourseObstacle],
) {
    // Shared unit primitives scaled per component so geometry memory stays tiny.
    let unit_cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let unit_cyl = meshes.add(Cylinder::new(0.5, 1.0));

    let white_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.93),
        perceptual_roughness: 0.55,
        ..default()
    });
    let wood_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.46, 0.30, 0.17),
        perceptual_roughness: 0.85,
        ..default()
    });
    let brush_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.14, 0.34, 0.15),
        perceptual_roughness: 0.92,
        ..default()
    });
    let red_flag_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.86, 0.14, 0.14),
        perceptual_roughness: 0.60,
        ..default()
    });
    let white_flag_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.96, 0.97, 0.99),
        perceptual_roughness: 0.60,
        ..default()
    });

    for (obs, spec) in obstacles.iter().zip(COURSE_SPECS.iter()) {
        let accent_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(spec.accent_rgb[0], spec.accent_rgb[1], spec.accent_rgb[2]),
            perceptual_roughness: 0.48,
            ..default()
        });

        let rot_y = Quat::from_rotation_y(obs.yaw);
        let upright_h = obs.rail_height + 0.34;
        let rail_len = obs.rail_half_width * 2.0;

        // Grounded helper for local (along_t = local X, along_n = local Z) offsets.
        let to_world = |local_t: f32, local_n: f32| -> (f32, f32, f32) {
            let p = obs.center_xz + obs.tangent_xz * local_t + obs.normal_xz * local_n;
            (p.x, terrain_height(p.x, p.y), p.y)
        };

        // 1. Main standards & lattice wings on both left (+1) and right (-1) sides.
        let wing_span = (obs.wing_half_width - obs.rail_half_width).max(0.35);
        for &side in &[-1.0_f32, 1.0_f32] {
            let post_t = side * (obs.rail_half_width + 0.05);
            let spread_offsets: &[f32] = if obs.spread > 0.40 {
                &[-obs.spread * 0.5, obs.spread * 0.5]
            } else {
                &[0.0]
            };

            for &n_off in spread_offsets {
                let (px, py, pz) = to_world(post_t, n_off);
                let total_post_h = upright_h + 0.20;
                commands.spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(white_mat.clone()),
                    Transform::from_xyz(px, py + upright_h * 0.5 - 0.10, pz)
                        .with_rotation(rot_y)
                        .with_scale(Vec3::new(0.11, total_post_h, 0.11)),
                ));
                // Base foot shoe seated on turf.
                commands.spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(accent_mat.clone()),
                    Transform::from_xyz(px, py + 0.03, pz)
                        .with_rotation(rot_y)
                        .with_scale(Vec3::new(0.24, 0.08, 0.42)),
                ));
            }

            // Outer wing upright (slightly shorter than main standard).
            let outer_t = side * obs.wing_half_width;
            let outer_h = upright_h * 0.78;
            let (ox, oy, oz) = to_world(outer_t, 0.0);
            commands.spawn((
                Mesh3d(unit_cube.clone()),
                MeshMaterial3d(white_mat.clone()),
                Transform::from_xyz(ox, oy + outer_h * 0.5 - 0.10, oz)
                    .with_rotation(rot_y)
                    .with_scale(Vec3::new(0.09, outer_h + 0.20, 0.09)),
            ));

            // Two horizontal wing bars + accent panel between inner & outer posts.
            let mid_t = side * (obs.rail_half_width + wing_span * 0.5);
            let (mx, my, mz) = to_world(mid_t, 0.0);
            for frac in [0.28_f32, 0.62_f32] {
                commands.spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(accent_mat.clone()),
                    Transform::from_xyz(mx, my + outer_h * frac, mz)
                        .with_rotation(rot_y)
                        .with_scale(Vec3::new(wing_span, 0.07, 0.06)),
                ));
            }

            // Top show-jumping flag (red on right/-1, white on left/+1).
            let (fx, fy, fz) = to_world(post_t + side * 0.09, 0.0);
            let flag_mat = if side < 0.0 {
                red_flag_mat.clone()
            } else {
                white_flag_mat.clone()
            };
            commands.spawn((
                Mesh3d(unit_cube.clone()),
                MeshMaterial3d(flag_mat),
                Transform::from_xyz(fx, fy + upright_h + 0.05, fz)
                    .with_rotation(rot_y)
                    .with_scale(Vec3::new(0.16, 0.10, 0.03)),
            ));
        }

        // 2. Ground line pole resting on turf beneath the jump.
        spawn_striped_pole(
            commands,
            &unit_cyl,
            &white_mat,
            &accent_mat,
            obs,
            obs.base_y + 0.045,
            0.0,
            rail_len * 0.96,
            0.075,
            0.0,
        );

        // 3. Style-specific rails & filler elements.
        match obs.style {
            JumpStyle::CrossRail => {
                // Two angled poles crossing cleanly at center height `rail_height`.
                let tilt = 0.19_f32;
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height,
                    -0.05,
                    rail_len * 1.02,
                    0.09,
                    tilt,
                );
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height,
                    0.05,
                    rail_len * 1.02,
                    0.09,
                    -tilt,
                );
            }
            JumpStyle::Vertical => {
                let lower_h = (obs.rail_height * 0.56).max(0.28);
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + lower_h,
                    0.0,
                    rail_len,
                    0.09,
                    0.0,
                );
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height,
                    0.0,
                    rail_len,
                    0.095,
                    0.0,
                );
            }
            JumpStyle::BrushBox => {
                // Timber planter box + trimmed evergreen brush hedge + top pole.
                let box_h = obs.rail_height * 0.48;
                commands.spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(wood_mat.clone()),
                    Transform::from_xyz(obs.center_xz.x, obs.base_y + box_h * 0.5 - 0.04, obs.center_xz.y)
                        .with_rotation(rot_y)
                        .with_scale(Vec3::new(rail_len * 0.94, box_h + 0.08, 0.34)),
                ));
                let brush_h = obs.rail_height * 0.34;
                commands.spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(brush_mat.clone()),
                    Transform::from_xyz(
                        obs.center_xz.x,
                        obs.base_y + box_h + brush_h * 0.45,
                        obs.center_xz.y,
                    )
                    .with_rotation(rot_y)
                    .with_scale(Vec3::new(rail_len * 0.90, brush_h, 0.28)),
                ));
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height,
                    0.0,
                    rail_len,
                    0.09,
                    0.0,
                );
            }
            JumpStyle::Oxer => {
                // Ascending front-to-back twin rails separated by `obs.spread`.
                let half_s = obs.spread * 0.5;
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height * 0.50,
                    -half_s,
                    rail_len,
                    0.085,
                    0.0,
                );
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height - 0.08,
                    -half_s,
                    rail_len,
                    0.095,
                    0.0,
                );
                spawn_striped_pole(
                    commands,
                    &unit_cyl,
                    &white_mat,
                    &accent_mat,
                    obs,
                    obs.base_y + obs.rail_height,
                    half_s,
                    rail_len,
                    0.095,
                    0.0,
                );
            }
            JumpStyle::TripleBar => {
                // Three stepped ascending rails spanning `obs.spread`.
                let half_s = obs.spread * 0.5;
                let tiers = [
                    (-half_s, obs.rail_height - 0.24),
                    (0.0, obs.rail_height - 0.12),
                    (half_s, obs.rail_height),
                ];
                for (n_off, h) in tiers {
                    spawn_striped_pole(
                        commands,
                        &unit_cyl,
                        &white_mat,
                        &accent_mat,
                        obs,
                        obs.base_y + h,
                        n_off,
                        rail_len,
                        0.095,
                        0.0,
                    );
                }
            }
        }
    }

    // Subtle white arena perimeter dressage markers around PLAY_AREA_RADIUS.
    const MARKER_COUNT: usize = 24;
    let marker_r = PLAY_AREA_RADIUS + 0.7;
    for i in 0..MARKER_COUNT {
        let angle = (i as f32 / MARKER_COUNT as f32) * TAU;
        let mx = angle.sin() * marker_r;
        let mz = angle.cos() * marker_r;
        let my = terrain_height(mx, mz);
        commands.spawn((
            Mesh3d(unit_cube.clone()),
            MeshMaterial3d(white_mat.clone()),
            Transform::from_xyz(mx, my + 0.22, mz)
                .with_rotation(Quat::from_rotation_y(angle))
                .with_scale(Vec3::new(0.45, 0.44, 0.12)),
        ));
    }
}

/// Spawns one horizontal (or angled) striped equestrian jump rail built from 5
/// alternating color segments along the obstacle's tangent axis.
#[allow(clippy::too_many_arguments)]
fn spawn_striped_pole(
    commands: &mut Commands,
    cyl_mesh: &Handle<Mesh>,
    white_mat: &Handle<StandardMaterial>,
    accent_mat: &Handle<StandardMaterial>,
    obs: &CourseObstacle,
    center_y: f32,
    normal_offset: f32,
    total_len: f32,
    diameter: f32,
    roll_tilt: f32,
) {
    const SEGMENTS: usize = 5;
    let seg_len = total_len / SEGMENTS as f32;
    let base_rot = Quat::from_rotation_y(obs.yaw) * Quat::from_rotation_z(roll_tilt);
    // Align Cylinder (+Y axis) along local +X (tangent axis).
    let cyl_rot = base_rot * Quat::from_rotation_z(FRAC_PI_2);

    let rail_center = Vec3::new(
        obs.center_xz.x + obs.normal_xz.x * normal_offset,
        center_y,
        obs.center_xz.y + obs.normal_xz.y * normal_offset,
    );
    let axis_dir = base_rot * Vec3::X;

    for i in 0..SEGMENTS {
        let frac = (i as f32 + 0.5) / SEGMENTS as f32 - 0.5;
        let pos = rail_center + axis_dir * (frac * total_len);
        let mat = if i % 2 == 0 {
            accent_mat.clone()
        } else {
            white_mat.clone()
        };
        commands.spawn((
            Mesh3d(cyl_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_translation(pos)
                .with_rotation(cyl_rot)
                .with_scale(Vec3::new(diameter, seg_len + 0.005, diameter)),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn course_has_eight_well_spaced_obstacles_inside_arena() {
        let obs = build_course_obstacles();
        assert_eq!(obs.len(), 8);

        for o in &obs {
            assert!(
                o.center_xz.length() < PLAY_AREA_RADIUS - 4.0,
                "obstacle '{}' too close to arena boundary",
                o.name
            );
            assert!((o.normal_xz.length() - 1.0).abs() < 1e-4);
            assert!((o.tangent_xz.length() - 1.0).abs() < 1e-4);
            assert!(o.normal_xz.dot(o.tangent_xz).abs() < 1e-4);
        }

        for i in 0..obs.len() {
            for j in (i + 1)..obs.len() {
                let d = obs[i].center_xz.distance(obs[j].center_xz);
                assert!(
                    d > 8.0,
                    "obstacles '{}' and '{}' too close ({d:.2}m)",
                    obs[i].name,
                    obs[j].name
                );
            }
        }
    }

    #[test]
    fn local_fence_coords_matches_normal_and_tangent() {
        let obs = build_course_obstacles();
        let o = &obs[0];
        let pt = o.center_xz + o.normal_xz * 1.8 - o.tangent_xz * 0.75;
        let (dn, dt) = local_fence_coords(o, pt);
        assert!((dn - 1.8).abs() < 1e-4);
        assert!((dt + 0.75).abs() < 1e-4);
    }

    #[test]
    fn grounded_capsule_pushout_prevents_rail_clipping() {
        let obs = build_course_obstacles();
        let o = &obs[1];
        let radius = 0.38;

        // Place fox right inside the front edge of the rail.
        let penetrating = o.center_xz + o.normal_xz * 0.08;
        let resolved = grounded_capsule_pushout(penetrating, radius, &obs);
        let (dn, _) = local_fence_coords(o, resolved);
        assert!(
            dn.abs() >= 0.16 + radius - 1e-3,
            "fox should be pushed clear of rail, got dn={dn}"
        );

        // Far away position should be completely untouched.
        let far = Vec2::new(0.0, 0.0);
        let resolved_far = grounded_capsule_pushout(far, radius, &obs);
        assert!((resolved_far - far).length() < 1e-5);
    }
}
