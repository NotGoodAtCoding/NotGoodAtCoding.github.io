//! 6-stage pure-Rust procedural Inverse Kinematics (IK) for the 19-bone Fox rig.
//!
//! Runs in `PostUpdate` with zero rotation drift:
//! `restore_pre_ik_pose` (`.before(AnimationSystems)`) -> `AnimationSystems` ->
//! `apply_fox_ik` (`.after(AnimationSystems).before(TransformSystems::Propagate)`).
//!
//! Stages:
//! 1. `FoxBody` pelvis height, terrain/flight pitch, centripetal banking, crouch/coil/spring.
//! 2. Spine yaw curvature, phase-locked gallop bounce, and jump bascule arch (`b_Spine01_02`, `b_Spine02_03`).
//! 3. Cascaded tail counterbalance (`b_Tail01_012` -> `b_Tail03_014`).
//! 4. Roll-free priority-stack look-at IK (`b_Spine02_03` 15%, `b_Neck_04` 35%, `b_Head_05` 50%) + creep counter-pitch.
//! 5. Hybrid grounded terrain + airborne tuck/reach 4-leg analytical 2-bone IK.
//! 6. Color-coded debug gizmos (`G` key).

use std::f32::consts::PI;

use bevy::math::Affine3A;
use bevy::prelude::*;

use crate::controls::{GazeFocus, LocomotionState, PlayerIntent};
use crate::gait::GaitPosture;
use crate::jump::{ApproachVerdict, JumpPhase, JumpState, GATHER_DURATION};
use crate::obstacles::CourseObstacles;
use crate::terrain::{terrain_height, terrain_normal};

/// Uniform scale applied to `Fox.glb` (authored in centimeters, ~100 units tall).
pub const FOX_SCALE: f32 = 0.02;

/// Rest-pose vertical clearance of front wrist joint (`b_*Hand`) above ground (m).
const FRONT_PAW_HEIGHT: f32 = 6.69 * FOX_SCALE;

/// Rest-pose vertical clearance of hind toe joint (`b_*Foot02`) above ground (m).
const HIND_PAW_HEIGHT: f32 = 1.0 * FOX_SCALE;

/// Minimum reach ratio relative to total limb length `(l1 + l2)`.
const MIN_REACH_RATIO: f32 = 0.01;

/// Maximum reach ratio relative to total limb length `(l1 + l2)` to prevent hyperextension popping.
const MAX_REACH_RATIO: f32 = 0.995;

/// Runtime toggles shared between controls, HUD, and IK systems.
#[derive(Resource, Debug, Clone)]
pub struct IkSettings {
    pub enabled: bool,
    pub show_gizmos: bool,
}

impl Default for IkSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            show_gizmos: false,
        }
    }
}

/// Marker on the locomotion root entity (`x, y, z` + heading yaw).
#[derive(Component)]
pub struct FoxRoot;

/// Marker on the child entity holding `SceneRoot` where pelvis height,
/// terrain/flight pitch, and centripetal banking roll are applied.
#[derive(Component)]
pub struct FoxBody;

/// Smoothed per-fox procedural IK state updated each frame.
#[derive(Component, Debug, Clone)]
pub struct FoxIkState {
    pub pelvis_offset: f32,
    pub body_pitch: f32,
    pub body_roll: f32,
    pub spine_yaw: f32,
    pub tail_yaw: [f32; 3],
    pub tail_pitch: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
    pub leg_weights: [f32; 4],
    pub airborne_blend: f32,
}

impl Default for FoxIkState {
    fn default() -> Self {
        Self {
            pelvis_offset: 0.0,
            body_pitch: 0.0,
            body_roll: 0.0,
            spine_yaw: 0.0,
            tail_yaw: [0.0; 3],
            tail_pitch: 0.0,
            look_yaw: 0.0,
            look_pitch: 0.0,
            leg_weights: [1.0; 4],
            airborne_blend: 0.0,
        }
    }
}

/// Cached bone entities for a single 2-bone leg chain (with optional hind paw tip).
#[derive(Debug, Clone, Copy)]
pub struct LegChain {
    pub upper: Entity,
    pub lower: Entity,
    pub end: Entity,
    pub paw_tip: Option<Entity>,
    pub is_front: bool,
}

/// Cached 19-bone `Fox.glb` rig populated once on scene spawn (zero runtime string lookups).
#[derive(Component, Debug, Clone)]
pub struct FoxRig {
    pub spine1: Entity,
    pub spine2: Entity,
    pub neck: Entity,
    pub head: Entity,
    pub tail: [Entity; 3],
    pub legs: [LegChain; 4],
    /// Pre-IK animated local transforms restored before `AnimationSystems` each frame.
    animated_pose: Vec<(Entity, Transform)>,
}

/// Pure geometric result of the analytical 2-bone IK solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoneSolution {
    pub mid_pos: Vec3,
    pub end_pos: Vec3,
}

/// Pure analytical 2-bone IK solver using the Law of Cosines.
pub fn solve_two_bone_positions(
    root: Vec3,
    mid: Vec3,
    end: Vec3,
    target: Vec3,
    pole_dir: Vec3,
) -> Option<TwoBoneSolution> {
    let l1 = (mid - root).length();
    let l2 = (end - mid).length();
    if l1 < 1e-5 || l2 < 1e-5 {
        return None;
    }

    let total_len = l1 + l2;
    let min_dist = (l1 - l2).abs().max(total_len * MIN_REACH_RATIO) + 1e-4;
    let max_dist = (total_len * MAX_REACH_RATIO).max(min_dist);

    let to_target = target - root;
    let raw_dist = to_target.length();
    let dir = if raw_dist > 1e-5 {
        to_target / raw_dist
    } else {
        (end - root).normalize_or(Vec3::NEG_Y)
    };
    let dist = raw_dist.clamp(min_dist, max_dist);

    let cos_alpha = ((l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)).clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    let proj_pole = pole_dir - dir * pole_dir.dot(dir);
    let bend_dir = if proj_pole.length_squared() > 1e-6 {
        proj_pole.normalize()
    } else {
        let cur_bend = (mid - root) - dir * (mid - root).dot(dir);
        if cur_bend.length_squared() > 1e-6 {
            cur_bend.normalize()
        } else {
            dir.any_orthogonal_vector().normalize()
        }
    };

    let mid_pos = root + dir * (l1 * cos_alpha) + bend_dir * (l1 * sin_alpha);
    let end_pos = root + dir * dist;

    Some(TwoBoneSolution { mid_pos, end_pos })
}

/// Computes the continuous fox-local `(0, dy, dz)` paw offset (in meters, fox `+Z`
/// forward, `+Y` up) across normalized airborne flight progress `u in [0, 1]`.
///
/// - **Front legs**: tuck high under chest on ascent (`u < 0.52`), then extend
///   forward and downward to reach the landing turf (`u >= 0.52`).
/// - **Hind legs**: push off backward (`u < 0.24`), snap knees high over the
///   rail (`0.20..0.80`), then unfold slightly after front paws for staggered landing.
pub fn airborne_leg_offset(is_front: bool, u: f32) -> Vec3 {
    let u = u.clamp(0.0, 1.0);
    if is_front {
        // Smooth tuck crest around u = 0.28, transitioning cleanly at u = 0.52 into reach.
        let tuck_bell = if u < 0.52 {
            (PI * (u / 0.52)).sin().max(0.0).powf(0.85)
        } else {
            0.0
        };
        let reach_ramp = smoothstep(0.46, 0.88, u);
        let dy = 0.24 * tuck_bell - 0.12 * reach_ramp;
        let dz = -0.07 * tuck_bell + 0.22 * reach_ramp;
        Vec3::new(0.0, dy, dz)
    } else {
        // Push-off (0..0.24) -> high rail snap (0.20..0.80) -> staggered unfold (0.74..0.98).
        let push = if u < 0.24 {
            (PI * (u / 0.24)).sin().max(0.0)
        } else {
            0.0
        };
        let tuck = if (0.20..=0.80).contains(&u) {
            (PI * ((u - 0.20) / 0.60)).sin().max(0.0).powf(0.9)
        } else {
            0.0
        };
        let unfold = smoothstep(0.74, 0.98, u);
        let dy = -0.08 * push + 0.24 * tuck - 0.10 * unfold;
        let dz = -0.20 * push + 0.11 * tuck - 0.06 * unfold;
        Vec3::new(0.0, dy, dz)
    }
}

/// Discovers and caches the 19 IK-driven `Fox.glb` bone entities onto [`FoxBody`]
/// once the glTF scene's [`AnimationPlayer`] spawns.
pub fn bind_fox_rig(
    mut commands: Commands,
    added_players: Query<Entity, Added<AnimationPlayer>>,
    parents: Query<&ChildOf>,
    children_query: Query<&Children>,
    names: Query<&Name>,
    fox_bodies: Query<Entity, (With<FoxBody>, Without<FoxRig>)>,
) {
    for player_entity in &added_players {
        let mut current = player_entity;
        let mut body_entity = None;
        for _ in 0..16 {
            if fox_bodies.contains(current) {
                body_entity = Some(current);
                break;
            }
            if let Ok(parent) = parents.get(current) {
                current = parent.parent();
            } else {
                break;
            }
        }
        let Some(body_entity) = body_entity else {
            continue;
        };

        let mut stack = vec![body_entity];
        let mut by_name: Vec<(&str, Entity)> = Vec::with_capacity(32);
        while let Some(e) = stack.pop() {
            if let Ok(name) = names.get(e) {
                by_name.push((name.as_str(), e));
            }
            if let Ok(children) = children_query.get(e) {
                for child in children.iter() {
                    stack.push(child);
                }
            }
        }

        let find = |target: &str| -> Option<Entity> {
            by_name.iter().find(|(n, _)| *n == target).map(|(_, e)| *e)
        };

        let (
            Some(spine1),
            Some(spine2),
            Some(neck),
            Some(head),
            Some(tail1),
            Some(tail2),
            Some(tail3),
            Some(fl_u),
            Some(fl_l),
            Some(fl_e),
            Some(fr_u),
            Some(fr_l),
            Some(fr_e),
            Some(hl_u),
            Some(hl_l),
            Some(hl_e),
            Some(hl_t),
            Some(hr_u),
            Some(hr_l),
            Some(hr_e),
            Some(hr_t),
        ) = (
            find("b_Spine01_02"),
            find("b_Spine02_03"),
            find("b_Neck_04"),
            find("b_Head_05"),
            find("b_Tail01_012"),
            find("b_Tail02_013"),
            find("b_Tail03_014"),
            find("b_LeftUpperArm_09"),
            find("b_LeftForeArm_010"),
            find("b_LeftHand_011"),
            find("b_RightUpperArm_06"),
            find("b_RightForeArm_07"),
            find("b_RightHand_08"),
            find("b_LeftLeg01_015"),
            find("b_LeftLeg02_016"),
            find("b_LeftFoot01_017"),
            find("b_LeftFoot02_018"),
            find("b_RightLeg01_019"),
            find("b_RightLeg02_020"),
            find("b_RightFoot01_021"),
            find("b_RightFoot02_022"),
        )
        else {
            warn!("Fox.glb rig missing expected bone names; IK disabled for entity {body_entity:?}");
            continue;
        };

        let legs = [
            LegChain {
                upper: fl_u,
                lower: fl_l,
                end: fl_e,
                paw_tip: None,
                is_front: true,
            },
            LegChain {
                upper: fr_u,
                lower: fr_l,
                end: fr_e,
                paw_tip: None,
                is_front: true,
            },
            LegChain {
                upper: hl_u,
                lower: hl_l,
                end: hl_e,
                paw_tip: Some(hl_t),
                is_front: false,
            },
            LegChain {
                upper: hr_u,
                lower: hr_l,
                end: hr_e,
                paw_tip: Some(hr_t),
                is_front: false,
            },
        ];

        commands.entity(body_entity).insert((
            FoxRig {
                spine1,
                spine2,
                neck,
                head,
                tail: [tail1, tail2, tail3],
                legs,
                animated_pose: Vec::with_capacity(19),
            },
            FoxIkState::default(),
        ));
    }
}

/// Restores the previous frame's pre-IK animated pose before [`AnimationSystems`]
/// runs so un-keyframed bones never accumulate drift.
pub fn restore_pre_ik_pose(
    rig_query: Query<&FoxRig, With<FoxBody>>,
    mut transforms: Query<&mut Transform>,
) {
    for rig in &rig_query {
        for &(bone, saved) in &rig.animated_pose {
            if let Ok(mut tf) = transforms.get_mut(bone) {
                *tf = saved;
            }
        }
    }
}

/// Executes the 6-stage procedural IK pipeline in `PostUpdate` on top of Bevy's
/// freshly evaluated skeletal pose before `TransformSystems::Propagate`.
#[allow(clippy::too_many_arguments)]
pub fn apply_fox_ik(
    time: Res<Time>,
    settings: Res<IkSettings>,
    intent: Res<PlayerIntent>,
    loco: Res<LocomotionState>,
    posture: Res<GaitPosture>,
    jump: Res<JumpState>,
    gaze: Res<GazeFocus>,
    obstacles: Res<CourseObstacles>,
    parents: Query<&ChildOf>,
    mut rig_query: Query<(Entity, &mut FoxRig, &mut FoxIkState), With<FoxBody>>,
    mut transforms: Query<&mut Transform>,
    mut gizmos: Gizmos,
) {
    let dt = time.delta_secs().min(0.1);

    for (body_entity, mut rig, mut state) in &mut rig_query {
        // Snapshot pre-IK animated pose for next frame's `restore_pre_ik_pose`.
        let ik_bones = rig.ik_bone_entities();
        rig.animated_pose.clear();
        for bone in ik_bones {
            if let Ok(tf) = transforms.get(bone) {
                rig.animated_pose.push((bone, *tf));
            }
        }

        let Ok(body_parent) = parents.get(body_entity) else {
            continue;
        };
        let root_entity = body_parent.parent();
        let Ok(root_tf) = transforms.get(root_entity).copied() else {
            continue;
        };

        if !settings.enabled {
            state.pelvis_offset.smooth_nudge(&0.0, 14.0, dt);
            state.body_pitch.smooth_nudge(&0.0, 14.0, dt);
            state.body_roll.smooth_nudge(&0.0, 14.0, dt);
            state.look_yaw.smooth_nudge(&0.0, 14.0, dt);
            state.look_pitch.smooth_nudge(&0.0, 14.0, dt);

            if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
                body_tf.translation = Vec3::new(0.0, state.pelvis_offset, 0.0);
                body_tf.rotation = Quat::from_rotation_x(state.body_pitch)
                    * Quat::from_rotation_z(state.body_roll);
                body_tf.scale = Vec3::splat(FOX_SCALE);
            }
            if settings.show_gizmos {
                draw_course_gizmos(&mut gizmos, &intent, &loco, &jump, &obstacles);
            }
            continue;
        }

        // Smooth airborne transition blend so takeoff & touchdown never pop.
        let target_air = if jump.phase == JumpPhase::Airborne {
            1.0
        } else {
            0.0
        };
        let air_rate = if target_air > state.airborne_blend {
            18.0
        } else {
            14.0
        };
        state.airborne_blend.smooth_nudge(&target_air, air_rate, dt);
        let air_w = state.airborne_blend;

        // =====================================================================
        // Stage 1 — FoxBody Pelvis, Pitch & Centripetal Banking
        // =====================================================================
        let root_pos = root_tf.translation;
        let root_fwd = (root_tf.rotation * Vec3::Z).normalize_or(Vec3::Z);
        let root_left = (root_tf.rotation * Vec3::X).normalize_or(Vec3::X);

        let h_front = terrain_height(root_pos.x + root_fwd.x * 0.45, root_pos.z + root_fwd.z * 0.45);
        let h_back = terrain_height(root_pos.x - root_fwd.x * 0.45, root_pos.z - root_fwd.z * 0.45);
        let h_left = terrain_height(root_pos.x + root_left.x * 0.25, root_pos.z + root_left.z * 0.25);
        let h_right = terrain_height(root_pos.x - root_left.x * 0.25, root_pos.z - root_left.z * 0.25);

        // Positive X rotation tilts +Z downward; positive Z rotation tilts +X upward.
        let slope_pitch = (-((h_front - h_back) / 0.90).atan() * 0.72).clamp(-0.42, 0.42);
        let slope_roll = (((h_left - h_right) / 0.50).atan() * 0.50).clamp(-0.35, 0.35);

        let gather_t = if jump.phase == JumpPhase::Gather {
            (jump.phase_time / GATHER_DURATION).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let gather_pitch = -0.14 * (PI * gather_t).sin();
        let flight_pitch = (-(jump.current_vy.atan2(loco.speed.max(1.5))) * 0.78).clamp(-0.46, 0.46);

        let target_pitch = (slope_pitch * (1.0 - air_w) + flight_pitch * air_w + gather_pitch)
            .clamp(-0.52, 0.52);
        let target_roll =
            ((slope_roll + loco.bank_roll) * (1.0 - air_w * 0.85)).clamp(-0.44, 0.44);

        state.body_pitch.smooth_nudge(&target_pitch, 11.0, dt);
        state.body_roll.smooth_nudge(&target_roll, 11.0, dt);

        if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
            body_tf.translation = Vec3::ZERO;
            body_tf.rotation =
                Quat::from_rotation_x(state.body_pitch) * Quat::from_rotation_z(state.body_roll);
            body_tf.scale = Vec3::splat(FOX_SCALE);
        }

        // Measure lowest paw delta for grounded anti-hyperextension pelvis drop.
        let (body_affine_unshifted, _) = compute_world_pose(body_entity, &parents, &transforms);
        let body_inv_unshifted = body_affine_unshifted.inverse();

        let mut min_delta_y = f32::INFINITY;
        let mut max_delta_y = f32::NEG_INFINITY;
        for leg in &rig.legs {
            let paw_entity = leg.paw_tip.unwrap_or(leg.end);
            let (paw_affine, _) = compute_world_pose(paw_entity, &parents, &transforms);
            let paw_world: Vec3 = paw_affine.translation.into();
            let rest_h = if leg.is_front {
                FRONT_PAW_HEIGHT
            } else {
                HIND_PAW_HEIGHT
            };
            let paw_model_y = body_inv_unshifted.transform_point3(paw_world).y * FOX_SCALE;
            let anim_lift = (paw_model_y - rest_h).max(0.0);
            let ground_y = terrain_height(paw_world.x, paw_world.z);
            let planted_y = paw_world.y - anim_lift;
            let delta_y = (ground_y + rest_h) - planted_y;
            min_delta_y = min_delta_y.min(delta_y);
            max_delta_y = max_delta_y.max(delta_y);
        }

        let slope_flex_margin = -0.02 * smoothstep(0.03, 0.14, max_delta_y - min_delta_y);
        let terrain_pelvis_drop =
            ((min_delta_y.min(0.0) + slope_flex_margin).clamp(-0.30, 0.0)) * (1.0 - air_w);
        let crouch_drop = -0.14 * posture.creep_w * (1.0 - air_w);
        let sprint_lift = 0.03 * posture.sprint_w * (1.0 - air_w);
        let gather_coil = -0.11 * (PI * gather_t).sin();

        let target_pelvis =
            (terrain_pelvis_drop + crouch_drop + sprint_lift + gather_coil).clamp(-0.36, 0.08);
        state.pelvis_offset.smooth_nudge(&target_pelvis, 12.0, dt);

        // Add damped landing spring directly on top of smoothed pelvis offset so
        // impact compression responds instantaneously without double-smoothing lag.
        let final_pelvis_y = (state.pelvis_offset + jump.landing_spring_y).clamp(-0.40, 0.10);
        if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
            body_tf.translation = Vec3::new(0.0, final_pelvis_y, 0.0);
        }

        let (body_affine, body_world_rot) = compute_world_pose(body_entity, &parents, &transforms);
        let body_inv = body_affine.inverse();
        let fox_fwd = (body_world_rot * Vec3::Z).normalize_or(Vec3::Z);

        // =====================================================================
        // Stage 2 — Spine Yaw, Gallop Bounce & Jump Bascule Arch
        // =====================================================================
        let target_spine_yaw = (loco.turn_rate * 0.16).clamp(-0.34, 0.34) * (1.0 - air_w);
        state.spine_yaw.smooth_nudge(&target_spine_yaw, 10.0, dt);

        let gallop_s1 = posture.gallop_phase.sin() * 0.11 * posture.sprint_w * (1.0 - air_w);
        let gallop_s2 =
            (posture.gallop_phase - 0.55).sin() * 0.08 * posture.sprint_w * (1.0 - air_w);
        let bascule = (PI * jump.flight_u).sin().max(0.0).powf(1.4) * air_w;
        let creep_flat = 0.05 * posture.creep_w * (1.0 - air_w);

        let spine1_local = Quat::from_rotation_y(state.spine_yaw * 0.55)
            * Quat::from_rotation_x(gallop_s1 + 0.15 * bascule + creep_flat);
        let spine2_local = Quat::from_rotation_y(state.spine_yaw * 0.45)
            * Quat::from_rotation_x(gallop_s2 + 0.12 * bascule);

        apply_fox_local_delta(rig.spine1, spine1_local, body_world_rot, &parents, &mut transforms);
        apply_fox_local_delta(rig.spine2, spine2_local, body_world_rot, &parents, &mut transforms);

        // =====================================================================
        // Stage 3 — Cascaded Tail Counterbalance (`b_Tail01` -> `b_Tail03`)
        // =====================================================================
        // Tail points along `-Z`; positive local Y rotation sweeps `-Z` toward `-X`
        // (outward right during a left turn `turn_rate > 0`).
        let target_tail_yaw = (loco.turn_rate * 0.24).clamp(-0.55, 0.55);
        let tail_rates = [10.0_f32, 7.0, 5.0];
        let mut upstream = target_tail_yaw;
        for i in 0..3 {
            state.tail_yaw[i].smooth_nudge(&upstream, tail_rates[i], dt);
            upstream = state.tail_yaw[i];
        }

        // Positive local X lifts `-Z` tail up; negative lowers it.
        let sprint_flutter =
            0.055 * posture.sprint_w * (posture.gallop_phase * 1.4).sin() * (1.0 - air_w);
        let target_tail_pitch = -0.22 * posture.creep_w
            + 0.16 * posture.sprint_w
            + sprint_flutter
            + 0.20 * air_w * (0.45 - jump.flight_u);
        state.tail_pitch.smooth_nudge(&target_tail_pitch, 8.5, dt);

        let tail_yaw_fracs = [0.42_f32, 0.34, 0.26];
        let tail_pitch_fracs = [0.50_f32, 0.30, 0.20];
        for i in 0..3 {
            let delta_local = Quat::from_rotation_y(state.tail_yaw[i] * tail_yaw_fracs[i])
                * Quat::from_rotation_x(state.tail_pitch * tail_pitch_fracs[i]);
            apply_fox_local_delta(
                rig.tail[i],
                delta_local,
                body_world_rot,
                &parents,
                &mut transforms,
            );
        }

        // =====================================================================
        // Stage 4 — Roll-Free Priority-Stack Look-At IK + Creep Counter-Pitch
        // =====================================================================
        // Applied BEFORE front-leg IK because front shoulders descend from `b_Spine02_03`.
        let (head_affine, _) = compute_world_pose(rig.head, &parents, &transforms);
        let head_world_pos: Vec3 = head_affine.translation.into();

        let to_target = (gaze.target - head_world_pos).normalize_or(fox_fwd);
        let local_dir = body_world_rot.inverse() * to_target;
        let raw_yaw = local_dir.x.atan2(local_dir.z);
        let horiz = (local_dir.x * local_dir.x + local_dir.z * local_dir.z)
            .sqrt()
            .max(1e-4);
        let raw_pitch = (-local_dir.y).atan2(horiz);

        let rear_fade = 1.0 - smoothstep(1.35, 2.05, raw_yaw.abs());
        let target_look_yaw = raw_yaw.clamp(-1.08, 1.08) * rear_fade;
        let target_look_pitch = raw_pitch.clamp(-0.56, 0.50) * rear_fade;

        state.look_yaw.smooth_nudge(&target_look_yaw, 9.0, dt);
        state.look_pitch.smooth_nudge(&target_look_pitch, 9.0, dt);

        let total_look_local =
            Quat::from_rotation_y(state.look_yaw) * Quat::from_rotation_x(state.look_pitch);
        let look_chain = [(rig.spine2, 0.15_f32), (rig.neck, 0.35), (rig.head, 0.50)];
        for (bone, fraction) in look_chain {
            let delta_local = Quat::IDENTITY.slerp(total_look_local, fraction);
            apply_fox_local_delta(bone, delta_local, body_world_rot, &parents, &mut transforms);
        }

        // Creep lowered neck (+0.24) + level-eyes head counter-pitch (-0.20),
        // and aerodynamic stretched neck during Sprint.
        let neck_posture = 0.24 * posture.creep_w + 0.10 * posture.sprint_w * (1.0 - air_w);
        let head_counter = -0.20 * posture.creep_w - 0.08 * posture.sprint_w * (1.0 - air_w);
        if neck_posture.abs() > 1e-4 {
            apply_fox_local_delta(
                rig.neck,
                Quat::from_rotation_x(neck_posture),
                body_world_rot,
                &parents,
                &mut transforms,
            );
        }
        if head_counter.abs() > 1e-4 {
            apply_fox_local_delta(
                rig.head,
                Quat::from_rotation_x(head_counter),
                body_world_rot,
                &parents,
                &mut transforms,
            );
        }

        // =====================================================================
        // Stage 5 — Hybrid Grounded + Airborne 4-Leg Analytical 2-Bone IK
        // =====================================================================
        for (idx, leg) in rig.legs.iter().enumerate() {
            let (root_affine, _) = compute_world_pose(leg.upper, &parents, &transforms);
            let (mid_affine, _) = compute_world_pose(leg.lower, &parents, &transforms);
            let (end_affine, orig_end_world_rot) = compute_world_pose(leg.end, &parents, &transforms);

            let root_pos_w: Vec3 = root_affine.translation.into();
            let mid_pos_w: Vec3 = mid_affine.translation.into();
            let end_pos_w: Vec3 = end_affine.translation.into();

            let paw_pos_w: Vec3 = if let Some(tip) = leg.paw_tip {
                let (tip_affine, _) = compute_world_pose(tip, &parents, &transforms);
                tip_affine.translation.into()
            } else {
                end_pos_w
            };

            let rest_h = if leg.is_front {
                FRONT_PAW_HEIGHT
            } else {
                HIND_PAW_HEIGHT
            };
            let paw_model_y = body_inv.transform_point3(paw_pos_w).y * FOX_SCALE;
            let anim_lift = (paw_model_y - rest_h).max(0.0);

            let ground_y = terrain_height(paw_pos_w.x, paw_pos_w.z);
            let ground_n = terrain_normal(paw_pos_w.x, paw_pos_w.z);

            let target_paw_ground = Vec3::new(paw_pos_w.x, ground_y + rest_h + anim_lift, paw_pos_w.z);
            let stance_factor = (1.0 - smoothstep(0.04, 0.28, anim_lift)) * (1.0 - air_w);
            let penetrating = paw_pos_w.y < ground_y + rest_h;

            let desired_weight = if air_w > 0.05 {
                0.95
            } else if penetrating {
                1.0
            } else {
                0.32 + 0.68 * stance_factor
            };
            state.leg_weights[idx].smooth_nudge(&desired_weight, 16.0, dt);
            let weight = state.leg_weights[idx];

            let slope_align = Quat::from_rotation_arc(Vec3::Y, ground_n);
            let slope_delta = Quat::IDENTITY.slerp(slope_align, 0.45 * stance_factor);
            let grounded_end_target = target_paw_ground - slope_delta * (paw_pos_w - end_pos_w);

            // Blend smoothly toward analytical airborne tuck/reach target during jumps.
            let final_end_target = if air_w > 1e-3 {
                let local_offset = airborne_leg_offset(leg.is_front, jump.flight_u);
                let mut air_target = end_pos_w + body_world_rot * local_offset;
                let air_ground = terrain_height(air_target.x, air_target.z) + rest_h;
                if air_target.y < air_ground {
                    air_target.y = air_ground;
                }
                grounded_end_target.lerp(air_target, air_w)
            } else {
                grounded_end_target
            };

            let pole_dir = if leg.is_front { -fox_fwd } else { fox_fwd };
            let Some(sol) = solve_two_bone_positions(
                root_pos_w,
                mid_pos_w,
                end_pos_w,
                final_end_target,
                pole_dir,
            ) else {
                continue;
            };

            // 1. Rotate upper bone toward solved mid-joint position.
            let old_upper_dir = (mid_pos_w - root_pos_w).normalize_or(Vec3::NEG_Y);
            let new_upper_dir = (sol.mid_pos - root_pos_w).normalize_or(old_upper_dir);
            let upper_delta = Quat::from_rotation_arc(old_upper_dir, new_upper_dir);
            apply_world_rotation_delta(leg.upper, upper_delta, weight, &parents, &mut transforms);

            // 2. Aim lower bone toward clamped end-effector position.
            let (mid_after_affine, _) = compute_world_pose(leg.lower, &parents, &transforms);
            let (end_after_affine, _) = compute_world_pose(leg.end, &parents, &transforms);
            let cur_mid_w: Vec3 = mid_after_affine.translation.into();
            let cur_end_w: Vec3 = end_after_affine.translation.into();

            let old_lower_dir = (cur_end_w - cur_mid_w).normalize_or(Vec3::NEG_Y);
            let new_lower_dir = (sol.end_pos - cur_mid_w).normalize_or(old_lower_dir);
            let lower_delta = Quat::from_rotation_arc(old_lower_dir, new_lower_dir);
            apply_world_rotation_delta(leg.lower, lower_delta, weight, &parents, &mut transforms);

            // 3. Preserve paw orientation with grounded slope conformity.
            let desired_end_world_rot = (slope_delta * orig_end_world_rot).normalize();
            set_world_rotation_blended(
                leg.end,
                desired_end_world_rot,
                weight * (1.0 - air_w * 0.55),
                &parents,
                &mut transforms,
            );

            // Leg bone gizmos.
            if settings.show_gizmos {
                let (u_a, _) = compute_world_pose(leg.upper, &parents, &transforms);
                let (l_a, _) = compute_world_pose(leg.lower, &parents, &transforms);
                let (e_a, _) = compute_world_pose(leg.end, &parents, &transforms);
                let u_p: Vec3 = u_a.translation.into();
                let l_p: Vec3 = l_a.translation.into();
                let e_p: Vec3 = e_a.translation.into();

                let chain_color = if leg.is_front {
                    Color::srgb(0.20, 0.90, 1.0)
                } else {
                    Color::srgb(1.0, 0.58, 0.20)
                };
                gizmos.line(u_p, l_p, chain_color);
                gizmos.line(l_p, e_p, chain_color);
                gizmos.sphere(Isometry3d::from_translation(l_p), 0.035, chain_color);

                if let Some(tip) = leg.paw_tip {
                    let (t_a, _) = compute_world_pose(tip, &parents, &transforms);
                    let t_p: Vec3 = t_a.translation.into();
                    gizmos.line(e_p, t_p, chain_color);
                }

                let ground_pt = Vec3::new(paw_pos_w.x, ground_y, paw_pos_w.z);
                gizmos.sphere(
                    Isometry3d::from_translation(ground_pt),
                    0.042,
                    Color::srgb(0.25, 1.0, 0.45),
                );
                gizmos.line(
                    ground_pt,
                    ground_pt + ground_n * 0.20,
                    Color::srgb(0.25, 1.0, 0.45),
                );
            }
        }

        // =====================================================================
        // Stage 6 — Color-Coded Debug Gizmos (`G` key)
        // =====================================================================
        if settings.show_gizmos {
            // Spine & tail bone overlay.
            let s1: Vec3 = compute_world_pose(rig.spine1, &parents, &transforms)
                .0
                .translation
                .into();
            let s2: Vec3 = compute_world_pose(rig.spine2, &parents, &transforms)
                .0
                .translation
                .into();
            let nk: Vec3 = compute_world_pose(rig.neck, &parents, &transforms)
                .0
                .translation
                .into();
            let hd: Vec3 = compute_world_pose(rig.head, &parents, &transforms)
                .0
                .translation
                .into();
            let spine_col = Color::srgb(0.95, 0.85, 0.25);
            gizmos.line(s1, s2, spine_col);
            gizmos.line(s2, nk, spine_col);
            gizmos.line(nk, hd, spine_col);

            let t0: Vec3 = compute_world_pose(rig.tail[0], &parents, &transforms)
                .0
                .translation
                .into();
            let t1: Vec3 = compute_world_pose(rig.tail[1], &parents, &transforms)
                .0
                .translation
                .into();
            let t2: Vec3 = compute_world_pose(rig.tail[2], &parents, &transforms)
                .0
                .translation
                .into();
            let tail_col = Color::srgb(1.0, 0.45, 0.85);
            gizmos.line(t0, t1, tail_col);
            gizmos.line(t1, t2, tail_col);

            // Yellow look-at gaze line.
            gizmos.line(hd, gaze.target, Color::srgb(1.0, 0.92, 0.25));
            gizmos.sphere(
                Isometry3d::from_translation(gaze.target),
                0.06,
                Color::srgb(1.0, 0.92, 0.25),
            );

            draw_course_gizmos(&mut gizmos, &intent, &loco, &jump, &obstacles);
        }
    }
}

/// Renders steering vectors (cyan suggested, green blended, magenta avoidance),
/// obstacle approach gates (green/amber/muted), and the gold ballistic jump arc.
fn draw_course_gizmos(
    gizmos: &mut Gizmos,
    intent: &PlayerIntent,
    loco: &LocomotionState,
    jump: &JumpState,
    obstacles: &CourseObstacles,
) {
    let base = loco.position + Vec3::Y * 0.35;

    // Cyan: player's suggested direction.
    if intent.has_input {
        let s_end = base + Vec3::new(intent.suggested_dir.x, 0.0, intent.suggested_dir.y) * 1.9;
        gizmos.arrow(base, s_end, Color::srgb(0.15, 0.92, 1.0));
    }

    // Green: final blended steering vector.
    let b_end = base + Vec3::new(loco.blended_dir.x, 0.0, loco.blended_dir.y) * 2.2;
    gizmos.arrow(base, b_end, Color::srgb(0.20, 1.0, 0.40));

    // Magenta: wing-avoidance tangent flow when active.
    if loco.avoid_weight > 0.05 {
        let a_end = base + Vec3::new(loco.avoid_tangent.x, 0.0, loco.avoid_tangent.y) * 2.3;
        gizmos.arrow(base, a_end, Color::srgb(1.0, 0.22, 0.85));
    }

    // Obstacle gate boxes & bidirectional approach lines.
    for obs in &obstacles.list {
        let color = match jump.last_eval_status {
            Some((id, ApproachVerdict::CommitJump)) if id == obs.id => {
                Color::srgb(0.18, 1.0, 0.38)
            }
            Some((id, ApproachVerdict::ApproachingOk)) if id == obs.id => {
                Color::srgb(0.35, 0.95, 0.55)
            }
            Some((id, ApproachVerdict::RefuseAvoid)) if id == obs.id => {
                Color::srgb(1.0, 0.65, 0.12)
            }
            _ => Color::srgba(0.85, 0.88, 0.92, 0.35),
        };

        let y = obs.base_y + 0.12;
        let left_gate = obs.center_xz + obs.tangent_xz * (obs.rail_half_width - 0.24);
        let right_gate = obs.center_xz - obs.tangent_xz * (obs.rail_half_width - 0.24);
        let p_l = Vec3::new(left_gate.x, y, left_gate.y);
        let p_r = Vec3::new(right_gate.x, y, right_gate.y);
        let n3 = Vec3::new(obs.normal_xz.x, 0.0, obs.normal_xz.y) * 2.2;
        gizmos.line(p_l - n3, p_l + n3, color);
        gizmos.line(p_r - n3, p_r + n3, color);
        gizmos.line(p_l - n3, p_r - n3, color);
        gizmos.line(p_l + n3, p_r + n3, color);
    }

    // Gold parabolic trajectory arc during Gather & Airborne.
    if let Some(traj) = jump.trajectory {
        let gold = Color::srgb(1.0, 0.82, 0.18);
        const STEPS: usize = 16;
        let mut prev = traj.sample_pos(0.0);
        for i in 1..=STEPS {
            let t = traj.total_time * (i as f32 / STEPS as f32);
            let cur = traj.sample_pos(t);
            gizmos.line(prev, cur, gold);
            prev = cur;
        }
        gizmos.sphere(Isometry3d::from_translation(traj.apex_pos), 0.08, gold);
        gizmos.sphere(
            Isometry3d::from_translation(traj.landing_pos + Vec3::Y * 0.05),
            0.09,
            Color::srgb(0.20, 1.0, 0.45),
        );
    }
}

impl FoxRig {
    /// Returns all 19 bone entities modified by IK in hierarchy order.
    fn ik_bone_entities(&self) -> [Entity; 19] {
        [
            self.spine1,
            self.spine2,
            self.neck,
            self.head,
            self.tail[0],
            self.tail[1],
            self.tail[2],
            self.legs[0].upper,
            self.legs[0].lower,
            self.legs[0].end,
            self.legs[1].upper,
            self.legs[1].lower,
            self.legs[1].end,
            self.legs[2].upper,
            self.legs[2].lower,
            self.legs[2].end,
            self.legs[3].upper,
            self.legs[3].lower,
            self.legs[3].end,
        ]
    }
}

/// Applies a fox-body-local delta rotation onto `bone` by conjugating into world
/// space via `body_world_rot`.
#[inline]
fn apply_fox_local_delta(
    bone: Entity,
    delta_local_fox: Quat,
    body_world_rot: Quat,
    parents: &Query<&ChildOf>,
    transforms: &mut Query<&mut Transform>,
) {
    let delta_world = (body_world_rot * delta_local_fox * body_world_rot.inverse()).normalize();
    apply_world_rotation_delta(bone, delta_world, 1.0, parents, transforms);
}

/// Computes an entity's current-frame world [`Affine3A`] and world [`Quat`] by
/// walking `&ChildOf` ancestors on local `Transform`s (valid before `TransformSystems::Propagate`).
fn compute_world_pose(
    entity: Entity,
    parents: &Query<&ChildOf>,
    transforms: &Query<&mut Transform>,
) -> (Affine3A, Quat) {
    let mut chain = [Transform::IDENTITY; 16];
    let mut len = 0usize;
    let mut cur = entity;

    while len < chain.len() {
        if let Ok(tf) = transforms.get(cur) {
            chain[len] = *tf;
            len += 1;
        }
        if let Ok(parent) = parents.get(cur) {
            cur = parent.parent();
        } else {
            break;
        }
    }

    let mut world_affine = Affine3A::IDENTITY;
    let mut world_rot = Quat::IDENTITY;
    for i in (0..len).rev() {
        let tf = chain[i];
        world_affine = world_affine
            * Affine3A::from_scale_rotation_translation(tf.scale, tf.rotation, tf.translation);
        world_rot = (world_rot * tf.rotation).normalize();
    }

    (world_affine, world_rot)
}

fn apply_world_rotation_delta(
    bone: Entity,
    world_delta: Quat,
    weight: f32,
    parents: &Query<&ChildOf>,
    transforms: &mut Query<&mut Transform>,
) {
    if weight <= 1e-4 {
        return;
    }
    let parent_world_rot = parents
        .get(bone)
        .map(|p| compute_world_pose(p.parent(), parents, transforms).1)
        .unwrap_or(Quat::IDENTITY);
    let Ok(mut bone_tf) = transforms.get_mut(bone) else {
        return;
    };
    let anim_local = bone_tf.rotation;
    let cur_world_rot = (parent_world_rot * anim_local).normalize();
    let target_world_rot = (world_delta * cur_world_rot).normalize();
    let target_local = (parent_world_rot.inverse() * target_world_rot).normalize();
    bone_tf.rotation = anim_local.slerp(target_local, weight.clamp(0.0, 1.0));
}

fn set_world_rotation_blended(
    bone: Entity,
    desired_world_rot: Quat,
    weight: f32,
    parents: &Query<&ChildOf>,
    transforms: &mut Query<&mut Transform>,
) {
    if weight <= 1e-4 {
        return;
    }
    let parent_world_rot = parents
        .get(bone)
        .map(|p| compute_world_pose(p.parent(), parents, transforms).1)
        .unwrap_or(Quat::IDENTITY);
    let Ok(mut bone_tf) = transforms.get_mut(bone) else {
        return;
    };
    let target_local = (parent_world_rot.inverse() * desired_world_rot).normalize();
    bone_tf.rotation = bone_tf.rotation.slerp(target_local, weight.clamp(0.0, 1.0));
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_bone_ik_preserves_lengths_and_respects_pole_directions() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mid = Vec3::new(0.0, 0.55, 0.0);
        let end = Vec3::new(0.0, 0.15, 0.0);
        let l1 = (mid - root).length();
        let l2 = (end - mid).length();

        let target = Vec3::new(0.08, 0.36, 0.14);
        let front = solve_two_bone_positions(root, mid, end, target, Vec3::NEG_Z).unwrap();
        assert!(((front.mid_pos - root).length() - l1).abs() < 1e-4);
        assert!(((front.end_pos - front.mid_pos).length() - l2).abs() < 1e-4);
        assert!((front.end_pos - target).length() < 1e-4);
        assert!(front.mid_pos.z < 0.0, "front elbow should bend toward -Z");

        let hind = solve_two_bone_positions(root, mid, end, target, Vec3::Z).unwrap();
        assert!(hind.mid_pos.z > 0.1, "hind knee should bend toward +Z");
    }

    #[test]
    fn airborne_leg_tuck_and_reach_are_continuous_and_expressive() {
        // Check continuity across 100 steps in [0, 1] for both front and hind legs.
        for is_front in [true, false] {
            let mut prev = airborne_leg_offset(is_front, 0.0);
            for i in 1..=100 {
                let u = i as f32 / 100.0;
                let cur = airborne_leg_offset(is_front, u);
                assert!(cur.is_finite());
                assert!(
                    (cur - prev).length() < 0.05,
                    "discontinuity at u={u:.2} for is_front={is_front}"
                );
                prev = cur;
            }
        }

        // Front legs tuck upward on ascent (u=0.26) and reach forward on descent (u=0.85).
        let front_ascent = airborne_leg_offset(true, 0.26);
        let front_descent = airborne_leg_offset(true, 0.85);
        assert!(front_ascent.y > 0.20, "front paws should tuck high on ascent");
        assert!(front_descent.z > 0.18 && front_descent.y < -0.08, "front paws should reach for landing");

        // Hind legs push back at takeoff (u=0.12) and snap high over rail at apex (u=0.50).
        let hind_push = airborne_leg_offset(false, 0.12);
        let hind_apex = airborne_leg_offset(false, 0.50);
        assert!(hind_push.z < -0.15, "hind legs should push off backward");
        assert!(hind_apex.y > 0.20, "hind legs should tuck high over rail");
    }
}
