//! Custom pure-Rust Inverse Kinematics (IK) for the animated Fox skeleton.
//!
//! Layers three real-time procedural IK stages on top of Bevy's glTF skeletal
//! animation in `PostUpdate` (`.after(AnimationSystems).before(TransformSystems::Propagate)`):
//! 1. Smoothed pelvis height adjustment and terrain-slope body alignment.
//! 2. Damped multi-bone look-at IK distributed across `b_Spine02_03` (20%),
//!    `b_Neck_04` (35%), and `b_Head_05` (45%).
//! 3. Analytical 2-bone Law-of-Cosines foot-placement IK for all four legs with
//!    stance-aware weight feathering and hind paw-tip offset compensation.

use bevy::math::Affine3A;
use bevy::prelude::*;

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

/// Runtime toggles and look-at target shared between controls, UI, and IK systems.
#[derive(Resource, Debug, Clone)]
pub struct IkSettings {
    pub enabled: bool,
    pub show_gizmos: bool,
    pub look_target: Vec3,
    pub look_active: bool,
}

impl Default for IkSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            show_gizmos: false,
            look_target: Vec3::new(0.0, 1.0, 3.0),
            look_active: true,
        }
    }
}

/// Marker on the locomotion root entity (`x, terrain_y, z`, heading yaw).
#[derive(Component)]
pub struct FoxRoot;

/// Marker on the child entity holding `SceneRoot` where smoothed pelvis drop
/// and terrain-slope pitch/roll are applied.
#[derive(Component)]
pub struct FoxBody;

/// Smoothed per-fox IK state updated each frame with [`StableInterpolate::smooth_nudge`].
#[derive(Component, Debug, Clone)]
pub struct FoxIkState {
    pub pelvis_offset: f32,
    pub slope_pitch: f32,
    pub slope_roll: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
    pub leg_weights: [f32; 4],
}

impl Default for FoxIkState {
    fn default() -> Self {
        Self {
            pelvis_offset: 0.0,
            slope_pitch: 0.0,
            slope_roll: 0.0,
            look_yaw: 0.0,
            look_pitch: 0.0,
            leg_weights: [1.0; 4],
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

/// Cached `Fox.glb` bone entity references populated once on scene spawn so
/// runtime IK performs zero string lookups.
#[derive(Component, Debug, Clone)]
pub struct FoxRig {
    pub spine2: Entity,
    pub neck: Entity,
    pub head: Entity,
    pub legs: [LegChain; 4],
    /// Snapshot of pre-IK animated local transforms so pausing or toggling IK
    /// never accumulates or drifts rotations.
    animated_pose: Vec<(Entity, Transform)>,
}

/// Pure geometric result of the analytical 2-bone IK solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoneSolution {
    /// Solved world-space mid-joint (elbow / knee) position.
    pub mid_pos: Vec3,
    /// Clamped world-space end-effector (wrist / ankle) position.
    pub end_pos: Vec3,
}

/// Pure analytical 2-bone IK solver using the Law of Cosines.
///
/// Given world-space joint positions `(root, mid, end)`, a desired `target`
/// position for `end`, and a `pole_dir` hint specifying which side of the
/// `root -> target` line the mid-joint should bend toward:
/// - Preserves segment lengths `l1 = |mid - root|` and `l2 = |end - mid|`.
/// - Clamps reach distance to `[MIN_REACH_RATIO, MAX_REACH_RATIO] * (l1 + l2)`
///   and `>= |l1 - l2| + 1e-4` to eliminate NaNs and straight-leg snapping.
/// - Handles collinear or zero `pole_dir` inputs gracefully by falling back to
///   the current animated bend plane or an arbitrary orthogonal axis.
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

    // Law of cosines for angle alpha at `root` between `dir` and upper bone.
    let cos_alpha = ((l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)).clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    // Project pole hint orthogonal to `dir`, with safe fallbacks for collinear inputs.
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

/// Discovers and caches the `Fox.glb` bone entities onto [`FoxBody`] once the
/// glTF scene's [`AnimationPlayer`] spawns.
pub fn bind_fox_rig(
    mut commands: Commands,
    added_players: Query<Entity, Added<AnimationPlayer>>,
    parents: Query<&ChildOf>,
    children_query: Query<&Children>,
    names: Query<&Name>,
    fox_bodies: Query<Entity, (With<FoxBody>, Without<FoxRig>)>,
) {
    for player_entity in &added_players {
        // Walk upward from AnimationPlayer to locate its FoxBody ancestor.
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

        // Collect all named descendant bones under FoxBody.
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
            Some(spine2),
            Some(neck),
            Some(head),
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
            find("b_Spine02_03"),
            find("b_Neck_04"),
            find("b_Head_05"),
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
                spine2,
                neck,
                head,
                legs,
                animated_pose: Vec::with_capacity(16),
            },
            FoxIkState::default(),
        ));
    }
}

/// Restores the previous frame's pre-IK animated pose before [`AnimationSystems`]
/// runs so un-keyframed bones never accumulate IK drift and clip changes while
/// paused always start from a clean pose.
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

/// Applies terrain-adaptive pelvis alignment, multi-bone look-at IK, and 4-leg
/// analytical foot-placement IK directly on top of the current frame's animated
/// pose before transform propagation.
pub fn apply_fox_ik(
    time: Res<Time>,
    settings: Res<IkSettings>,
    parents: Query<&ChildOf>,
    mut rig_query: Query<(Entity, &mut FoxRig, &mut FoxIkState), With<FoxBody>>,
    mut transforms: Query<&mut Transform>,
    mut gizmos: Gizmos,
) {
    let dt = time.delta_secs().min(0.1);

    for (body_entity, mut rig, mut state) in &mut rig_query {
        // Snapshot the freshly evaluated pre-IK animated pose so `restore_pre_ik_pose`
        // can cleanly reset these bones before the next frame's `AnimationSystems`.
        let ik_bones = rig.ik_bone_entities();
        rig.animated_pose.clear();
        for bone in ik_bones {
            if let Ok(tf) = transforms.get(bone) {
                rig.animated_pose.push((bone, *tf));
            }
        }

        // Locate FoxRoot (direct parent of FoxBody) for heading & footprint sampling.
        let Ok(body_parent) = parents.get(body_entity) else {
            continue;
        };
        let root_entity = body_parent.parent();
        let Ok(root_tf) = transforms.get(root_entity).copied() else {
            continue;
        };

        if !settings.enabled {
            // Smoothly return body alignment to neutral when IK is toggled off.
            state.pelvis_offset.smooth_nudge(&0.0, 14.0, dt);
            state.slope_pitch.smooth_nudge(&0.0, 14.0, dt);
            state.slope_roll.smooth_nudge(&0.0, 14.0, dt);
            state.look_yaw.smooth_nudge(&0.0, 14.0, dt);
            state.look_pitch.smooth_nudge(&0.0, 14.0, dt);

            if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
                body_tf.translation = Vec3::new(0.0, state.pelvis_offset, 0.0);
                body_tf.rotation =
                    Quat::from_rotation_x(state.slope_pitch) * Quat::from_rotation_z(state.slope_roll);
                body_tf.scale = Vec3::splat(FOX_SCALE);
            }
            continue;
        }

        // --- Stage 1: Terrain-slope body alignment + pelvis vertical drop ---
        let root_pos = root_tf.translation;
        let root_fwd = (root_tf.rotation * Vec3::Z).normalize_or(Vec3::Z);
        let root_left = (root_tf.rotation * Vec3::X).normalize_or(Vec3::X);

        let h_front = terrain_height(root_pos.x + root_fwd.x * 0.45, root_pos.z + root_fwd.z * 0.45);
        let h_back = terrain_height(root_pos.x - root_fwd.x * 0.45, root_pos.z - root_fwd.z * 0.45);
        let h_left = terrain_height(root_pos.x + root_left.x * 0.25, root_pos.z + root_left.z * 0.25);
        let h_right = terrain_height(root_pos.x - root_left.x * 0.25, root_pos.z - root_left.z * 0.25);

        // Positive X rotation tilts +Z downward; positive Z rotation tilts +X upward.
        let target_pitch = (-((h_front - h_back) / 0.90).atan() * 0.72).clamp(-0.42, 0.42);
        let target_roll = (((h_left - h_right) / 0.50).atan() * 0.50).clamp(-0.35, 0.35);

        state.slope_pitch.smooth_nudge(&target_pitch, 10.0, dt);
        state.slope_roll.smooth_nudge(&target_roll, 10.0, dt);

        // Apply smoothed slope rotation with zero pelvis offset first so we can
        // measure how far the terrain dips beneath all four paws.
        if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
            body_tf.translation = Vec3::ZERO;
            body_tf.rotation =
                Quat::from_rotation_x(state.slope_pitch) * Quat::from_rotation_z(state.slope_roll);
            body_tf.scale = Vec3::splat(FOX_SCALE);
        }

        let (body_affine_unshifted, _) = compute_world_pose(body_entity, &parents, &transforms);
        let body_inv_unshifted = body_affine_unshifted.inverse();

        let mut min_delta_y = f32::INFINITY;
        let mut max_delta_y = f32::NEG_INFINITY;
        for leg in &rig.legs {
            let paw_entity = leg.paw_tip.unwrap_or(leg.end);
            let (paw_affine, _) = compute_world_pose(paw_entity, &parents, &transforms);
            let paw_world = paw_affine.translation.into();
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

        // Drop pelvis toward the lowest paw so legs never hyperextend over crests.
        let slope_flex_margin = -0.02 * smoothstep(0.03, 0.14, max_delta_y - min_delta_y);
        let target_pelvis_offset = (min_delta_y.min(0.0) + slope_flex_margin).clamp(-0.32, 0.0);
        state.pelvis_offset.smooth_nudge(&target_pelvis_offset, 11.0, dt);

        if let Ok(mut body_tf) = transforms.get_mut(body_entity) {
            body_tf.translation = Vec3::new(0.0, state.pelvis_offset, 0.0);
        }

        let (body_affine, body_world_rot) = compute_world_pose(body_entity, &parents, &transforms);
        let body_inv = body_affine.inverse();
        let fox_fwd = (body_world_rot * Vec3::Z).normalize_or(Vec3::Z);

        // --- Stage 2: Damped multi-bone Look-At IK (Spine02 -> Neck -> Head) ---
        // Applied BEFORE front-leg IK because front shoulders are children of `b_Spine02_03`;
        // solving legs afterward keeps front paws locked to the terrain while the torso turns!
        let (head_affine, _) = compute_world_pose(rig.head, &parents, &transforms);
        let head_world_pos: Vec3 = head_affine.translation.into();

        let (target_yaw, target_pitch) = if settings.look_active {
            let to_target = (settings.look_target - head_world_pos).normalize_or(fox_fwd);
            let local_dir = body_world_rot.inverse() * to_target;
            let raw_yaw = local_dir.x.atan2(local_dir.z);
            let horiz = (local_dir.x * local_dir.x + local_dir.z * local_dir.z)
                .sqrt()
                .max(1e-4);
            let raw_pitch = (-local_dir.y).atan2(horiz);

            // Fade smoothly to neutral if the target moves behind the fox's shoulders.
            let rear_fade = 1.0 - smoothstep(1.35, 2.05, raw_yaw.abs());
            (
                raw_yaw.clamp(-1.10, 1.10) * rear_fade,
                raw_pitch.clamp(-0.58, 0.50) * rear_fade,
            )
        } else {
            (0.0, 0.0)
        };

        state.look_yaw.smooth_nudge(&target_yaw, 8.0, dt);
        state.look_pitch.smooth_nudge(&target_pitch, 8.0, dt);

        let total_delta_local =
            Quat::from_rotation_y(state.look_yaw) * Quat::from_rotation_x(state.look_pitch);
        let look_chain = [(rig.spine2, 0.20_f32), (rig.neck, 0.35), (rig.head, 0.45)];
        for (bone, fraction) in look_chain {
            let delta_local_fox = Quat::IDENTITY.slerp(total_delta_local, fraction);
            let delta_world =
                (body_world_rot * delta_local_fox * body_world_rot.inverse()).normalize();
            apply_world_rotation_delta(bone, delta_world, 1.0, &parents, &mut transforms);
        }

        // --- Stage 3: 4-Leg Analytical Terrain Foot-Placement IK ---
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

            let target_paw = Vec3::new(paw_pos_w.x, ground_y + rest_h + anim_lift, paw_pos_w.z);

            // Stance-aware weight feathering: full weight when planted or when
            // preventing terrain penetration during swing; softer pull downward
            // during high swing apex to preserve authored stride arc.
            let stance_factor = 1.0 - smoothstep(0.04, 0.28, anim_lift);
            let penetrating = paw_pos_w.y < ground_y + rest_h;
            let desired_weight = if penetrating {
                1.0
            } else {
                0.30 + 0.70 * stance_factor
            };
            state.leg_weights[idx].smooth_nudge(&desired_weight, 16.0, dt);
            let weight = state.leg_weights[idx];

            // Compute subtle slope conformity delta up front so hind `Foot01 -> Foot02`
            // offset accounts for the final rotated paw orientation.
            let slope_align = Quat::from_rotation_arc(Vec3::Y, ground_n);
            let slope_delta = Quat::IDENTITY.slerp(slope_align, 0.45 * stance_factor);
            let end_target = target_paw - slope_delta * (paw_pos_w - end_pos_w);

            // Quadruped pole hints: front elbows bend backward, hind knees bend forward.
            let pole_dir = if leg.is_front { -fox_fwd } else { fox_fwd };

            let Some(sol) =
                solve_two_bone_positions(root_pos_w, mid_pos_w, end_pos_w, end_target, pole_dir)
            else {
                continue;
            };

            // 1. Rotate upper bone toward solved mid-joint position.
            let old_upper_dir = (mid_pos_w - root_pos_w).normalize_or(Vec3::NEG_Y);
            let new_upper_dir = (sol.mid_pos - root_pos_w).normalize_or(old_upper_dir);
            let upper_delta = Quat::from_rotation_arc(old_upper_dir, new_upper_dir);
            apply_world_rotation_delta(leg.upper, upper_delta, weight, &parents, &mut transforms);

            // 2. Recompute lower/end world positions after upper bone rotation, then aim lower bone.
            let (mid_after_affine, _) = compute_world_pose(leg.lower, &parents, &transforms);
            let (end_after_affine, _) = compute_world_pose(leg.end, &parents, &transforms);
            let cur_mid_w: Vec3 = mid_after_affine.translation.into();
            let cur_end_w: Vec3 = end_after_affine.translation.into();

            let old_lower_dir = (cur_end_w - cur_mid_w).normalize_or(Vec3::NEG_Y);
            let new_lower_dir = (sol.end_pos - cur_mid_w).normalize_or(old_lower_dir);
            let lower_delta = Quat::from_rotation_arc(old_lower_dir, new_lower_dir);
            apply_world_rotation_delta(leg.lower, lower_delta, weight, &parents, &mut transforms);

            // 3. Preserve end-effector world orientation (+ subtle slope conformity during stance)
            //    so hind `Foot02` lands right on `target_paw` instead of swinging with `Leg02`.
            let desired_end_world_rot = (slope_delta * orig_end_world_rot).normalize();
            set_world_rotation_blended(
                leg.end,
                desired_end_world_rot,
                weight,
                &parents,
                &mut transforms,
            );

            // --- Stage 4: Optional Gizmos Debug Overlay ---
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

                // Ground target & surface normal marker
                let ground_pt = Vec3::new(paw_pos_w.x, ground_y, paw_pos_w.z);
                gizmos.sphere(
                    Isometry3d::from_translation(ground_pt),
                    0.045,
                    Color::srgb(0.25, 1.0, 0.45),
                );
                gizmos.line(
                    ground_pt,
                    ground_pt + ground_n * 0.22,
                    Color::srgb(0.25, 1.0, 0.45),
                );
            }
        }

        if settings.show_gizmos {
            let (head_after, _) = compute_world_pose(rig.head, &parents, &transforms);
            let head_p: Vec3 = head_after.translation.into();
            gizmos.line(head_p, settings.look_target, Color::srgb(1.0, 0.92, 0.25));
            gizmos.sphere(
                Isometry3d::from_translation(settings.look_target),
                0.06,
                Color::srgb(1.0, 0.92, 0.25),
            );
        }
    }
}

impl FoxRig {
    /// Returns all bone entities modified by IK in hierarchy order.
    fn ik_bone_entities(&self) -> [Entity; 15] {
        [
            self.spine2,
            self.neck,
            self.head,
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

/// Computes an entity's current-frame world [`Affine3A`] and world [`Quat`] by
/// walking `&ChildOf` ancestors on local `Transform`s (valid in `PostUpdate`
/// before `TransformSystems::Propagate`).
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

/// Applies a world-space delta rotation on top of a bone's current animated
/// orientation, blended by `weight` in `[0.0, 1.0]`.
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

/// Blends a bone's local rotation toward an explicit desired world rotation.
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

/// Hermite smoothstep interpolation in `[0.0, 1.0]`.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_bone_ik_reaches_target_and_preserves_bone_lengths() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mid = Vec3::new(0.0, 0.54, -0.02);
        let end = Vec3::new(0.0, 0.15, 0.0);
        let l1 = (mid - root).length();
        let l2 = (end - mid).length();

        let target = Vec3::new(0.12, 0.35, 0.18);
        let pole = Vec3::NEG_Z;
        let sol = solve_two_bone_positions(root, mid, end, target, pole).expect("valid solution");

        assert!(
            ((sol.mid_pos - root).length() - l1).abs() < 1e-4,
            "upper bone length changed"
        );
        assert!(
            ((sol.end_pos - sol.mid_pos).length() - l2).abs() < 1e-4,
            "lower bone length changed"
        );
        assert!(
            (sol.end_pos - target).length() < 1e-4,
            "end effector missed reachable target: got {:?}, expected {:?}",
            sol.end_pos,
            target
        );
    }

    #[test]
    fn two_bone_ik_respects_pole_direction_front_and_hind() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mid = Vec3::new(0.0, 0.55, 0.0);
        let end = Vec3::new(0.0, 0.15, 0.0);
        let target = Vec3::new(0.0, 0.40, 0.0);

        // Front leg: elbow bends backward (-Z)
        let front = solve_two_bone_positions(root, mid, end, target, Vec3::NEG_Z).unwrap();
        assert!(front.mid_pos.z < -0.1, "front elbow should bend toward -Z, got {:?}", front.mid_pos);

        // Hind leg: knee bends forward (+Z)
        let hind = solve_two_bone_positions(root, mid, end, target, Vec3::Z).unwrap();
        assert!(hind.mid_pos.z > 0.1, "hind knee should bend toward +Z, got {:?}", hind.mid_pos);
    }

    #[test]
    fn two_bone_ik_clamps_unreachable_and_collapsed_targets_safely() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mid = Vec3::new(0.0, 0.5, 0.0);
        let end = Vec3::new(0.0, 0.0, 0.0);
        let total = 1.0;

        // Way out of reach
        let far_target = Vec3::new(0.0, -5.0, 0.0);
        let far_sol = solve_two_bone_positions(root, mid, end, far_target, Vec3::NEG_Z).unwrap();
        let far_dist = (far_sol.end_pos - root).length();
        assert!(
            (far_dist - total * MAX_REACH_RATIO).abs() < 1e-3,
            "far reach should clamp to MAX_REACH_RATIO, got {far_dist}"
        );
        assert!(far_sol.mid_pos.is_finite() && far_sol.end_pos.is_finite());

        // Collapsed onto root
        let near_sol = solve_two_bone_positions(root, mid, end, root, Vec3::Z).unwrap();
        assert!(near_sol.mid_pos.is_finite() && near_sol.end_pos.is_finite());
        assert!((near_sol.mid_pos - root).length() > 0.49);
    }

    #[test]
    fn two_bone_ik_handles_collinear_and_zero_pole_vectors() {
        let root = Vec3::new(0.0, 1.0, 0.0);
        let mid = Vec3::new(0.0, 0.5, -0.05);
        let end = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(0.0, 0.3, 0.0);

        // Pole parallel to root->target (-Y)
        let sol_collinear = solve_two_bone_positions(root, mid, end, target, Vec3::NEG_Y).unwrap();
        assert!(sol_collinear.mid_pos.is_finite() && sol_collinear.end_pos.is_finite());
        assert!((sol_collinear.end_pos - target).length() < 1e-4);

        // Zero pole vector
        let sol_zero = solve_two_bone_positions(root, mid, end, target, Vec3::ZERO).unwrap();
        assert!(sol_zero.mid_pos.is_finite() && sol_zero.end_pos.is_finite());
        assert!((sol_zero.end_pos - target).length() < 1e-4);
    }
}
