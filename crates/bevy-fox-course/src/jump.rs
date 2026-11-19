//! Bidirectional OBB show-jump feasibility evaluator, apex-targeted parabolic
//! ballistic solver, flight integrator, and critically damped spring-damper
//! landing recovery.

use bevy::prelude::*;

use crate::controls::{resolve_avoidance_side, AvoidanceLock, LocomotionState, PlayerIntent};
use crate::ik::FoxRoot;
use crate::obstacles::{
    grounded_capsule_pushout, local_fence_coords, wing_world_position, CourseObstacle,
    CourseObstacles,
};
use crate::terrain::{terrain_height, PLAY_AREA_RADIUS};

/// Ballistic gravity used during show jumps (m/s^2) — slightly snappier than
/// Earth gravity so arcs feel athletic and responsive at fox scale.
pub const JUMP_GRAVITY: f32 = 11.5;

/// Hindquarter plant / coil / forehand-lift preparation duration before liftoff (s).
pub const GATHER_DURATION: f32 = 0.22;

/// Post-touchdown spring absorption phase duration (s).
pub const LANDING_DURATION: f32 = 0.26;

/// Cooldown after touchdown before committing to another jump (s).
pub const JUMP_COOLDOWN: f32 = 0.65;

/// Critically damped landing spring stiffness (`k = 140`, `zeta = 0.88`).
pub const LANDING_SPRING_K: f32 = 140.0;
pub const LANDING_SPRING_ZETA: f32 = 0.88;

/// Grounded 2D safety capsule radius (m) preventing slow creep/walk rail clipping.
pub const FOX_CAPSULE_RADIUS: f32 = 0.38;

/// Active phase of the fox's jump lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JumpPhase {
    #[default]
    Grounded,
    Gather,
    Airborne,
    Landing,
}

impl JumpPhase {
    pub const fn label(self) -> &'static str {
        match self {
            JumpPhase::Grounded => "Grounded",
            JumpPhase::Gather => "Gather",
            JumpPhase::Airborne => "Airborne",
            JumpPhase::Landing => "Landing",
        }
    }
}

/// Outcome of evaluating an obstacle's bidirectional OBB approach zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproachVerdict {
    /// Inside pre-approach zone and on a valid line/speed toward takeoff.
    ApproachingOk,
    /// All 5 commit criteria passed inside the dynamic takeoff zone.
    CommitJump,
    /// Too slow, too oblique, aimed at a wing, or landing unsafe -> smooth refusal.
    RefuseAvoid,
}

/// Closed-form parabolic trajectory solved for a single show jump.
#[derive(Debug, Clone, Copy)]
pub struct JumpTrajectory {
    pub start_pos: Vec3,
    pub vel_xz: Vec2,
    pub vy0: f32,
    pub total_time: f32,
    pub apex_time: f32,
    pub apex_pos: Vec3,
    pub landing_pos: Vec3,
    pub rail_top_pos: Vec3,
}

impl JumpTrajectory {
    /// Samples the ballistic position at elapsed flight time `t` in `[0, total_time]`.
    #[inline]
    pub fn sample_pos(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, self.total_time);
        let xz = Vec2::new(self.start_pos.x, self.start_pos.z) + self.vel_xz * t;
        let y = self.start_pos.y + self.vy0 * t - 0.5 * JUMP_GRAVITY * t * t;
        Vec3::new(xz.x, y, xz.y)
    }

    /// Vertical velocity `vy(t)` at elapsed flight time `t`.
    #[inline]
    pub fn sample_vy(&self, t: f32) -> f32 {
        self.vy0 - JUMP_GRAVITY * t
    }
}

/// Runtime state of the jump lifecycle, landing spring, and HUD counters.
#[derive(Resource, Debug, Clone, Default)]
pub struct JumpState {
    pub phase: JumpPhase,
    pub phase_time: f32,
    pub cooldown: f32,
    pub active_obstacle: Option<usize>,
    /// `+1.0` when jumping along `+obs.normal_xz`, `-1.0` when jumping along `-obs.normal_xz`.
    pub approach_sign: f32,
    pub trajectory: Option<JumpTrajectory>,
    /// Normalized airborne progress `u = t / total_time` in `[0, 1]`.
    pub flight_u: f32,
    /// Current vertical velocity during airborne flight (m/s).
    pub current_vy: f32,
    /// Damped impact spring pelvis displacement (m, typically negative on impact).
    pub landing_spring_y: f32,
    /// Damped impact spring vertical velocity (m/s).
    pub landing_spring_vy: f32,
    pub jumps_cleared: u32,
    pub refusals: u32,
    /// Most relevant nearby obstacle verdict for debug gizmo coloring.
    pub last_eval_status: Option<(usize, ApproachVerdict)>,
}

/// Dynamic takeoff distance (m) scaled by speed and rail height (includes Gather runway).
#[inline]
pub fn compute_takeoff_distance(speed: f32, rail_height: f32) -> f32 {
    (0.92 + 0.29 * speed + 0.45 * rail_height).clamp(1.38, 2.95)
}

/// Solves an apex-targeted parabolic trajectory clearing `obs` from `(pos_xz, fwd_xz, speed)`.
///
/// - Sets apex height to `base_y + rail_height + 0.18 + 0.08 * spread` (combined with
///   `+0.22 m` procedural leg tuck, paws clear rails cleanly by `~0.40 m`).
/// - Adjusts horizontal speed within `+-12%` so apex lands cleanly over the fence center.
/// - Iterates touchdown terrain elevation twice for exact `total_time = t_up + t_down`.
pub fn solve_jump_trajectory(
    obs: &CourseObstacle,
    pos_xz: Vec2,
    fwd_xz: Vec2,
    speed: f32,
    approach_normal: Vec2,
) -> JumpTrajectory {
    let start_y = terrain_height(pos_xz.x, pos_xz.y);
    let start_pos = Vec3::new(pos_xz.x, start_y, pos_xz.y);

    let clearance_margin = 0.18 + 0.08 * obs.spread;
    let y_apex = (obs.base_y + obs.rail_height + clearance_margin).max(start_y + 0.38);
    let vy0 = (2.0 * JUMP_GRAVITY * (y_apex - start_y)).sqrt();
    let t_up = vy0 / JUMP_GRAVITY;

    let (along_n, _) = local_fence_coords(obs, pos_xz);
    let dist_to_center = along_n.abs().max(0.55);

    // Gently square launch direction toward the fence normal (30%) so oblique
    // valid approaches still fly squarely across wide oxers.
    let launch_dir = (fwd_xz * 0.70 + approach_normal * 0.30).normalize_or(approach_normal);
    let cos_launch = launch_dir.dot(approach_normal).max(0.65);

    let v_n_raw = (speed * cos_launch).max(1.8);
    let v_n_ideal = dist_to_center / t_up;
    let speed_scale = (v_n_ideal / v_n_raw).clamp(0.88, 1.12);
    let launch_speed = (speed * speed_scale).max(obs.min_jump_speed * 0.92);
    let vel_xz = launch_dir * launch_speed;

    // 2-step terrain height iteration for exact landing point & flight duration.
    let mut total_time = 2.0 * t_up;
    let mut land_xz = pos_xz + vel_xz * total_time;
    let mut y_land = terrain_height(land_xz.x, land_xz.y);
    for _ in 0..2 {
        let drop = (y_apex - y_land).max(0.05);
        let t_down = (2.0 * drop / JUMP_GRAVITY).sqrt();
        total_time = t_up + t_down;
        land_xz = pos_xz + vel_xz * total_time;
        y_land = terrain_height(land_xz.x, land_xz.y);
    }

    let apex_xz = pos_xz + vel_xz * t_up;
    let apex_pos = Vec3::new(apex_xz.x, y_apex, apex_xz.y);
    let landing_pos = Vec3::new(land_xz.x, y_land, land_xz.y);
    let rail_top_pos = Vec3::new(
        obs.center_xz.x,
        obs.base_y + obs.rail_height + 0.08,
        obs.center_xz.y,
    );

    JumpTrajectory {
        start_pos,
        vel_xz,
        vy0,
        total_time,
        apex_time: t_up,
        apex_pos,
        landing_pos,
        rail_top_pos,
    }
}

/// Checks whether a predicted landing point stays inside the arena and clear of
/// every other obstacle's physical footprint.
pub fn is_landing_safe(
    landing_xz: Vec2,
    jumping_obstacle_id: usize,
    obstacles: &[CourseObstacle],
) -> bool {
    if landing_xz.length() > PLAY_AREA_RADIUS - 0.5 {
        return false;
    }
    for other in obstacles {
        if other.id == jumping_obstacle_id {
            continue;
        }
        let (dn, dt) = local_fence_coords(other, landing_xz);
        let half_d = 0.65 + other.spread * 0.5;
        let half_w = other.wing_half_width + 0.55;
        if dn.abs() < half_d && dt.abs() < half_w {
            return false;
        }
    }
    true
}

/// Evaluates whether the fox is approaching `obs` and whether it should commit
/// to a jump, keep approaching, or refuse and smoothly arc around a wing.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_obstacle_approach(
    obs: &CourseObstacle,
    pos_xz: Vec2,
    fwd_xz: Vec2,
    speed: f32,
    suggested_dir: Vec2,
    has_input: bool,
    all_obstacles: &[CourseObstacle],
) -> Option<(ApproachVerdict, f32, JumpTrajectory)> {
    let (along_n, along_t) = local_fence_coords(obs, pos_xz);
    let side_sign = if along_n < 0.0 { 1.0 } else { -1.0 };
    let approach_normal = obs.normal_xz * side_sign;
    let dist_to_front = along_n.abs() - obs.spread * 0.5;

    // Must be on the approaching side, not already past or far away.
    let fwd_dot = fwd_xz.dot(approach_normal);
    if fwd_dot <= 0.15 {
        return None;
    }

    let takeoff_dist = compute_takeoff_distance(speed, obs.rail_height);
    let detect_dist = takeoff_dist + 2.5;
    if dist_to_front <= 0.12 || dist_to_front > detect_dist {
        return None;
    }
    if along_t.abs() > obs.wing_half_width + 1.05 {
        return None;
    }

    let in_gate = along_t.abs() <= obs.rail_half_width - 0.24;
    let speed_ok = speed >= obs.min_jump_speed;
    let angle_ok = fwd_dot >= obs.max_approach_cos;
    let intent_ok = !has_input || suggested_dir.dot(approach_normal) > 0.25;

    let traj = solve_jump_trajectory(obs, pos_xz, fwd_xz, speed, approach_normal);
    let landing_ok = is_landing_safe(
        Vec2::new(traj.landing_pos.x, traj.landing_pos.z),
        obs.id,
        all_obstacles,
    );

    // Inside commit takeoff distance: decide commit vs refusal.
    // Enforce enough room for the Gather coil so late commits never clip the front rail.
    let min_commit_dist = (0.35 + speed * GATHER_DURATION * 0.78).min(takeoff_dist * 0.72);
    if dist_to_front <= takeoff_dist {
        if dist_to_front >= min_commit_dist
            && in_gate
            && speed_ok
            && angle_ok
            && intent_ok
            && landing_ok
        {
            return Some((ApproachVerdict::CommitJump, side_sign, traj));
        } else {
            return Some((ApproachVerdict::RefuseAvoid, side_sign, traj));
        }
    }

    // Pre-approach anticipation zone: start steering smoothly around a wing early
    // if aimed at a wing upright, too slow to realistically jump, or too oblique.
    let aimed_at_wing = !in_gate && along_t.abs() <= obs.wing_half_width + 0.55;
    let clearly_too_slow = speed < obs.min_jump_speed * 0.82 && dist_to_front < 2.6;
    let too_oblique = !angle_ok && dist_to_front < 2.4;
    let intent_peeling_off = !intent_ok && dist_to_front < 2.2;

    if aimed_at_wing || clearly_too_slow || too_oblique || intent_peeling_off || !landing_ok {
        Some((ApproachVerdict::RefuseAvoid, side_sign, traj))
    } else {
        Some((ApproachVerdict::ApproachingOk, side_sign, traj))
    }
}

/// Integrates the near-critically damped landing spring-damper (`k = 140`, `zeta = 0.88`)
/// using 2 substeps for unconditional stability.
pub fn step_landing_spring(y: &mut f32, vy: &mut f32, dt: f32) {
    let c = 2.0 * LANDING_SPRING_ZETA * LANDING_SPRING_K.sqrt();
    let sub_dt = (dt * 0.5).min(0.025);
    for _ in 0..2 {
        let accel = -LANDING_SPRING_K * (*y) - c * (*vy);
        *vy += accel * sub_dt;
        *y = (*y + (*vy) * sub_dt).clamp(-0.24, 0.04);
    }
    if y.abs() < 1e-4 && vy.abs() < 1e-3 {
        *y = 0.0;
        *vy = 0.0;
    }
}

/// Evaluates nearby obstacles while grounded to either initiate a `Gather` jump
/// commit or acquire/release the zero-oscillation [`AvoidanceLock`].
pub fn evaluate_jump_decision(
    time: Res<Time>,
    obstacles: Res<CourseObstacles>,
    intent: Res<PlayerIntent>,
    loco: Res<LocomotionState>,
    mut jump: ResMut<JumpState>,
    mut avoid: ResMut<AvoidanceLock>,
) {
    let dt = time.delta_secs().min(0.1);
    if intent.paused {
        return;
    }

    if jump.cooldown > 0.0 {
        jump.cooldown = (jump.cooldown - dt).max(0.0);
    }

    // Only evaluate new jump commits / avoidance while grounded.
    if jump.phase != JumpPhase::Grounded {
        avoid.clear();
        return;
    }

    let pos_xz = Vec2::new(loco.position.x, loco.position.z);
    let fwd_xz = Vec2::new(loco.heading.sin(), loco.heading.cos());

    // First, check whether an existing AvoidanceLock should stay active or release.
    if let Some(lock_id) = avoid.active_obstacle {
        if let Some(obs) = obstacles.list.get(lock_id) {
            let (along_n, along_t) = local_fence_coords(obs, pos_xz);
            let approach_sign = avoid.approach_sign;
            let approach_normal = obs.normal_xz * approach_sign;
            let passed_fence_plane =
                along_n * approach_sign > 0.35 + obs.spread * 0.5 || along_n.abs() > 4.2;
            let cleared_wing =
                along_t * avoid.side > obs.wing_half_width + 0.55 && along_n.abs() < 2.4;
            let turned_away = fwd_xz.dot(approach_normal) < -0.30 && along_n.abs() > 1.5;

            if passed_fence_plane || cleared_wing || turned_away {
                avoid.clear();
            } else {
                // Keep side and approach_sign locked! Refresh clearance target smoothly.
                let pass_n = (0.65 + obs.spread * 0.5) * approach_sign;
                avoid.target_xz = obs.center_xz
                    + obs.tangent_xz * (avoid.side * (obs.wing_half_width + 0.65))
                    + obs.normal_xz * pass_n;
                avoid.wing_world_pos = wing_world_position(obs, avoid.side);
            }
        } else {
            avoid.clear();
        }
    }

    // Scan obstacles for nearest active approach zone.
    let mut best_commit: Option<(f32, usize, f32, JumpTrajectory)> = None;
    let mut best_refuse: Option<(f32, usize, f32)> = None;
    let mut best_status: Option<(f32, usize, ApproachVerdict)> = None;

    for obs in &obstacles.list {
        let Some((verdict, side_sign, traj)) = evaluate_obstacle_approach(
            obs,
            pos_xz,
            fwd_xz,
            loco.speed,
            intent.suggested_dir,
            intent.has_input,
            &obstacles.list,
        ) else {
            continue;
        };

        let dist = pos_xz.distance(obs.center_xz);
        if best_status.map_or(true, |(d, _, _)| dist < d) {
            best_status = Some((dist, obs.id, verdict));
        }

        match verdict {
            ApproachVerdict::CommitJump => {
                if jump.cooldown <= 0.0 && best_commit.map_or(true, |(d, _, _, _)| dist < d) {
                    best_commit = Some((dist, obs.id, side_sign, traj));
                }
            }
            ApproachVerdict::RefuseAvoid => {
                if best_refuse.map_or(true, |(d, _, _)| dist < d) {
                    best_refuse = Some((dist, obs.id, side_sign));
                }
            }
            ApproachVerdict::ApproachingOk => {}
        }
    }

    jump.last_eval_status = best_status.map(|(_, id, v)| (id, v));

    // Commit jump takes priority if ready and not already locked into avoiding this obstacle.
    if let Some((_, obs_id, side_sign, traj)) = best_commit {
        if avoid.active_obstacle != Some(obs_id) {
            jump.phase = JumpPhase::Gather;
            jump.phase_time = 0.0;
            jump.active_obstacle = Some(obs_id);
            jump.approach_sign = side_sign;
            jump.trajectory = Some(traj);
            jump.flight_u = 0.0;
            jump.current_vy = 0.0;
            avoid.clear();
            return;
        }
    }

    // Acquire hysteresis side lock on refusal if not already locked.
    if avoid.active_obstacle.is_none() {
        if let Some((_, obs_id, side_sign)) = best_refuse {
            let obs = &obstacles.list[obs_id];
            let chosen_side =
                resolve_avoidance_side(&mut avoid, obs, pos_xz, fwd_xz, intent.suggested_dir);
            avoid.approach_sign = side_sign;
            let pass_n = (0.65 + obs.spread * 0.5) * side_sign;
            let target_xz = obs.center_xz
                + obs.tangent_xz * (chosen_side * (obs.wing_half_width + 0.65))
                + obs.normal_xz * pass_n;

            avoid.target_xz = target_xz;
            avoid.wing_world_pos = wing_world_position(obs, chosen_side);
            if !avoid.refusal_counted {
                jump.refusals += 1;
                avoid.refusal_counted = true;
            }
        }
    }
}

/// Integrates either the ballistic jump trajectory (`Gather` -> `Airborne` -> `Landing`)
/// or grounded non-holonomic locomotion + safety-capsule pushout onto `FoxRoot`.
pub fn step_jump_or_ground_locomotion(
    time: Res<Time>,
    obstacles: Res<CourseObstacles>,
    intent: Res<PlayerIntent>,
    mut loco: ResMut<LocomotionState>,
    mut jump: ResMut<JumpState>,
    mut fox_query: Query<&mut Transform, With<FoxRoot>>,
) {
    let dt = time.delta_secs().min(0.1);
    let Ok(mut fox_tf) = fox_query.single_mut() else {
        return;
    };
    if intent.paused {
        return;
    }

    // Always tick the landing spring-damper so residual bounce decays smoothly.
    let jump_ref = &mut *jump;
    step_landing_spring(
        &mut jump_ref.landing_spring_y,
        &mut jump_ref.landing_spring_vy,
        dt,
    );

    match jump.phase {
        JumpPhase::Grounded => {
            let mut next_xz = Vec2::new(loco.position.x, loco.position.z) + loco.velocity_xz * dt;

            // Grounded 2D safety capsule prevents slow creep/walk from clipping rails/wings.
            next_xz = grounded_capsule_pushout(next_xz, FOX_CAPSULE_RADIUS, &obstacles.list);

            // Soft circular arena containment.
            let dist = next_xz.length();
            if dist > PLAY_AREA_RADIUS {
                let radial = next_xz / dist;
                next_xz = radial * PLAY_AREA_RADIUS;
                let outward = loco.velocity_xz.dot(radial);
                if outward > 0.0 {
                    loco.velocity_xz -= radial * outward;
                    loco.speed = loco.velocity_xz.length();
                }
            }

            let y = terrain_height(next_xz.x, next_xz.y);
            loco.position = Vec3::new(next_xz.x, y, next_xz.y);
            jump.current_vy = 0.0;
            jump.flight_u = 0.0;
        }
        JumpPhase::Gather => {
            jump.phase_time += dt;
            let gather_t = (jump.phase_time / GATHER_DURATION).clamp(0.0, 1.0);

            // Slightly check stride speed during the coil (hind feet plant & grip turf),
            // then drive forward into launch at the end of Gather.
            let stride_scale = 0.82 - 0.14 * (std::f32::consts::PI * gather_t).sin();
            let mut next_xz = Vec2::new(loco.position.x, loco.position.z)
                + loco.velocity_xz * (dt * stride_scale);
            let dist = next_xz.length();
            if dist > PLAY_AREA_RADIUS {
                next_xz = (next_xz / dist) * PLAY_AREA_RADIUS;
            }
            let y = terrain_height(next_xz.x, next_xz.y);
            loco.position = Vec3::new(next_xz.x, y, next_xz.y);

            // Refresh trajectory from the actual liftoff point when Gather finishes.
            if let Some(obs_id) = jump.active_obstacle {
                let obs = &obstacles.list[obs_id];
                let fwd_xz = Vec2::new(loco.heading.sin(), loco.heading.cos());
                let approach_normal = obs.normal_xz * jump.approach_sign;
                let traj = solve_jump_trajectory(obs, next_xz, fwd_xz, loco.speed, approach_normal);
                jump.trajectory = Some(traj);

                if jump.phase_time >= GATHER_DURATION {
                    jump.phase = JumpPhase::Airborne;
                    jump.phase_time = 0.0;
                    jump.flight_u = 0.0;
                    jump.current_vy = traj.vy0;
                    loco.velocity_xz = traj.vel_xz;
                    loco.speed = traj.vel_xz.length();
                    loco.heading = traj.vel_xz.x.atan2(traj.vel_xz.y);
                    loco.turn_rate = 0.0;
                }
            } else {
                jump.phase = JumpPhase::Grounded;
            }
        }
        JumpPhase::Airborne => {
            jump.phase_time += dt;
            if let Some(traj) = jump.trajectory {
                let t = jump.phase_time.min(traj.total_time);
                let pos = traj.sample_pos(t);
                let ground_y = terrain_height(pos.x, pos.z);
                jump.flight_u = (t / traj.total_time.max(0.05)).clamp(0.0, 1.0);
                jump.current_vy = traj.sample_vy(t);

                // Touchdown when elapsed flight completes or descending paws meet turf.
                let descending = jump.current_vy < 0.0;
                if jump.phase_time >= traj.total_time || (descending && pos.y <= ground_y + 0.02) {
                    let land_pos = Vec3::new(pos.x, ground_y, pos.z);
                    loco.position = land_pos;
                    jump.phase = JumpPhase::Landing;
                    jump.phase_time = 0.0;
                    jump.cooldown = JUMP_COOLDOWN;
                    jump.flight_u = 1.0;
                    jump.jumps_cleared += 1;

                    // Trigger critically damped landing compression spring proportional to impact.
                    jump.landing_spring_y = 0.0;
                    jump.landing_spring_vy = (jump.current_vy * 0.28).clamp(-2.8, -0.8);
                    jump.current_vy = 0.0;
                } else {
                    loco.position = pos;
                }
            } else {
                jump.phase = JumpPhase::Grounded;
            }
        }
        JumpPhase::Landing => {
            jump.phase_time += dt;

            // Maintain forward momentum smoothly on touchdown without snapping.
            let mut next_xz = Vec2::new(loco.position.x, loco.position.z) + loco.velocity_xz * dt;
            next_xz = grounded_capsule_pushout(next_xz, FOX_CAPSULE_RADIUS, &obstacles.list);
            let dist = next_xz.length();
            if dist > PLAY_AREA_RADIUS {
                next_xz = (next_xz / dist) * PLAY_AREA_RADIUS;
            }
            let y = terrain_height(next_xz.x, next_xz.y);
            loco.position = Vec3::new(next_xz.x, y, next_xz.y);

            if jump.phase_time >= LANDING_DURATION {
                jump.phase = JumpPhase::Grounded;
                jump.phase_time = 0.0;
                jump.active_obstacle = None;
                jump.trajectory = None;
                jump.flight_u = 0.0;
            }
        }
    }

    fox_tf.translation = loco.position;
    fox_tf.rotation = Quat::from_rotation_y(loco.heading);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacles::build_course_obstacles;

    #[test]
    fn bidirectional_obb_jump_matrix_valid_vs_slow_vs_oblique_vs_wing() {
        let obstacles = build_course_obstacles();
        let obs = &obstacles[1]; // Meadow Vertical (v_min = 3.1, h = 0.64)

        // Test BOTH approach directions (+normal and -normal)!
        for &side_sign in &[1.0_f32, -1.0_f32] {
            let approach_n = obs.normal_xz * side_sign;
            let pos = obs.center_xz - approach_n * 1.65;

            // 1. Valid centered fast approach -> CommitJump
            let (verdict, sign, _) = evaluate_obstacle_approach(
                obs,
                pos,
                approach_n,
                3.8,
                approach_n,
                true,
                &obstacles,
            )
            .expect("should detect valid approach");
            assert_eq!(verdict, ApproachVerdict::CommitJump);
            assert_eq!(sign, side_sign);

            // 2. Too slow -> RefuseAvoid
            let (verdict_slow, _, _) = evaluate_obstacle_approach(
                obs,
                pos,
                approach_n,
                1.6,
                approach_n,
                true,
                &obstacles,
            )
            .expect("should detect slow refusal");
            assert_eq!(verdict_slow, ApproachVerdict::RefuseAvoid);

            // 3. Too oblique (~55 deg off normal) -> RefuseAvoid
            let oblique_dir = (approach_n * 0.55 + obs.tangent_xz * 0.835).normalize();
            let (verdict_oblique, _, _) = evaluate_obstacle_approach(
                obs,
                pos,
                oblique_dir,
                4.2,
                oblique_dir,
                true,
                &obstacles,
            )
            .expect("should detect oblique refusal");
            assert_eq!(verdict_oblique, ApproachVerdict::RefuseAvoid);

            // 4. Aimed at wing upright -> RefuseAvoid
            let wing_pos = pos + obs.tangent_xz * (obs.rail_half_width + 0.10);
            let (verdict_wing, _, _) = evaluate_obstacle_approach(
                obs,
                wing_pos,
                approach_n,
                4.2,
                approach_n,
                true,
                &obstacles,
            )
            .expect("should detect wing refusal");
            assert_eq!(verdict_wing, ApproachVerdict::RefuseAvoid);
        }
    }

    #[test]
    fn parabolic_trajectory_clears_every_obstacle_vertical_and_oxer() {
        let obstacles = build_course_obstacles();
        for obs in &obstacles {
            let speed = obs.min_jump_speed + 0.35;
            let takeoff = compute_takeoff_distance(speed, obs.rail_height);
            let pos = obs.center_xz - obs.normal_xz * (takeoff + obs.spread * 0.5 - 0.15);
            let traj = solve_jump_trajectory(obs, pos, obs.normal_xz, speed, obs.normal_xz);

            assert!(traj.total_time > 0.35 && traj.total_time < 1.6);
            assert!(
                traj.apex_pos.y >= obs.base_y + obs.rail_height + 0.17,
                "apex too low for '{}'",
                obs.name
            );

            // Sample flight along the rail spread zone [-spread/2, +spread/2] and
            // verify trajectory + procedural tuck (+0.20m) stays cleanly above top rail.
            let half_s = obs.spread * 0.5;
            for step in 0..=40 {
                let t = traj.total_time * (step as f32 / 40.0);
                let p = traj.sample_pos(t);
                let (dn, _) = local_fence_coords(obs, Vec2::new(p.x, p.z));
                if dn.abs() <= half_s + 0.08 {
                    let effective_paw_y = p.y + 0.20;
                    assert!(
                        effective_paw_y > obs.base_y + obs.rail_height + 0.05,
                        "insufficient clearance over '{}' at dn={dn:.2}: paw_y={effective_paw_y:.2} vs rail={:.2}",
                        obs.name,
                        obs.base_y + obs.rail_height
                    );
                }
            }
        }
    }

    #[test]
    fn critically_damped_landing_spring_compresses_and_recovers_without_oscillation() {
        let mut y = 0.0;
        let mut vy = -2.0;
        let mut min_y = 0.0_f32;
        let mut max_overshoot = 0.0_f32;
        for _ in 0..80 {
            step_landing_spring(&mut y, &mut vy, 1.0 / 60.0);
            min_y = min_y.min(y);
            max_overshoot = max_overshoot.max(y);
        }
        assert!(min_y < -0.05, "spring should visibly compress on landing");
        assert!(
            max_overshoot < 0.015,
            "critically damped spring should not bounce wildly upward, got {max_overshoot}"
        );
        assert!(y.abs() < 0.005, "spring should settle near zero, got {y}");
    }
}
