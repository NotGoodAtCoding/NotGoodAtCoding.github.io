//! Suggestive intent-based quadruped controls, autonomous Show-Jump Course Demo,
//! non-holonomic speed-limited steering with centripetal banking, zero-oscillation
//! wing-avoidance flow field, and priority-stack gaze focus.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::prelude::*;

use crate::ik::IkSettings;
use crate::jump::{JumpPhase, JumpState};
use crate::obstacles::{local_fence_coords, CourseObstacle};
use crate::terrain::{raycast_terrain, terrain_height};

/// Starting world XZ coordinates and heading for the fox (lined up for Jump #1).
pub const FOX_START_XZ: Vec2 = Vec2::new(-14.2, 11.0);
pub const FOX_START_HEADING: f32 = FRAC_PI_2;

/// Marker for the smooth 3/4 follow camera.
#[derive(Component)]
pub struct FollowCamera;

/// Marker for the glowing ambient firefly orb tracked in cruising/idle gaze fallback.
#[derive(Component)]
pub struct FireflyOrb;

/// Pace hint used by each waypoint in the autonomous Show-Jump Course Demo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoPace {
    Walk,
    Canter,
    Sprint,
}

#[derive(Debug, Clone, Copy)]
struct DemoWaypoint {
    pos: Vec2,
    pace: DemoPace,
    reach_radius: f32,
}

/// Autonomous equestrian circuit that flows cleanly over all 8 show jumps
/// (ramping into Sprint for the Grand Prix Oxer and Triple Bar Finale) and
/// includes an intentional slow Walk approach into the Narrow Skinny Gate so
/// visitors see both clean jumping and graceful wing-avoidance refusal.
const DEMO_COURSE: [DemoWaypoint; 21] = [
    // Approach & clear #0 Starter Cross-Rail (-8.5, 11.0), normal (+1, 0)
    DemoWaypoint { pos: Vec2::new(-11.0, 11.0), pace: DemoPace::Canter, reach_radius: 1.5 },
    DemoWaypoint { pos: Vec2::new(-3.8, 11.3), pace: DemoPace::Canter, reach_radius: 1.8 },
    // Line up & clear #1 Meadow Vertical (3.5, 11.5), normal (0.96, -0.28)
    DemoWaypoint { pos: Vec2::new(0.2, 12.4), pace: DemoPace::Sprint, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(8.5, 10.0), pace: DemoPace::Sprint, reach_radius: 1.8 },
    // Curve onto & clear #2 Hunter Brush-Box (13.5, 4.5), normal (0.19, -0.98)
    DemoWaypoint { pos: Vec2::new(12.6, 8.6), pace: DemoPace::Canter, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(14.5, -0.8), pace: DemoPace::Sprint, reach_radius: 1.8 },
    // Line up & clear #3 Ascending Oxer (12.0, -8.0), normal (-0.71, -0.71)
    DemoWaypoint { pos: Vec2::new(15.0, -4.8), pace: DemoPace::Sprint, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(7.8, -12.0), pace: DemoPace::Sprint, reach_radius: 1.9 },
    // Line up & clear #4 Hillcrest Vertical (0.5, -13.5), normal (-0.99, 0.12)
    DemoWaypoint { pos: Vec2::new(4.8, -14.0), pace: DemoPace::Sprint, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(-4.8, -12.9), pace: DemoPace::Sprint, reach_radius: 1.8 },
    // Open into full Sprint along west straight through #5 Grand Prix Oxer (-12.0, -8.5)
    DemoWaypoint { pos: Vec2::new(-10.5, -13.4), pace: DemoPace::Sprint, reach_radius: 1.8 },
    DemoWaypoint { pos: Vec2::new(-13.3, -3.0), pace: DemoPace::Sprint, reach_radius: 1.9 },
    // Stay in full Sprint through #7 Triple Bar Finale (-12.5, 3.0)
    DemoWaypoint { pos: Vec2::new(-13.3, -1.2), pace: DemoPace::Sprint, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(-11.4, 8.2), pace: DemoPace::Sprint, reach_radius: 1.9 },
    // Slow down and approach #6 Narrow Skinny Gate (-2.0, -1.5) at leisurely Walk pace:
    // below v_min (3.2 m/s), naturally triggering Refusal + graceful wing avoidance!
    DemoWaypoint { pos: Vec2::new(-3.2, 6.2), pace: DemoPace::Canter, reach_radius: 1.8 },
    DemoWaypoint { pos: Vec2::new(1.2, 1.7), pace: DemoPace::Walk, reach_radius: 1.5 },
    DemoWaypoint { pos: Vec2::new(-5.6, -5.1), pace: DemoPace::Walk, reach_radius: 1.8 },
    // Swing around behind #6 and accelerate squarely along +normal (0.70, 0.71) to clear it!
    DemoWaypoint { pos: Vec2::new(-6.2, -5.8), pace: DemoPace::Sprint, reach_radius: 1.6 },
    DemoWaypoint { pos: Vec2::new(2.2, 2.7), pace: DemoPace::Sprint, reach_radius: 1.8 },
    // Flow back to the northwest start straight for a seamless continuous loop.
    DemoWaypoint { pos: Vec2::new(-6.2, 8.8), pace: DemoPace::Canter, reach_radius: 1.9 },
    DemoWaypoint { pos: Vec2::new(-14.2, 11.0), pace: DemoPace::Canter, reach_radius: 1.9 },
];

/// Suggestive player intent and mode flags (Option A ergonomics).
#[derive(Resource, Debug, Clone)]
pub struct PlayerIntent {
    /// Camera-relative suggested direction on the XZ plane (unit vector or ZERO).
    pub suggested_dir: Vec2,
    /// Whether any directional suggestion (manual or demo) is currently active.
    pub has_input: bool,
    /// Continuous directional hold duration (s) for the initial 0.45s Walk cap.
    pub hold_duration: f32,
    /// `Shift` held (or Sprint demo leg) urging faster acceleration into full Sprint.
    pub sprint_urged: bool,
    /// `C` key toggle locking into deliberate Creep / Stalk.
    pub creep_toggle: bool,
    /// `Ctrl` held locking into deliberate Creep / Stalk.
    pub creep_held: bool,
    /// Autonomous Show-Jump Course Demo enabled (starts `true`; any WASD/Arrow disables).
    pub auto_course: bool,
    /// Current demo waypoint index.
    pub demo_index: usize,
    /// Demo pace override (`Some(DemoPace::Walk)` caps demo speed for natural refusal).
    pub demo_pace: Option<DemoPace>,
    /// Pause simulation (`Space`).
    pub paused: bool,
    /// Follow camera azimuth offset (radians), adjustable with `Q` / `E`.
    pub camera_yaw: f32,
    /// Latest cursor terrain hit for Gaze priority #4.
    pub cursor_target: Option<Vec3>,
    /// Seconds elapsed since last cursor motion over the canvas.
    pub cursor_idle_secs: f32,
}

impl Default for PlayerIntent {
    fn default() -> Self {
        Self {
            suggested_dir: Vec2::ZERO,
            has_input: false,
            hold_duration: 0.0,
            sprint_urged: false,
            creep_toggle: false,
            creep_held: false,
            auto_course: true,
            demo_index: 0,
            demo_pace: Some(DemoPace::Canter),
            paused: false,
            camera_yaw: -0.55,
            cursor_target: None,
            cursor_idle_secs: 10.0,
        }
    }
}

impl PlayerIntent {
    #[inline]
    pub fn is_stalking(&self) -> bool {
        self.creep_toggle || self.creep_held
    }
}

/// Non-holonomic quadruped locomotion state published to `gait.rs`, `jump.rs`, and `ik.rs`.
#[derive(Resource, Debug, Clone)]
pub struct LocomotionState {
    pub position: Vec3,
    pub velocity_xz: Vec2,
    pub speed: f32,
    /// Facing yaw in radians (`0 = +Z`, `+PI/2 = +X`).
    pub heading: f32,
    /// Smoothed yaw angular velocity (rad/s; positive = turning left).
    pub turn_rate: f32,
    /// Centripetal lateral banking roll (rad; negative leans left into a left turn).
    pub bank_roll: f32,
    /// Final blended target direction after obstacle-avoidance flow field.
    pub blended_dir: Vec2,
    /// Active tangential wing-avoidance direction (for magenta gizmo).
    pub avoid_tangent: Vec2,
    /// Blend weight `[0, 1]` of tangential wing avoidance.
    pub avoid_weight: f32,
    /// True while active reversal braking is engaged.
    pub is_braking: bool,
    /// Dynamic extra follow-camera distance during high speed / airborne jumps.
    pub camera_zoom_extra: f32,
}

impl Default for LocomotionState {
    fn default() -> Self {
        let start_y = terrain_height(FOX_START_XZ.x, FOX_START_XZ.y);
        let fwd = Vec2::new(FOX_START_HEADING.sin(), FOX_START_HEADING.cos());
        Self {
            position: Vec3::new(FOX_START_XZ.x, start_y, FOX_START_XZ.y),
            velocity_xz: fwd * 0.4,
            speed: 0.4,
            heading: FOX_START_HEADING,
            turn_rate: 0.0,
            bank_roll: 0.0,
            blended_dir: fwd,
            avoid_tangent: Vec2::ZERO,
            avoid_weight: 0.0,
            is_braking: false,
            camera_zoom_extra: 0.0,
        }
    }
}

/// Zero-oscillation wing-avoidance side lock.
///
/// Once acquired on refusal, `side` (`+1.0` or `-1.0`) stays locked until the
/// fox clears the obstacle so dead-center approaches never oscillate left/right.
#[derive(Resource, Debug, Clone, Default)]
pub struct AvoidanceLock {
    pub active_obstacle: Option<usize>,
    /// `+1.0` for `+tangent_xz` wing, `-1.0` for `-tangent_xz` wing.
    pub side: f32,
    /// Locked approach direction sign (`+1.0` or `-1.0` along `obs.normal_xz`)
    /// captured on refusal entry so crossing `along_n = 0` never flips the target behind the fox.
    pub approach_sign: f32,
    /// World XZ clearance waypoint just outside the chosen wing tip.
    pub target_xz: Vec2,
    /// World 3D upright position glanced at via Look-At IK during refusal.
    pub wing_world_pos: Vec3,
    pub refusal_counted: bool,
}

impl AvoidanceLock {
    pub fn clear(&mut self) {
        self.active_obstacle = None;
        self.side = 0.0;
        self.approach_sign = 0.0;
        self.target_xz = Vec2::ZERO;
        self.refusal_counted = false;
    }
}

/// Active driver of the fox's multi-bone Look-At IK priority stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GazeSource {
    JumpFocus,
    AvoidedWing,
    TurnAnticipation,
    #[default]
    CursorOrFirefly,
}

impl GazeSource {
    pub const fn label(self) -> &'static str {
        match self {
            GazeSource::JumpFocus => "Jump",
            GazeSource::AvoidedWing => "Wing",
            GazeSource::TurnAnticipation => "Turn",
            GazeSource::CursorOrFirefly => "Orb",
        }
    }
}

/// Priority-stack gaze target resource consumed by `ik.rs`.
#[derive(Resource, Debug, Clone)]
pub struct GazeFocus {
    pub target: Vec3,
    pub raw_target: Vec3,
    pub source: GazeSource,
}

impl Default for GazeFocus {
    fn default() -> Self {
        let start_y = terrain_height(FOX_START_XZ.x, FOX_START_XZ.y);
        let initial = Vec3::new(FOX_START_XZ.x + 3.0, start_y + 0.9, FOX_START_XZ.y);
        Self {
            target: initial,
            raw_target: initial,
            source: GazeSource::CursorOrFirefly,
        }
    }
}

/// Speed-dependent maximum yaw turn rate (rad/s):
/// `4.2 rad/s` at low speed smoothly tightening to `1.45 rad/s` at full sprint
/// (`R_min = 6.2 / 1.45 ~= 4.28 m`).
#[inline]
pub fn compute_max_turn_rate(speed: f32) -> f32 {
    let t = smoothstep(0.5, 5.8, speed);
    4.2 + (1.45 - 4.2) * t
}

/// Computes desired speed and acceleration/deceleration rate (m/s, m/s^2) under
/// Option A suggestive input ergonomics + sharp reversal braking.
pub fn compute_speed_profile(
    has_input: bool,
    stalking: bool,
    sprint_urged: bool,
    hold_duration: f32,
    demo_pace: Option<DemoPace>,
    suggested_dir: Vec2,
    fwd_xz: Vec2,
    current_speed: f32,
) -> (f32, f32, bool) {
    if !has_input {
        // Natural 2-3 stride coast down (`Sprint -> Run -> Walk -> Creep -> Idle`).
        return (0.0, 3.8, false);
    }

    // Active braking on sharp reversal (`dot < -0.35` at `speed > 2.0 m/s`).
    if current_speed > 2.0 && suggested_dir.dot(fwd_xz) < -0.35 {
        return (1.1, 7.5, true);
    }

    // Demo pace override when Auto-Course is driving.
    if let Some(pace) = demo_pace {
        return match pace {
            DemoPace::Walk => (1.55, 2.8, false),
            DemoPace::Canter => (3.55, 3.4, false),
            DemoPace::Sprint => (5.65, 5.4, false),
        };
    }

    if stalking {
        let rate = if current_speed > 0.72 { 4.2 } else { 2.0 };
        return (0.72, rate, false);
    }

    if sprint_urged {
        return (6.20, 5.6, false);
    }

    // Normal WASD hold: first 0.45s stays gently capped in Walk (~1.75 m/s),
    // then sustained hold ramps smoothly to a comfortable Run (~2.95 m/s).
    let target = if hold_duration < 0.45 { 1.75 } else { 2.95 };
    let rate = if current_speed > target { 3.8 } else { 2.8 };
    (target, rate, false)
}

/// Centripetal lateral body banking roll (radians) for `+Z` forward, `+X` left:
/// turning left (`turn_rate > 0`) leans the left side (`+X`) down via negative Z rotation.
#[inline]
pub fn compute_bank_roll(speed: f32, turn_rate: f32) -> f32 {
    (-((speed * turn_rate).atan2(9.81)) * 0.65).clamp(-0.32, 0.32)
}

/// Pure helper that chooses an avoidance wing side (`+1.0` or `-1.0`) once on
/// obstacle entry and keeps that side locked without flipping while active.
pub fn resolve_avoidance_side(
    lock: &mut AvoidanceLock,
    obs: &CourseObstacle,
    pos_xz: Vec2,
    fwd_xz: Vec2,
    suggested_dir: Vec2,
) -> f32 {
    if lock.active_obstacle == Some(obs.id) && lock.side.abs() > 0.5 {
        return lock.side;
    }
    let (_, along_t) = local_fence_coords(obs, pos_xz);
    let score = along_t * 1.4
        + suggested_dir.dot(obs.tangent_xz) * 0.9
        + fwd_xz.dot(obs.tangent_xz) * 0.6;
    let side = if score >= 0.0 { 1.0 } else { -1.0 };
    lock.active_obstacle = Some(obs.id);
    lock.side = side;
    side
}

/// Reads keyboard input, camera-relative direction, Option A hold timer, and
/// Auto-Course Demo waypoints into [`PlayerIntent`].
#[allow(clippy::too_many_arguments)]
pub fn read_player_intent(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut intent: ResMut<PlayerIntent>,
    mut ik_settings: ResMut<IkSettings>,
    loco: Res<LocomotionState>,
    cam_query: Query<&Transform, With<FollowCamera>>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let dt = time.delta_secs().min(0.1);

    // Global toggles.
    if keys.just_pressed(KeyCode::KeyI) {
        ik_settings.enabled = !ik_settings.enabled;
    }
    if keys.just_pressed(KeyCode::KeyG) {
        ik_settings.show_gizmos = !ik_settings.show_gizmos;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        intent.auto_course = !intent.auto_course;
        if intent.auto_course {
            intent.creep_toggle = false;
        }
    }
    if keys.just_pressed(KeyCode::KeyC) {
        intent.creep_toggle = !intent.creep_toggle;
    }
    if keys.just_pressed(KeyCode::Space) {
        intent.paused = !intent.paused;
        for mut player in &mut players {
            if intent.paused {
                player.pause_all();
            } else {
                player.resume_all();
            }
        }
    }

    // Gentle camera orbit via Q / E.
    if keys.pressed(KeyCode::KeyQ) {
        intent.camera_yaw -= 1.45 * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        intent.camera_yaw += 1.45 * dt;
    }

    // Read manual directional keys (WASD + Arrows).
    let mut axes = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        axes.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        axes.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        axes.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        axes.x -= 1.0;
    }

    intent.creep_held =
        keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let shift_held = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // Pressing any directional key immediately hands control cleanly to the player.
    if axes.length_squared() > 0.01 {
        intent.auto_course = false;
    }

    if intent.paused {
        return;
    }

    if intent.auto_course {
        advance_demo_waypoint(&mut intent, &loco);
    } else if axes.length_squared() > 0.01 {
        let cam_fwd_xz = cam_query
            .single()
            .map(|c| {
                let f = c.forward().as_vec3();
                Vec2::new(f.x, f.z).normalize_or(Vec2::NEG_Y)
            })
            .unwrap_or(Vec2::new(0.0, -1.0));
        let cam_right_xz = Vec2::new(-cam_fwd_xz.y, cam_fwd_xz.x);

        intent.suggested_dir = (cam_fwd_xz * axes.y + cam_right_xz * axes.x).normalize_or_zero();
        intent.has_input = intent.suggested_dir.length_squared() > 0.01;
        intent.hold_duration = (intent.hold_duration + dt).min(3.0);
        intent.sprint_urged = shift_held;
        intent.demo_pace = None;
    } else {
        intent.suggested_dir = Vec2::ZERO;
        intent.has_input = false;
        intent.hold_duration = 0.0;
        intent.sprint_urged = false;
        intent.demo_pace = None;
    }
}

fn advance_demo_waypoint(intent: &mut PlayerIntent, loco: &LocomotionState) {
    let pos_xz = Vec2::new(loco.position.x, loco.position.z);
    let wp = DEMO_COURSE[intent.demo_index];
    let to_wp = wp.pos - pos_xz;
    let dist = to_wp.length();
    if dist < wp.reach_radius {
        intent.demo_index = (intent.demo_index + 1) % DEMO_COURSE.len();
    }
    let next_wp = DEMO_COURSE[intent.demo_index];
    let dir = (next_wp.pos - pos_xz).normalize_or(Vec2::new(
        loco.heading.sin(),
        loco.heading.cos(),
    ));
    intent.suggested_dir = dir;
    intent.has_input = true;
    intent.hold_duration = 1.0;
    intent.sprint_urged = next_wp.pace == DemoPace::Sprint;
    intent.demo_pace = Some(next_wp.pace);
}

#[cfg(test)]
pub fn step_auto_course_intent(
    mut intent: ResMut<PlayerIntent>,
    loco: Res<LocomotionState>,
) {
    advance_demo_waypoint(&mut intent, &loco);
}

/// Blends suggestive heading with tangential wing avoidance, enforces speed-dependent
/// non-holonomic turn limits, integrates heading & speed, and computes centripetal banking.
pub fn update_steering_and_avoidance(
    time: Res<Time>,
    intent: Res<PlayerIntent>,
    avoid: Res<AvoidanceLock>,
    jump: Res<JumpState>,
    mut loco: ResMut<LocomotionState>,
) {
    let dt = time.delta_secs().min(0.1);
    if intent.paused {
        return;
    }

    // While airborne, ballistic physics govern motion; smoothly relax banking roll to level.
    if jump.phase == JumpPhase::Airborne {
        loco.avoid_weight = 0.0;
        loco.avoid_tangent = Vec2::ZERO;
        loco.is_braking = false;
        loco.turn_rate.smooth_nudge(&0.0, 8.0, dt);
        loco.bank_roll.smooth_nudge(&0.0, 9.0, dt);
        return;
    }

    // During the brief Gather takeoff coil, freeze speed and smoothly square heading
    // toward the solved launch vector so liftoff has zero yaw snap.
    if jump.phase == JumpPhase::Gather {
        loco.avoid_weight = 0.0;
        loco.avoid_tangent = Vec2::ZERO;
        loco.is_braking = false;
        if let Some(traj) = jump.trajectory {
            let launch_heading = traj.vel_xz.x.atan2(traj.vel_xz.y);
            let diff = wrap_angle(launch_heading - loco.heading);
            let step = diff * (1.0 - (-18.0 * dt).exp());
            loco.heading = wrap_angle(loco.heading + step);
            loco.blended_dir = traj.vel_xz.normalize_or(Vec2::new(loco.heading.sin(), loco.heading.cos()));
        }
        loco.turn_rate.smooth_nudge(&0.0, 12.0, dt);
        loco.bank_roll.smooth_nudge(&0.0, 12.0, dt);
        let fwd = Vec2::new(loco.heading.sin(), loco.heading.cos());
        loco.velocity_xz = fwd * loco.speed;
        return;
    }

    let pos_xz = Vec2::new(loco.position.x, loco.position.z);
    let fwd_xz = Vec2::new(loco.heading.sin(), loco.heading.cos());

    // 1. Compute tangential wing-avoidance flow vector if AvoidanceLock is active.
    let (avoid_dir, avoid_w) = if avoid.active_obstacle.is_some() {
        let to_clear = avoid.target_xz - pos_xz;
        let d = to_clear.length();
        let dir = if d > 0.05 {
            to_clear / d
        } else {
            fwd_xz
        };
        let w = (1.0 - smoothstep(1.2, 4.4, d)).clamp(0.42, 0.88);
        (dir, w)
    } else {
        (Vec2::ZERO, 0.0)
    };
    loco.avoid_tangent = avoid_dir;
    loco.avoid_weight = avoid_w;

    // 2. Blend player's suggested heading with avoidance tangent.
    let blended_dir = if intent.has_input && avoid_w > 0.01 {
        (intent.suggested_dir * (1.0 - avoid_w) + avoid_dir * avoid_w).normalize_or(avoid_dir)
    } else if avoid_w > 0.01 {
        avoid_dir
    } else if intent.has_input {
        intent.suggested_dir
    } else {
        fwd_xz
    };
    loco.blended_dir = blended_dir;

    // 3. Speed-dependent non-holonomic turn limit & angular inertia.
    let omega_max = compute_max_turn_rate(loco.speed);
    let mut delta_theta = 0.0_f32;
    let omega_des = if intent.has_input || avoid_w > 0.01 {
        let desired_heading = blended_dir.x.atan2(blended_dir.y);
        delta_theta = wrap_angle(desired_heading - loco.heading);
        // Proportional turn demand clamped strictly by speed-dependent omega_max(v).
        (delta_theta * 4.8).clamp(-omega_max, omega_max)
    } else {
        0.0
    };

    loco.turn_rate.smooth_nudge(&omega_des, 9.0, dt);
    loco.heading = wrap_angle(loco.heading + loco.turn_rate * dt);

    // 4. Option A speed profile + reversal braking + avoidance easing + tight-corner scrub.
    let (mut target_speed, accel_rate, is_braking) = compute_speed_profile(
        intent.has_input,
        intent.is_stalking(),
        intent.sprint_urged,
        intent.hold_duration,
        intent.demo_pace,
        blended_dir,
        fwd_xz,
        loco.speed,
    );
    loco.is_braking = is_braking;

    if avoid_w > 0.01 {
        target_speed *= 1.0 - 0.32 * avoid_w;
    }
    if intent.has_input || avoid_w > 0.01 {
        let corner_factor = 0.70 + 0.30 * delta_theta.cos().max(0.0);
        target_speed *= corner_factor;
    }

    // Move speed linearly toward target_speed using calibrated accel/decel (m/s^2).
    let diff = target_speed - loco.speed;
    let max_step = accel_rate * dt;
    if diff.abs() <= max_step {
        loco.speed = target_speed;
    } else {
        loco.speed += diff.signum() * max_step;
    }
    loco.speed = loco.speed.max(0.0);

    // Strictly non-holonomic forward motion along the fox's facing heading.
    let new_fwd = Vec2::new(loco.heading.sin(), loco.heading.cos());
    loco.velocity_xz = new_fwd * loco.speed;

    // 5. Centripetal banking roll.
    let target_bank = compute_bank_roll(loco.speed, loco.turn_rate);
    loco.bank_roll.smooth_nudge(&target_bank, 9.5, dt);
}

/// Evaluates the 4-tier gaze priority stack:
/// 1. Jump Focus (top rail on Gather/ascent -> landing turf on descent)
/// 2. Avoided Fence Wing (during refusal/run-out)
/// 3. Turn Anticipation (projecting ahead into curves)
/// 4. Mouse Cursor terrain raycast / Orbiting Firefly fallback
#[allow(clippy::too_many_arguments)]
pub fn update_gaze_priority(
    time: Res<Time>,
    windows: Query<&Window>,
    mut cursor_events: MessageReader<CursorMoved>,
    cam_query: Query<(&Camera, &GlobalTransform), With<FollowCamera>>,
    mut intent: ResMut<PlayerIntent>,
    loco: Res<LocomotionState>,
    jump: Res<JumpState>,
    avoid: Res<AvoidanceLock>,
    mut gaze: ResMut<GazeFocus>,
    mut orb_query: Query<&mut Transform, (With<FireflyOrb>, Without<FollowCamera>)>,
) {
    let dt = time.delta_secs().min(0.1);

    // Track live pointer motion over the canvas for priority #4.
    let mut latest_cursor = None;
    for ev in cursor_events.read() {
        latest_cursor = Some(ev.position);
    }
    if let Some(cursor_pos) = latest_cursor {
        if let Ok((camera, cam_gt)) = cam_query.single() {
            if let Ok(ray) = camera.viewport_to_world(cam_gt, cursor_pos) {
                if let Some(hit) = raycast_terrain(ray, 70.0) {
                    intent.cursor_target = Some(hit + Vec3::Y * 0.55);
                    intent.cursor_idle_secs = 0.0;
                }
            }
        }
    } else {
        let cursor_inside = windows
            .single()
            .ok()
            .and_then(|w| w.cursor_position())
            .is_some();
        intent.cursor_idle_secs += if cursor_inside { dt } else { dt * 2.5 };
    }

    // Compute ambient firefly / cursor orb position (always moves smoothly in world).
    let firefly_pos = if intent.cursor_idle_secs < 4.5 {
        intent
            .cursor_target
            .unwrap_or_else(|| orbit_firefly_position(&loco, time.elapsed_secs()))
    } else {
        orbit_firefly_position(&loco, time.elapsed_secs())
    };
    if let Ok(mut orb_tf) = orb_query.single_mut() {
        orb_tf.translation.smooth_nudge(&firefly_pos, 9.5, dt);
    }

    let fwd_xz = Vec2::new(loco.heading.sin(), loco.heading.cos());
    let desired_heading = loco.blended_dir.x.atan2(loco.blended_dir.y);
    let turn_error = wrap_angle(desired_heading - loco.heading);

    // Priority stack selection.
    let (raw_target, source) = if matches!(jump.phase, JumpPhase::Gather | JumpPhase::Airborne) {
        if let Some(traj) = jump.trajectory {
            let apex_u = (traj.apex_time / traj.total_time.max(0.05)).clamp(0.35, 0.55);
            let pt = if jump.phase == JumpPhase::Gather || jump.flight_u < apex_u {
                traj.rail_top_pos
            } else {
                traj.landing_pos + Vec3::Y * 0.35
            };
            (pt, GazeSource::JumpFocus)
        } else {
            (firefly_pos, GazeSource::CursorOrFirefly)
        }
    } else if avoid.active_obstacle.is_some() {
        (avoid.wing_world_pos, GazeSource::AvoidedWing)
    } else if loco.speed > 0.65 && (turn_error.abs() > 0.16 || loco.turn_rate.abs() > 0.32) {
        let look_ahead = 2.2 + 0.45 * loco.speed;
        let inward_bias = (loco.blended_dir * 0.78 + fwd_xz * 0.22).normalize_or(fwd_xz);
        let target_xz = Vec2::new(loco.position.x, loco.position.z) + inward_bias * look_ahead;
        let y = terrain_height(target_xz.x, target_xz.y) + 0.65;
        (Vec3::new(target_xz.x, y, target_xz.y), GazeSource::TurnAnticipation)
    } else {
        (firefly_pos, GazeSource::CursorOrFirefly)
    };

    gaze.raw_target = raw_target;
    gaze.source = source;
    gaze.target.smooth_nudge(&raw_target, 10.5, dt);
}

fn orbit_firefly_position(loco: &LocomotionState, t: f32) -> Vec3 {
    let fwd = Vec2::new(loco.heading.sin(), loco.heading.cos());
    let left = Vec2::new(loco.heading.cos(), -loco.heading.sin());
    let sweep = (t * 0.85).sin() * 1.55;
    let ahead = 2.4 + (t * 0.55).cos() * 0.60;
    let pos_xz = Vec2::new(loco.position.x, loco.position.z) + fwd * ahead + left * sweep;
    let ground = terrain_height(pos_xz.x, pos_xz.y);
    let hover = 0.95 + (t * 1.9).sin() * 0.35;
    Vec3::new(pos_xz.x, ground + hover, pos_xz.y)
}

/// Wraps an angle in radians into `[-PI, PI]`.
#[inline]
pub fn wrap_angle(mut a: f32) -> f32 {
    while a > PI {
        a -= TAU;
    }
    while a < -PI {
        a += TAU;
    }
    a
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacles::build_course_obstacles;

    #[test]
    fn speed_dependent_turn_rate_tightens_with_speed() {
        let low = compute_max_turn_rate(0.3);
        let mid = compute_max_turn_rate(3.0);
        let sprint = compute_max_turn_rate(6.2);

        assert!((low - 4.2).abs() < 1e-4);
        assert!(mid < low && mid > sprint);
        assert!((sprint - 1.45).abs() < 1e-4);

        let r_min_sprint = 6.2 / sprint;
        assert!(
            (r_min_sprint - 4.2758).abs() < 0.05,
            "sprint min turning radius should be ~4.28m, got {r_min_sprint}"
        );
    }

    #[test]
    fn option_a_speed_curves_and_reversal_braking() {
        let fwd = Vec2::new(0.0, 1.0);

        // No input -> coast down
        let (t, decel, brk) = compute_speed_profile(false, false, false, 0.0, None, Vec2::ZERO, fwd, 4.0);
        assert_eq!(t, 0.0);
        assert!((decel - 3.8).abs() < 1e-4);
        assert!(!brk);

        // Creep / Stalk
        let (t, acc, _) = compute_speed_profile(true, true, false, 0.8, None, fwd, fwd, 0.2);
        assert!((t - 0.72).abs() < 1e-4);
        assert!((acc - 2.0).abs() < 1e-4);

        // Initial 0.45s WASD hold stays in Walk cap (~1.75 m/s)
        let (t, _, _) = compute_speed_profile(true, false, false, 0.20, None, fwd, fwd, 0.5);
        assert!((t - 1.75).abs() < 1e-4);

        // Sustained WASD hold (>0.45s) ramps to comfortable Run (~2.95 m/s)
        let (t, acc, _) = compute_speed_profile(true, false, false, 0.70, None, fwd, fwd, 1.8);
        assert!((t - 2.95).abs() < 1e-4);
        assert!((acc - 2.8).abs() < 1e-4);

        // Shift urges full Sprint (6.20 m/s, 5.6 m/s^2)
        let (t, acc, _) = compute_speed_profile(true, false, true, 0.10, None, fwd, fwd, 2.0);
        assert!((t - 6.20).abs() < 1e-4);
        assert!((acc - 5.6).abs() < 1e-4);

        // Sharp reversal braking when dot < -0.35 at speed > 2.0
        let (t, decel, brk) = compute_speed_profile(true, false, true, 0.5, None, -fwd, fwd, 4.5);
        assert!(brk);
        assert!((t - 1.1).abs() < 1e-4);
        assert!((decel - 7.5).abs() < 1e-4);
    }

    #[test]
    fn centripetal_bank_roll_leans_into_turns() {
        // Left turn (turn_rate > 0) -> negative Z roll drops +X (left side down).
        let left_bank = compute_bank_roll(5.0, 1.2);
        assert!(left_bank < -0.15 && left_bank >= -0.32);

        // Right turn (turn_rate < 0) -> positive Z roll drops -X (right side down).
        let right_bank = compute_bank_roll(5.0, -1.2);
        assert!(right_bank > 0.15 && right_bank <= 0.32);
    }

    #[test]
    fn avoidance_side_lock_never_oscillates_dead_center() {
        let obstacles = build_course_obstacles();
        let obs = &obstacles[6]; // Narrow Skinny Gate
        let mut lock = AvoidanceLock::default();

        // Approach dead center (along_t == 0.0) and slightly jitter left/right across frames.
        let center_pos = obs.center_xz - obs.normal_xz * 2.0;
        let first_side = resolve_avoidance_side(&mut lock, obs, center_pos, obs.normal_xz, obs.normal_xz);

        for i in 0..10 {
            let jitter = if i % 2 == 0 { -0.25 } else { 0.25 };
            let p = center_pos + obs.tangent_xz * jitter;
            let side = resolve_avoidance_side(&mut lock, obs, p, obs.normal_xz, obs.normal_xz);
            assert_eq!(
                side, first_side,
                "avoidance side must stay locked without oscillating on frame {i}"
            );
        }
    }
}
