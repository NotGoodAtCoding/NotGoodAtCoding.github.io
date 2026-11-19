//! Continuous 5-gait state machine (`Idle` <-> `Creep` <-> `Walk` <-> `Run` <-> `Sprint`)
//! with hysteresis, anti-chatter dwell, `AnimationTransitions` over `Fox.glb`'s 3 clips,
//! phase extraction for gallop spine bounce, and continuous procedural posture weights.

use std::f32::consts::TAU;
use std::time::Duration;

use bevy::prelude::*;

use crate::controls::{LocomotionState, PlayerIntent};
use crate::jump::{JumpPhase, JumpState};

/// Reference ground speeds (m/s) matched to `Fox.glb` Walk and Run stride lengths.
pub const WALK_REF_SPEED: f32 = 1.65;
pub const RUN_REF_SPEED: f32 = 4.20;

/// Cycle duration of `Fox.glb`'s embedded `Run` clip (20 frames @ 24 fps = 0.8333 s).
const RUN_CLIP_PERIOD: f32 = 0.8333;

/// Minimum time (seconds) a gait stays active before transitioning again to
/// eliminate chatter around speed boundaries.
pub const GAIT_DWELL_SECS: f32 = 0.14;

pub const CLIP_SURVEY: usize = 0;
pub const CLIP_WALK: usize = 1;
pub const CLIP_RUN: usize = 2;

/// Five continuous locomotion gaits synthesized from 3 glTF clips plus
/// procedural posture modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gait {
    #[default]
    Idle,
    Creep,
    Walk,
    Run,
    Sprint,
}

impl Gait {
    pub const fn label(self) -> &'static str {
        match self {
            Gait::Idle => "Idle",
            Gait::Creep => "Creep",
            Gait::Walk => "Walk",
            Gait::Run => "Run",
            Gait::Sprint => "Sprint",
        }
    }
}

/// Embedded `Fox.glb` animation graph handles (`Survey`, `Walk`, `Run`).
#[derive(Resource)]
pub struct FoxAnimations {
    pub nodes: [AnimationNodeIndex; 3],
    pub graph: Handle<AnimationGraph>,
}

/// Active gait state and clip phase read back from `AnimationPlayer`.
#[derive(Resource, Debug, Clone)]
pub struct GaitState {
    pub current: Gait,
    pub dwell_secs: f32,
    pub active_clip: usize,
    pub clip_seek_secs: f32,
}

impl Default for GaitState {
    fn default() -> Self {
        Self {
            current: Gait::Idle,
            dwell_secs: GAIT_DWELL_SECS,
            active_clip: CLIP_SURVEY,
            clip_seek_secs: 0.0,
        }
    }
}

/// Continuous procedural posture weights consumed by `ik.rs` so `Creep <-> Walk`
/// and `Run <-> Sprint` morph the fox's silhouette seamlessly without clip restarts.
#[derive(Resource, Debug, Clone, Default)]
pub struct GaitPosture {
    /// `[0.0, 1.0]` crouch/stalk posture weight.
    pub creep_w: f32,
    /// `[0.0, 1.0]` full-gallop posture weight.
    pub sprint_w: f32,
    /// Phase angle (radians) derived from `ActiveAnimation::seek_time()` for
    /// phase-locked gallop spine bounce and tail flutter.
    pub gallop_phase: f32,
}

/// Maps each of the 5 gaits onto one of `Fox.glb`'s 3 embedded clips.
///
/// Crucially, `Creep`/`Walk` share `CLIP_WALK` and `Run`/`Sprint` share `CLIP_RUN`
/// so crossing those boundaries triggers zero clip restart.
#[inline]
pub const fn clip_for_gait(gait: Gait) -> usize {
    match gait {
        Gait::Idle => CLIP_SURVEY,
        Gait::Creep | Gait::Walk => CLIP_WALK,
        Gait::Run | Gait::Sprint => CLIP_RUN,
    }
}

/// Returns the cross-fade duration when changing underlying glTF clips.
#[inline]
pub const fn blend_duration_for_clips(from_clip: usize, to_clip: usize) -> Duration {
    if from_clip == to_clip {
        Duration::ZERO
    } else if from_clip == CLIP_SURVEY || to_clip == CLIP_SURVEY {
        Duration::from_millis(320)
    } else {
        Duration::from_millis(240)
    }
}

/// Computes stride-matched playback speed for a given gait and ground speed.
#[inline]
pub fn playback_rate_for_gait(gait: Gait, speed: f32) -> f32 {
    match gait {
        Gait::Idle => 1.0,
        Gait::Creep => (speed / WALK_REF_SPEED).clamp(0.32, 0.72),
        Gait::Walk => (speed / WALK_REF_SPEED).clamp(0.68, 1.45),
        Gait::Run => (speed / RUN_REF_SPEED).clamp(0.62, 1.25),
        Gait::Sprint => (speed / RUN_REF_SPEED).clamp(1.10, 1.58),
    }
}

/// Pure hysteresis state machine step for the 5 gaits with anti-chatter dwell.
///
/// Rising / falling thresholds:
/// - `Idle  <-> Creep`:  enter `>= 0.20`, exit `< 0.14`
/// - `Creep <-> Walk`:   enter `>= 1.15` (when not stalking), exit `< 0.90` (or stalking)
/// - `Walk  <-> Run`:    enter `>= 2.65`, exit `< 2.25`
/// - `Run   <-> Sprint`: enter `>= 4.65`, exit `< 4.15`
pub fn step_gait_machine(
    current: Gait,
    speed: f32,
    stalking: bool,
    dwell_secs: f32,
    dt: f32,
) -> (Gait, f32) {
    let next_dwell = dwell_secs + dt;
    if next_dwell < GAIT_DWELL_SECS {
        return (current, next_dwell);
    }

    let candidate = match current {
        Gait::Idle => {
            if speed >= 0.20 {
                Gait::Creep
            } else {
                Gait::Idle
            }
        }
        Gait::Creep => {
            if speed < 0.14 {
                Gait::Idle
            } else if !stalking && speed >= 1.15 {
                Gait::Walk
            } else {
                Gait::Creep
            }
        }
        Gait::Walk => {
            if stalking || speed < 0.90 {
                Gait::Creep
            } else if speed >= 2.65 {
                Gait::Run
            } else {
                Gait::Walk
            }
        }
        Gait::Run => {
            if speed < 2.25 {
                Gait::Walk
            } else if speed >= 4.65 {
                Gait::Sprint
            } else {
                Gait::Run
            }
        }
        Gait::Sprint => {
            if speed < 4.15 {
                Gait::Run
            } else {
                Gait::Sprint
            }
        }
    };

    if candidate != current {
        (candidate, 0.0)
    } else {
        (current, next_dwell)
    }
}

/// Attaches the [`AnimationGraphHandle`] and [`AnimationTransitions`] to the
/// fox's [`AnimationPlayer`] once the glTF scene spawns.
pub fn setup_animation_player(
    mut commands: Commands,
    animations: Res<FoxAnimations>,
    intent: Res<PlayerIntent>,
    gait_state: Res<GaitState>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    for (entity, mut player) in &mut players {
        let mut transitions = AnimationTransitions::new();
        let anim = transitions
            .play(
                &mut player,
                animations.nodes[gait_state.active_clip],
                Duration::ZERO,
            )
            .repeat();
        if intent.paused {
            anim.pause();
        }

        commands
            .entity(entity)
            .insert((AnimationGraphHandle(animations.graph.clone()), transitions));
    }
}

/// Advances the 5-gait hysteresis state machine, updates Bevy `AnimationTransitions`
/// without restarting shared clips, extracts `seek_time` for gallop spine bounce,
/// and updates continuous `GaitPosture` weights.
pub fn update_gait_and_animations(
    time: Res<Time>,
    animations: Res<FoxAnimations>,
    intent: Res<PlayerIntent>,
    loco: Res<LocomotionState>,
    jump: Res<JumpState>,
    mut gait_state: ResMut<GaitState>,
    mut posture: ResMut<GaitPosture>,
    mut query: Query<(&mut AnimationPlayer, &mut AnimationTransitions)>,
) {
    let dt = time.delta_secs().min(0.1);
    if intent.paused {
        return;
    }

    let stalking = intent.is_stalking() && !intent.auto_course;

    // While airborne, keep gait aligned to Run/Sprint without dropping to Walk mid-jump.
    let effective_speed = match jump.phase {
        JumpPhase::Airborne => loco.speed.max(3.4),
        _ => loco.speed,
    };

    let (next_gait, next_dwell) = step_gait_machine(
        gait_state.current,
        effective_speed,
        stalking && jump.phase == JumpPhase::Grounded,
        gait_state.dwell_secs,
        dt,
    );
    gait_state.current = next_gait;
    gait_state.dwell_secs = next_dwell;

    let desired_clip = match jump.phase {
        JumpPhase::Gather | JumpPhase::Airborne => CLIP_RUN,
        _ => clip_for_gait(gait_state.current),
    };

    let target_rate = match jump.phase {
        // Freeze/slow the run cycle during takeoff coil and flight so IK owns the
        // jump silhouette cleanly regardless of prior stride phase.
        JumpPhase::Airborne => 0.12,
        JumpPhase::Gather => 0.25,
        _ => playback_rate_for_gait(gait_state.current, loco.speed),
    };

    // Detect fresh touchdown so we sync `Run` to its front-paw strike phase (0.50 s)
    // and transition out of landing without mid-stride popping.
    let sync_landing_stride = jump.phase == JumpPhase::Landing && jump.phase_time <= dt * 1.5;

    for (mut player, mut transitions) in &mut query {
        if desired_clip != gait_state.active_clip {
            let blend = blend_duration_for_clips(gait_state.active_clip, desired_clip);
            gait_state.active_clip = desired_clip;
            transitions
                .play(&mut player, animations.nodes[desired_clip], blend)
                .repeat();
        }

        let active_node = animations.nodes[gait_state.active_clip];
        for (node_idx, active) in player.playing_animations_mut() {
            if sync_landing_stride && *node_idx == animations.nodes[CLIP_RUN] {
                active.seek_to(0.50);
            }
            active.set_speed(target_rate);
            if *node_idx == active_node {
                gait_state.clip_seek_secs = active.seek_time();
            }
        }
    }

    // Continuous procedural posture weights.
    let target_creep_w = if jump.phase != JumpPhase::Grounded {
        0.0
    } else if gait_state.current == Gait::Creep {
        1.0
    } else if stalking && gait_state.current == Gait::Idle {
        0.70
    } else {
        0.0
    };

    let target_sprint_w = if jump.phase == JumpPhase::Airborne {
        0.35
    } else {
        smoothstep(3.85, 5.55, loco.speed)
    };

    posture.creep_w.smooth_nudge(&target_creep_w, 8.5, dt);
    posture.sprint_w.smooth_nudge(&target_sprint_w, 7.5, dt);
    posture.gallop_phase = (gait_state.clip_seek_secs / RUN_CLIP_PERIOD) * TAU;
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
    fn gait_hysteresis_prevents_chatter_across_all_boundaries() {
        // Rising sequence: Idle -> Creep -> Walk -> Run -> Sprint
        let mut g = Gait::Idle;
        let mut dwell = GAIT_DWELL_SECS;

        (g, dwell) = step_gait_machine(g, 0.25, false, dwell, 0.02);
        assert_eq!(g, Gait::Creep);
        assert_eq!(dwell, 0.0);

        // Inside dwell window: must not immediately jump to Walk.
        (g, dwell) = step_gait_machine(g, 1.50, false, dwell, 0.05);
        assert_eq!(g, Gait::Creep);

        // Past dwell: enters Walk.
        (g, dwell) = step_gait_machine(g, 1.50, false, dwell, 0.12);
        assert_eq!(g, Gait::Walk);

        // Speed dips into hysteresis band (0.95 is between 0.90 exit and 1.15 enter): stays Walk!
        (g, dwell) = step_gait_machine(g, 0.95, false, dwell + 0.2, 0.02);
        assert_eq!(g, Gait::Walk);

        // Rises into Run, then Sprint.
        (g, dwell) = step_gait_machine(g, 2.80, false, dwell + 0.2, 0.02);
        assert_eq!(g, Gait::Run);
        (g, dwell) = step_gait_machine(g, 4.80, false, dwell + 0.2, 0.02);
        assert_eq!(g, Gait::Sprint);

        // Speed dips to 4.30 (inside Sprint/Run hysteresis band 4.15..4.65): stays Sprint!
        (g, dwell) = step_gait_machine(g, 4.30, false, dwell + 0.2, 0.02);
        assert_eq!(g, Gait::Sprint);

        // Drops below 4.15: returns to Run.
        (g, _) = step_gait_machine(g, 4.00, false, dwell + 0.2, 0.02);
        assert_eq!(g, Gait::Run);
    }

    #[test]
    fn stalk_mode_holds_creep_and_shared_clips_avoid_restart() {
        let (g, _) = step_gait_machine(Gait::Creep, 1.50, true, 0.5, 0.02);
        assert_eq!(g, Gait::Creep, "stalking should cap gait in Creep");

        assert_eq!(clip_for_gait(Gait::Creep), clip_for_gait(Gait::Walk));
        assert_eq!(clip_for_gait(Gait::Run), clip_for_gait(Gait::Sprint));
        assert_eq!(
            blend_duration_for_clips(clip_for_gait(Gait::Creep), clip_for_gait(Gait::Walk)),
            Duration::ZERO
        );
        assert_eq!(
            blend_duration_for_clips(clip_for_gait(Gait::Run), clip_for_gait(Gait::Sprint)),
            Duration::ZERO
        );
    }
}
