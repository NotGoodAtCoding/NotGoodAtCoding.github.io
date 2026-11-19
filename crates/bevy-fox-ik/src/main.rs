mod ik;
mod terrain;

use std::f32::consts::{PI, TAU};
use std::time::Duration;

use bevy::app::AnimationSystems;
use bevy::asset::AssetMetaCheck;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::ik::{
    apply_fox_ik, bind_fox_rig, restore_pre_ik_pose, FoxBody, FoxRig, FoxRoot, IkSettings,
    FOX_SCALE,
};
use crate::terrain::{
    build_terrain_mesh, raycast_terrain, terrain_height, PLAY_AREA_RADIUS, TERRAIN_HALF_EXTENT,
};

const FOX_PATH: &str = "models/Fox.glb";

/// Reference ground speeds (m/s) matched to `Fox.glb` Walk and Run stride lengths
/// to minimize foot sliding when scaling playback speed.
const WALK_REF_SPEED: f32 = 1.65;
const RUN_REF_SPEED: f32 = 4.20;

const CLIP_SURVEY: usize = 0;
const CLIP_WALK: usize = 1;
const CLIP_RUN: usize = 2;
const CLIP_NAMES: [&str; 3] = ["Survey", "Walk", "Run"];

#[derive(Resource)]
struct FoxAnimations {
    nodes: [AnimationNodeIndex; 3],
    graph: Handle<AnimationGraph>,
}

#[derive(Resource)]
struct FoxController {
    auto_roam: bool,
    /// When `Some(idx)`, user explicitly cycled clips via `Enter`.
    manual_clip: Option<usize>,
    active_clip: usize,
    speed_multiplier: f32,
    paused: bool,
    /// Current horizontal velocity in world space (m/s).
    velocity: Vec2,
    /// Current heading yaw in radians (0 = +Z).
    heading: f32,
    /// Auto-roam waypoint & dwell timer.
    roam_target: Vec2,
    roam_wait: f32,
    roam_index: usize,
    /// Camera azimuth offset (radians) adjustable via Q/E.
    camera_yaw: f32,
    /// Seconds since last pointer hover over the canvas.
    cursor_idle_secs: f32,
    cursor_target: Option<Vec3>,
}

impl Default for FoxController {
    fn default() -> Self {
        Self {
            auto_roam: true,
            manual_clip: None,
            active_clip: CLIP_WALK,
            speed_multiplier: 1.0,
            paused: false,
            velocity: Vec2::ZERO,
            heading: 0.4,
            roam_target: Vec2::new(2.8, 2.0),
            roam_wait: 0.0,
            roam_index: 0,
            camera_yaw: 0.65,
            cursor_idle_secs: 10.0,
            cursor_target: None,
        }
    }
}

#[derive(Component)]
struct FireflyOrb;

#[derive(Component)]
struct FollowCamera;

#[derive(Component)]
struct HudText;

/// Roam loop waypoints crossing rolling hills and stepping mounds so all four
/// legs continually demonstrate terrain-adaptive IK on page load.
const ROAM_WAYPOINTS: [(f32, f32, bool); 8] = [
    (2.6, 1.8, false),
    (5.2, -1.5, true),
    (3.6, -3.2, false),
    (-1.8, -3.5, false),
    (-5.2, -0.8, true),
    (-3.2, 2.5, false),
    (0.0, 5.2, false),
    (4.6, 3.0, true),
];

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.56, 0.73, 0.88)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.84, 0.90, 1.0),
            brightness: 420.0,
            ..default()
        })
        .init_resource::<IkSettings>()
        .init_resource::<FoxController>()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy Fox IK".into(),
                        canvas: Some("#bevy-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    #[cfg(not(target_arch = "wasm32"))]
                    file_path: "../../playground/bevy-fox-ik/assets".into(),
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        .add_systems(Startup, (setup_scene, setup_hud))
        .add_systems(
            Update,
            (
                setup_animation_player,
                bind_fox_rig,
                handle_controls,
                update_locomotion,
                update_animation_state,
                update_look_target,
                update_follow_camera,
                update_hud,
            )
                .chain(),
        )
        .add_systems(
            PostUpdate,
            (
                restore_pre_ik_pose.before(AnimationSystems),
                apply_fox_ik
                    .after(AnimationSystems)
                    .before(TransformSystems::Propagate),
            ),
        )
        .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    // Build AnimationGraph from Fox.glb's three embedded clips: Survey, Walk, Run.
    let (graph, indices) = AnimationGraph::from_clips([
        asset_server.load(GltfAssetLabel::Animation(0).from_asset(FOX_PATH)),
        asset_server.load(GltfAssetLabel::Animation(1).from_asset(FOX_PATH)),
        asset_server.load(GltfAssetLabel::Animation(2).from_asset(FOX_PATH)),
    ]);
    let graph_handle = graphs.add(graph);
    commands.insert_resource(FoxAnimations {
        nodes: [indices[0], indices[1], indices[2]],
        graph: graph_handle,
    });

    // Undulating analytic terrain mesh (120×120 grid over 36m × 36m).
    let terrain_mesh = build_terrain_mesh(TERRAIN_HALF_EXTENT, 120);
    commands.spawn((
        Mesh3d(meshes.add(terrain_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.92,
            metallic: 0.0,
            ..default()
        })),
    ));

    // Fox hierarchy:
    // FoxRoot (locomotion: x, terrain_y, z + yaw)
    //   └─ FoxBody (IK pelvis drop + smoothed slope tilt + FOX_SCALE)
    //        └─ SceneRoot(Fox.glb)
    let start_x = 0.0;
    let start_z = 0.0;
    let start_y = terrain_height(start_x, start_z);
    commands
        .spawn((
            FoxRoot,
            Transform::from_xyz(start_x, start_y, start_z)
                .with_rotation(Quat::from_rotation_y(0.4)),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((
                FoxBody,
                SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(FOX_PATH))),
                Transform::from_scale(Vec3::splat(FOX_SCALE)),
            ));
        });

    // Glowing firefly orb tracked by the fox's multi-bone look-at IK.
    commands.spawn((
        FireflyOrb,
        Mesh3d(meshes.add(Sphere::new(0.085))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.92, 0.45),
            emissive: LinearRgba::rgb(4.5, 3.2, 0.8),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(1.2, start_y + 1.1, 2.2),
        PointLight {
            color: Color::srgb(1.0, 0.88, 0.45),
            intensity: 18_000.0,
            range: 5.0,
            shadows_enabled: false,
            ..default()
        },
    ));

    // Warm afternoon directional sunlight with soft shadows.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 13_500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 16.0, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Subtle cool sky rim directional fill (no shadows to keep WebGL2 fast).
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.55, 0.72, 1.0),
            illuminance: 2_600.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-10.0, 8.0, -9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Smooth follow camera with gentle horizon fog.
    commands.spawn((
        Camera3d::default(),
        FollowCamera,
        DistanceFog {
            color: Color::srgb(0.56, 0.73, 0.88),
            falloff: FogFalloff::Linear {
                start: 14.0,
                end: 34.0,
            },
            ..default()
        },
        Transform::from_xyz(3.2, start_y + 2.4, 4.4)
            .looking_at(Vec3::new(0.0, start_y + 0.75, 0.0), Vec3::Y),
    ));
}

fn setup_hud(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(7.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.06, 0.08, 0.11, 0.72)),
        ))
        .with_children(|parent| {
            parent.spawn((
                HudText,
                Text::new("Loading Fox.glb..."),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.93, 0.95, 0.98)),
            ));
        });
}

/// Attaches the [`AnimationGraphHandle`] and [`AnimationTransitions`] to the
/// fox's [`AnimationPlayer`] once the glTF scene spawns.
fn setup_animation_player(
    mut commands: Commands,
    animations: Res<FoxAnimations>,
    controller: Res<FoxController>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    for (entity, mut player) in &mut players {
        let mut transitions = AnimationTransitions::new();
        let anim = transitions
            .play(&mut player, animations.nodes[controller.active_clip], Duration::ZERO)
            .repeat();
        if controller.paused {
            anim.pause();
        }

        commands
            .entity(entity)
            .insert((AnimationGraphHandle(animations.graph.clone()), transitions));
    }
}

/// Processes keyboard shortcuts for IK toggles, playback control, clip cycling,
/// and mode switching.
fn handle_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut controller: ResMut<FoxController>,
    mut ik_settings: ResMut<IkSettings>,
    mut players: Query<&mut AnimationPlayer>,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        ik_settings.enabled = !ik_settings.enabled;
    }

    if keys.just_pressed(KeyCode::KeyG) {
        ik_settings.show_gizmos = !ik_settings.show_gizmos;
    }

    if keys.just_pressed(KeyCode::KeyR) {
        controller.auto_roam = !controller.auto_roam;
        if controller.auto_roam {
            controller.manual_clip = None;
        }
    }

    if keys.just_pressed(KeyCode::Space) {
        controller.paused = !controller.paused;
        for mut player in &mut players {
            if controller.paused {
                player.pause_all();
            } else {
                player.resume_all();
            }
        }
    }

    if keys.just_pressed(KeyCode::Enter) {
        let next = (controller.active_clip + 1) % 3;
        controller.manual_clip = Some(next);
        controller.auto_roam = false;
    }

    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::BracketRight) {
        controller.speed_multiplier = (controller.speed_multiplier + 0.15).min(2.5);
    }
    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::BracketLeft) {
        controller.speed_multiplier = (controller.speed_multiplier - 0.15).max(0.25);
    }
}

/// Moves the fox across the undulating terrain either under smooth camera-relative
/// WASD/Arrow steering or autonomous hill-roaming.
fn update_locomotion(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut controller: ResMut<FoxController>,
    mut fox_query: Query<&mut Transform, With<FoxRoot>>,
    cam_query: Query<&Transform, (With<FollowCamera>, Without<FoxRoot>)>,
) {
    let dt = time.delta_secs().min(0.1);
    let Ok(mut fox_tf) = fox_query.single_mut() else {
        return;
    };

    // Optional gentle camera orbit with Q / E.
    if keys.pressed(KeyCode::KeyQ) {
        controller.camera_yaw -= 1.4 * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        controller.camera_yaw += 1.4 * dt;
    }

    // Read manual steering axes (WASD + Arrows).
    let mut input = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        input.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        input.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        input.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        input.x -= 1.0;
    }

    if input.length_squared() > 0.01 {
        controller.auto_roam = false;
        controller.manual_clip = None;
    }

    if controller.paused {
        return;
    }

    let pos_xz = Vec2::new(fox_tf.translation.x, fox_tf.translation.z);
    let mut desired_vel = Vec2::ZERO;

    if controller.auto_roam {
        if controller.roam_wait > 0.0 {
            controller.roam_wait -= dt;
        } else {
            let to_wp = controller.roam_target - pos_xz;
            let dist = to_wp.length();
            if dist < 0.65 {
                controller.roam_index = (controller.roam_index + 1) % ROAM_WAYPOINTS.len();
                let (wx, wz, _sprint) = ROAM_WAYPOINTS[controller.roam_index];
                controller.roam_target = Vec2::new(wx, wz);
                // Pause briefly every few waypoints to trigger Survey idle & look-around.
                controller.roam_wait = if controller.roam_index % 3 == 0 {
                    2.6
                } else {
                    0.0
                };
            } else {
                let sprint = ROAM_WAYPOINTS[controller.roam_index].2 && dist > 2.2;
                let base_speed = if sprint { 3.55 } else { 1.75 };
                desired_vel = (to_wp / dist) * (base_speed * controller.speed_multiplier);
            }
        }
    } else if input.length_squared() > 0.01 {
        let cam_fwd_xz = cam_query
            .single()
            .map(|c| {
                let f = c.forward().as_vec3();
                Vec2::new(f.x, f.z).normalize_or(Vec2::NEG_Y)
            })
            .unwrap_or(Vec2::new(0.0, -1.0));
        let cam_right_xz = Vec2::new(-cam_fwd_xz.y, cam_fwd_xz.x);

        let move_dir = (cam_fwd_xz * input.y + cam_right_xz * input.x).normalize_or_zero();
        let running = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        let speed = if running { RUN_REF_SPEED } else { WALK_REF_SPEED };
        desired_vel = move_dir * (speed * controller.speed_multiplier);
    }

    // Smooth acceleration / deceleration.
    controller.velocity.smooth_nudge(&desired_vel, 6.5, dt);

    let mut next_xz = pos_xz + controller.velocity * dt;
    let dist_from_origin = next_xz.length();
    if dist_from_origin > PLAY_AREA_RADIUS {
        let radial = next_xz / dist_from_origin;
        next_xz = radial * PLAY_AREA_RADIUS;
        let outward = controller.velocity.dot(radial);
        if outward > 0.0 {
            controller.velocity -= radial * outward;
        }
    }

    // Smoothly turn fox heading (+Z forward) toward velocity vector.
    let speed = controller.velocity.length();
    if speed > 0.12 {
        let target_heading = controller.velocity.x.atan2(controller.velocity.y);
        let diff = wrap_angle(target_heading - controller.heading);
        let turn_step = (1.0 - (-8.5 * dt).exp()) * diff;
        controller.heading = wrap_angle(controller.heading + turn_step);
    }

    fox_tf.translation = Vec3::new(next_xz.x, terrain_height(next_xz.x, next_xz.y), next_xz.y);
    fox_tf.rotation = Quat::from_rotation_y(controller.heading);
}

/// Drives smooth [`AnimationTransitions`] between Survey, Walk, and Run and
/// matches playback rate to ground velocity to minimize foot sliding.
fn update_animation_state(
    animations: Res<FoxAnimations>,
    mut controller: ResMut<FoxController>,
    mut query: Query<(&mut AnimationPlayer, &mut AnimationTransitions)>,
) {
    let speed = controller.velocity.length();
    let desired_clip = if let Some(forced) = controller.manual_clip {
        forced
    } else if speed < 0.22 {
        CLIP_SURVEY
    } else if speed < 2.65 * controller.speed_multiplier {
        CLIP_WALK
    } else {
        CLIP_RUN
    };

    for (mut player, mut transitions) in &mut query {
        if desired_clip != controller.active_clip {
            controller.active_clip = desired_clip;
            let blend = if controller.paused {
                Duration::ZERO
            } else {
                Duration::from_millis(260)
            };
            let anim = transitions
                .play(&mut player, animations.nodes[desired_clip], blend)
                .repeat();
            if controller.paused {
                anim.pause();
            }
        }

        // Sync clip speed with actual horizontal ground velocity (unless manually cycling clips).
        let target_rate = if controller.manual_clip.is_some() {
            controller.speed_multiplier
        } else {
            match controller.active_clip {
                CLIP_WALK => (speed / WALK_REF_SPEED).clamp(0.45, 2.0),
                CLIP_RUN => (speed / RUN_REF_SPEED).clamp(0.65, 2.2),
                _ => controller.speed_multiplier,
            }
        };

        for (_, active) in player.playing_animations_mut() {
            active.set_speed(target_rate);
        }
    }
}

/// Updates the glowing firefly orb & look-at IK target from either the cursor's
/// raycast onto the undulating terrain or an autonomous orbiting firefly path.
fn update_look_target(
    time: Res<Time>,
    windows: Query<&Window>,
    mut cursor_events: MessageReader<CursorMoved>,
    cam_query: Query<(&Camera, &GlobalTransform), With<FollowCamera>>,
    fox_query: Query<&Transform, With<FoxRoot>>,
    mut controller: ResMut<FoxController>,
    mut ik_settings: ResMut<IkSettings>,
    mut orb_query: Query<&mut Transform, (With<FireflyOrb>, Without<FoxRoot>)>,
) {
    let dt = time.delta_secs().min(0.1);
    let Ok(fox_tf) = fox_query.single() else {
        return;
    };

    // Detect live cursor motion over the canvas.
    let mut latest_cursor = None;
    for ev in cursor_events.read() {
        latest_cursor = Some(ev.position);
    }

    if let Some(cursor_pos) = latest_cursor {
        if let Ok((camera, cam_gt)) = cam_query.single() {
            if let Ok(ray) = camera.viewport_to_world(cam_gt, cursor_pos) {
                if let Some(hit) = raycast_terrain(ray, 55.0) {
                    controller.cursor_target = Some(hit + Vec3::Y * 0.55);
                    controller.cursor_idle_secs = 0.0;
                }
            }
        }
    } else {
        let cursor_inside = windows
            .single()
            .ok()
            .and_then(|w| w.cursor_position())
            .is_some();
        if !cursor_inside {
            controller.cursor_idle_secs += dt * 2.5;
        } else {
            controller.cursor_idle_secs += dt;
        }
    }

    // Use cursor terrain target while active; otherwise gently orbit a firefly ahead of the fox.
    let desired_target = if controller.cursor_idle_secs < 4.5 {
        controller.cursor_target.unwrap_or_else(|| {
            orbit_firefly_position(fox_tf, time.elapsed_secs())
        })
    } else {
        orbit_firefly_position(fox_tf, time.elapsed_secs())
    };

    ik_settings.look_target.smooth_nudge(&desired_target, 9.0, dt);
    ik_settings.look_active = true;

    if let Ok(mut orb_tf) = orb_query.single_mut() {
        orb_tf.translation = ik_settings.look_target;
    }
}

/// Computes a lively figure-8 / arc firefly orbit ahead of the fox so head/neck
/// look-at IK is immediately engaging even before the mouse enters the canvas.
fn orbit_firefly_position(fox_tf: &Transform, t: f32) -> Vec3 {
    let fwd = (fox_tf.rotation * Vec3::Z).normalize_or(Vec3::Z);
    let left = (fox_tf.rotation * Vec3::X).normalize_or(Vec3::X);

    let sweep = (t * 0.85).sin() * 1.45;
    let ahead = 2.0 + (t * 0.55).cos() * 0.55;
    let pos_xz = Vec2::new(
        fox_tf.translation.x + fwd.x * ahead + left.x * sweep,
        fox_tf.translation.z + fwd.z * ahead + left.z * sweep,
    );
    let ground = terrain_height(pos_xz.x, pos_xz.y);
    let hover = 0.95 + (t * 1.9).sin() * 0.35;
    Vec3::new(pos_xz.x, ground + hover, pos_xz.y)
}

/// Smoothly frames the fox from a 3/4 elevated follow viewpoint.
fn update_follow_camera(
    time: Res<Time>,
    controller: Res<FoxController>,
    fox_query: Query<&Transform, With<FoxRoot>>,
    mut cam_query: Query<&mut Transform, (With<FollowCamera>, Without<FoxRoot>)>,
) {
    let dt = time.delta_secs().min(0.1);
    let (Ok(fox_tf), Ok(mut cam_tf)) = (fox_query.single(), cam_query.single_mut()) else {
        return;
    };

    let focus = fox_tf.translation + Vec3::Y * 0.72;
    let dist = 5.2;
    let offset = Vec3::new(
        controller.camera_yaw.sin() * dist,
        2.35,
        controller.camera_yaw.cos() * dist,
    );
    let mut desired_pos = focus + offset;
    let cam_ground = terrain_height(desired_pos.x, desired_pos.z);
    desired_pos.y = desired_pos.y.max(cam_ground + 1.15);

    cam_tf.translation.smooth_nudge(&desired_pos, 4.5, dt);
    let target_rot = cam_tf.looking_at(focus, Vec3::Y).rotation;
    cam_tf.rotation.smooth_nudge(&target_rot, 6.5, dt);
}

fn update_hud(
    controller: Res<FoxController>,
    ik_settings: Res<IkSettings>,
    rig_query: Query<(), With<FoxRig>>,
    mut hud_query: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = hud_query.single_mut() else {
        return;
    };

    if rig_query.is_empty() {
        return;
    }

    let mode = if controller.auto_roam {
        "Auto-Roam"
    } else if controller.manual_clip.is_some() {
        "Clip Preview"
    } else {
        "Manual (WASD)"
    };
    let clip = CLIP_NAMES[controller.active_clip];
    let ik_label = if ik_settings.enabled { "ON" } else { "OFF" };
    let gizmo_label = if ik_settings.show_gizmos { "ON" } else { "OFF" };
    let state_label = if controller.paused { " [PAUSED]" } else { "" };

    text.0 = format!(
        "Mode: {mode} | Clip: {clip}{state_label} | Speed: {:.2}x\nIK [I]: {ik_label} | Gizmos [G]: {gizmo_label} | Roam [R] | Orbit [Q/E]",
        controller.speed_multiplier
    );
}

/// Wraps an angle in radians into `[-PI, PI]`.
#[inline]
fn wrap_angle(mut a: f32) -> f32 {
    while a > PI {
        a -= TAU;
    }
    while a < -PI {
        a += TAU;
    }
    a
}
