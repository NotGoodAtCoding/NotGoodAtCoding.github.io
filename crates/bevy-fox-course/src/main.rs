mod controls;
mod gait;
mod ik;
mod jump;
mod obstacles;
mod terrain;

use bevy::app::AnimationSystems;
use bevy::asset::AssetMetaCheck;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::controls::{
    read_player_intent, update_gaze_priority, update_steering_and_avoidance, AvoidanceLock,
    FireflyOrb, FollowCamera, GazeFocus, LocomotionState, PlayerIntent, FOX_START_HEADING,
    FOX_START_XZ,
};
use crate::gait::{
    setup_animation_player, update_gait_and_animations, FoxAnimations, GaitPosture, GaitState,
};
use crate::ik::{
    apply_fox_ik, bind_fox_rig, restore_pre_ik_pose, FoxBody, FoxRig, FoxRoot, IkSettings,
    FOX_SCALE,
};
use crate::jump::{
    evaluate_jump_decision, step_jump_or_ground_locomotion, JumpPhase, JumpState,
};
use crate::obstacles::{spawn_course_obstacles, CourseObstacles};
use crate::terrain::{build_terrain_mesh, terrain_height, TERRAIN_HALF_EXTENT};

const FOX_PATH: &str = "models/Fox.glb";

#[derive(Component)]
struct HudText;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.55, 0.74, 0.90)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.86, 0.91, 1.0),
            brightness: 440.0,
            ..default()
        })
        .init_resource::<CourseObstacles>()
        .init_resource::<IkSettings>()
        .init_resource::<PlayerIntent>()
        .init_resource::<LocomotionState>()
        .init_resource::<AvoidanceLock>()
        .init_resource::<GazeFocus>()
        .init_resource::<GaitState>()
        .init_resource::<GaitPosture>()
        .init_resource::<JumpState>()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy Fox Show-Jumping Course".into(),
                        canvas: Some("#bevy-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    #[cfg(target_arch = "wasm32")]
                    file_path: "/playground/bevy-fox-ik/assets".into(),
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
                (
                    read_player_intent,
                    evaluate_jump_decision,
                    update_steering_and_avoidance,
                    step_jump_or_ground_locomotion,
                    update_gait_and_animations,
                    update_gaze_priority,
                )
                    .chain()
                    .run_if(rig_is_ready),
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

fn rig_is_ready(rig_query: Query<(), With<FoxRig>>) -> bool {
    !rig_query.is_empty()
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    obstacles: Res<CourseObstacles>,
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

    // Undulating 56m x 56m equestrian meadow mesh (144x144 grid).
    let terrain_mesh = build_terrain_mesh(TERRAIN_HALF_EXTENT, 144);
    commands.spawn((
        Mesh3d(meshes.add(terrain_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.92,
            metallic: 0.0,
            ..default()
        })),
    ));

    // Procedural 8-jump equestrian show course + arena boundary markers.
    spawn_course_obstacles(&mut commands, &mut meshes, &mut materials, &obstacles.list);

    // Fox hierarchy:
    // FoxRoot (locomotion / ballistic flight x, y, z + heading yaw)
    //   └─ FoxBody (IK pelvis, crouch/coil/spring, pitch & centripetal bank + FOX_SCALE)
    //        └─ SceneRoot(Fox.glb)
    let start_y = terrain_height(FOX_START_XZ.x, FOX_START_XZ.y);
    commands
        .spawn((
            FoxRoot,
            Transform::from_xyz(FOX_START_XZ.x, start_y, FOX_START_XZ.y)
                .with_rotation(Quat::from_rotation_y(FOX_START_HEADING)),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((
                FoxBody,
                SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(FOX_PATH))),
                Transform::from_scale(Vec3::splat(FOX_SCALE)),
            ));
        });

    // Glowing ambient firefly orb (Gaze Priority #4 fallback).
    commands.spawn((
        FireflyOrb,
        Mesh3d(meshes.add(Sphere::new(0.085))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.92, 0.45),
            emissive: LinearRgba::rgb(4.5, 3.2, 0.8),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(FOX_START_XZ.x + 2.5, start_y + 1.0, FOX_START_XZ.y),
        PointLight {
            color: Color::srgb(1.0, 0.88, 0.45),
            intensity: 16_000.0,
            range: 5.0,
            shadows_enabled: false,
            ..default()
        },
    ));

    // Warm afternoon directional sunlight with soft shadows.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 14_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(14.0, 22.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Subtle cool sky fill light (no shadows for crisp WebGL2 frame times).
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.55, 0.72, 1.0),
            illuminance: 2_800.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-14.0, 10.0, -12.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Smooth elevated 3/4 follow camera with atmospheric meadow horizon fog.
    commands.spawn((
        Camera3d::default(),
        FollowCamera,
        DistanceFog {
            color: Color::srgb(0.55, 0.74, 0.90),
            falloff: FogFalloff::Linear {
                start: 24.0,
                end: 58.0,
            },
            ..default()
        },
        Transform::from_xyz(FOX_START_XZ.x - 4.8, start_y + 3.0, FOX_START_XZ.y + 4.2)
            .looking_at(Vec3::new(FOX_START_XZ.x, start_y + 0.8, FOX_START_XZ.y), Vec3::Y),
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
            BackgroundColor(Color::srgba(0.06, 0.08, 0.11, 0.74)),
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

/// Smoothly frames the fox with subtle dynamic zoom-out on sprint and jumps so
/// sweeping turns and tall oxers stay comfortably inside the viewport.
fn update_follow_camera(
    time: Res<Time>,
    intent: Res<PlayerIntent>,
    mut loco: ResMut<LocomotionState>,
    jump: Res<JumpState>,
    mut cam_query: Query<&mut Transform, With<FollowCamera>>,
) {
    let dt = time.delta_secs().min(0.1);
    let Ok(mut cam_tf) = cam_query.single_mut() else {
        return;
    };

    let sprint_zoom = ((loco.speed - 2.6) / 3.6).clamp(0.0, 1.0) * 1.15;
    let jump_zoom = if jump.phase == JumpPhase::Airborne {
        0.75
    } else if jump.phase == JumpPhase::Gather {
        0.35
    } else {
        0.0
    };
    let target_extra = sprint_zoom + jump_zoom;
    loco.camera_zoom_extra.smooth_nudge(&target_extra, 3.5, dt);

    let fwd = Vec3::new(loco.heading.sin(), 0.0, loco.heading.cos());
    let look_lead = fwd * (0.45 + 0.18 * loco.speed.min(5.0));
    let focus = loco.position + Vec3::Y * 0.78 + look_lead;

    let dist = 5.6 + loco.camera_zoom_extra;
    let height = 2.55 + loco.camera_zoom_extra * 0.35;
    let offset = Vec3::new(
        intent.camera_yaw.sin() * dist,
        height,
        intent.camera_yaw.cos() * dist,
    );
    let mut desired_pos = loco.position + offset;
    let cam_ground = terrain_height(desired_pos.x, desired_pos.z);
    desired_pos.y = desired_pos.y.max(cam_ground + 1.25);

    cam_tf.translation.smooth_nudge(&desired_pos, 4.6, dt);
    let target_rot = cam_tf.looking_at(focus, Vec3::Y).rotation;
    cam_tf.rotation.smooth_nudge(&target_rot, 6.8, dt);
}

#[allow(clippy::too_many_arguments)]
fn update_hud(
    intent: Res<PlayerIntent>,
    loco: Res<LocomotionState>,
    gait: Res<GaitState>,
    jump: Res<JumpState>,
    avoid: Res<AvoidanceLock>,
    obstacles: Res<CourseObstacles>,
    gaze: Res<GazeFocus>,
    ik_settings: Res<IkSettings>,
    rig_query: Query<(), With<FoxRig>>,
    mut hud_query: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = hud_query.single_mut() else {
        return;
    };

    // Keep "Loading Fox.glb..." visible until the glTF rig binds.
    if rig_query.is_empty() {
        return;
    }

    let mode = if intent.auto_course {
        "Auto-Course"
    } else if intent.is_stalking() {
        "Manual (Stalk)"
    } else {
        "Manual"
    };
    let paused = if intent.paused { " [PAUSED]" } else { "" };
    let brake = if loco.is_braking { " [BRAKE]" } else { "" };
    let ik_label = if ik_settings.enabled { "ON" } else { "OFF" };
    let gizmo_label = if ik_settings.show_gizmos { "ON" } else { "OFF" };

    let fence_tag = if let Some(id) = jump.active_obstacle {
        obstacles
            .list
            .get(id)
            .map(|o| format!(" | Jump: {}", o.name))
            .unwrap_or_default()
    } else if let Some(id) = avoid.active_obstacle {
        obstacles
            .list
            .get(id)
            .map(|o| format!(" | Avoid: {}", o.name))
            .unwrap_or_default()
    } else {
        String::new()
    };

    text.0 = format!(
        "Mode: {mode} | Gait: {} | Speed: {:.1} m/s | Phase: {}{fence_tag}{brake}{paused}\nJumps: {} | Refusals: {} | Gaze: {} | IK [I]: {ik_label} | Gizmos [G]: {gizmo_label}",
        gait.current.label(),
        loco.speed,
        jump.phase.label(),
        jump.jumps_cleared,
        jump.refusals,
        gaze.source.label(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::step_auto_course_intent;
    use crate::terrain::PLAY_AREA_RADIUS;
    use std::time::Duration;

    #[test]
    fn auto_course_demo_clears_jumps_and_demonstrates_wing_avoidance() {
        let mut world = World::new();
        let mut time = Time::<()>::default();
        time.advance_by(Duration::from_secs_f32(1.0 / 60.0));
        world.insert_resource(time);
        world.insert_resource(CourseObstacles::default());
        world.insert_resource(PlayerIntent::default());
        world.insert_resource(LocomotionState::default());
        world.insert_resource(AvoidanceLock::default());
        world.insert_resource(JumpState::default());
        world.spawn((FoxRoot, Transform::default()));

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                step_auto_course_intent,
                evaluate_jump_decision,
                update_steering_and_avoidance,
                step_jump_or_ground_locomotion,
            )
                .chain(),
        );

        let dt = Duration::from_secs_f32(1.0 / 60.0);
        for _ in 0..(60 * 38) {
            world.resource_mut::<Time<()>>().advance_by(dt);
            schedule.run(&mut world);

            let loco = world.resource::<LocomotionState>();
            assert!(loco.position.is_finite(), "fox position became non-finite");
            let r = Vec2::new(loco.position.x, loco.position.z).length();
            assert!(
                r <= PLAY_AREA_RADIUS + 0.1,
                "fox left arena bounds: r={r:.2}"
            );
        }

        let jump = world.resource::<JumpState>();
        assert!(
            jump.jumps_cleared >= 6,
            "auto-course demo should clear most/all course jumps in 38s, cleared {}",
            jump.jumps_cleared
        );
        assert!(
            jump.refusals >= 1,
            "auto-course demo should naturally demonstrate at least one wing refusal, got {}",
            jump.refusals
        );
    }
}
