mod bridge;
mod mesher;
mod rng;
mod wfc;

use std::f32::consts::FRAC_PI_2;

use bevy::app::{App, Startup, Update};
use bevy::asset::{AssetMetaCheck, AssetPlugin, Assets, Handle};
use bevy::camera::{Camera3d, ClearColor};
use bevy::color::Color;
use bevy::ecs::message::MessageReader;
use bevy::ecs::prelude::{
    Commands, Component, IntoScheduleConfigs, Query, Res, ResMut, Resource, With,
};
use bevy::input::keyboard::KeyCode;
use bevy::input::mouse::{MouseButton, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::ButtonInput;
use bevy::light::{DirectionalLight, GlobalAmbientLight};
use bevy::math::Vec3;
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{AlphaMode, Cuboid, DefaultPlugins, PluginGroup};
use bevy::text::{TextColor, TextFont};
use bevy::time::Time;
use bevy::transform::components::Transform;
use bevy::ui::{BackgroundColor, Node, PositionType, UiRect, Val};
use bevy::ui::widget::Text;
use bevy::utils::default;
use bevy::window::{Window, WindowPlugin};

use crate::bridge::{drain_cmds, set_seed_label, set_status, BridgeCmd};
use crate::mesher::build_diorama_meshes;
use crate::rng::Prng;
use crate::wfc::{DioramaStats, WfcPhase, WfcSolver};

const DEFAULT_SEED: &str = "alpine-42";
const PASS1_STEPS_PER_FRAME: usize = 110;
const PASS2_STEPS_PER_FRAME: usize = 8;
const PASS3_STEPS_PER_FRAME: usize = 40;

#[derive(Resource)]
struct WfcState {
    solver: WfcSolver,
    cached_stats: DioramaStats,
    last_hud_text: String,
    animate: bool,
    paused: bool,
    mesh_dirty: bool,
    random_rng: Prng,
}

impl WfcState {
    fn new(seed: &str, animate: bool) -> Self {
        let mut solver = WfcSolver::new(seed);
        if !animate {
            solver.solve_all();
        }
        set_seed_label(solver.seed_label().to_owned());
        let cached_stats = solver.stats();
        Self {
            solver,
            cached_stats,
            last_hud_text: String::new(),
            animate,
            paused: false,
            mesh_dirty: true,
            random_rng: Prng::from_seed(0x7365_6564_2026_0042),
        }
    }

    fn restart(&mut self, seed: &str, animate: bool) {
        self.animate = animate;
        self.paused = false;
        self.solver = WfcSolver::new(seed);
        if !animate {
            self.solver.solve_all();
        }
        set_seed_label(self.solver.seed_label().to_owned());
        self.cached_stats = self.solver.stats();
        self.mesh_dirty = true;
    }
}

#[derive(Resource)]
struct DioramaMeshHandles {
    opaque: Handle<Mesh>,
    water: Handle<Mesh>,
}

#[derive(Component)]
struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    auto_turntable: bool,
}

impl OrbitCamera {
    fn compute_transform(&self) -> Transform {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let offset = Vec3::new(
            self.distance * cos_p * sin_y,
            self.distance * sin_p,
            self.distance * cos_p * cos_y,
        );
        Transform::from_translation(self.target + offset).looking_at(self.target, Vec3::Y)
    }
}

#[derive(Component)]
struct HudText;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.68, 0.81, 0.92)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.86, 0.91, 1.0),
            brightness: 460.0,
            ..default()
        })
        .insert_resource(WfcState::new(DEFAULT_SEED, true))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy 3D Voxel WFC Diorama".into(),
                        canvas: Some("#bevy-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        .add_systems(Startup, (setup_scene, setup_hud))
        .add_systems(
            Update,
            (
                poll_bridge_commands,
                handle_keyboard_shortcuts,
                step_wfc_solver,
                rebuild_diorama_meshes,
                update_orbit_camera,
                update_hud,
            )
                .chain(),
        )
        .run();
}

fn setup_scene(
    mut commands: Commands,
    state: Res<WfcState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let initial_meshes = build_diorama_meshes(&state.solver);
    let opaque_handle = meshes.add(initial_meshes.opaque);
    let water_handle = meshes.add(initial_meshes.water);

    // Opaque terrain & foliage voxels.
    commands.spawn((
        Mesh3d(opaque_handle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.92,
            metallic: 0.0,
            ..default()
        })),
    ));

    // Translucent sea & alpine pond water voxels.
    commands.spawn((
        Mesh3d(water_handle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.14,
            metallic: 0.04,
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
    ));

    commands.insert_resource(DioramaMeshHandles {
        opaque: opaque_handle,
        water: water_handle,
    });

    // Subtle tabletop plinth under the 20 x 20 cutaway block.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(20.5, 0.55, 20.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.14, 0.16, 0.19),
            perceptual_roughness: 0.85,
            metallic: 0.05,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.276, 0.0),
    ));

    // Warm key sunlight with crisp shadows.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.89),
            illuminance: 14_500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(22.0, 34.0, 18.0).looking_at(Vec3::new(0.0, 6.0, 0.0), Vec3::Y),
    ));

    // Cool sky fill light (no shadows to keep WebGL2 frame times snappy).
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.58, 0.76, 1.0),
            illuminance: 3_200.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-20.0, 14.0, -18.0).looking_at(Vec3::new(0.0, 6.0, 0.0), Vec3::Y),
    ));

    // Elevated 3/4 orbit camera framing the centered 20 x 20 x 20 diorama.
    let orbit = OrbitCamera {
        target: Vec3::new(0.0, 6.5, 0.0),
        yaw: 0.72,
        pitch: 0.50,
        distance: 32.0,
        auto_turntable: true,
    };
    let transform = orbit.compute_transform();
    commands.spawn((Camera3d::default(), orbit, transform));
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
            BackgroundColor(Color::srgba(0.06, 0.08, 0.11, 0.76)),
        ))
        .with_children(|parent| {
            parent.spawn((
                HudText,
                Text::new("Initializing 20x20x20 WFC diorama..."),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.93, 0.95, 0.98)),
            ));
        });
}

fn poll_bridge_commands(mut state: ResMut<WfcState>) {
    for cmd in drain_cmds() {
        match cmd {
            BridgeCmd::SetSeed { seed, animate } => {
                state.restart(&seed, animate);
            }
            BridgeCmd::SetAnimate(animate) => {
                state.animate = animate;
                if !animate && !state.solver.is_complete() {
                    state.solver.solve_all();
                    state.mesh_dirty = true;
                }
            }
            BridgeCmd::FinishNow => {
                if !state.solver.is_complete() {
                    state.solver.solve_all();
                    state.paused = false;
                    state.mesh_dirty = true;
                }
            }
            BridgeCmd::TogglePause => {
                state.paused = !state.paused;
            }
        }
    }
}

fn handle_keyboard_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<WfcState>,
    mut cam_query: Query<&mut OrbitCamera>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        let next_seed = state.random_rng.next_seed_label();
        let animate = state.animate;
        state.restart(&next_seed, animate);
    }
    if keys.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }
    if keys.just_pressed(KeyCode::KeyF) && !state.solver.is_complete() {
        state.solver.solve_all();
        state.paused = false;
        state.mesh_dirty = true;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        for mut orbit in &mut cam_query {
            orbit.auto_turntable = !orbit.auto_turntable;
        }
    }
}

fn step_wfc_solver(mut state: ResMut<WfcState>) {
    if state.paused || state.solver.is_complete() {
        return;
    }

    if !state.animate {
        state.solver.solve_all();
        state.mesh_dirty = true;
        return;
    }

    let budget = match state.solver.phase() {
        WfcPhase::Pass1Geology => PASS1_STEPS_PER_FRAME,
        WfcPhase::Pass2Ponds => PASS2_STEPS_PER_FRAME,
        WfcPhase::Pass3Flora => PASS3_STEPS_PER_FRAME,
        WfcPhase::Complete => 0,
    };

    if state.solver.step(budget) {
        state.mesh_dirty = true;
    }
}

fn rebuild_diorama_meshes(
    mut state: ResMut<WfcState>,
    handles: Res<DioramaMeshHandles>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !state.mesh_dirty {
        return;
    }
    state.mesh_dirty = false;
    state.cached_stats = state.solver.stats();

    let built = build_diorama_meshes(&state.solver);
    if let Some(opaque_mesh) = meshes.get_mut(&handles.opaque) {
        *opaque_mesh = built.opaque;
    }
    if let Some(water_mesh) = meshes.get_mut(&handles.water) {
        *water_mesh = built.water;
    }
}

fn update_orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_reader: MessageReader<MouseMotion>,
    mut wheel_reader: MessageReader<MouseWheel>,
    mut cam_query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let dt = time.delta_secs();

    let mut drag_delta = bevy::math::Vec2::ZERO;
    if mouse_buttons.pressed(MouseButton::Left) {
        for event in motion_reader.read() {
            drag_delta += event.delta;
        }
    } else {
        motion_reader.clear();
    }

    let mut scroll_delta = 0.0_f32;
    for wheel in wheel_reader.read() {
        scroll_delta += match wheel.unit {
            MouseScrollUnit::Line => wheel.y * 2.2,
            MouseScrollUnit::Pixel => wheel.y * 0.04,
        };
    }

    let mut yaw_input = 0.0_f32;
    if keys.pressed(KeyCode::KeyQ) || keys.pressed(KeyCode::ArrowLeft) {
        yaw_input -= 1.0;
    }
    if keys.pressed(KeyCode::KeyE) || keys.pressed(KeyCode::ArrowRight) {
        yaw_input += 1.0;
    }

    let mut pitch_input = 0.0_f32;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        pitch_input += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        pitch_input -= 1.0;
    }

    for (mut orbit, mut transform) in &mut cam_query {
        if drag_delta.length_squared() > 0.0 {
            orbit.yaw -= drag_delta.x * 0.008;
            orbit.pitch = (orbit.pitch + drag_delta.y * 0.007).clamp(0.14, FRAC_PI_2 - 0.08);
            orbit.auto_turntable = false;
        }

        if yaw_input != 0.0 {
            orbit.yaw += yaw_input * 1.45 * dt;
            orbit.auto_turntable = false;
        }
        if pitch_input != 0.0 {
            orbit.pitch = (orbit.pitch + pitch_input * 1.15 * dt).clamp(0.14, FRAC_PI_2 - 0.08);
        }

        if orbit.auto_turntable {
            orbit.yaw += 0.14 * dt;
        }

        if scroll_delta != 0.0 {
            orbit.distance = (orbit.distance - scroll_delta).clamp(15.0, 52.0);
        }

        *transform = orbit.compute_transform();
    }
}

fn update_hud(
    mut state: ResMut<WfcState>,
    cam_query: Query<&OrbitCamera>,
    mut hud_query: Query<&mut Text, With<HudText>>,
) {
    let stats = state.cached_stats;
    let pct = state.solver.progress_percent();
    let phase_label = state.solver.phase().label();
    let pause_tag = if state.paused && !state.solver.is_complete() {
        " [PAUSED]"
    } else {
        ""
    };
    let spin_tag = if cam_query.iter().next().is_some_and(|c| c.auto_turntable) {
        "ON"
    } else {
        "OFF"
    };

    let line1 = format!(
        "Seed: {} | {} ({:.0}%){}",
        state.solver.seed_label(),
        phase_label,
        pct,
        pause_tag,
    );
    let line2 = format!(
        "Sea: y={} | Ponds: {} ({} vox) | Trees: {} | Bushes: {} | Land: {}",
        state.solver.sea_level(),
        stats.pond_count,
        stats.pond_voxels,
        stats.tree_count,
        stats.bush_count,
        stats.land_voxels,
    );
    let line3 = format!(
        "Drag: orbit | Scroll: zoom | Q/E/W/S: cam | T: spin ({spin_tag}) \
         | R: seed | Space: pause | F: instant"
    );

    let full_text = format!("{line1}\n{line2}\n{line3}");
    if state.last_hud_text == full_text {
        return;
    }
    state.last_hud_text = full_text.clone();
    set_status(format!("{line1} | {line2}"));

    for mut text in &mut hud_query {
        text.0 = full_text.clone();
    }
}
