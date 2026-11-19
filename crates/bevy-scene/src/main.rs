use bevy::asset::AssetMetaCheck;
use bevy::prelude::*;

#[derive(Component)]
struct Spin {
    speed_x: f32,
    speed_y: f32,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.06, 0.08)))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Bevy 3D Scene".into(),
                        canvas: Some("#bevy-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    // Avoids 404 .meta requests when hosted on GitHub Pages.
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        .add_systems(Startup, setup)
        .add_systems(Update, spin_meshes)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground disc
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.14, 0.16, 0.20),
            perceptual_roughness: 0.85,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // Spinning center cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.2, 1.2, 1.2))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.38, 0.65, 0.98),
            metallic: 0.25,
            perceptual_roughness: 0.35,
            ..default()
        })),
        Transform::from_xyz(0.0, 1.1, 0.0),
        Spin {
            speed_x: 0.5,
            speed_y: 0.9,
        },
    ));

    // Orbiting accent torus
    commands.spawn((
        Mesh3d(meshes.add(Torus::new(0.18, 0.95))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.96, 0.45, 0.71),
            metallic: 0.4,
            perceptual_roughness: 0.3,
            ..default()
        })),
        Transform::from_xyz(0.0, 1.1, 0.0),
        Spin {
            speed_x: -0.7,
            speed_y: 0.35,
        },
    ));

    // Key light with shadows
    commands.spawn((
        PointLight {
            intensity: 1_500_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 7.0, 4.0),
    ));

    // Soft fill light
    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            color: Color::srgb(0.7, 0.85, 1.0),
            ..default()
        },
        Transform::from_xyz(-5.0, 4.0, -3.0),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.6, 3.0, 5.5).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

fn spin_meshes(time: Res<Time>, mut query: Query<(&Spin, &mut Transform)>) {
    let dt = time.delta_secs();
    for (spin, mut transform) in &mut query {
        transform.rotate_x(spin.speed_x * dt);
        transform.rotate_y(spin.speed_y * dt);
    }
}
