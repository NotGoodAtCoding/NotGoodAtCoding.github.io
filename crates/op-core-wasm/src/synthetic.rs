//! Procedural anatomical O&P limb & foot-ankle generators (TT, TF, and AFO).

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;
use std::f64::consts::PI;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::Tri;
use crate::mesh::TriMesh;

/// Procedural limb or foot-ankle mesh paired with anatomical landmark vertex indices.
#[derive(Debug, Clone)]
pub struct SyntheticLimbCase {
    pub mesh: TriMesh,
    pub landmarks: BTreeMap<String, usize>,
    pub landmark_coords: BTreeMap<String, Vec3>,
    pub device_type: String,
}

/// Deterministic lightweight PRNG (SplitMix64) so zero RNG crates are needed in WASM.
fn deterministic_scale_var(seed: u64, half_range: f64) -> f64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    let u = (z >> 11) as f64 / ((1u64 << 53) as f64);
    1.0 + (u * 2.0 - 1.0) * half_range
}

fn close_proximal_dome(
    vertices: &mut Vec<Vec3>,
    faces: &mut Vec<Tri>,
    n_seg: usize,
    height_mm: f64,
    dome_rings: usize,
) {
    let inv = 1.0 / n_seg as f64;
    let cx = vertices[..n_seg].iter().map(|v| v.x).sum::<f64>() * inv;
    let cy = vertices[..n_seg].iter().map(|v| v.y).sum::<f64>() * inv;
    let cz = vertices[..n_seg].iter().map(|v| v.z).sum::<f64>() * inv;
    let base_offsets: Vec<Vec3> = vertices[..n_seg]
        .iter()
        .map(|v| Vec3::new(v.x - cx, v.y - cy, v.z - cz))
        .collect();

    let mut prev_ring: Vec<usize> = (0..n_seg).collect();
    for k in 1..dome_rings {
        let phi = (k as f64 / dome_rings as f64) * FRAC_PI_2;
        let r_scale = phi.cos();
        let z_lift = height_mm * phi.sin();
        let mut next_ring = Vec::with_capacity(n_seg);
        for d in &base_offsets {
            next_ring.push(vertices.len());
            vertices.push(Vec3::new(
                cx + d.x * r_scale,
                cy + d.y * r_scale,
                cz + d.z * r_scale + z_lift,
            ));
        }
        for s in 0..n_seg {
            let sn = (s + 1) % n_seg;
            let a = prev_ring[s];
            let b = prev_ring[sn];
            let c = next_ring[sn];
            let d = next_ring[s];
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
        prev_ring = next_ring;
    }

    let prox_pole = vertices.len();
    vertices.push(Vec3::new(cx, cy, cz + height_mm));
    for s in 0..n_seg {
        let sn = (s + 1) % n_seg;
        faces.push([prev_ring[s], prev_ring[sn], prox_pole]);
    }
}

/// Generates an anatomically contoured transtibial residual limb in canonical coordinates.
#[allow(clippy::too_many_arguments)]
pub fn generate_transtibial_limb(
    length_mm: f64,
    proximal_radius: f64,
    distal_radius: f64,
    cuff_height_mm: f64,
    open_cuff: bool,
    include_knee_thigh: bool,
    n_rings: usize,
    n_seg: usize,
    seed: u64,
) -> SyntheticLimbCase {
    let scale_var = if seed == 0 {
        1.0
    } else {
        deterministic_scale_var(seed, 0.08)
    };
    let pr = proximal_radius * scale_var;
    let dr = distal_radius * scale_var;

    let z_top = if include_knee_thigh && cuff_height_mm <= 40.0 {
        135.0
    } else {
        cuff_height_mm
    };
    let is_full_knee = z_top > 60.0;
    let eff_rings = if is_full_knee {
        n_rings.max(76)
    } else {
        n_rings
    };
    let eff_seg = if is_full_knee { n_seg.max(72) } else { n_seg };
    let z_bot = -length_mm;

    let mut vertices = Vec::with_capacity((eff_rings + 6) * eff_seg + 2);
    let mut faces = Vec::with_capacity((eff_rings + 5) * eff_seg * 2);

    let z_start = z_top - if open_cuff { 0.0 } else { 22.0 };
    for r in 0..eff_rings {
        let u = r as f64 / (eff_rings - 1) as f64;
        let z = z_start + u * (z_bot + 10.0 - z_start);

        let base_r = if z > 0.0 {
            let t_thigh = ((z - 22.0) / 115.0).max(0.0);
            let waist = -4.2 * (-((z - 52.0) / 16.0).powi(2)).exp();
            pr * (1.0 + 0.28 * t_thigh.powf(0.92)) + waist
        } else {
            let t_below = (-z / length_mm.max(1e-6)).clamp(0.0, 1.0);
            let mut br = pr * (1.0 - 0.36 * t_below);
            if z < z_bot + 28.0 {
                let cap_u = ((z_bot + 28.0 - z) / 28.0).clamp(0.0, 1.0);
                br = (dr * (1.0 - 0.88 * cap_u * cap_u).max(0.02).sqrt()).max(8.0);
            }
            br
        };

        for s in 0..eff_seg {
            let theta = 2.0 * PI * s as f64 / eff_seg as f64;
            let ct = theta.cos();
            let st = theta.sin();
            let d_ant = (-ct).atan2(st);

            let tri_w = if z <= 5.0 {
                (-((z + 65.0) / 60.0).powi(2)).exp()
            } else {
                0.0
            };
            let tri_shape = -1.6 * tri_w * (3.0 * theta).cos();

            // Broad, low-profile shield-shaped patella with cylindrical flatness compensation.
            let pat_z_rel = (z - 33.0) / 18.5;
            let pat_ang_w = 0.44 + 0.12 * ((z - 28.0) / 18.0).clamp(-1.0, 1.0);
            let pat_u = (d_ant / pat_ang_w.max(0.20)).hypot(pat_z_rel);
            let pat_w = if st > 0.25 {
                (-(pat_u.powf(3.6)) * 1.1).exp()
            } else {
                0.0
            };
            let pat_center_mm = if is_full_knee { 5.8 } else { 4.8 };
            let pat_flat_comp = (if is_full_knee { 12.0 } else { 8.0 }) * (1.0 - d_ant.cos());
            let patella = (pat_center_mm + pat_flat_comp) * pat_w;

            let gutter_ang =
                (-((d_ant - 0.66) / 0.20).powi(2)).exp() + (-((d_ant + 0.66) / 0.20).powi(2)).exp();
            let gutter_z = (-((z - 12.0) / 24.0).powi(2)).exp();
            let knee_gutters = (if is_full_knee { -3.8 } else { -2.2 }) * gutter_ang * gutter_z;

            let supra_dip = if is_full_knee {
                -2.8 * (-(d_ant / 0.48).powi(2)).exp() * (-((z - 60.0) / 12.0).powi(2)).exp()
            } else {
                0.0
            };

            let pt_shelf = (if is_full_knee { -3.2 } else { -2.2 })
                * (-(d_ant / 0.42).powi(2)).exp()
                * (-((z - 2.0) / 9.5).powi(2)).exp();
            let pt_cord = (if is_full_knee { 2.4 } else { 1.0 })
                * (-(d_ant / 0.14).powi(2)).exp()
                * (-((z - 1.0) / 12.0).powi(2)).exp();

            let tib_tub = (if is_full_knee { 4.6 } else { 3.2 })
                * (-(d_ant / 0.20).powi(2)).exp()
                * (-((z + 18.0) / 11.0).powi(2)).exp();

            let crest_fade = if z > z_bot + 30.0 {
                1.0
            } else {
                ((z - z_bot) / 30.0).max(0.0)
            };
            let tibial_crest = 3.8
                * crest_fade
                * (-((st - 1.0).powi(2) / 0.055)).exp()
                * (-((z + 75.0) / 52.0).powi(2)).exp();

            let med_condyle = (if is_full_knee { 7.2 } else { 5.2 })
                * (-(((ct + 0.98).powi(2) + (st + 0.05).powi(2)) / 0.16)).exp()
                * (-((z - 25.0) / 19.0).powi(2)).exp();
            let lat_condyle = (if is_full_knee { 6.2 } else { 4.6 })
                * (-(((ct - 0.98).powi(2) + (st + 0.05).powi(2)) / 0.16)).exp()
                * (-((z - 27.0) / 19.0).powi(2)).exp();

            let pop_fossa = (if is_full_knee { -6.8 } else { -4.2 })
                * (-((st + 1.0).powi(2) / 0.16)).exp()
                * (-((z - 4.0) / 16.0).powi(2)).exp();
            let ham_tendons = if is_full_knee {
                4.2 * ((-(((st + 0.86).powi(2) + (ct + 0.50).powi(2)) / 0.14)).exp()
                    + (-(((st + 0.86).powi(2) + (ct - 0.50).powi(2)) / 0.14)).exp())
                    * (-((z - 44.0) / 34.0).powi(2)).exp()
            } else {
                0.0
            };

            let post_env = (-((z + 72.0) / 46.0).powi(2)).exp();
            let calf_med =
                5.8 * (-(((st + 0.92).powi(2) + (ct + 0.38).powi(2)) / 0.24)).exp() * post_env;
            let calf_lat =
                5.4 * (-(((st + 0.92).powi(2) + (ct - 0.38).powi(2)) / 0.24)).exp() * post_env;

            let vmo = if is_full_knee {
                6.2 * (-(((st - 0.72).powi(2) + (ct + 0.68).powi(2)) / 0.22)).exp()
                    * (-((z - 76.0) / 34.0).powi(2)).exp()
            } else {
                0.0
            };
            let quad_vl = if is_full_knee {
                5.8 * (-(((st - 0.68).powi(2) + (ct - 0.72).powi(2)) / 0.28)).exp()
                    * (-((z - 94.0) / 44.0).powi(2)).exp()
            } else {
                0.0
            };

            let fib_head = 4.6
                * (-(((ct - 0.96).powi(2) + (st + 0.26).powi(2)) / 0.11)).exp()
                * (-((z + 24.0) / 14.0).powi(2)).exp();

            let rad = (base_r
                + tri_shape
                + patella
                + knee_gutters
                + supra_dip
                + pt_shelf
                + pt_cord
                + tib_tub
                + tibial_crest
                + med_condyle
                + lat_condyle
                + pop_fossa
                + ham_tendons
                + calf_med
                + calf_lat
                + vmo
                + quad_vl
                + fib_head)
                .max(7.0);

            let y_shift = if is_full_knee {
                if z > 20.0 {
                    -0.15 * (z - 20.0)
                } else {
                    -0.07 * (20.0 - z).max(0.0)
                }
            } else if z > 10.0 {
                -0.04 * (z - 10.0)
            } else {
                -0.05 * (-z).max(0.0)
            };

            vertices.push(Vec3::new(rad * ct, rad * st + y_shift, z));
        }
    }

    for r in 0..(eff_rings - 1) {
        for s in 0..eff_seg {
            let sn = (s + 1) % eff_seg;
            let a = r * eff_seg + s;
            let b = r * eff_seg + sn;
            let c = (r + 1) * eff_seg + sn;
            let d = (r + 1) * eff_seg + s;
            faces.push([a, d, c]);
            faces.push([a, c, b]);
        }
    }

    let dist_y_shift = if is_full_knee {
        -0.07 * (20.0 + length_mm)
    } else {
        -0.05 * length_mm
    };
    let distal_pole = vertices.len();
    vertices.push(Vec3::new(0.0, dist_y_shift, z_bot));
    let last_ring = (eff_rings - 1) * eff_seg;
    for s in 0..eff_seg {
        let sn = (s + 1) % eff_seg;
        faces.push([last_ring + s, distal_pole, last_ring + sn]);
    }

    if !open_cuff {
        close_proximal_dome(&mut vertices, &mut faces, eff_seg, 22.0, 5);
    }

    let mesh = TriMesh::new(vertices, faces);
    let targets = [
        ("mpt", Vec3::new(0.0, pr, 0.0)),
        ("popliteal", Vec3::new(0.0, -pr, -5.0)),
        ("fibular_head", Vec3::new(pr + 3.5, -6.0, -24.0)),
        ("tibial_crest", Vec3::new(0.0, pr * 0.85, -75.0)),
        ("patella", Vec3::new(0.0, pr + 8.0, 33.0)),
        ("distal_end", Vec3::new(0.0, dist_y_shift, z_bot)),
    ];
    build_case(mesh, &targets, "transtibial_socket")
}

/// Generates an anatomically contoured transfemoral (TF) residual limb with hip/gluteal flare.
#[allow(clippy::too_many_arguments)]
pub fn generate_transfemoral_limb(
    length_mm: f64,
    proximal_radius: f64,
    hip_height_mm: f64,
    open_cuff: bool,
    n_rings: usize,
    n_seg: usize,
    seed: u64,
) -> SyntheticLimbCase {
    let scale_var = deterministic_scale_var(seed, 0.06);
    let pr = proximal_radius * scale_var;

    let z_top = hip_height_mm;
    let z_bot = -length_mm;
    let z_start = z_top - if open_cuff { 0.0 } else { 34.0 };

    let mut vertices = Vec::with_capacity((n_rings + 7) * n_seg + 2);
    let mut faces = Vec::with_capacity((n_rings + 6) * n_seg * 2);

    for r in 0..n_rings {
        let u = r as f64 / (n_rings - 1) as f64;
        let z = z_start + u * (z_bot + 10.0 - z_start);

        let base_r = if z >= 0.0 {
            let t_hip = z / hip_height_mm.max(1e-6);
            pr * (1.0 + 0.19 * t_hip.powf(1.05))
        } else {
            let t_len = (-z / length_mm.max(1e-6)).min(1.0);
            let mut br = pr * (1.0 - 0.36 * t_len.powf(0.92));
            if z < z_bot + 32.0 {
                let cap_u = ((z_bot + 32.0 - z) / 32.0).clamp(0.0, 1.0);
                br = (br * (1.0 - 0.86 * cap_u * cap_u).max(0.03).sqrt()).max(9.0);
            }
            br
        };

        for s in 0..n_seg {
            let theta = 2.0 * PI * s as f64 / n_seg as f64;
            let ct = theta.cos();
            let st = theta.sin();

            let glute = 12.5
                * (-(((st + 0.85).powi(2) + (ct - 0.25).powi(2)) / 0.45)).exp()
                * (-((z - 48.0) / 42.0).powi(2)).exp();
            let troch =
                6.8 * (-((ct - 1.0).powi(2) / 0.14)).exp() * (-((z - 12.0) / 24.0).powi(2)).exp();
            let ischial = 5.2
                * (-(((st + 0.78).powi(2) + (ct + 0.62).powi(2)) / 0.22)).exp()
                * (-((z - 4.0) / 18.0).powi(2)).exp();
            let scarpa = -4.0
                * (-(((st - 0.92).powi(2) + (ct + 0.25).powi(2)) / 0.22)).exp()
                * (-((z + 15.0) / 24.0).powi(2)).exp();
            let quad =
                4.2 * (-((st - 0.82).powi(2) / 0.35)).exp() * (-((z + 95.0) / 68.0).powi(2)).exp();
            let ham =
                4.6 * (-((st + 0.95).powi(2) / 0.38)).exp() * (-((z + 95.0) / 62.0).powi(2)).exp();

            let rad = (base_r + glute + troch + ischial + scarpa + quad + ham).max(8.0);
            let x_shift = -0.04 * (-z).max(0.0);
            let y_shift = 0.03 * z.max(0.0) - 0.04 * (-z).max(0.0);
            vertices.push(Vec3::new(rad * ct + x_shift, rad * st + y_shift, z));
        }
    }

    for r in 0..(n_rings - 1) {
        for s in 0..n_seg {
            let sn = (s + 1) % n_seg;
            let a = r * n_seg + s;
            let b = r * n_seg + sn;
            let c = (r + 1) * n_seg + sn;
            let d = (r + 1) * n_seg + s;
            faces.push([a, d, c]);
            faces.push([a, c, b]);
        }
    }

    let dist_pole = vertices.len();
    vertices.push(Vec3::new(-0.04 * length_mm, -0.04 * length_mm, z_bot));
    let last_ring = (n_rings - 1) * n_seg;
    for s in 0..n_seg {
        let sn = (s + 1) % n_seg;
        faces.push([last_ring + s, dist_pole, last_ring + sn]);
    }

    if !open_cuff {
        close_proximal_dome(&mut vertices, &mut faces, n_seg, 34.0, 6);
    }

    let mesh = TriMesh::new(vertices, faces);
    let targets = [
        ("mpt", Vec3::new(0.0, pr, 0.0)),
        ("popliteal", Vec3::new(-pr * 0.45, -pr * 0.82, 4.0)),
        ("fibular_head", Vec3::new(pr + 6.0, 0.0, 12.0)),
        ("ischial_shelf", Vec3::new(-pr * 0.45, -pr * 0.82, 4.0)),
        ("scarpas_triangle", Vec3::new(-12.0, pr * 0.88, -15.0)),
        ("trochanter", Vec3::new(pr + 6.0, 0.0, 12.0)),
        (
            "distal_end",
            Vec3::new(-0.04 * length_mm, -0.04 * length_mm, z_bot),
        ),
    ];
    build_case(mesh, &targets, "transfemoral_socket")
}

/// Generates an anatomically contoured lower-leg, heel, ankle, and foot mesh for AFO workflows.
pub fn generate_foot_ankle_mesh(
    plantarflexion_deg: f64,
    inversion_deg: f64,
    calf_height_mm: f64,
    foot_length_mm: f64,
    n_rings: usize,
    n_seg: usize,
) -> SyntheticLimbCase {
    let spow = |val: f64, exp: f64| val.abs().powf(exp).copysign(val);
    let scale_f = foot_length_mm / 230.0;
    let y_piv = 46.0;
    let z_piv = 84.0;
    let y_toe_end = 186.0 * scale_f;

    let n_shank = (n_rings / 3).max(14);
    let n_fan = (n_rings / 3).max(14);
    let n_foot = n_rings
        .saturating_sub(n_shank + n_fan)
        .saturating_add(2)
        .max(18);

    let apply_deform = |pt: Vec3, deform_w: f64| -> Vec3 {
        if deform_w <= 1e-6 {
            return pt;
        }
        let mut out = pt;
        let ankle = Vec3::new(0.0, 0.0, 58.0);
        if plantarflexion_deg != 0.0 {
            out = math3d::rodrigues_rotate(
                out,
                Vec3::new(1.0, 0.0, 0.0),
                (-plantarflexion_deg * deform_w).to_radians(),
                ankle,
            );
        }
        if inversion_deg != 0.0 {
            out = math3d::rodrigues_rotate(
                out,
                Vec3::new(0.0, 1.0, 0.0),
                (inversion_deg * deform_w).to_radians(),
                ankle,
            );
        }
        out
    };

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Tri> = Vec::new();
    let mut ring_count = 0usize;

    // Section 1: Vertical lower leg (calf belly down to supramalleolar ankle z = z_piv).
    for r in 0..n_shank {
        let t = r as f64 / (n_shank - 1) as f64;
        let z = calf_height_mm * (1.0 - t) + z_piv * t;
        let calf_w = (-((t - 0.28) / 0.26).powi(2)).exp();
        let end_fade = 1.0 - math3d::smoothstep5((t - 0.72) / 0.28);
        let y_front = 29.0 + 17.0 * (1.0 - t);
        let y_back = -(27.0 + 18.0 * (1.0 - t) + 12.0 * calf_w * end_fade);
        let half_w = 31.0 + 14.0 * (1.0 - t) + 6.5 * calf_w * end_fade;

        let cy = 0.5 * (y_front + y_back);
        let ry = 0.5 * (y_front - y_back);
        let rx = half_w;
        let achilles_w = (-((t - 0.94) / 0.16).powi(2)).exp();
        let deform_w = math3d::smoothstep5((t - 0.85) / 0.15) * 0.15;

        for s in 0..n_seg {
            let theta = 2.0 * PI * s as f64 / n_seg as f64;
            let ct = theta.cos();
            let st = theta.sin();
            let shin_ridge = 3.0
                * end_fade
                * (-((st - 1.0).powi(2) / 0.06)).exp()
                * (-((t - 0.42) / 0.34).powi(2)).exp();
            let calf_lobes = 4.5
                * calf_w
                * end_fade
                * ((-(((st + 0.92).powi(2) + (ct + 0.38).powi(2)) / 0.24)).exp()
                    + (-(((st + 0.92).powi(2) + (ct - 0.38).powi(2)) / 0.24)).exp());
            let pinch = -4.5
                * achilles_w
                * (-(((ct.abs() - 0.68).powi(2) + (st + 0.72).powi(2)) / 0.14)).exp();
            let ach_ridge = 2.6 * achilles_w * (-((st + 1.0).powi(2) / 0.06)).exp();

            let px = (rx + pinch) * ct;
            let py = cy + (ry + shin_ridge + calf_lobes + ach_ridge) * st;
            vertices.push(apply_deform(Vec3::new(px, py, z), deform_w));
        }
        ring_count += 1;
    }

    // Section 2: External-pivot ankle & heel fan sweep.
    for r in 1..n_fan {
        let sf = r as f64 / (n_fan - 1) as f64;
        let alpha = sf * FRAC_PI_2;
        let ca = alpha.cos();
        let sa = alpha.sin();

        let d_front = 17.0 + 7.0 * math3d::smoothstep5(sf) + 1.2 * (PI * sf).sin();
        let y_post_wall = -27.0 - 19.0 * math3d::smoothstep5(sf / 0.44);
        let d_wall = (y_piv - y_post_wall) / ca.max(0.04);
        let d_floor = z_piv / sa.max(0.04);
        let p_norm = 7.5;
        let d_back = (d_wall.powf(-p_norm) + d_floor.powf(-p_norm)).powf(-1.0 / p_norm);

        let d_mid = 0.5 * (d_front + d_back);
        let r_dp = 0.5 * (d_back - d_front);

        let x_med = -(31.0 + 1.5 * (PI * sf).sin() + 0.5 * math3d::smoothstep5(sf));
        let x_lat = 31.0 + 2.0 * (PI * sf).sin() + 4.5 * math3d::smoothstep5(sf);
        let cx = 0.5 * (x_med + x_lat);
        let rx = 0.5 * (x_lat - x_med);

        let med_mal = 6.2 * (-((sf - 0.18) / 0.09).powi(2)).exp();
        let lat_mal = 6.0 * (-((sf - 0.30) / 0.09).powi(2)).exp();
        let ach_w = (-(sf / 0.22).powi(2)).exp();
        let sole_flat_w = math3d::smoothstep5((sf - 0.36) / 0.50);
        let arch_start_w = math3d::smoothstep5((sf - 0.72) / 0.28);
        let deform_w = 0.15 + 0.85 * math3d::smoothstep5(sf / 0.50);

        for s in 0..n_seg {
            let theta = 2.0 * PI * s as f64 / n_seg as f64;
            let ct = theta.cos();
            let st = theta.sin();

            let mal_boost = med_mal * (-(((ct + 0.96).powi(2) + (st - 0.16).powi(2)) / 0.16)).exp()
                + lat_mal * (-(((ct - 0.96).powi(2) + (st + 0.18).powi(2)) / 0.16)).exp();
            let pinch =
                -4.5 * ach_w * (-(((ct.abs() - 0.68).powi(2) + (st + 0.72).powi(2)) / 0.14)).exp();
            let ach_ridge = 2.6 * ach_w * (-((st + 1.0).powi(2) / 0.06)).exp();

            let st_shaped = if st < 0.0 {
                spow(st, 1.0 - 0.28 * sole_flat_w)
            } else {
                st
            };
            let ct_shaped = spow(ct, 1.0 - 0.12 * sole_flat_w);

            let arch_lift = if st < 0.0 && arch_start_w > 0.0 {
                5.5 * arch_start_w * (-ct).max(0.0).powf(1.15) * (-st)
            } else {
                0.0
            };
            let dorsal_slope = if st > 0.0 {
                -4.0 * math3d::smoothstep5((sf - 0.60) / 0.40) * ct.max(0.0) * st
            } else {
                0.0
            };

            let d_ray = d_mid - (r_dp + ach_ridge) * st_shaped - arch_lift - dorsal_slope;
            let px = cx + (rx + mal_boost + pinch) * ct_shaped;
            let py = y_piv - d_ray * ca;
            let pz = z_piv - d_ray * sa;
            vertices.push(apply_deform(Vec3::new(px, py, pz), deform_w));
        }
        ring_count += 1;
    }

    // Section 3: Coronal foot slices (midfoot arch -> metatarsal flare -> asymmetrical toe box).
    for r in 1..n_foot {
        let tf = r as f64 / (n_foot - 1) as f64;
        let y_base = y_piv + tf * (y_toe_end - y_piv);

        let medial_waist = -3.0 * (-((tf - 0.24) / 0.15).powi(2)).exp();
        let mpj1_flare = 15.0 * (-((tf - 0.66) / 0.24).powi(2)).exp();
        let x_med_raw = -(31.5 + medial_waist + mpj1_flare);
        let mpj5_flare = 10.5 * (-((tf - 0.52) / 0.26).powi(2)).exp();
        let x_lat_raw = 35.5 + mpj5_flare;

        let (x_med, x_lat) = if tf > 0.72 {
            let tt = (tf - 0.72) / 0.28;
            let med_taper = (1.0 - 0.82 * tt.powf(2.1)).max(0.06).sqrt();
            let lat_taper = (1.0 - 0.90 * tt.powf(1.75)).max(0.05).sqrt();
            let toe_cx_shift = -8.5 * math3d::smoothstep5(tt);
            (
                toe_cx_shift + (x_med_raw - toe_cx_shift) * med_taper,
                toe_cx_shift + (x_lat_raw - toe_cx_shift) * lat_taper,
            )
        } else {
            (x_med_raw, x_lat_raw)
        };

        let cx = 0.5 * (x_med + x_lat);
        let rx = (0.5 * (x_lat - x_med)).max(6.0);

        let mut z_top_base = 60.0 * (1.0 - tf).powf(1.35) + 21.0 * (1.0 - (1.0 - tf).powf(1.35));
        if tf > 0.80 {
            let tt_z = (tf - 0.80) / 0.20;
            z_top_base *= (1.0 - 0.78 * tt_z * tt_z).max(0.14).sqrt();
        }
        let z_bot_base = 3.5 * math3d::smoothstep5((tf - 0.80) / 0.20);
        let cz = 0.5 * (z_top_base + z_bot_base);
        let rz = (0.5 * (z_top_base - z_bot_base)).max(4.5);

        let arch_env = (-((tf - 0.22) / 0.20).powi(2)).exp();
        let arch_amp = 5.5 * (1.0 - math3d::smoothstep5(tf / 0.14))
            + 17.5 * arch_env * math3d::smoothstep5(tf / 0.12);
        let lat_arch = 2.5 * (-((tf - 0.26) / 0.18).powi(2)).exp();

        for s in 0..n_seg {
            let theta = 2.0 * PI * s as f64 / n_seg as f64;
            let ct = theta.cos();
            let st = theta.sin();

            let st_shaped = if st < 0.0 {
                spow(st, 0.72)
            } else {
                spow(st, 0.92)
            };
            let ct_shaped = spow(ct, 0.88);

            let px = cx + rx * ct_shaped;
            let mut pz = cz + rz * st_shaped;
            if st < 0.0 {
                let med_w = (-ct).max(0.0).powf(1.15);
                let lat_w = ct.max(0.0).powf(1.2);
                pz += (arch_amp * med_w + lat_arch * lat_w) * (-st);
            } else {
                let slope_w = 1.0 - 0.55 * tf;
                pz -= 4.0 * slope_w * ct.max(0.0) * st;
            }

            let mut py = y_base;
            if tf > 0.62 {
                let t_ob = math3d::smoothstep5((tf - 0.62) / 0.38);
                py += t_ob * scale_f * (-11.0 * ct.max(0.0).powf(1.15) + 3.5 * (-ct).max(0.0));
            }
            vertices.push(apply_deform(Vec3::new(px, py, pz), 1.0));
        }
        ring_count += 1;
    }

    for r in 0..(ring_count - 1) {
        for s in 0..n_seg {
            let sn = (s + 1) % n_seg;
            let a = r * n_seg + s;
            let b = r * n_seg + sn;
            let c = (r + 1) * n_seg + sn;
            let d = (r + 1) * n_seg + s;
            faces.push([a, d, c]);
            faces.push([a, c, b]);
        }
    }

    close_proximal_dome(&mut vertices, &mut faces, n_seg, 18.0, 4);

    // Smooth rounded distal toe-tip dome aligned with plantarflexed forefoot axis.
    let last_start = (ring_count - 1) * n_seg;
    let last_ring: Vec<usize> = (last_start..(last_start + n_seg)).collect();
    let inv = 1.0 / n_seg as f64;
    let tcx = last_ring.iter().map(|&i| vertices[i].x).sum::<f64>() * inv;
    let tcy = last_ring.iter().map(|&i| vertices[i].y).sum::<f64>() * inv;
    let tcz = last_ring.iter().map(|&i| vertices[i].z).sum::<f64>() * inv;
    let toe_offsets: Vec<Vec3> = last_ring
        .iter()
        .map(|&i| {
            Vec3::new(
                vertices[i].x - tcx,
                vertices[i].y - tcy,
                vertices[i].z - tcz,
            )
        })
        .collect();

    let toe_dir_p1 = apply_deform(Vec3::new(0.0, 1.0, 58.0), 1.0);
    let toe_dir_p0 = apply_deform(Vec3::new(0.0, 0.0, 58.0), 1.0);
    let toe_dir = math3d::v_sub(toe_dir_p1, toe_dir_p0);
    let toe_dome_len = 11.0 * scale_f;

    let mut prev_toe = last_ring;
    for k in 1..3 {
        let phi = (k as f64 / 3.0) * FRAC_PI_2;
        let r_sc = phi.cos();
        let ext = toe_dome_len * phi.sin();
        let mut next_toe = Vec::with_capacity(n_seg);
        for d in &toe_offsets {
            next_toe.push(vertices.len());
            vertices.push(Vec3::new(
                tcx + d.x * r_sc + toe_dir.x * ext,
                tcy + d.y * r_sc + toe_dir.y * ext,
                tcz + d.z * r_sc + toe_dir.z * ext,
            ));
        }
        for s in 0..n_seg {
            let sn = (s + 1) % n_seg;
            faces.push([prev_toe[s], next_toe[s], next_toe[sn]]);
            faces.push([prev_toe[s], next_toe[sn], prev_toe[sn]]);
        }
        prev_toe = next_toe;
    }

    let toe_pole = vertices.len();
    vertices.push(Vec3::new(
        tcx + toe_dir.x * toe_dome_len,
        tcy + toe_dir.y * toe_dome_len,
        tcz + toe_dir.z * toe_dome_len,
    ));
    for s in 0..n_seg {
        let sn = (s + 1) % n_seg;
        faces.push([prev_toe[s], toe_pole, prev_toe[sn]]);
    }

    let mesh = TriMesh::new(vertices, faces);
    let targets = [
        ("medial_malleolus", Vec3::new(-36.0, 2.0, 72.0)),
        ("lateral_malleolus", Vec3::new(36.0, -7.0, 60.0)),
        ("heel", Vec3::new(0.0, -45.0, 18.0)),
        ("metatarsal_head", Vec3::new(-10.0, 136.0 * scale_f, 4.0)),
        (
            "proximal_calf",
            Vec3::new(0.0, -52.0, calf_height_mm - 45.0),
        ),
    ];
    build_case(mesh, &targets, "afo")
}

fn build_case(mesh: TriMesh, targets: &[(&str, Vec3)], device_type: &str) -> SyntheticLimbCase {
    let mut landmarks = BTreeMap::new();
    let mut landmark_coords = BTreeMap::new();
    for &(name, pt) in targets {
        let idx = mesh.nearest_vertex(pt);
        landmarks.insert(name.to_string(), idx);
        landmark_coords.insert(name.to_string(), mesh.vertices[idx]);
    }
    SyntheticLimbCase {
        mesh,
        landmarks,
        landmark_coords,
        device_type: device_type.to_string(),
    }
}
