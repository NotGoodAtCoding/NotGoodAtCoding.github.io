//! 4-axis rotary CNC foam-carver G-code generator and ball-nose stock-removal simulator.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write;

use serde::Deserialize;
use serde::Serialize;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::TriMesh;

/// Generated 4-axis rotary G-code and ball-nose stock-removal verification metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarverCamResult {
    pub gcode: String,
    pub line_count: usize,
    pub rms_error_mm: f64,
    pub max_gouge_mm: f64,
    pub max_excess_mm: f64,
    pub mandrel_collisions: usize,
    pub stock_radius_mm: f64,
    pub ball_radius_mm: f64,
    pub warnings: Vec<String>,
}

fn ray_tri_x_dist(origin: Vec3, direction: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f64> {
    let e1 = math3d::v_sub(v1, v0);
    let e2 = math3d::v_sub(v2, v0);
    let pvec = math3d::v_cross(direction, e2);
    let det = math3d::v_dot(e1, pvec);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = math3d::v_sub(origin, v0);
    let u = math3d::v_dot(tvec, pvec) * inv_det;
    if !(-1e-5..=1.00001).contains(&u) {
        return None;
    }
    let qvec = math3d::v_cross(tvec, e1);
    let v = math3d::v_dot(direction, qvec) * inv_det;
    if v < -1e-5 || u + v > 1.00001 {
        return None;
    }
    let t = math3d::v_dot(e2, qvec) * inv_det;
    if t > 1e-4 {
        Some(t)
    } else {
        None
    }
}

fn sample_cylindrical_heightmap(
    positive_mesh: &TriMesh,
    z_step_mm: f64,
    n_theta: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (bmin, bmax) = positive_mesh.bounds();
    let z0 = bmin.z + 1.0;
    let z1 = bmax.z - 1.0;
    let c = positive_mesh.centroid();

    let n_z = ((((z1 - z0) / z_step_mm) as usize) + 1).max(8);
    let z_vals: Vec<f64> = (0..n_z)
        .map(|i| z0 + i as f64 * (z1 - z0) / (n_z - 1) as f64)
        .collect();
    let thetas: Vec<f64> = (0..n_theta)
        .map(|k| 2.0 * PI * k as f64 / n_theta as f64)
        .collect();

    let bucket_size = (z_step_mm * 2.0).max(2.0);
    let mut buckets: HashMap<i32, Vec<usize>> = HashMap::new();
    for (fi, &[a, b, c_idx]) in positive_mesh.faces.iter().enumerate() {
        let za = positive_mesh.vertices[a].z;
        let zb = positive_mesh.vertices[b].z;
        let zc = positive_mesh.vertices[c_idx].z;
        let b_lo = (za.min(zb).min(zc) / bucket_size).floor() as i32;
        let b_hi = (za.max(zb).max(zc) / bucket_size).floor() as i32;
        for bk in b_lo..=b_hi {
            buckets.entry(bk).or_default().push(fi);
        }
    }

    let mut grid = vec![vec![0.0f64; n_theta]; n_z];
    for (iz, &z) in z_vals.iter().enumerate() {
        let bk = (z / bucket_size).floor() as i32;
        let origin = Vec3::new(c.x, c.y, z);
        if let Some(cand_faces) = buckets.get(&bk) {
            for (it, &th) in thetas.iter().enumerate() {
                let dir = Vec3::new(th.cos(), th.sin(), 0.0);
                let mut max_r = 0.0f64;
                for &fi in cand_faces {
                    let [a, b, c_idx] = positive_mesh.faces[fi];
                    if let Some(t) = ray_tri_x_dist(
                        origin,
                        dir,
                        positive_mesh.vertices[a],
                        positive_mesh.vertices[b],
                        positive_mesh.vertices[c_idx],
                    ) {
                        max_r = max_r.max(t);
                    }
                }
                grid[iz][it] = max_r;
            }
        }
    }

    for iz in 0..n_z {
        for it in 0..n_theta {
            if grid[iz][it] <= 1e-3 {
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for dz in -1isize..=1 {
                    let nz = (iz as isize + dz).clamp(0, (n_z - 1) as isize) as usize;
                    for dt in -1isize..=1 {
                        let nt = (it as isize + dt).rem_euclid(n_theta as isize) as usize;
                        let val = grid[nz][nt];
                        if val > 1e-3 {
                            sum += val;
                            cnt += 1;
                        }
                    }
                }
                grid[iz][it] = if cnt > 0 { sum / cnt as f64 } else { 15.0 };
            }
        }
    }

    (z_vals, grid)
}

/// Generates gouge-free 4-axis rotary foam-carver G-code (`X, Z, A`) and verifies stock removal.
pub fn generate_4axis_gcode_and_simulate(
    positive_mesh: &TriMesh,
    ball_radius_mm: f64,
    z_step_mm: f64,
    n_theta: usize,
) -> CarverCamResult {
    let rough_allowance_mm = 3.0;
    let feed_rough = 3200.0;
    let feed_finish = 2000.0;

    let (z_vals, r_target) = sample_cylindrical_heightmap(positive_mesh, z_step_mm, n_theta);
    let n_z = z_vals.len();
    let max_r = r_target
        .iter()
        .flat_map(|row| row.iter().copied())
        .fold(0.0f64, f64::max);
    let stock_r = max_r + 8.0;
    let dz_step = (z_vals[n_z - 1] - z_vals[0]) / (n_z - 1).max(1) as f64;
    let max_dz_idx = ((ball_radius_mm / dz_step.max(0.5)).ceil() as usize).max(1);
    let dtheta = 2.0 * PI / n_theta as f64;

    let mut r_center = vec![vec![0.0f64; n_theta]; n_z];
    for iz in 0..n_z {
        let z_cur = z_vals[iz];
        let j_lo = iz.saturating_sub(max_dz_idx);
        let j_hi = (iz + max_dz_idx + 1).min(n_z);
        for it in 0..n_theta {
            let mut best_rc = r_target[iz][it] + ball_radius_mm;
            for (jz, row) in r_target.iter().enumerate().take(j_hi).skip(j_lo) {
                let dz = z_vals[jz] - z_cur;
                if dz.abs() >= ball_radius_mm {
                    continue;
                }
                for dt_i in -6isize..=6 {
                    let jt = (it as isize + dt_i).rem_euclid(n_theta as isize) as usize;
                    let rs = row[jt];
                    let d_arc = rs * (dt_i as f64 * dtheta);
                    let rad2 = ball_radius_mm * ball_radius_mm - dz * dz - d_arc * d_arc;
                    if rad2 > 0.0 {
                        best_rc = best_rc.max(rs + rad2.sqrt());
                    }
                }
            }
            r_center[iz][it] = best_rc;
        }
    }

    let retract_x = stock_r + 15.0;
    let mut gcode = String::with_capacity(n_z * n_theta * 36);
    let mut line_count = 0usize;

    let _ = writeln!(
        gcode,
        "; op-core 4-Axis Rotary Foam Carver G-Code (Rust/WASM)"
    );
    let _ = writeln!(
        gcode,
        "; StockRadius={stock_r:.2}mm BallRadius={ball_radius_mm:.2}mm ZStep={dz_step:.2}mm"
    );
    let _ = writeln!(gcode, "G21 ; Millimeters");
    let _ = writeln!(gcode, "G90 ; Absolute positioning");
    let _ = writeln!(gcode, "M3 S14000 ; Spindle ON");
    let _ = writeln!(gcode, "G0 X{retract_x:.3} Z{:.3} A0.000", z_vals[0]);
    let _ = writeln!(gcode, "; --- Roughing Pass ---");
    let _ = writeln!(gcode, "G1 F{feed_rough:.0}");
    line_count += 8;

    let mut a_cum = 0.0;
    let mut iz = 0;
    while iz < n_z {
        let z = z_vals[iz];
        for (k, &rc) in r_center[iz].iter().enumerate() {
            let x_tip = (rc + rough_allowance_mm) - ball_radius_mm;
            let a_deg = a_cum + k as f64 * (360.0 / n_theta as f64);
            let _ = writeln!(gcode, "G1 X{x_tip:.3} Z{z:.3} A{a_deg:.2}");
            line_count += 1;
        }
        a_cum += 360.0;
        iz += 2;
    }

    let _ = writeln!(gcode, "; --- Finishing Pass ---");
    let _ = writeln!(gcode, "G1 F{feed_finish:.0}");
    line_count += 2;

    for iz_f in 0..n_z {
        let z0_ring = z_vals[iz_f];
        let z1_ring = z_vals[(iz_f + 1).min(n_z - 1)];
        for (k, &rc) in r_center[iz_f].iter().enumerate() {
            let frac = k as f64 / n_theta as f64;
            let z_cur = z0_ring * (1.0 - frac) + z1_ring * frac;
            let x_tip = rc - ball_radius_mm;
            let a_deg = a_cum + k as f64 * (360.0 / n_theta as f64);
            let _ = writeln!(gcode, "G1 X{x_tip:.3} Z{z_cur:.3} A{a_deg:.2}");
            line_count += 1;
        }
        a_cum += 360.0;
    }

    let _ = writeln!(gcode, "G0 X{retract_x:.3}");
    let _ = writeln!(gcode, "M5 ; Spindle OFF");
    let _ = writeln!(gcode, "M30 ; Program End");
    line_count += 3;

    let mut sq_err_sum = 0.0;
    let mut count = 0usize;
    let mut max_gouge = 0.0f64;
    let mut max_excess = 0.0f64;
    let mut mandrel_collisions = 0usize;
    let z_top_limit = z_vals[n_z - 1] + 1.0;

    for iz_s in 0..n_z {
        let z_s = z_vals[iz_s];
        let j_lo = iz_s.saturating_sub(max_dz_idx);
        let j_hi = (iz_s + max_dz_idx + 1).min(n_z);
        for (it, &rs) in r_target[iz_s].iter().enumerate() {
            let mut r_carved = stock_r;
            for (jz, row) in r_center.iter().enumerate().take(j_hi).skip(j_lo) {
                let dz = z_vals[jz] - z_s;
                if dz.abs() >= ball_radius_mm {
                    continue;
                }
                for dt_i in -6isize..=6 {
                    let jt = (it as isize + dt_i).rem_euclid(n_theta as isize) as usize;
                    let d_arc = rs * (dt_i as f64 * dtheta);
                    let rad2 = ball_radius_mm * ball_radius_mm - dz * dz - d_arc * d_arc;
                    if rad2 > 0.0 {
                        r_carved = r_carved.min(row[jt] - rad2.sqrt());
                    }
                }
            }
            if r_carved < 12.0 && z_s > z_top_limit - 10.0 {
                mandrel_collisions += 1;
            }
            let diff = r_carved - rs;
            sq_err_sum += diff * diff;
            count += 1;
            if diff < -1e-6 {
                max_gouge = max_gouge.max(-diff);
            } else if diff > 0.0 {
                max_excess = max_excess.max(diff);
            }
        }
    }

    let rms = (sq_err_sum / count.max(1) as f64).sqrt();
    let mut warnings = Vec::new();
    if max_excess > 1.5 {
        warnings.push(format!(
            "Ball cutter radius ({ball_radius_mm:.1} mm) leaves {max_excess:.2} mm scallop in tight concavity"
        ));
    }

    CarverCamResult {
        gcode,
        line_count,
        rms_error_mm: rms,
        max_gouge_mm: max_gouge,
        max_excess_mm: max_excess,
        mandrel_collisions,
        stock_radius_mm: stock_r,
        ball_radius_mm,
        warnings,
    }
}
