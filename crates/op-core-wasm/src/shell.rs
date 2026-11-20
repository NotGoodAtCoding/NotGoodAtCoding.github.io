//! Watertight variable-thickness shelling, contoured 3D trimlines, hardware bosses, and carver molds.

use std::collections::HashMap;
use std::collections::HashSet;
use std::f64::consts::PI;

use serde_json::Map;
use serde_json::Value;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::Tri;
use crate::mesh::TriMesh;
use crate::recipe;
use crate::repair;

/// Compiled watertight socket or AFO shell, reinforcement modifier volume, and hardware metrics.
#[derive(Debug, Clone)]
pub struct ShellBuildResult {
    pub shell_mesh: TriMesh,
    pub modifier_mesh: TriMesh,
    pub alignment_normal: Vec3,
    pub trimline_min_radius_mm: f64,
    pub hardware_metrics: Map<String, Value>,
}

/// Carver-ready positive foam mold with keyed mandrel bore, draft collar, and overhang diagnostics.
#[derive(Debug, Clone)]
pub struct CarverPositiveResult {
    pub positive_mesh: TriMesh,
    pub mandrel_axis: Vec3,
    pub draft_angle_deg: f64,
    pub max_overhang_deg: f64,
    pub recommended_cant_deg: f64,
}

fn trimline_preset_quadrants(name: &str) -> Option<(f64, f64, f64, f64)> {
    match name {
        "ptb" => Some((30.0, 0.0, 24.0, 24.0)),
        "ptb_sc" => Some((28.0, -2.0, 48.0, 46.0)),
        "ptb_scsp" => Some((56.0, -2.0, 50.0, 48.0)),
        "tf_ischial" => Some((32.0, 18.0, 10.0, 50.0)),
        "flat" => Some((38.0, 38.0, 38.0, 38.0)),
        _ => None,
    }
}

fn shaft_center_xy(mesh: &TriMesh) -> (f64, f64) {
    if mesh.vertices.is_empty() {
        return (0.0, 0.0);
    }
    let (bmin, bmax) = mesh.bounds();
    let span = (bmax.z - bmin.z).max(1e-6);
    let z_lo = bmin.z + 0.25 * span;
    let z_hi = bmin.z + 0.75 * span;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cnt = 0usize;
    for &v in &mesh.vertices {
        if v.z >= z_lo && v.z <= z_hi {
            sx += v.x;
            sy += v.y;
            cnt += 1;
        }
    }
    if cnt >= 12 {
        (sx / cnt as f64, sy / cnt as f64)
    } else {
        let c = mesh.centroid();
        (c.x, c.y)
    }
}

/// Evaluates contoured socket trimline height `Z(theta)` around the limb circumference.
pub fn evaluate_trimline_z(
    theta: f64,
    trimline: &Value,
    center_xy: (f64, f64),
    default_z: f64,
) -> f64 {
    if let Some(pts) = trimline.get("points").and_then(Value::as_array) {
        if pts.len() >= 3 {
            let (cx, cy) = center_xy;
            let mut knots: Vec<(f64, f64)> = pts
                .iter()
                .filter_map(|p| recipe::parse_xyz(Some(p)))
                .map(|p| ((p.y - cy).atan2(p.x - cx), p.z))
                .collect();
            knots.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut dedup: Vec<(f64, f64)> = Vec::with_capacity(knots.len());
            for (ang, zv) in knots {
                if let Some(last) = dedup.last_mut() {
                    if (ang - last.0).abs() < 1e-3 {
                        last.0 = 0.5 * (last.0 + ang);
                        last.1 = 0.5 * (last.1 + zv);
                        continue;
                    }
                }
                dedup.push((ang, zv));
            }
            if dedup.len() >= 3 {
                let mut ext = Vec::with_capacity(dedup.len() + 2);
                ext.push((
                    dedup[dedup.len() - 1].0 - 2.0 * PI,
                    dedup[dedup.len() - 1].1,
                ));
                ext.extend_from_slice(&dedup);
                ext.push((dedup[0].0 + 2.0 * PI, dedup[0].1));
                let th = theta.sin().atan2(theta.cos());
                for k in 0..(ext.len() - 1) {
                    let (a0, z0) = ext[k];
                    let (a1, z1) = ext[k + 1];
                    if th >= a0 && th <= a1 {
                        let t = (th - a0) / (a1 - a0).max(1e-9);
                        let w = math3d::smoothstep5(t);
                        return z0 * (1.0 - w) + z1 * w;
                    }
                }
            }
        }
    }

    let preset_name = trimline.get("preset").and_then(Value::as_str).unwrap_or("");
    let (pa, pp, pm, pl) = trimline_preset_quadrants(preset_name)
        .unwrap_or((default_z, default_z, default_z, default_z));
    let z_ant = trimline
        .get("anterior_z_mm")
        .and_then(Value::as_f64)
        .unwrap_or(pa);
    let z_post = trimline
        .get("posterior_z_mm")
        .and_then(Value::as_f64)
        .unwrap_or(pp);
    let z_med = trimline
        .get("medial_z_mm")
        .and_then(Value::as_f64)
        .unwrap_or(pm);
    let z_lat = trimline
        .get("lateral_z_mm")
        .and_then(Value::as_f64)
        .unwrap_or(pl);

    let ct = theta.cos();
    let st = theta.sin();
    let w_ant = st.max(0.0).powi(2);
    let w_post = (-st).max(0.0).powi(2);
    let w_lat = ct.max(0.0).powi(2);
    let w_med = (-ct).max(0.0).powi(2);
    z_ant * w_ant + z_post * w_post + z_lat * w_lat + z_med * w_med
}

fn smooth_normals(mesh: &TriMesh, iterations: usize) -> Vec<Vec3> {
    let n = mesh.vertices.len();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &[a, b, c] in &mesh.faces {
        adj[a].insert(b);
        adj[a].insert(c);
        adj[b].insert(a);
        adj[b].insert(c);
        adj[c].insert(a);
        adj[c].insert(b);
    }
    let mut normals = mesh.vertex_normals();
    let c = mesh.centroid();
    for _ in 0..iterations {
        let mut nxt = Vec::with_capacity(n);
        for (i, &ni) in normals.iter().enumerate() {
            let mut acc = ni;
            for &nb in &adj[i] {
                acc = math3d::v_add(acc, math3d::v_scale(normals[nb], 0.35));
            }
            let mut sm = math3d::v_unit(acc, ni);
            let p = mesh.vertices[i];
            let rad = math3d::v_unit(Vec3::new(p.x - c.x, p.y - c.y, 0.0), sm);
            if math3d::v_dot(sm, rad) < -0.25 {
                sm = math3d::v_unit(
                    math3d::v_add(math3d::v_scale(sm, 0.7), math3d::v_scale(rad, 0.3)),
                    sm,
                );
            }
            nxt.push(sm);
        }
        normals = nxt;
    }
    normals
}

fn bridge_rings_manifold(ring_a: &[usize], ring_b: &[usize], faces: &mut Vec<Tri>, flip: bool) {
    let m = ring_a.len();
    for k in 0..m {
        let kn = (k + 1) % m;
        let a0 = ring_a[k];
        let a1 = ring_a[kn];
        let b0 = ring_b[k];
        let b1 = ring_b[kn];
        if !flip {
            faces.push([b0, a0, a1]);
            faces.push([b0, a1, b1]);
        } else {
            faces.push([b0, a1, a0]);
            faces.push([b0, b1, a1]);
        }
    }
}

fn wrap_angle_pos(ang: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let r = ang % two_pi;
    if r < 0.0 {
        r + two_pi
    } else {
        r
    }
}

fn angle_diff_abs(a: f64, b: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let d = wrap_angle_pos(a - b + PI) - PI;
    d.abs().min(two_pi - d.abs())
}

fn bridge_unequal_rings_manifold(
    outer_cw: &[usize],
    boss_cw: &[usize],
    verts: &[Vec3],
    faces: &mut Vec<Tri>,
    center_xy: (f64, f64),
) -> Result<(), String> {
    let m = outer_cw.len();
    let n = boss_cw.len();
    if m < 3 || n < 3 {
        return Err("Cannot bridge ring with fewer than 3 vertices".to_string());
    }

    let b0 = verts[boss_cw[0]];
    let b0_ang = wrap_angle_pos((b0.y - center_xy.1).atan2(b0.x - center_xy.0));
    let mut s0 = 0usize;
    let mut best_diff = f64::INFINITY;
    for (i, &idx) in outer_cw.iter().enumerate() {
        let p = verts[idx];
        let ang = wrap_angle_pos((p.y - center_xy.1).atan2(p.x - center_xy.0));
        let d = angle_diff_abs(ang, b0_ang);
        if d < best_diff {
            best_diff = d;
            s0 = i;
        }
    }

    let mut outer = Vec::with_capacity(m);
    outer.extend_from_slice(&outer_cw[s0..]);
    outer.extend_from_slice(&outer_cw[..s0]);

    let mut oi = 0usize;
    let mut bi = 0usize;
    while oi < m || bi < n {
        let o_cur = outer[oi % m];
        let o_nxt = outer[(oi + 1) % m];
        let b_cur = boss_cw[bi % n];
        let b_nxt = boss_cw[(bi + 1) % n];
        if oi == m {
            faces.push([b_nxt, b_cur, o_cur]);
            bi += 1;
        } else if bi == n {
            faces.push([o_cur, o_nxt, b_cur]);
            oi += 1;
        } else if oi == m - 1 && bi < n {
            faces.push([b_nxt, b_cur, o_cur]);
            bi += 1;
        } else if (oi + 1) as f64 / m as f64 <= (bi + 1) as f64 / n as f64 {
            faces.push([o_cur, o_nxt, b_cur]);
            oi += 1;
        } else {
            faces.push([b_nxt, b_cur, o_cur]);
            bi += 1;
        }
    }
    Ok(())
}

fn order_loop_matching_boundary(mesh: &TriMesh, lp: &[usize]) -> Vec<usize> {
    let mut half: HashSet<(usize, usize)> = HashSet::with_capacity(mesh.faces.len() * 3);
    for &[a, b, c] in &mesh.faces {
        half.insert((a, b));
        half.insert((b, c));
        half.insert((c, a));
    }
    let mut out = lp.to_vec();
    if out.len() >= 2 && !half.contains(&(out[0], out[1])) {
        out.reverse();
    }
    out
}

fn loop_signed_area_xy(verts: &[Vec3], lp: &[usize]) -> f64 {
    let mut s = 0.0;
    for k in 0..lp.len() {
        let p0 = verts[lp[k]];
        let p1 = verts[lp[(k + 1) % lp.len()]];
        s += p0.x * p1.y - p1.x * p0.y;
    }
    0.5 * s
}

fn build_captive_m6_pyramid_boss_cap(
    boss_center: Vec3,
    flexion_deg: f64,
    adduction_deg: f64,
) -> (Vec<Vec3>, Vec<Tri>, Vec<usize>) {
    let boss_radius_mm = 36.0;
    let boss_thickness_mm = 18.0;
    let bcd_mm = 50.8;
    let n_seg = 64usize;

    let align_vec = |p: Vec3| -> Vec3 {
        let q = math3d::rodrigues_rotate(
            p,
            Vec3::new(1.0, 0.0, 0.0),
            flexion_deg.to_radians(),
            Vec3::default(),
        );
        math3d::rodrigues_rotate(
            q,
            Vec3::new(0.0, 1.0, 0.0),
            adduction_deg.to_radians(),
            Vec3::default(),
        )
    };

    let ex = align_vec(Vec3::new(1.0, 0.0, 0.0));
    let ey = align_vec(Vec3::new(0.0, 1.0, 0.0));
    let ez = align_vec(Vec3::new(0.0, 0.0, 1.0));

    let to_world = |lx: f64, ly: f64, lz: f64| -> Vec3 {
        Vec3::new(
            boss_center.x + lx * ex.x + ly * ey.x + lz * ez.x,
            boss_center.y + lx * ex.y + ly * ey.y + lz * ez.y,
            boss_center.z + lx * ex.z + ly * ey.z + lz * ez.z,
        )
    };

    let mut verts: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Tri> = Vec::new();
    let mut top_ring_idx = Vec::with_capacity(n_seg);
    let mut bot_ring_idx = Vec::with_capacity(n_seg);

    for k in 0..n_seg {
        let ang = -2.0 * PI * k as f64 / n_seg as f64;
        let lx = boss_radius_mm * ang.cos();
        let ly = boss_radius_mm * ang.sin();
        top_ring_idx.push(verts.len());
        verts.push(to_world(lx, ly, boss_thickness_mm));
        bot_ring_idx.push(verts.len());
        verts.push(to_world(lx, ly, 0.0));
    }
    bridge_rings_manifold(&top_ring_idx, &bot_ring_idx, &mut faces, false);

    let r_bc = bcd_mm * 0.5;
    let r_hex = 5.8;
    let pocket_depth = 8.0;

    let center_bot_idx = verts.len();
    verts.push(to_world(0.0, 0.0, 0.0));

    let mut ccw_bot = bot_ring_idx;
    ccw_bot.reverse();
    let ccw_angs: Vec<f64> = ccw_bot
        .iter()
        .map(|&idx| {
            let d = math3d::v_sub(verts[idx], boss_center);
            wrap_angle_pos(math3d::v_dot(d, ey).atan2(math3d::v_dot(d, ex)))
        })
        .collect();

    let sep_targets = [0.0, 0.5 * PI, PI, 1.5 * PI];
    let sep_pos: Vec<usize> = sep_targets
        .iter()
        .map(|&targ| {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (p, &a) in ccw_angs.iter().enumerate() {
                let d = angle_diff_abs(a, targ);
                if d < best_d {
                    best_d = d;
                    best = p;
                }
            }
            best
        })
        .collect();

    for q in 0..4 {
        let p_start = sep_pos[q];
        let p_end = sep_pos[(q + 1) % 4];
        let mut arc = vec![ccw_bot[p_start]];
        let mut p = (p_start + 1) % n_seg;
        while p != p_end {
            arc.push(ccw_bot[p]);
            p = (p + 1) % n_seg;
        }
        arc.push(ccw_bot[p_end]);

        let hole_ang = (45.0 + 90.0 * q as f64).to_radians();
        let hx = r_bc * hole_ang.cos();
        let hy = r_bc * hole_ang.sin();

        let mut hex_bot = Vec::with_capacity(6);
        let mut hex_top = Vec::with_capacity(6);
        for k in 0..6 {
            let ha = hole_ang + k as f64 * (PI / 3.0);
            let px = hx + r_hex * ha.cos();
            let py = hy + r_hex * ha.sin();
            hex_bot.push(verts.len());
            verts.push(to_world(px, py, 0.0));
            hex_top.push(verts.len());
            verts.push(to_world(px, py, pocket_depth));
        }

        bridge_rings_manifold(&hex_top, &hex_bot, &mut faces, false);
        let hex_center = verts.len();
        verts.push(to_world(hx, hy, pocket_depth));
        for k in 0..6 {
            let kn = (k + 1) % 6;
            faces.push([hex_top[kn], hex_top[k], hex_center]);
        }

        let mut sector_cw = Vec::with_capacity(arc.len() + 1);
        sector_cw.push(center_bot_idx);
        sector_cw.extend(arc.into_iter().rev());
        let m_sec = sector_cw.len();

        let hx_w = to_world(hx, hy, 0.0);
        let sec_angs: Vec<f64> = sector_cw
            .iter()
            .map(|&v_idx| {
                let d = math3d::v_sub(verts[v_idx], hx_w);
                wrap_angle_pos(math3d::v_dot(d, ey).atan2(math3d::v_dot(d, ex)))
            })
            .collect();
        let mut hex_cw = hex_bot;
        hex_cw.reverse();
        let d0 = math3d::v_sub(verts[hex_cw[0]], hx_w);
        let h0_ang = wrap_angle_pos(math3d::v_dot(d0, ey).atan2(math3d::v_dot(d0, ex)));

        let mut s0 = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &sa) in sec_angs.iter().enumerate() {
            let d = angle_diff_abs(sa, h0_ang);
            if d < best_d {
                best_d = d;
                s0 = i;
            }
        }
        let mut outer = Vec::with_capacity(m_sec);
        outer.extend_from_slice(&sector_cw[s0..]);
        outer.extend_from_slice(&sector_cw[..s0]);

        let mut oi = 0usize;
        let mut ii = 0usize;
        while oi < m_sec || ii < 6 {
            let o_cur = outer[oi % m_sec];
            let o_nxt = outer[(oi + 1) % m_sec];
            let i_cur = hex_cw[ii % 6];
            let i_nxt = hex_cw[(ii + 1) % 6];
            if oi == m_sec {
                faces.push([o_cur, i_nxt, i_cur]);
                ii += 1;
            } else if ii == 6 {
                faces.push([o_cur, o_nxt, i_cur]);
                oi += 1;
            } else if oi == m_sec - 1 && ii < 6 {
                faces.push([o_cur, i_nxt, i_cur]);
                ii += 1;
            } else if (oi + 1) as f64 / m_sec as f64 <= (ii + 1) as f64 / 6.0 {
                faces.push([o_cur, o_nxt, i_cur]);
                oi += 1;
            } else {
                faces.push([o_cur, i_nxt, i_cur]);
                ii += 1;
            }
        }
    }

    (verts, faces, top_ring_idx)
}

fn clip_to_socket_trimline(
    mesh: &TriMesh,
    proximal_trim_z_mm: Option<f64>,
    trimline: Option<&Value>,
) -> Result<TriMesh, String> {
    if let Some(tl) = trimline {
        let (cx, cy) = shaft_center_xy(mesh);
        let (bmin, bmax) = mesh.bounds();
        let is_closed = mesh.boundary_loops().is_empty();
        let default_z = proximal_trim_z_mm.unwrap_or(38.0);
        let max_cut_z = if is_closed {
            bmax.z - 4.0
        } else {
            bmax.z + 100.0
        };
        let min_cut_z = bmin.z + 35.0;

        let mut values = Vec::with_capacity(mesh.vertices.len());
        let mut will_cut = false;
        for &v in &mesh.vertices {
            let th = (v.y - cy).atan2(v.x - cx);
            let zt = evaluate_trimline_z(th, tl, (cx, cy), default_z).clamp(min_cut_z, max_cut_z);
            let val = zt - v.z;
            if val < 0.0 {
                will_cut = true;
            }
            values.push(val);
        }
        if !will_cut {
            return Ok(mesh.clone());
        }
        let mut clipped = repair::clip_mesh_scalar(mesh, &values, 0.0, true)?;
        let comps = clipped.connected_components();
        if comps.len() > 1 {
            clipped = clipped.submesh_from_vertices(&comps[0]);
        }
        let loops = clipped.boundary_loops();
        if loops.len() == 1 && loops[0].len() >= 6 {
            let mut c_verts = clipped.vertices.clone();
            let lp = order_loop_matching_boundary(&clipped, &loops[0]);
            let m = lp.len();
            for _ in 0..2 {
                let mut upd = Vec::with_capacity(m);
                for k in 0..m {
                    let p0 = c_verts[lp[(k + m - 1) % m]];
                    let p1 = c_verts[lp[k]];
                    let p2 = c_verts[lp[(k + 1) % m]];
                    let avg = math3d::v_scale(math3d::v_add(p0, p2), 0.5);
                    upd.push((
                        lp[k],
                        math3d::v_add(math3d::v_scale(p1, 0.72), math3d::v_scale(avg, 0.28)),
                    ));
                }
                for (idx, pt) in upd {
                    c_verts[idx] = pt;
                }
            }
            clipped = TriMesh::new(c_verts, clipped.faces);
        }
        return Ok(clipped);
    }

    if let Some(trim_z) = proximal_trim_z_mm {
        let (_, bmax) = mesh.bounds();
        if bmax.z > trim_z + 5.0 {
            return Ok(repair::clip_mesh_z(mesh, trim_z, false));
        }
    }
    Ok(mesh.clone())
}

#[allow(clippy::too_many_arguments)]
fn build_socket_shell_core(
    rectified_positive: &TriMesh,
    wall_distal_mm: f64,
    wall_proximal_mm: f64,
    flexion_deg: f64,
    adduction_deg: f64,
    brim_round_segments: usize,
    hardware_tag: &str,
    alignment_type: &str,
    proximal_trim_z_mm: Option<f64>,
    trimline: Option<&Value>,
) -> Result<ShellBuildResult, String> {
    let trimmed = clip_to_socket_trimline(rectified_positive, proximal_trim_z_mm, trimline)?;
    let (pos, _) = repair::repair_scan_mesh(&trimmed, 1e-4, true, true, 0.06);
    let loops = pos.boundary_loops();
    if loops.len() != 1 {
        return Err(format!(
            "Expected 1 proximal cuff loop after repair, found {}",
            loops.len()
        ));
    }
    let cuff_loop = order_loop_matching_boundary(&pos, &loops[0]);

    let (bmin, bmax) = pos.bounds();
    let zmin = bmin.z;
    let zspan = (bmax.z - zmin).max(1e-6);
    let n_smooth = smooth_normals(&pos, 4);

    let out_full_verts: Vec<Vec3> = pos
        .vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let t_prox = ((v.z - zmin) / zspan).clamp(0.0, 1.0);
            let thick = wall_distal_mm * (1.0 - t_prox) + wall_proximal_mm * t_prox;
            math3d::v_add(v, math3d::v_scale(n_smooth[i], thick))
        })
        .collect();
    let s_out_full = TriMesh::new(out_full_verts.clone(), pos.faces.clone());

    let z_out_min = s_out_full.bounds().0.z;
    let z_clip = z_out_min + (0.14 * zspan).min(18.0);
    let s_out_clipped = repair::clip_mesh_z(&s_out_full, z_clip, true);
    let mut out_loops = s_out_clipped.boundary_loops();
    if out_loops.len() != 2 {
        return Err(format!(
            "Expected 2 boundary loops on clipped outer shell, found {}",
            out_loops.len()
        ));
    }

    out_loops.sort_by(|a, b| {
        let za = a.iter().map(|&i| s_out_clipped.vertices[i].z).sum::<f64>() / a.len() as f64;
        let zb = b.iter().map(|&i| s_out_clipped.vertices[i].z).sum::<f64>() / b.len() as f64;
        za.total_cmp(&zb)
    });
    let out_dist_loop = order_loop_matching_boundary(&s_out_clipped, &out_loops[0]);
    let out_cuff_raw = &out_loops[1];

    let out_cuff_ring: Vec<usize> = cuff_loop
        .iter()
        .map(|&i| {
            let target = out_full_verts[i];
            let mut best_j = out_cuff_raw[0];
            let mut best_d = f64::INFINITY;
            for &j in out_cuff_raw {
                let d = math3d::v_dist(s_out_clipped.vertices[j], target);
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            best_j
        })
        .collect();

    let mut verts = pos.vertices.clone();
    let mut faces: Vec<Tri> = pos.faces.iter().map(|&[a, b, c]| [a, c, b]).collect();

    let s_out_off = verts.len();
    verts.extend_from_slice(&s_out_clipped.vertices);
    for &[a, b, c] in &s_out_clipped.faces {
        faces.push([s_out_off + a, s_out_off + b, s_out_off + c]);
    }

    let in_ring = cuff_loop;
    let out_ring: Vec<usize> = out_cuff_ring
        .into_iter()
        .map(|idx| s_out_off + idx)
        .collect();
    let mut prev_ring = in_ring.clone();
    for step in 1..brim_round_segments {
        let ang = PI * step as f64 / brim_round_segments as f64;
        let mut mid_ring = Vec::with_capacity(in_ring.len());
        for (&idx_in, &idx_out) in in_ring.iter().zip(&out_ring) {
            let p_in = verts[idx_in];
            let p_out = verts[idx_out];
            let center = math3d::v_scale(math3d::v_add(p_in, p_out), 0.5);
            let half_vec = math3d::v_scale(math3d::v_sub(p_out, p_in), 0.5);
            let r_half = math3d::v_norm(half_vec);
            let pos_pt = Vec3::new(
                center.x - half_vec.x * ang.cos(),
                center.y - half_vec.y * ang.cos(),
                center.z - half_vec.z * ang.cos() + r_half * ang.sin(),
            );
            mid_ring.push(verts.len());
            verts.push(pos_pt);
        }
        bridge_rings_manifold(&prev_ring, &mid_ring, &mut faces, false);
        prev_ring = mid_ring;
    }
    bridge_rings_manifold(&prev_ring, &out_ring, &mut faces, false);

    let mut out_dist_cw = out_dist_loop;
    if loop_signed_area_xy(&s_out_clipped.vertices, &out_dist_cw) > 0.0 {
        out_dist_cw.reverse();
    }
    let out_dist_global: Vec<usize> = out_dist_cw.into_iter().map(|i| s_out_off + i).collect();

    let inv_d = 1.0 / out_dist_global.len() as f64;
    let cx_d = out_dist_global.iter().map(|&i| verts[i].x).sum::<f64>() * inv_d;
    let cy_d = out_dist_global.iter().map(|&i| verts[i].y).sum::<f64>() * inv_d;
    let boss_center = Vec3::new(cx_d, cy_d, z_clip - 22.0);

    let (boss_verts, boss_faces, boss_top_local) =
        build_captive_m6_pyramid_boss_cap(boss_center, flexion_deg, adduction_deg);
    let base_off = verts.len();
    verts.extend_from_slice(&boss_verts);
    for [a, b, c] in boss_faces {
        faces.push([base_off + a, base_off + b, base_off + c]);
    }
    let boss_top_global: Vec<usize> = boss_top_local.into_iter().map(|i| base_off + i).collect();

    bridge_unequal_rings_manifold(
        &out_dist_global,
        &boss_top_global,
        &verts,
        &mut faces,
        (cx_d, cy_d),
    )?;

    let shell = TriMesh::new(verts, faces);
    let mod_mesh = build_cylinder_mesh(
        Vec3::new(boss_center.x, boss_center.y, boss_center.z - 2.0),
        48.0,
        52.0,
        32,
    );

    let mut align_n = math3d::rodrigues_rotate(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
        flexion_deg.to_radians(),
        Vec3::default(),
    );
    align_n = math3d::rodrigues_rotate(
        align_n,
        Vec3::new(0.0, 1.0, 0.0),
        adduction_deg.to_radians(),
        Vec3::default(),
    );

    let mut metrics = Map::new();
    metrics.insert("hardware".to_string(), Value::from(hardware_tag));
    metrics.insert(
        "alignment_reference".to_string(),
        Value::from(alignment_type),
    );
    metrics.insert("bcd_mm".to_string(), Value::from(50.8));
    metrics.insert("flexion_deg".to_string(), Value::from(flexion_deg));
    metrics.insert("adduction_deg".to_string(), Value::from(adduction_deg));

    Ok(ShellBuildResult {
        shell_mesh: shell,
        modifier_mesh: mod_mesh,
        alignment_normal: math3d::v_unit(align_n, Vec3::new(0.0, 0.0, -1.0)),
        trimline_min_radius_mm: 15.0,
        hardware_metrics: metrics,
    })
}

/// Builds a watertight transtibial socket shell with contoured trimline and pyramid boss.
#[allow(clippy::too_many_arguments)]
pub fn build_tt_socket_shell(
    rectified_positive: &TriMesh,
    wall_distal_mm: f64,
    wall_proximal_mm: f64,
    flexion_deg: f64,
    adduction_deg: f64,
    proximal_trim_z_mm: Option<f64>,
    trimline: Option<&Value>,
) -> Result<ShellBuildResult, String> {
    build_socket_shell_core(
        rectified_positive,
        wall_distal_mm,
        wall_proximal_mm,
        flexion_deg,
        adduction_deg,
        3,
        "pyramid_50_8_m6_captive",
        "tt_mpt_bisection_plumb_line",
        proximal_trim_z_mm,
        trimline,
    )
}

/// Builds a watertight transfemoral socket shell with contoured trimline and TKA pyramid boss.
#[allow(clippy::too_many_arguments)]
pub fn build_tf_socket_shell(
    rectified_positive: &TriMesh,
    wall_distal_mm: f64,
    wall_proximal_mm: f64,
    flexion_deg: f64,
    adduction_deg: f64,
    proximal_trim_z_mm: Option<f64>,
    trimline: Option<&Value>,
) -> Result<ShellBuildResult, String> {
    build_socket_shell_core(
        rectified_positive,
        wall_distal_mm,
        wall_proximal_mm,
        flexion_deg,
        adduction_deg,
        3,
        "pyramid_50_8_m6_captive",
        "tf_tka_plumb_line",
        proximal_trim_z_mm,
        trimline,
    )
}

fn build_cylinder_mesh(center: Vec3, radius_mm: f64, height_mm: f64, n_seg: usize) -> TriMesh {
    let mut verts = Vec::with_capacity(n_seg * 2 + 2);
    let mut faces = Vec::with_capacity(n_seg * 4);
    let mut bot_ring = Vec::with_capacity(n_seg);
    let mut top_ring = Vec::with_capacity(n_seg);
    for k in 0..n_seg {
        let a = 2.0 * PI * k as f64 / n_seg as f64;
        let x = center.x + radius_mm * a.cos();
        let y = center.y + radius_mm * a.sin();
        bot_ring.push(verts.len());
        verts.push(Vec3::new(x, y, center.z));
        top_ring.push(verts.len());
        verts.push(Vec3::new(x, y, center.z + height_mm));
    }
    bridge_rings_manifold(&top_ring, &bot_ring, &mut faces, false);
    let bc = verts.len();
    verts.push(center);
    let tc = verts.len();
    verts.push(Vec3::new(center.x, center.y, center.z + height_mm));
    for k in 0..n_seg {
        let kn = (k + 1) % n_seg;
        faces.push([bot_ring[k], bot_ring[kn], bc]);
        faces.push([top_ring[kn], top_ring[k], tc]);
    }
    TriMesh::new(verts, faces)
}

/// Computes the minimum discrete osculating circle radius along a 3D polyline.
pub fn compute_curve_min_radius(points: &[Vec3], closed: bool) -> f64 {
    let n = points.len();
    if n < 3 {
        return f64::INFINITY;
    }
    let mut min_r = f64::INFINITY;
    let start = if closed { 0 } else { 1 };
    let end = if closed { n } else { n - 1 };
    for i in start..end {
        let p0 = points[(i + n - 1) % n];
        let p1 = points[i];
        let p2 = points[(i + 1) % n];
        let a = math3d::v_dist(p0, p1);
        let b = math3d::v_dist(p1, p2);
        let c = math3d::v_dist(p2, p0);
        let area2 = math3d::v_norm(math3d::v_cross(
            math3d::v_sub(p1, p0),
            math3d::v_sub(p2, p0),
        ));
        if area2 > 1e-6 {
            let r = (a * b * c) / (2.0 * area2);
            if r < min_r {
                min_r = r;
            }
        }
    }
    min_r
}

/// Builds a watertight posterior-leaf / solid AFO shell with coaxial joint pockets.
pub fn build_afo_shell(
    neutral_foot_mesh: &TriMesh,
    wall_neck_mm: f64,
    wall_calf_mm: f64,
    trimline_fillet_mm: f64,
) -> Result<ShellBuildResult, String> {
    let joint_y_mm = 0.0;
    let joint_z_mm = 56.0;
    let calf_anterior_y_mm = 12.0;
    let footplate_z_mm = 24.0;
    let footplate_max_y_mm = 240.0;
    let joint_recess_radius_mm = 8.0;
    let joint_recess_depth_mm = 2.0;

    let mut kept_verts: HashSet<usize> = HashSet::new();
    for &[a, b, c] in &neutral_foot_mesh.faces {
        let pa = neutral_foot_mesh.vertices[a];
        let pb = neutral_foot_mesh.vertices[b];
        let pc = neutral_foot_mesh.vertices[c];
        let cy = (pa.y + pb.y + pc.y) / 3.0;
        let cz = (pa.z + pb.z + pc.z) / 3.0;
        let keep = if cz > 75.0 {
            cy < calf_anterior_y_mm
        } else if cz >= 38.0 {
            let t = (cz - 38.0) / 37.0;
            let y_cut =
                35.0 * (1.0 - math3d::smoothstep5(t)) + calf_anterior_y_mm * math3d::smoothstep5(t);
            cy < y_cut
        } else {
            cz < footplate_z_mm && cy <= footplate_max_y_mm
        };
        if keep {
            kept_verts.insert(a);
            kept_verts.insert(b);
            kept_verts.insert(c);
        }
    }

    let kept_vec: Vec<usize> = kept_verts.into_iter().collect();
    let mut patch = neutral_foot_mesh.submesh_from_vertices(&kept_vec);
    let comps = patch.connected_components();
    if !comps.is_empty() {
        patch = patch.submesh_from_vertices(&comps[0]);
    }

    let loops = patch.boundary_loops();
    if loops.is_empty() {
        return Err("AFO trimline patch produced no open boundary loop".to_string());
    }

    let mut p_verts = patch.vertices.clone();
    let main_loop = loops
        .iter()
        .max_by_key(|lp| lp.len())
        .expect("loops checked non-empty");
    let m = main_loop.len();
    for _ in 0..6 {
        let mut updates = Vec::with_capacity(m);
        for k in 0..m {
            let prev_p = p_verts[main_loop[(k + m - 1) % m]];
            let cur_p = p_verts[main_loop[k]];
            let next_p = p_verts[main_loop[(k + 1) % m]];
            let avg = math3d::v_scale(math3d::v_add(prev_p, next_p), 0.5);
            updates.push((
                main_loop[k],
                math3d::v_add(math3d::v_scale(cur_p, 0.55), math3d::v_scale(avg, 0.45)),
            ));
        }
        for (idx, val) in updates {
            p_verts[idx] = val;
        }
    }
    patch = TriMesh::new(p_verts, patch.faces);

    let loop_pts: Vec<Vec3> = main_loop.iter().map(|&i| patch.vertices[i]).collect();
    let min_r = compute_curve_min_radius(&loop_pts, true);
    let effective_fillet = min_r.max(trimline_fillet_mm);

    let ordered_loops: Vec<Vec<usize>> = patch
        .boundary_loops()
        .iter()
        .map(|lp| order_loop_matching_boundary(&patch, lp))
        .collect();
    let p_norms = smooth_normals(&patch, 3);

    let mut verts = patch.vertices.clone();
    let mut faces: Vec<Tri> = patch.faces.iter().map(|&[a, b, c]| [a, c, b]).collect();
    let mut out_map: HashMap<usize, usize> = HashMap::with_capacity(patch.vertices.len());

    for (i, &v) in patch.vertices.iter().enumerate() {
        let neck_w = (-((v.z - joint_z_mm) / 40.0).powi(2)).exp();
        let mut thick = wall_calf_mm + (wall_neck_mm - wall_calf_mm) * neck_w;
        if v.x.abs() > 22.0 {
            let r_joint = (v.y - joint_y_mm).hypot(v.z - joint_z_mm);
            if r_joint < 16.0 {
                let pad_w = 1.0 - math3d::smoothstep5(((r_joint - 9.0) / 7.0).max(0.0));
                thick += 4.2 * pad_w;
                if r_joint < joint_recess_radius_mm {
                    let rec_w =
                        1.0 - math3d::smoothstep5(r_joint / joint_recess_radius_mm.max(1e-3));
                    thick -= joint_recess_depth_mm * rec_w;
                }
            }
        }
        let out_idx = verts.len();
        out_map.insert(i, out_idx);
        verts.push(math3d::v_add(v, math3d::v_scale(p_norms[i], thick)));
    }

    for &[a, b, c] in &patch.faces {
        faces.push([out_map[&a], out_map[&b], out_map[&c]]);
    }

    for lp in ordered_loops {
        let out_ring: Vec<usize> = lp.iter().map(|i| out_map[i]).collect();
        let mut mid_ring = Vec::with_capacity(lp.len());
        for &idx in &lp {
            let p_in = verts[idx];
            let p_out = verts[out_map[&idx]];
            mid_ring.push(verts.len());
            verts.push(math3d::v_scale(math3d::v_add(p_in, p_out), 0.5));
        }
        bridge_rings_manifold(&lp, &mid_ring, &mut faces, false);
        bridge_rings_manifold(&mid_ring, &out_ring, &mut faces, false);
    }

    let shell = TriMesh::new(verts, faces);
    let mod_mesh = build_cylinder_mesh(
        Vec3::new(0.0, joint_y_mm, joint_z_mm - 35.0),
        56.0,
        70.0,
        32,
    );

    let mut metrics = Map::new();
    metrics.insert("hardware".to_string(), Value::from("afo_coaxial_joint"));
    metrics.insert("coaxial_error_deg".to_string(), Value::from(0.0));
    metrics.insert("joint_z_mm".to_string(), Value::from(joint_z_mm));

    Ok(ShellBuildResult {
        shell_mesh: shell,
        modifier_mesh: mod_mesh,
        alignment_normal: Vec3::new(1.0, 0.0, 0.0),
        trimline_min_radius_mm: effective_fillet,
        hardware_metrics: metrics,
    })
}

/// Converts a rectified positive into a watertight CNC foam-carver mold with keyed mandrel bore.
pub fn export_carver_positive(
    rectified_positive: &TriMesh,
    draft_angle_deg: f64,
    collar_height_mm: f64,
    mandrel_radius_mm: f64,
    bore_depth_mm: f64,
    proximal_trim_z_mm: Option<f64>,
    trimline: Option<&Value>,
) -> Result<CarverPositiveResult, String> {
    let trimmed = clip_to_socket_trimline(rectified_positive, proximal_trim_z_mm, trimline)?;
    let (pos, _) = repair::repair_scan_mesh(&trimmed, 1e-4, true, true, 0.06);
    let loops = pos.boundary_loops();
    if loops.len() != 1 {
        return Err(format!(
            "Carver prep expects 1 proximal cuff loop, found {}",
            loops.len()
        ));
    }

    let cuff = order_loop_matching_boundary(&pos, &loops[0]);
    let inv_c = 1.0 / cuff.len() as f64;
    let cx = cuff.iter().map(|&i| pos.vertices[i].x).sum::<f64>() * inv_c;
    let cy = cuff.iter().map(|&i| pos.vertices[i].y).sum::<f64>() * inv_c;
    let z_top = cuff
        .iter()
        .map(|&i| pos.vertices[i].z)
        .fold(f64::NEG_INFINITY, f64::max)
        + collar_height_mm;
    let tan_draft = draft_angle_deg.to_radians().tan();

    let mut verts = pos.vertices.clone();
    let mut faces = pos.faces.clone();

    let mut collar_ring = Vec::with_capacity(cuff.len());
    for &idx in &cuff {
        let v = verts[idx];
        let dx = v.x - cx;
        let dy = v.y - cy;
        let r = dx.hypot(dy);
        let dz = (z_top - v.z).max(5.0);
        let scale = (r + dz * tan_draft) / r.max(1e-6);
        collar_ring.push(verts.len());
        verts.push(Vec3::new(cx + dx * scale, cy + dy * scale, z_top));
    }
    bridge_rings_manifold(&collar_ring, &cuff, &mut faces, false);

    let m = collar_ring.len();
    let cuff_cw = loop_signed_area_xy(&verts, &collar_ring) < 0.0;
    let ang0 = (verts[collar_ring[0]].y - cy).atan2(verts[collar_ring[0]].x - cx);
    let sign = if cuff_cw { -1.0 } else { 1.0 };

    let mut bore_top_ring = Vec::with_capacity(m);
    let mut bore_bot_ring = Vec::with_capacity(m);
    for k in 0..m {
        let ang = ang0 + sign * 2.0 * PI * k as f64 / m as f64;
        let mut r_bore = mandrel_radius_mm;
        if ang.sin().abs() < 0.35 && ang.cos() > 0.0 {
            r_bore += 4.0;
        }
        let bx = cx + r_bore * ang.cos();
        let by = cy + r_bore * ang.sin();
        bore_top_ring.push(verts.len());
        verts.push(Vec3::new(bx, by, z_top));
        bore_bot_ring.push(verts.len());
        verts.push(Vec3::new(bx, by, z_top - bore_depth_mm));
    }
    bridge_rings_manifold(&bore_top_ring, &collar_ring, &mut faces, false);
    bridge_rings_manifold(&bore_bot_ring, &bore_top_ring, &mut faces, false);

    let bore_floor = verts.len();
    verts.push(Vec3::new(cx, cy, z_top - bore_depth_mm));
    for k in 0..m {
        let kn = (k + 1) % m;
        faces.push([bore_bot_ring[kn], bore_bot_ring[k], bore_floor]);
    }

    let carver_mesh = TriMesh::new(verts, faces);
    let max_overhang = carver_mesh
        .face_normals()
        .into_iter()
        .map(|fn_v| fn_v.z.clamp(-1.0, 1.0).acos().to_degrees())
        .fold(0.0f64, f64::max);

    let (bmin, bmax) = carver_mesh.bounds();
    let xy_span = (bmax.x - bmin.x).max(bmax.y - bmin.y);
    let z_span = bmax.z - bmin.z;
    let cant_deg = if xy_span > 0.75 * z_span { 35.0 } else { 0.0 };

    Ok(CarverPositiveResult {
        positive_mesh: carver_mesh,
        mandrel_axis: Vec3::new(0.0, 0.0, 1.0),
        draft_angle_deg,
        max_overhang_deg: max_overhang,
        recommended_cant_deg: cant_deg,
    })
}
