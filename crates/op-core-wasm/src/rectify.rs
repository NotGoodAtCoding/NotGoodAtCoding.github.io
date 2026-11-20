//! Non-destructive M_0-domain rectification, AFO posture correction, sculpting, and Taubin smoothing.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::Value;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::TriMesh;
use crate::recipe;
use crate::recipe::OpRecipe;

/// Output of evaluating an `.oprecipe.json` against a canonical limb scan `M_0`.
#[derive(Debug, Clone)]
pub struct RectificationResult {
    pub rectified_mesh: TriMesh,
    pub base_compressed_mesh: TriMesh,
    pub relief_field_mm: Vec<f64>,
    pub resolved_landmarks: BTreeMap<String, usize>,
}

fn zone_weight(z: f64, z_lo: f64, z_hi: f64, transition_mm: f64) -> f64 {
    if transition_mm <= 1e-6 {
        return if z >= z_lo && z <= z_hi { 1.0 } else { 0.0 };
    }
    let w_lo = math3d::smoothstep5((z - (z_lo - transition_mm)) / transition_mm);
    let w_hi = 1.0 - math3d::smoothstep5((z - z_hi) / transition_mm);
    (w_lo * w_hi).clamp(0.0, 1.0)
}

fn slice_centerline_xy(mesh: &TriMesh, bins: usize) -> (f64, f64, Vec<(f64, f64, f64)>) {
    let (bmin, bmax) = mesh.bounds();
    let zmin = bmin.z;
    let span = (bmax.z - zmin).max(1e-6);
    let mut buckets: Vec<Vec<Vec3>> = vec![Vec::new(); bins];
    for &v in &mesh.vertices {
        let b = (((v.z - zmin) / span * bins as f64) as usize).min(bins - 1);
        buckets[b].push(v);
    }

    let c_global = mesh.centroid();
    let mut centers = Vec::with_capacity(bins);
    let mut last_xy = (c_global.x, c_global.y);
    for (b, pts) in buckets.iter().enumerate() {
        let z_mid = zmin + (b as f64 + 0.5) * span / bins as f64;
        if pts.len() >= 3 {
            let inv = 1.0 / pts.len() as f64;
            last_xy = (
                pts.iter().map(|p| p.x).sum::<f64>() * inv,
                pts.iter().map(|p| p.y).sum::<f64>() * inv,
            );
        }
        centers.push((z_mid, last_xy.0, last_xy.1));
    }
    (zmin, span, centers)
}

fn interp_center_xy(z: f64, zmin: f64, span: f64, centers: &[(f64, f64, f64)]) -> (f64, f64) {
    let t = ((z - zmin) / span * centers.len() as f64 - 0.5).clamp(0.0, (centers.len() - 1) as f64);
    let i0 = t.floor() as usize;
    let i1 = (i0 + 1).min(centers.len() - 1);
    let frac = t - i0 as f64;
    (
        centers[i0].1 * (1.0 - frac) + centers[i1].1 * frac,
        centers[i0].2 * (1.0 - frac) + centers[i1].2 * frac,
    )
}

/// Smooth quintic ankle articulation and plantar sole flattening at footwear heel pitch.
pub fn apply_afo_posture(
    mesh: &TriMesh,
    dorsiflexion_deg: f64,
    eversion_deg: f64,
    heel_pitch_mm: f64,
) -> TriMesh {
    let talocrural_axis = math3d::v_unit(Vec3::new(1.0, -0.05, -0.14), Vec3::new(1.0, 0.0, 0.0));
    let subtalar_axis = math3d::v_unit(Vec3::new(0.0, 1.0, 0.15), Vec3::new(0.0, 1.0, 0.0));
    let ankle_origin = Vec3::new(0.0, 0.0, 58.0);

    let verts_rot: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|&p| {
            let w_z = 1.0 - math3d::smoothstep5((p.z - 35.0) / 50.0);
            let mut pt = p;
            if w_z > 1e-6 {
                if dorsiflexion_deg != 0.0 {
                    pt = math3d::rodrigues_rotate(
                        pt,
                        talocrural_axis,
                        (dorsiflexion_deg * w_z).to_radians(),
                        ankle_origin,
                    );
                }
                if eversion_deg != 0.0 {
                    pt = math3d::rodrigues_rotate(
                        pt,
                        subtalar_axis,
                        (-eversion_deg * w_z).to_radians(),
                        ankle_origin,
                    );
                }
            }
            pt
        })
        .collect();

    let z_floor_ref = verts_rot
        .iter()
        .filter(|v| v.y > -55.0)
        .map(|v| v.z)
        .fold(f64::INFINITY, f64::min);
    let z_floor_ref = if z_floor_ref.is_finite() {
        z_floor_ref
    } else {
        0.0
    };
    let y_heel = -22.0;
    let y_met = 132.0;

    let verts_flat = verts_rot
        .into_iter()
        .map(|mut v| {
            let t_y = ((v.y - y_heel) / (y_met - y_heel).max(1.0)).clamp(0.0, 1.0);
            let sole_target_z = z_floor_ref + heel_pitch_mm * (1.0 - t_y);
            if v.z < sole_target_z + 2.5 && v.y > -55.0 {
                if v.z < sole_target_z {
                    v.z = sole_target_z;
                } else {
                    let blend = 1.0 - math3d::smoothstep5((v.z - sole_target_z) / 2.5);
                    v.z = v.z * (1.0 - 0.35 * blend) + sole_target_z * (0.35 * blend);
                }
            }
            v
        })
        .collect();

    TriMesh::new(verts_flat, mesh.faces.clone())
}

fn accumulate_geodesic_kernel(
    relief_mm: &mut [f64],
    dists: &[f64],
    offset: f64,
    radius: f64,
    falloff: &str,
) {
    let rad = radius.max(1.0);
    if falloff == "flat_pad" {
        let inner_r = 0.45 * rad;
        let outer_r = (inner_r + 1.0).max(rad);
        for (slot, &d) in relief_mm.iter_mut().zip(dists) {
            if d <= inner_r {
                *slot += offset;
            } else if d < outer_r {
                let t_pad = (d - inner_r) / (outer_r - inner_r);
                *slot += offset * (1.0 - math3d::smoothstep5(t_pad));
            }
        }
    } else {
        let sigma = rad * 0.48;
        let inv_2s2 = 0.5 / (sigma * sigma);
        let cutoff = rad * 2.2;
        for (slot, &d) in relief_mm.iter_mut().zip(dists) {
            if d < cutoff {
                *slot += offset * (-d * d * inv_2s2).exp();
            }
        }
    }
}

/// Volume-preserving Taubin (lambda/mu) mesh smoothing with optional per-vertex mask weights.
pub fn taubin_smooth_mesh(
    mesh: &TriMesh,
    iterations: usize,
    strength: f64,
    weights: Option<&[f64]>,
) -> TriMesh {
    let n = mesh.vertices.len();
    if n == 0 || iterations == 0 || strength <= 0.0 {
        return mesh.clone();
    }

    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &[a, b, c] in &mesh.faces {
        adj[a].insert(b);
        adj[a].insert(c);
        adj[b].insert(a);
        adj[b].insert(c);
        adj[c].insert(a);
        adj[c].insert(b);
    }
    let nbrs: Vec<Vec<usize>> = adj.into_iter().map(|s| s.into_iter().collect()).collect();

    let mut boundary_verts: HashSet<usize> = HashSet::new();
    for lp in mesh.boundary_loops() {
        boundary_verts.extend(lp);
    }

    let s = strength.clamp(0.05, 0.95);
    let lam = 0.50 * s;
    let mu = -0.53 * s;
    let mut verts = mesh.vertices.clone();

    for _ in 0..iterations {
        for factor in [lam, mu] {
            let mut nxt = verts.clone();
            for i in 0..n {
                if boundary_verts.contains(&i) || nbrs[i].is_empty() {
                    continue;
                }
                let w = weights.map_or(1.0, |ws| ws[i]);
                if w <= 1e-6 {
                    continue;
                }
                let nb = &nbrs[i];
                let inv_k = 1.0 / nb.len() as f64;
                let mut avg = Vec3::default();
                for &j in nb {
                    avg = math3d::v_add(avg, verts[j]);
                }
                avg = math3d::v_scale(avg, inv_k);
                let v = verts[i];
                let fw = factor * w;
                nxt[i] = Vec3::new(
                    v.x + fw * (avg.x - v.x),
                    v.y + fw * (avg.y - v.y),
                    v.z + fw * (avg.z - v.z),
                );
            }
            verts = nxt;
        }
    }
    TriMesh::new(verts, mesh.faces.clone())
}

/// Evaluates an `.oprecipe.json` deterministically in the undeformed reference domain `M_0`.
pub fn apply_recipe(
    canonical_mesh: &TriMesh,
    recipe: &OpRecipe,
) -> Result<RectificationResult, String> {
    recipe.validate()?;
    let mut m0 = canonical_mesh.clone();
    if (recipe.scale_factor - 1.0).abs() > 1e-12 {
        let sf = recipe.scale_factor;
        m0 = m0.transformed(|p| math3d::v_scale(p, sf));
    }

    for op in &recipe.operations {
        if op.get("op").and_then(Value::as_str) == Some("afo_posture") {
            let df = op
                .get("dorsiflexion_deg")
                .and_then(Value::as_f64)
                .unwrap_or(12.0);
            let ev = op
                .get("eversion_deg")
                .and_then(Value::as_f64)
                .unwrap_or(5.0);
            let hp = op
                .get("heel_pitch_mm")
                .and_then(Value::as_f64)
                .unwrap_or(10.0);
            m0 = apply_afo_posture(&m0, df, ev, hp);
        }
    }

    let resolved_landmarks = recipe.resolve_landmarks(&m0);
    let n0 = m0.vertex_normals();
    let (zmin, span, centers) = slice_centerline_xy(&m0, 32);
    let n_verts = m0.vertices.len();

    // Step A: Slice-wise radial compression + liner thickness offset in reference domain M_0.
    let mut pct_field = vec![0.0f64; n_verts];
    for op in &recipe.operations {
        let kind = op.get("op").and_then(Value::as_str).unwrap_or("");
        if kind == "global_compression" || kind == "zoned_compression" {
            let (z_lo, z_hi) = op
                .get("z_range_mm")
                .and_then(Value::as_array)
                .filter(|a| a.len() >= 2)
                .map(|a| (a[0].as_f64().unwrap_or(-1e6), a[1].as_f64().unwrap_or(1e6)))
                .unwrap_or((-1e6, 1e6));
            let trans = op
                .get("transition_mm")
                .and_then(Value::as_f64)
                .unwrap_or(20.0);
            let pct = op.get("percent").and_then(Value::as_f64).unwrap_or(0.0);
            for (i, &v) in m0.vertices.iter().enumerate() {
                let w = zone_weight(v.z, z_lo, z_hi, trans);
                if w > 0.0 {
                    pct_field[i] += pct * w;
                }
            }
        }
    }

    let liner = recipe.liner_thickness_mm;
    let mut base_verts = Vec::with_capacity(n_verts);
    for (i, &v) in m0.vertices.iter().enumerate() {
        let (cx, cy) = interp_center_xy(v.z, zmin, span, &centers);
        let scale = 1.0 + pct_field[i] / 100.0;
        let mut bx = cx + (v.x - cx) * scale;
        let mut by = cy + (v.y - cy) * scale;
        let mut bz = v.z;
        if liner != 0.0 {
            let n = n0[i];
            bx -= liner * n.x;
            by -= liner * n.y;
            bz -= liner * n.z;
        }
        base_verts.push(Vec3::new(bx, by, bz));
    }
    let base_mesh = TriMesh::new(base_verts.clone(), m0.faces.clone());

    // Step B: Geodesic landmark reliefs & additive/reductive sculpt strokes along fixed N_0.
    let mut relief_mm = vec![0.0f64; n_verts];
    let mut geo_cache: HashMap<usize, Vec<f64>> = HashMap::new();

    for op in &recipe.operations {
        let kind = op.get("op").and_then(Value::as_str).unwrap_or("");
        if kind == "landmark_relief" {
            let lm_name = op.get("landmark").and_then(Value::as_str).unwrap_or("");
            let seed_idx = if let Some(&idx) = resolved_landmarks.get(lm_name) {
                idx
            } else if let Some(xyz) = recipe::parse_xyz(op.get("xyz")) {
                m0.nearest_vertex(xyz)
            } else {
                return Err(format!(
                    "Landmark {lm_name:?} not found in recipe landmarks"
                ));
            };
            let dists = geo_cache
                .entry(seed_idx)
                .or_insert_with(|| m0.geodesic_distances(&[seed_idx]));
            let offset = op.get("offset_mm").and_then(Value::as_f64).unwrap_or(0.0);
            let radius = op.get("radius_mm").and_then(Value::as_f64).unwrap_or(22.0);
            let falloff = op
                .get("falloff")
                .and_then(Value::as_str)
                .unwrap_or("gaussian");
            accumulate_geodesic_kernel(&mut relief_mm, dists, offset, radius, falloff);
        } else if kind == "sculpt_stroke" {
            let seeds = collect_op_seeds(&m0, op);
            if seeds.is_empty() {
                continue;
            }
            let dists = m0.geodesic_distances(&seeds);
            let mode = op.get("mode").and_then(Value::as_str).unwrap_or("");
            let raw_off = op.get("offset_mm").and_then(Value::as_f64).unwrap_or(3.0);
            let signed_off = if mode.eq_ignore_ascii_case("add") {
                raw_off.abs()
            } else if mode.eq_ignore_ascii_case("remove") {
                -raw_off.abs()
            } else {
                raw_off
            };
            let radius = op.get("radius_mm").and_then(Value::as_f64).unwrap_or(22.0);
            let falloff = op
                .get("falloff")
                .and_then(Value::as_str)
                .unwrap_or("gaussian");
            accumulate_geodesic_kernel(&mut relief_mm, &dists, signed_off, radius, falloff);
        }
    }

    let final_verts: Vec<Vec3> = (0..n_verts)
        .map(|i| math3d::v_add(base_verts[i], math3d::v_scale(n0[i], relief_mm[i])))
        .collect();
    let mut rectified = TriMesh::new(final_verts, m0.faces.clone());

    // Step C: Optional global or localized Taubin smoothing operations.
    for op in &recipe.operations {
        if op.get("op").and_then(Value::as_str) == Some("smooth") {
            let iters = op
                .get("iterations")
                .and_then(Value::as_u64)
                .map(|v| (v as usize).clamp(1, 20))
                .unwrap_or(4);
            let strength = op.get("strength").and_then(Value::as_f64).unwrap_or(0.5);
            let seeds = collect_op_seeds(&m0, op);
            if seeds.is_empty() {
                rectified = taubin_smooth_mesh(&rectified, iters, strength, None);
            } else {
                let dists = m0.geodesic_distances(&seeds);
                let rad = op
                    .get("radius_mm")
                    .and_then(Value::as_f64)
                    .unwrap_or(30.0)
                    .max(2.0);
                let weights: Vec<f64> = dists
                    .into_iter()
                    .map(|d| {
                        if d < rad {
                            1.0 - math3d::smoothstep5(d / rad)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                rectified = taubin_smooth_mesh(&rectified, iters, strength, Some(&weights));
            }
        }
    }

    Ok(RectificationResult {
        rectified_mesh: rectified,
        base_compressed_mesh: base_mesh,
        relief_field_mm: relief_mm,
        resolved_landmarks,
    })
}

fn collect_op_seeds(mesh: &TriMesh, op: &Value) -> Vec<usize> {
    let mut seeds = Vec::new();
    if let Some(arr) = op.get("points").and_then(Value::as_array) {
        for item in arr {
            if let Some(p) = recipe::parse_xyz(Some(item)) {
                seeds.push(mesh.nearest_vertex(p));
            }
        }
    } else if let Some(p) = recipe::parse_xyz(op.get("xyz")) {
        seeds.push(mesh.nearest_vertex(p));
    }
    seeds.sort_unstable();
    seeds.dedup();
    seeds
}
