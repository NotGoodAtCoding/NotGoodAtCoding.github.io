//! Chairside mobile scan QA, artifact triage, canonical alignment, and girth slicing.

use std::collections::HashMap;
use std::collections::HashSet;

use serde::Deserialize;
use serde::Serialize;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::SliceSummary;
use crate::mesh::TriMesh;

/// Structured chairside scan quality assurance report and 20 mm girth slices.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanQaReport {
    pub status: String,
    pub floater_components_removed: usize,
    pub proximal_cuff_loops: usize,
    pub internal_tear_loops: usize,
    pub double_wall_vertices: usize,
    pub seam_step_score_mm: f64,
    pub detected_scale_ratio: Option<f64>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub slices: Vec<SliceSummary>,
}

/// Removes detached floater components outside `distance_margin_mm` of the main mesh bbox.
pub fn prune_floaters(mesh: &TriMesh, distance_margin_mm: f64) -> (TriMesh, usize) {
    let comps = mesh.connected_components();
    if comps.len() <= 1 {
        return (mesh.clone(), 0);
    }

    let mut mx0 = f64::INFINITY;
    let mut mx1 = f64::NEG_INFINITY;
    let mut my0 = f64::INFINITY;
    let mut my1 = f64::NEG_INFINITY;
    let mut mz0 = f64::INFINITY;
    let mut mz1 = f64::NEG_INFINITY;
    for &idx in &comps[0] {
        let v = mesh.vertices[idx];
        mx0 = mx0.min(v.x);
        mx1 = mx1.max(v.x);
        my0 = my0.min(v.y);
        my1 = my1.max(v.y);
        mz0 = mz0.min(v.z);
        mz1 = mz1.max(v.z);
    }
    mx0 -= distance_margin_mm;
    mx1 += distance_margin_mm;
    my0 -= distance_margin_mm;
    my1 += distance_margin_mm;
    mz0 -= distance_margin_mm;
    mz1 += distance_margin_mm;

    let mut keep = comps[0].clone();
    let mut removed = 0usize;
    for comp in &comps[1..] {
        let inv = 1.0 / comp.len() as f64;
        let cx = comp.iter().map(|&i| mesh.vertices[i].x).sum::<f64>() * inv;
        let cy = comp.iter().map(|&i| mesh.vertices[i].y).sum::<f64>() * inv;
        let cz = comp.iter().map(|&i| mesh.vertices[i].z).sum::<f64>() * inv;
        let inside =
            (mx0..=mx1).contains(&cx) && (my0..=my1).contains(&cy) && (mz0..=mz1).contains(&cz);
        if inside && comp.len() >= 20 {
            keep.extend(comp.iter().copied());
        } else {
            removed += 1;
        }
    }
    (mesh.submesh_from_vertices(&keep), removed)
}

/// Classifies boundary loops into intended proximal open cuff(s) vs internal scan tears.
pub fn classify_boundary_loops(mesh: &TriMesh) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let loops = mesh.boundary_loops();
    if loops.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let (bmin, bmax) = mesh.bounds();
    let z_span = (bmax.z - bmin.z).max(1e-6);
    let top_threshold = bmin.z + 0.75 * z_span;

    let mut scored: Vec<(Vec<usize>, f64, f64)> = loops
        .into_iter()
        .map(|lp| {
            let mean_z = lp.iter().map(|&i| mesh.vertices[i].z).sum::<f64>() / lp.len() as f64;
            let perim: f64 = (0..lp.len())
                .map(|k| {
                    math3d::v_dist(mesh.vertices[lp[k]], mesh.vertices[lp[(k + 1) % lp.len()]])
                })
                .sum();
            (lp, mean_z, perim)
        })
        .collect();

    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut cuffs = Vec::new();
    let mut tears = Vec::new();
    let mut cuff_assigned = false;
    for (lp, mean_z, _) in scored {
        if !cuff_assigned && mean_z >= top_threshold {
            cuffs.push(lp);
            cuff_assigned = true;
        } else {
            tears.push(lp);
        }
    }
    (cuffs, tears)
}

/// Detects non-adjacent overlapping surface patches within `distance_threshold_mm`.
pub fn detect_double_walls(mesh: &TriMesh, distance_threshold_mm: f64) -> usize {
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

    let mut near_ring: Vec<HashSet<usize>> = Vec::with_capacity(n);
    for (i, nbs) in adj.iter().enumerate() {
        let mut ring = nbs.clone();
        ring.insert(i);
        for &n1 in nbs {
            ring.extend(adj[n1].iter().copied());
        }
        near_ring.push(ring);
    }

    let cell = (distance_threshold_mm * 1.5).max(2.0);
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(n);
    for (i, &v) in mesh.vertices.iter().enumerate() {
        let key = (
            (v.x / cell).floor() as i32,
            (v.y / cell).floor() as i32,
            (v.z / cell).floor() as i32,
        );
        grid.entry(key).or_default().push(i);
    }

    let d2_thresh = distance_threshold_mm * distance_threshold_mm;
    let mut flagged: HashSet<usize> = HashSet::new();
    for (i, &v) in mesh.vertices.iter().enumerate() {
        let gx = (v.x / cell).floor() as i32;
        let gy = (v.y / cell).floor() as i32;
        let gz = (v.z / cell).floor() as i32;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &j in bucket {
                            if j <= i || near_ring[i].contains(&j) {
                                continue;
                            }
                            let p = mesh.vertices[j];
                            let d2 =
                                (v.x - p.x).powi(2) + (v.y - p.y).powi(2) + (v.z - p.z).powi(2);
                            if d2 < d2_thresh {
                                flagged.insert(i);
                                flagged.insert(j);
                            }
                        }
                    }
                }
            }
        }
    }
    flagged.len()
}

/// Measures anomalous sharp dihedral-angle ridges indicating loop-closure seam drift.
pub fn detect_seam_step(mesh: &TriMesh, dihedral_threshold_deg: f64) -> f64 {
    let fn_vecs = mesh.face_normals();
    let zmin = mesh.bounds().0.z;
    let mut edges: HashMap<(usize, usize), usize> = HashMap::with_capacity(mesh.faces.len() * 2);
    let mut max_excess = 0.0f64;
    let mut ridge_count = 0usize;

    for (fi, &[a, b, c]) in mesh.faces.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            if let Some(&fj) = edges.get(&key) {
                if mesh.vertices[u].z.min(mesh.vertices[v].z) < zmin + 12.0 {
                    continue;
                }
                let dot_val = math3d::v_dot(fn_vecs[fi], fn_vecs[fj]).clamp(-1.0, 1.0);
                let ang = dot_val.acos().to_degrees();
                if ang > dihedral_threshold_deg {
                    ridge_count += 1;
                    max_excess = max_excess.max((ang - dihedral_threshold_deg) * 0.25);
                }
            } else {
                edges.insert(key, fi);
            }
        }
    }
    if ridge_count >= 3 {
        max_excess
    } else {
        0.0
    }
}

/// Aligns a limb to the canonical frame (`+Z` proximal, `+Y` anterior, `z = 0` at MPT).
pub fn align_canonical_limb(
    mesh: &TriMesh,
    mpt_point: Option<Vec3>,
    distal_point: Option<Vec3>,
) -> TriMesh {
    let n = mesh.vertices.len();
    if n == 0 {
        return mesh.clone();
    }
    let centroid = mesh.centroid();

    let mut cov = [[0.0f64; 3]; 3];
    for &v in &mesh.vertices {
        let d = math3d::v_sub(v, centroid);
        cov[0][0] += d.x * d.x;
        cov[0][1] += d.x * d.y;
        cov[0][2] += d.x * d.z;
        cov[1][1] += d.y * d.y;
        cov[1][2] += d.y * d.z;
        cov[2][2] += d.z * d.z;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    let (_, eigvecs) = math3d::jacobi_eigen_3x3(cov, 50);
    let mut z_axis = eigvecs[0];

    for _ in 0..2 {
        let projs: Vec<f64> = mesh
            .vertices
            .iter()
            .map(|&v| math3d::v_dot(math3d::v_sub(v, centroid), z_axis))
            .collect();
        let p_min = projs.iter().copied().fold(f64::INFINITY, f64::min);
        let p_max = projs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = (p_max - p_min).max(1e-6);

        let mut low_c = Vec3::default();
        let mut high_c = Vec3::default();
        let mut n_low = 0usize;
        let mut n_high = 0usize;
        for (&p, &s) in mesh.vertices.iter().zip(&projs) {
            let t = (s - p_min) / span;
            if (0.25..=0.45).contains(&t) {
                low_c = math3d::v_add(low_c, p);
                n_low += 1;
            } else if (0.70..=0.90).contains(&t) {
                high_c = math3d::v_add(high_c, p);
                n_high += 1;
            }
        }
        if n_low >= 10 && n_high >= 10 {
            let p_lo = math3d::v_scale(low_c, 1.0 / n_low as f64);
            let p_hi = math3d::v_scale(high_c, 1.0 / n_high as f64);
            let mut cand = math3d::v_unit(math3d::v_sub(p_hi, p_lo), z_axis);
            if math3d::v_dot(cand, z_axis) < 0.0 {
                cand = math3d::v_scale(cand, -1.0);
            }
            z_axis = cand;
        }
    }

    let origin = if let (Some(mpt), Some(distal)) = (mpt_point, distal_point) {
        if math3d::v_dot(math3d::v_sub(mpt, distal), z_axis) < 0.0 {
            z_axis = math3d::v_scale(z_axis, -1.0);
        }
        mpt
    } else {
        let mut projs: Vec<(f64, Vec3)> = mesh
            .vertices
            .iter()
            .map(|&v| (math3d::v_dot(math3d::v_sub(v, centroid), z_axis), v))
            .collect();
        projs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q_n = (n / 6).max(10).min(n);
        let radial_rms = |pts: &[(f64, Vec3)]| -> f64 {
            let inv = 1.0 / pts.len().max(1) as f64;
            let mut mc = Vec3::default();
            for &(_, p) in pts {
                mc = math3d::v_add(mc, p);
            }
            mc = math3d::v_scale(mc, inv);
            pts.iter()
                .map(|&(_, p)| {
                    let d = math3d::v_sub(p, mc);
                    math3d::v_dot(d, d)
                })
                .sum::<f64>()
                * inv
        };
        if radial_rms(&projs[..q_n]) > radial_rms(&projs[n - q_n..]) {
            z_axis = math3d::v_scale(z_axis, -1.0);
        }
        centroid
    };

    let y_axis = if let Some(mpt) = mpt_point {
        let d_mpt = math3d::v_sub(mpt, centroid);
        let proj_z = math3d::v_dot(d_mpt, z_axis);
        let y_cand = math3d::v_sub(d_mpt, math3d::v_scale(z_axis, proj_z));
        if math3d::v_norm(y_cand) > 1e-3 {
            math3d::v_unit(y_cand, Vec3::new(0.0, 1.0, 0.0))
        } else {
            let ref_x = if z_axis.x.abs() < 0.85 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            math3d::v_unit(math3d::v_cross(z_axis, ref_x), Vec3::new(0.0, 1.0, 0.0))
        }
    } else {
        let ref_x = if z_axis.x.abs() < 0.85 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        math3d::v_unit(math3d::v_cross(z_axis, ref_x), Vec3::new(0.0, 1.0, 0.0))
    };

    let x_axis = math3d::v_unit(math3d::v_cross(y_axis, z_axis), Vec3::new(1.0, 0.0, 0.0));

    let mut aligned = mesh.transformed(|p| {
        let d = math3d::v_sub(p, origin);
        Vec3::new(
            math3d::v_dot(d, x_axis),
            math3d::v_dot(d, y_axis),
            math3d::v_dot(d, z_axis),
        )
    });

    if mpt_point.is_none() {
        let (bmin, bmax) = aligned.bounds();
        let z_mpt_est = bmin.z + 0.82 * (bmax.z - bmin.z);
        aligned = aligned.transformed(|p| Vec3::new(p.x, p.y, p.z - z_mpt_est));
    }
    aligned
}

/// Measures taut-tape circumferences downward from an anchor plane in `step_mm` increments.
pub fn measure_circumferences(
    mesh: &TriMesh,
    anchor_z_mm: f64,
    step_mm: f64,
    max_depth_mm: f64,
) -> Vec<SliceSummary> {
    let zmin = mesh.bounds().0.z;
    let mut results = Vec::new();
    let mut depth = 0.0;
    while depth <= max_depth_mm {
        let z = anchor_z_mm - depth;
        if z <= zmin + 2.0 {
            break;
        }
        let mut sl = mesh.slice_z(z);
        if sl.points >= 6 {
            sl.depth_from_anchor_mm = depth;
            results.push(sl);
        }
        depth += step_mm;
    }
    results
}

/// Executes the full chairside Scan QA pipeline and returns `(clean_aligned_mesh, report)`.
pub fn run_scan_qa(
    raw_mesh: &TriMesh,
    mpt_point: Option<Vec3>,
    distal_point: Option<Vec3>,
) -> (TriMesh, ScanQaReport) {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    let (clean, removed_floaters) = prune_floaters(raw_mesh, 10.0);
    if removed_floaters > 0 {
        warnings.push(format!(
            "Pruned {removed_floaters} disconnected floater component(s)"
        ));
    }

    let aligned = align_canonical_limb(&clean, mpt_point, distal_point);
    let (cuffs, tears) = classify_boundary_loops(&aligned);
    if !tears.is_empty() {
        errors.push(format!(
            "Detected {} internal surface tear hole(s)",
            tears.len()
        ));
    }

    let double_wall_verts = detect_double_walls(&aligned, 1.4);
    if double_wall_verts > 15 {
        errors.push(format!(
            "Detected SLAM double-walled overlap ({double_wall_verts} vertices)"
        ));
    }

    let seam_score = detect_seam_step(&aligned, 42.0);
    if seam_score > 1.0 {
        errors.push(format!(
            "Detected potential loop-closure seam step ({seam_score:.2} mm excess)"
        ));
    }

    let slices = measure_circumferences(&aligned, 0.0, 20.0, 160.0);
    let status = if !errors.is_empty() {
        "FAIL"
    } else if !warnings.is_empty() {
        "WARN"
    } else {
        "PASS"
    };

    (
        aligned,
        ScanQaReport {
            status: status.to_string(),
            floater_components_removed: removed_floaters,
            proximal_cuff_loops: cuffs.len(),
            internal_tear_loops: tears.len(),
            double_wall_vertices: double_wall_verts,
            seam_step_score_mm: seam_score,
            detected_scale_ratio: None,
            warnings,
            errors,
            slices,
        },
    )
}
