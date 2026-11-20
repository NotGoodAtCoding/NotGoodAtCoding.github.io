//! Mesh repair utilities: vertex welding, consistent winding, hole filling, and scalar clipping.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use serde::Deserialize;
use serde::Serialize;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::Tri;
use crate::mesh::TriMesh;

/// Summary of automated mesh cleanup and topological repairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairReport {
    pub welded_vertices_removed: usize,
    pub degenerate_faces_removed: usize,
    pub flipped_faces: usize,
    pub holes_filled: usize,
    pub clipped_proximal_cuff: bool,
    pub warnings: Vec<String>,
}

/// Welds near-coincident vertices and removes degenerate or duplicate triangles.
pub fn weld_and_clean(mesh: &TriMesh, tol: f64) -> (TriMesh, usize, usize) {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return (mesh.clone(), 0, 0);
    }

    let inv = 1.0 / tol.max(1e-9);
    let mut vmap: HashMap<(i64, i64, i64), usize> = HashMap::with_capacity(mesh.vertices.len());
    let mut new_verts: Vec<Vec3> = Vec::with_capacity(mesh.vertices.len());
    let mut old_to_new: Vec<usize> = Vec::with_capacity(mesh.vertices.len());

    for &v in &mesh.vertices {
        let key = (
            (v.x * inv).round() as i64,
            (v.y * inv).round() as i64,
            (v.z * inv).round() as i64,
        );
        let next_id = new_verts.len();
        let mapped = *vmap.entry(key).or_insert_with(|| {
            new_verts.push(v);
            next_id
        });
        old_to_new.push(mapped);
    }

    let welded_removed = mesh.vertices.len().saturating_sub(new_verts.len());

    let mut seen_tris: HashSet<[usize; 3]> = HashSet::with_capacity(mesh.faces.len());
    let mut clean_faces: Vec<Tri> = Vec::with_capacity(mesh.faces.len());
    let mut degen_removed = 0usize;

    for &[a, b, c] in &mesh.faces {
        let na = old_to_new[a];
        let nb = old_to_new[b];
        let nc = old_to_new[c];
        if na == nb || nb == nc || nc == na {
            degen_removed += 1;
            continue;
        }
        let area2 = math3d::v_norm(math3d::v_cross(
            math3d::v_sub(new_verts[nb], new_verts[na]),
            math3d::v_sub(new_verts[nc], new_verts[na]),
        ));
        if area2 < 1e-10 {
            degen_removed += 1;
            continue;
        }
        let mut sorted_key = [na, nb, nc];
        sorted_key.sort_unstable();
        if !seen_tris.insert(sorted_key) {
            degen_removed += 1;
            continue;
        }
        clean_faces.push([na, nb, nc]);
    }

    let mut used: Vec<usize> = clean_faces.iter().flat_map(|f| f.iter().copied()).collect();
    used.sort_unstable();
    used.dedup();

    if used.len() < new_verts.len() {
        let mut compact = vec![0usize; new_verts.len()];
        let mut final_verts = Vec::with_capacity(used.len());
        for (new_idx, &old_idx) in used.iter().enumerate() {
            compact[old_idx] = new_idx;
            final_verts.push(new_verts[old_idx]);
        }
        let final_faces = clean_faces
            .into_iter()
            .map(|[a, b, c]| [compact[a], compact[b], compact[c]])
            .collect();
        return (
            TriMesh::new(final_verts, final_faces),
            welded_removed,
            degen_removed,
        );
    }

    (
        TriMesh::new(new_verts, clean_faces),
        welded_removed,
        degen_removed,
    )
}

/// Enforces consistent outward face winding across every connected component via BFS.
pub fn fix_consistent_winding(mesh: &TriMesh) -> (TriMesh, usize) {
    if mesh.faces.is_empty() {
        return (mesh.clone(), 0);
    }

    let mut faces = mesh.faces.clone();
    let verts = &mesh.vertices;
    let n_faces = faces.len();

    let mut edge_to_faces: HashMap<(usize, usize), Vec<usize>> =
        HashMap::with_capacity(n_faces * 2);
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_to_faces.entry(key).or_default().push(fi);
        }
    }

    let mut visited = vec![false; n_faces];
    let mut flipped_total = 0usize;

    for start in 0..n_faces {
        if visited[start] {
            continue;
        }
        let mut comp_faces = Vec::new();
        visited[start] = true;
        let mut q = VecDeque::new();
        q.push_back(start);

        while let Some(fi) = q.pop_front() {
            comp_faces.push(fi);
            let [a, b, c] = faces[fi];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                if let Some(nbs) = edge_to_faces.get(&key) {
                    for &nb in nbs {
                        if visited[nb] {
                            continue;
                        }
                        let [na, nb_v, nc] = faces[nb];
                        if (na == u && nb_v == v) || (nb_v == u && nc == v) || (nc == u && na == v)
                        {
                            faces[nb] = [na, nc, nb_v];
                            flipped_total += 1;
                        }
                        visited[nb] = true;
                        q.push_back(nb);
                    }
                }
            }
        }

        let mut comp_undir: HashMap<(usize, usize), usize> = HashMap::new();
        for &fi in &comp_faces {
            let [a, b, c] = faces[fi];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *comp_undir.entry(key).or_insert(0) += 1;
            }
        }

        let has_boundary = comp_undir.values().any(|&cnt| cnt == 1);
        let flip_comp = if !has_boundary {
            let mut vol = 0.0;
            for &fi in &comp_faces {
                let [a, b, c] = faces[fi];
                vol += math3d::v_dot(verts[a], math3d::v_cross(verts[b], verts[c]));
            }
            vol < 0.0
        } else {
            let mut comp_v_ids: HashSet<usize> = HashSet::new();
            for &fi in &comp_faces {
                for idx in faces[fi] {
                    comp_v_ids.insert(idx);
                }
            }
            let inv = 1.0 / comp_v_ids.len().max(1) as f64;
            let mut center = Vec3::default();
            for &idx in &comp_v_ids {
                center = math3d::v_add(center, verts[idx]);
            }
            center = math3d::v_scale(center, inv);

            let mut score = 0.0;
            for &fi in &comp_faces {
                let [a, b, c] = faces[fi];
                let fc = math3d::v_scale(
                    math3d::v_add(verts[a], math3d::v_add(verts[b], verts[c])),
                    1.0 / 3.0,
                );
                let fn_vec = math3d::v_cross(
                    math3d::v_sub(verts[b], verts[a]),
                    math3d::v_sub(verts[c], verts[a]),
                );
                score += math3d::v_dot(math3d::v_sub(fc, center), fn_vec);
            }
            score < 0.0
        };

        if flip_comp {
            for &fi in &comp_faces {
                let [a, b, c] = faces[fi];
                faces[fi] = [a, c, b];
            }
            flipped_total += comp_faces.len();
        }
    }

    (TriMesh::new(verts.clone(), faces), flipped_total)
}

/// Fills internal tear holes while optionally keeping the primary proximal cuff open.
pub fn fill_small_holes(
    mesh: &TriMesh,
    max_loop_edges: usize,
    keep_proximal_cuff: bool,
) -> (TriMesh, usize) {
    let loops = mesh.boundary_loops();
    if loops.is_empty() {
        return (mesh.clone(), 0);
    }

    let (bmin, bmax) = mesh.bounds();
    let zspan = (bmax.z - bmin.z).max(1e-6);
    let top_thresh = bmin.z + 0.72 * zspan;

    let mut cuff_idx: Option<usize> = None;
    if keep_proximal_cuff {
        let mut best_perim = -1.0;
        for (idx, lp) in loops.iter().enumerate() {
            let mean_z = lp.iter().map(|&v| mesh.vertices[v].z).sum::<f64>() / lp.len() as f64;
            if mean_z >= top_thresh {
                let perim: f64 = (0..lp.len())
                    .map(|k| {
                        math3d::v_dist(mesh.vertices[lp[k]], mesh.vertices[lp[(k + 1) % lp.len()]])
                    })
                    .sum();
                if perim > best_perim {
                    best_perim = perim;
                    cuff_idx = Some(idx);
                }
            }
        }
    }

    let mut half_edges: HashSet<(usize, usize)> = HashSet::with_capacity(mesh.faces.len() * 3);
    for &[a, b, c] in &mesh.faces {
        half_edges.insert((a, b));
        half_edges.insert((b, c));
        half_edges.insert((c, a));
    }

    let mut verts = mesh.vertices.clone();
    let mut faces = mesh.faces.clone();
    let mut filled = 0usize;

    for (idx, lp) in loops.iter().enumerate() {
        if Some(idx) == cuff_idx || lp.len() > max_loop_edges || lp.len() < 3 {
            continue;
        }
        let mut ordered = lp.clone();
        if !half_edges.contains(&(ordered[0], ordered[1])) {
            ordered.reverse();
        }
        let inv = 1.0 / ordered.len() as f64;
        let center = Vec3::new(
            ordered.iter().map(|&v| verts[v].x).sum::<f64>() * inv,
            ordered.iter().map(|&v| verts[v].y).sum::<f64>() * inv,
            ordered.iter().map(|&v| verts[v].z).sum::<f64>() * inv,
        );
        let c_idx = verts.len();
        verts.push(center);

        let m = ordered.len();
        for k in 0..m {
            let u = ordered[k];
            let v = ordered[(k + 1) % m];
            faces.push([v, u, c_idx]);
            half_edges.insert((v, u));
        }
        filled += 1;
    }

    (TriMesh::new(verts, faces), filled)
}

/// Clips a mesh cleanly against an arbitrary per-vertex scalar iso-surface with zero T-junctions.
pub fn clip_mesh_scalar(
    mesh: &TriMesh,
    values: &[f64],
    threshold: f64,
    keep_above: bool,
) -> Result<TriMesh, String> {
    if values.len() != mesh.vertices.len() {
        return Err("values length must match mesh.vertices length".to_string());
    }

    let mut verts = mesh.vertices.clone();
    let signed: Vec<f64> = values
        .iter()
        .map(|&v| {
            let d = v - threshold;
            if d.abs() >= 1e-7 {
                d
            } else if keep_above {
                1e-7
            } else {
                -1e-7
            }
        })
        .collect();

    let is_kept = |idx: usize| -> bool {
        if keep_above {
            signed[idx] >= 0.0
        } else {
            signed[idx] <= 0.0
        }
    };

    let mut new_faces: Vec<Tri> = Vec::with_capacity(mesh.faces.len());
    let mut edge_cache: HashMap<(usize, usize), usize> = HashMap::new();

    let mut cut_vertex = |u: usize, v: usize, verts_buf: &mut Vec<Vec3>| -> usize {
        let key = if u < v { (u, v) } else { (v, u) };
        if let Some(&existing) = edge_cache.get(&key) {
            return existing;
        }
        let pu = verts_buf[u];
        let pv = verts_buf[v];
        let su = signed[u];
        let sv = signed[v];
        let denom = sv - su;
        let t = if denom.abs() < 1e-12 {
            0.5
        } else {
            (-su / denom).clamp(1e-5, 1.0 - 1e-5)
        };
        let pt = Vec3::new(
            pu.x + t * (pv.x - pu.x),
            pu.y + t * (pv.y - pu.y),
            pu.z + t * (pv.z - pu.z),
        );
        let idx = verts_buf.len();
        verts_buf.push(pt);
        edge_cache.insert(key, idx);
        idx
    };

    for &[a, b, c] in &mesh.faces {
        let ka = is_kept(a);
        let kb = is_kept(b);
        let kc = is_kept(c);
        let n_kept = usize::from(ka) + usize::from(kb) + usize::from(kc);
        if n_kept == 0 {
            continue;
        }
        if n_kept == 3 {
            new_faces.push([a, b, c]);
            continue;
        }

        let mut tri = [a, b, c];
        let mut flags = [ka, kb, kc];
        while !(flags[0] && (n_kept == 1 || flags[1])) {
            tri.rotate_left(1);
            flags.rotate_left(1);
        }

        let [v0, v1, v2] = tri;
        if n_kept == 1 {
            let p01 = cut_vertex(v0, v1, &mut verts);
            let p02 = cut_vertex(v0, v2, &mut verts);
            new_faces.push([v0, p01, p02]);
        } else {
            let p12 = cut_vertex(v1, v2, &mut verts);
            let p20 = cut_vertex(v2, v0, &mut verts);
            new_faces.push([v0, v1, p12]);
            new_faces.push([v0, p12, p20]);
        }
    }

    let mut used: Vec<usize> = new_faces.iter().flat_map(|f| f.iter().copied()).collect();
    used.sort_unstable();
    used.dedup();
    let mut remap = vec![0usize; verts.len()];
    let mut final_verts = Vec::with_capacity(used.len());
    for (new_idx, &old_idx) in used.iter().enumerate() {
        remap[old_idx] = new_idx;
        final_verts.push(verts[old_idx]);
    }
    let final_faces = new_faces
        .into_iter()
        .map(|[a, b, c]| [remap[a], remap[b], remap[c]])
        .collect();
    Ok(TriMesh::new(final_verts, final_faces))
}

/// Clips a mesh cleanly against horizontal plane `z = z_cut`.
pub fn clip_mesh_z(mesh: &TriMesh, z_cut: f64, keep_above: bool) -> TriMesh {
    let vals: Vec<f64> = mesh.vertices.iter().map(|v| v.z).collect();
    clip_mesh_scalar(mesh, &vals, z_cut, keep_above)
        .expect("z-values length always matches mesh.vertices length")
}

/// Runs full scan repair: weld/clean -> fix winding -> fill internal tears -> optional cuff crop.
pub fn repair_scan_mesh(
    mesh: &TriMesh,
    weld_tol: f64,
    fill_holes: bool,
    ensure_open_cuff: bool,
    cuff_crop_fraction: f64,
) -> (TriMesh, RepairReport) {
    let (cleaned, welded_rm, degen_rm) = weld_and_clean(mesh, weld_tol);
    let (mut wound_ok, flipped) = fix_consistent_winding(&cleaned);
    let mut holes_filled = 0usize;
    if fill_holes {
        let (filled_mesh, cnt) = fill_small_holes(&wound_ok, 80, true);
        wound_ok = filled_mesh;
        holes_filled = cnt;
    }

    let mut clipped_cuff = false;
    if ensure_open_cuff && wound_ok.boundary_loops().is_empty() && !wound_ok.vertices.is_empty() {
        let (bmin, bmax) = wound_ok.bounds();
        let z_cut = bmax.z - (cuff_crop_fraction * (bmax.z - bmin.z)).max(2.0);
        wound_ok = clip_mesh_z(&wound_ok, z_cut, false);
        clipped_cuff = true;
    }

    let mut warnings = Vec::new();
    if welded_rm > 0 {
        warnings.push(format!("Welded {welded_rm} duplicate/coincident vertices"));
    }
    if flipped > 0 {
        warnings.push(format!("Corrected winding on {flipped} faces"));
    }
    if holes_filled > 0 {
        warnings.push(format!("Filled {holes_filled} internal scan tear(s)"));
    }

    (
        wound_ok,
        RepairReport {
            welded_vertices_removed: welded_rm,
            degenerate_faces_removed: degen_rm,
            flipped_faces: flipped,
            holes_filled,
            clipped_proximal_cuff: clipped_cuff,
            warnings,
        },
    )
}
