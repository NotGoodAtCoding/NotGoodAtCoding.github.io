//! Core triangle mesh representation, binary/ASCII STL I/O, topology, geodesics, and slicing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use serde::Deserialize;
use serde::Serialize;

use crate::math3d;
use crate::math3d::Vec2;
use crate::math3d::Vec3;

/// Indexed triangle face `(a, b, c)`.
pub type Tri = [usize; 3];

/// Edge topology summary for manifoldness and watertightness verification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeTopology {
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub winding_mismatches: usize,
    pub degenerate_faces: usize,
}

/// Cross-sectional Z-slice taut-tape measurement.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SliceSummary {
    pub z_mm: f64,
    pub taut_circumference_mm: f64,
    pub hull_area_mm2: f64,
    pub points: usize,
    pub depth_from_anchor_mm: f64,
}

/// Lightweight indexed triangle mesh with pure-Rust geometric and topological queries.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Tri>,
}

#[derive(Debug, Clone, Copy)]
struct HeapItem {
    dist: f64,
    idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.dist.to_bits() == other.dist.to_bits()
    }
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TriMesh {
    /// Creates a new mesh from vertices and indexed triangle faces.
    pub fn new(vertices: Vec<Vec3>, faces: Vec<Tri>) -> Self {
        Self { vertices, faces }
    }

    /// Returns axis-aligned `(min, max)` bounds.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        if self.vertices.is_empty() {
            return (Vec3::default(), Vec3::default());
        }
        let mut min_v = self.vertices[0];
        let mut max_v = self.vertices[0];
        for &v in &self.vertices[1..] {
            min_v.x = min_v.x.min(v.x);
            min_v.y = min_v.y.min(v.y);
            min_v.z = min_v.z.min(v.z);
            max_v.x = max_v.x.max(v.x);
            max_v.y = max_v.y.max(v.y);
            max_v.z = max_v.z.max(v.z);
        }
        (min_v, max_v)
    }

    /// Returns the arithmetic vertex centroid.
    pub fn centroid(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::default();
        }
        let inv = 1.0 / self.vertices.len() as f64;
        let mut sum = Vec3::default();
        for &v in &self.vertices {
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
        }
        math3d::v_scale(sum, inv)
    }

    /// Computes unit face normals.
    pub fn face_normals(&self) -> Vec<Vec3> {
        let fallback = Vec3::new(0.0, 0.0, 1.0);
        self.faces
            .iter()
            .map(|&[a, b, c]| {
                let n = math3d::v_cross(
                    math3d::v_sub(self.vertices[b], self.vertices[a]),
                    math3d::v_sub(self.vertices[c], self.vertices[a]),
                );
                math3d::v_unit(n, fallback)
            })
            .collect()
    }

    /// Computes area-weighted unit vertex normals.
    pub fn vertex_normals(&self) -> Vec<Vec3> {
        let mut acc = vec![Vec3::default(); self.vertices.len()];
        for &[a, b, c] in &self.faces {
            let fn_vec = math3d::v_cross(
                math3d::v_sub(self.vertices[b], self.vertices[a]),
                math3d::v_sub(self.vertices[c], self.vertices[a]),
            );
            for idx in [a, b, c] {
                acc[idx] = math3d::v_add(acc[idx], fn_vec);
            }
        }
        let fallback = Vec3::new(0.0, 0.0, 1.0);
        acc.into_iter()
            .map(|n| math3d::v_unit(n, fallback))
            .collect()
    }

    /// Computes signed mesh volume (positive for outward-wound closed solids).
    pub fn signed_volume(&self) -> f64 {
        let mut total = 0.0;
        for &[a, b, c] in &self.faces {
            total += math3d::v_dot(
                self.vertices[a],
                math3d::v_cross(self.vertices[b], self.vertices[c]),
            );
        }
        total / 6.0
    }

    /// Audits half-edge consistency, boundaries, non-manifold edges, and degenerate triangles.
    pub fn edge_topology(&self) -> EdgeTopology {
        let mut half_edges: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(self.faces.len() * 3);
        let mut undirected: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(self.faces.len() * 2);
        let mut degenerate = 0;

        for &[a, b, c] in &self.faces {
            if a == b || b == c || c == a {
                degenerate += 1;
            } else {
                let area2 = math3d::v_norm(math3d::v_cross(
                    math3d::v_sub(self.vertices[b], self.vertices[a]),
                    math3d::v_sub(self.vertices[c], self.vertices[a]),
                ));
                if area2 < 1e-12 {
                    degenerate += 1;
                }
            }
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *half_edges.entry((u, v)).or_insert(0) += 1;
                let key = if u < v { (u, v) } else { (v, u) };
                *undirected.entry(key).or_insert(0) += 1;
            }
        }

        let boundary_edges = half_edges
            .iter()
            .filter(|&(&(u, v), &cnt)| cnt == 1 && !half_edges.contains_key(&(v, u)))
            .count();
        let non_manifold_edges = undirected.values().filter(|&&cnt| cnt > 2).count();
        let winding_mismatches = half_edges.values().filter(|&&cnt| cnt > 1).count();

        EdgeTopology {
            boundary_edges,
            non_manifold_edges,
            winding_mismatches,
            degenerate_faces: degenerate,
        }
    }

    /// Returns `true` if the mesh is a closed, outward/consistently wound 2-manifold.
    pub fn is_watertight_manifold(&self) -> bool {
        if self.faces.is_empty() {
            return false;
        }
        let topo = self.edge_topology();
        topo.boundary_edges == 0
            && topo.non_manifold_edges == 0
            && topo.winding_mismatches == 0
            && topo.degenerate_faces == 0
    }

    /// Extracts ordered closed vertex loops along mesh boundaries.
    pub fn boundary_loops(&self) -> Vec<Vec<usize>> {
        let mut half_edges: HashSet<(usize, usize)> = HashSet::with_capacity(self.faces.len() * 3);
        for &[a, b, c] in &self.faces {
            half_edges.insert((a, b));
            half_edges.insert((b, c));
            half_edges.insert((c, a));
        }

        let mut next_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(u, v) in &half_edges {
            if !half_edges.contains(&(v, u)) {
                next_map.entry(u).or_default().push(v);
            }
        }
        // Sort deterministically for reproducible loop extraction.
        let mut starts: Vec<usize> = next_map.keys().copied().collect();
        starts.sort_unstable();
        for targets in next_map.values_mut() {
            targets.sort_unstable();
        }

        let mut visited_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut loops: Vec<Vec<usize>> = Vec::new();
        for start in starts {
            let targets = next_map.get(&start).cloned().unwrap_or_default();
            for nxt in targets {
                if visited_edges.contains(&(start, nxt)) {
                    continue;
                }
                let mut lp = vec![start];
                let mut u = start;
                let mut v = nxt;
                while !visited_edges.contains(&(u, v)) {
                    visited_edges.insert((u, v));
                    if v == start {
                        break;
                    }
                    lp.push(v);
                    let next_cand = next_map.get(&v).and_then(|cands| {
                        cands
                            .iter()
                            .copied()
                            .find(|&w| !visited_edges.contains(&(v, w)))
                    });
                    match next_cand {
                        Some(w) => {
                            u = v;
                            v = w;
                        }
                        None => break,
                    }
                }
                if lp.len() >= 3 {
                    loops.push(lp);
                }
            }
        }
        loops
    }

    /// Returns face-connected vertex components sorted largest-first.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];
        for &[a, b, c] in &self.faces {
            adj[a].push(b);
            adj[a].push(c);
            adj[b].push(a);
            adj[b].push(c);
            adj[c].push(a);
            adj[c].push(b);
        }

        let mut seen = vec![false; self.vertices.len()];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for i in 0..self.vertices.len() {
            if seen[i] || adj[i].is_empty() {
                continue;
            }
            let mut q = VecDeque::new();
            q.push_back(i);
            seen[i] = true;
            let mut comp = Vec::new();
            while let Some(u) = q.pop_front() {
                comp.push(u);
                for &w in &adj[u] {
                    if !seen[w] {
                        seen[w] = true;
                        q.push_back(w);
                    }
                }
            }
            comps.push(comp);
        }
        comps.sort_by_key(|c| std::cmp::Reverse(c.len()));
        comps
    }

    /// Extracts a compacted submesh keeping triangles whose vertices are all in `keep_vertices`.
    pub fn submesh_from_vertices(&self, keep_vertices: &[usize]) -> Self {
        let keep_set: HashSet<usize> = keep_vertices.iter().copied().collect();
        let mut remap: HashMap<usize, usize> = HashMap::with_capacity(keep_set.len());
        let mut new_verts = Vec::with_capacity(keep_set.len());
        let mut new_faces = Vec::new();

        for &[a, b, c] in &self.faces {
            if keep_set.contains(&a) && keep_set.contains(&b) && keep_set.contains(&c) {
                let mut tri = [0usize; 3];
                for (slot, old_idx) in [a, b, c].into_iter().enumerate() {
                    let next_id = new_verts.len();
                    let mapped = *remap.entry(old_idx).or_insert_with(|| {
                        new_verts.push(self.vertices[old_idx]);
                        next_id
                    });
                    tri[slot] = mapped;
                }
                new_faces.push(tri);
            }
        }
        Self::new(new_verts, new_faces)
    }

    /// Computes multi-source Dijkstra geodesic edge distances on this mesh.
    pub fn geodesic_distances(&self, seeds: &[usize]) -> Vec<f64> {
        let n = self.vertices.len();
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut seen_edge: HashSet<(usize, usize)> = HashSet::with_capacity(self.faces.len() * 2);
        for &[a, b, c] in &self.faces {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                if seen_edge.insert(key) {
                    let w = math3d::v_dist(self.vertices[u], self.vertices[v]);
                    adj[u].push((v, w));
                    adj[v].push((u, w));
                }
            }
        }

        let mut dist = vec![f64::INFINITY; n];
        let mut pq = BinaryHeap::new();
        for &s in seeds {
            if s < n && dist[s] > 0.0 {
                dist[s] = 0.0;
                pq.push(HeapItem { dist: 0.0, idx: s });
            }
        }

        while let Some(HeapItem { dist: d, idx: u }) = pq.pop() {
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &adj[u] {
                let nd = d + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    pq.push(HeapItem { dist: nd, idx: v });
                }
            }
        }
        dist
    }

    /// Finds the vertex index closest to `point`.
    pub fn nearest_vertex(&self, point: Vec3) -> usize {
        let mut best_i = 0;
        let mut best_d2 = f64::INFINITY;
        for (i, &v) in self.vertices.iter().enumerate() {
            let dx = v.x - point.x;
            let dy = v.y - point.y;
            let dz = v.z - point.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_i = i;
            }
        }
        best_i
    }

    /// Slices the mesh with plane `z = z_height` and computes taut-tape hull perimeter and area.
    pub fn slice_z(&self, z_height: f64) -> SliceSummary {
        let mut pts: Vec<Vec2> = Vec::new();
        for &[a, b, c] in &self.faces {
            let va = self.vertices[a];
            let vb = self.vertices[b];
            let vc = self.vertices[c];
            let zmin = va.z.min(vb.z).min(vc.z);
            let zmax = va.z.max(vb.z).max(vc.z);
            if zmax < z_height || zmin > z_height {
                continue;
            }
            for (p0, p1) in [(va, vb), (vb, vc), (vc, va)] {
                let dz0 = p0.z - z_height;
                let dz1 = p1.z - z_height;
                if (dz0 <= 0.0 && 0.0 <= dz1) || (dz1 <= 0.0 && 0.0 <= dz0) {
                    let denom = p1.z - p0.z;
                    if denom.abs() < 1e-12 {
                        continue;
                    }
                    let t = (z_height - p0.z) / denom;
                    if (0.0..=1.0).contains(&t) {
                        pts.push(Vec2::new(
                            p0.x + t * (p1.x - p0.x),
                            p0.y + t * (p1.y - p0.y),
                        ));
                    }
                }
            }
        }
        if pts.len() < 3 {
            return SliceSummary {
                z_mm: z_height,
                taut_circumference_mm: 0.0,
                hull_area_mm2: 0.0,
                points: 0,
                depth_from_anchor_mm: 0.0,
            };
        }
        let hull = math3d::convex_hull_2d(&pts);
        SliceSummary {
            z_mm: z_height,
            taut_circumference_mm: math3d::polygon_perimeter_2d(&hull, true),
            hull_area_mm2: math3d::polygon_area_2d(&hull),
            points: pts.len(),
            depth_from_anchor_mm: 0.0,
        }
    }

    /// Returns a copy of this mesh with `f` applied to every vertex.
    pub fn transformed<F>(&self, f: F) -> Self
    where
        F: Fn(Vec3) -> Vec3,
    {
        Self::new(
            self.vertices.iter().copied().map(f).collect(),
            self.faces.clone(),
        )
    }
}

/// Serializes a `TriMesh` into binary STL bytes.
pub fn write_binary_stl_bytes(mesh: &TriMesh, header_text: &str) -> Vec<u8> {
    let tri_count = mesh.faces.len();
    let mut out = Vec::with_capacity(84 + tri_count * 50);
    let mut header = [b' '; 80];
    let src = header_text.as_bytes();
    let copy_len = src.len().min(80);
    header[..copy_len].copy_from_slice(&src[..copy_len]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(tri_count as u32).to_le_bytes());

    let fallback = Vec3::new(0.0, 0.0, 1.0);
    for &[a, b, c] in &mesh.faces {
        let va = mesh.vertices[a];
        let vb = mesh.vertices[b];
        let vc = mesh.vertices[c];
        let fn_vec = math3d::v_unit(
            math3d::v_cross(math3d::v_sub(vb, va), math3d::v_sub(vc, va)),
            fallback,
        );
        for val in [
            fn_vec.x as f32,
            fn_vec.y as f32,
            fn_vec.z as f32,
            va.x as f32,
            va.y as f32,
            va.z as f32,
            vb.x as f32,
            vb.y as f32,
            vb.z as f32,
            vc.x as f32,
            vc.y as f32,
            vc.z as f32,
        ] {
            out.extend_from_slice(&val.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// Parses binary or ASCII STL bytes and merges vertices within `merge_tol`.
pub fn read_stl_bytes(data: &[u8], merge_tol: f64) -> Result<TriMesh, String> {
    if data.len() < 84 {
        return Err("Invalid STL file: too short".to_string());
    }
    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected_bin = 84usize.saturating_add(tri_count.saturating_mul(50));
    let mut raw_tris: Vec<[Vec3; 3]> = Vec::new();

    if data.len() == expected_bin && tri_count > 0 {
        raw_tris.reserve(tri_count);
        let read_f32 = |off: usize| -> f64 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as f64
        };
        let mut off = 84;
        for _ in 0..tri_count {
            let v0 = Vec3::new(read_f32(off + 12), read_f32(off + 16), read_f32(off + 20));
            let v1 = Vec3::new(read_f32(off + 24), read_f32(off + 28), read_f32(off + 32));
            let v2 = Vec3::new(read_f32(off + 36), read_f32(off + 40), read_f32(off + 44));
            raw_tris.push([v0, v1, v2]);
            off += 50;
        }
    } else {
        let text = String::from_utf8_lossy(data);
        let mut buf: Vec<Vec3> = Vec::with_capacity(3);
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            if let (Some(kw), Some(xs), Some(ys), Some(zs), None) = (
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
            ) {
                if kw.eq_ignore_ascii_case("vertex") {
                    if let (Ok(x), Ok(y), Ok(z)) =
                        (xs.parse::<f64>(), ys.parse::<f64>(), zs.parse::<f64>())
                    {
                        buf.push(Vec3::new(x, y, z));
                        if buf.len() == 3 {
                            raw_tris.push([buf[0], buf[1], buf[2]]);
                            buf.clear();
                        }
                    }
                }
            }
        }
    }

    if raw_tris.is_empty() {
        return Err("STL contains no readable triangles".to_string());
    }

    let inv = 1.0 / merge_tol.max(1e-9);
    let mut vmap: HashMap<(i64, i64, i64), usize> = HashMap::with_capacity(raw_tris.len());
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Tri> = Vec::with_capacity(raw_tris.len());

    for tri in raw_tris {
        let mut idxs = [0usize; 3];
        for (slot, p) in tri.into_iter().enumerate() {
            let key = (
                (p.x * inv).round() as i64,
                (p.y * inv).round() as i64,
                (p.z * inv).round() as i64,
            );
            let next_id = vertices.len();
            let mapped = *vmap.entry(key).or_insert_with(|| {
                vertices.push(p);
                next_id
            });
            idxs[slot] = mapped;
        }
        if idxs[0] != idxs[1] && idxs[1] != idxs[2] && idxs[2] != idxs[0] {
            faces.push(idxs);
        }
    }

    Ok(TriMesh::new(vertices, faces))
}
