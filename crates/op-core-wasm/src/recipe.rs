//! Declarative `.oprecipe.json` v0.1 contract, validation, hashing, and revision diffing.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use crate::math3d::Vec3;
use crate::mesh::TriMesh;

pub const SCHEMA_VERSION: &str = "0.1";
pub const KERNEL_VERSION: &str = "0.1.0-wasm";

/// Portable, deterministic O&P modification recipe (`.oprecipe.json` v0.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpRecipe {
    pub device_type: String,
    #[serde(default = "default_side")]
    pub side: String,
    #[serde(default = "default_units")]
    pub units: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default = "default_kernel_version")]
    pub kernel_version: String,
    #[serde(default)]
    pub scan_sha256: String,
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,
    #[serde(default)]
    pub liner_thickness_mm: f64,
    #[serde(default = "default_alignment")]
    pub alignment: Map<String, Value>,
    #[serde(default)]
    pub landmarks: BTreeMap<String, Value>,
    #[serde(default)]
    pub operations: Vec<Value>,
    #[serde(default = "default_shell")]
    pub shell: Map<String, Value>,
}

fn default_side() -> String {
    "left".to_string()
}

fn default_units() -> String {
    "mm".to_string()
}

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_string()
}

fn default_kernel_version() -> String {
    KERNEL_VERSION.to_string()
}

fn default_scale_factor() -> f64 {
    1.0
}

fn default_alignment() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("flexion_deg".to_string(), Value::from(5.0));
    m.insert("adduction_deg".to_string(), Value::from(5.0));
    m.insert("mpt_offset_mm".to_string(), Value::from(vec![0.0, 0.0]));
    m
}

fn default_shell() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("wall_distal_mm".to_string(), Value::from(4.5));
    m.insert("wall_proximal_mm".to_string(), Value::from(2.5));
    m.insert("trimline_fillet_mm".to_string(), Value::from(10.0));
    m.insert(
        "hardware".to_string(),
        Value::from("pyramid_50_8_m6_captive"),
    );
    m
}

impl OpRecipe {
    /// Validates required `.oprecipe.json` schema constraints and clinical bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.units != "mm" {
            return Err(format!(
                "Unsupported units {:?}; .oprecipe.json v0.1 requires 'mm'",
                self.units
            ));
        }
        if self.side != "left" && self.side != "right" {
            return Err(format!(
                "Invalid side {:?}; expected 'left' or 'right'",
                self.side
            ));
        }
        if self.device_type != "transtibial_socket"
            && self.device_type != "transfemoral_socket"
            && self.device_type != "afo"
        {
            return Err(format!("Unsupported device_type {:?}", self.device_type));
        }
        if self.scale_factor <= 0.0 {
            return Err("scale_factor must be positive".to_string());
        }
        if !(0.0..=12.0).contains(&self.liner_thickness_mm) {
            return Err("liner_thickness_mm out of clinical bounds [0, 12]".to_string());
        }
        for (i, op) in self.operations.iter().enumerate() {
            if op.get("op").and_then(Value::as_str).is_none() {
                return Err(format!("Operation index {i} missing required 'op' key"));
            }
        }
        Ok(())
    }

    /// Resolves every named landmark specification onto its nearest mesh vertex index.
    pub fn resolve_landmarks(&self, mesh: &TriMesh) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for (name, spec) in &self.landmarks {
            if let Some(v_idx) = spec.get("vertex_idx").and_then(Value::as_u64) {
                let idx = v_idx as usize;
                if idx < mesh.vertices.len() {
                    out.insert(name.clone(), idx);
                    continue;
                }
            }
            if let Some(xyz) = parse_xyz(spec.get("xyz")) {
                out.insert(name.clone(), mesh.nearest_vertex(xyz));
            }
        }
        out
    }

    /// Reads a float parameter from `self.shell` with fallback.
    pub fn shell_f64(&self, key: &str, default_val: f64) -> f64 {
        self.shell
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or(default_val)
    }

    /// Reads a float parameter from `self.alignment` with fallback.
    pub fn alignment_f64(&self, key: &str, default_val: f64) -> f64 {
        self.alignment
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or(default_val)
    }
}

/// Parses a JSON `[x, y, z]` array into `Vec3`.
pub fn parse_xyz(val: Option<&Value>) -> Option<Vec3> {
    let arr = val?.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    Some(Vec3::new(
        arr[0].as_f64()?,
        arr[1].as_f64()?,
        arr[2].as_f64()?,
    ))
}

/// Computes a deterministic SHA-256 hex digest of a mesh's vertices and faces.
pub fn mesh_sha256_hex(mesh: &TriMesh) -> String {
    let mut data = Vec::with_capacity(mesh.vertices.len() * 24 + mesh.faces.len() * 12);
    for v in &mesh.vertices {
        data.extend_from_slice(&((v.x * 1e4).round() as i64).to_le_bytes());
        data.extend_from_slice(&((v.y * 1e4).round() as i64).to_le_bytes());
        data.extend_from_slice(&((v.z * 1e4).round() as i64).to_le_bytes());
    }
    for &[a, b, c] in &mesh.faces {
        data.extend_from_slice(&(a as u32).to_le_bytes());
        data.extend_from_slice(&(b as u32).to_le_bytes());
        data.extend_from_slice(&(c as u32).to_le_bytes());
    }
    sha256_hex(&data)
}

/// Computes a pure-Rust FIPS 180-4 SHA-256 lowercase hex digest.
pub fn sha256_hex(msg: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (msg.len() as u64) * 8;
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

/// Computes a structured revision diff between `baseline` and `current` recipes.
pub fn diff_recipes(baseline: &OpRecipe, current: &OpRecipe) -> Map<String, Value> {
    let mut diff = Map::new();
    let v_base = serde_json::to_value(baseline).unwrap_or(Value::Null);
    let v_cur = serde_json::to_value(current).unwrap_or(Value::Null);

    for field in [
        "device_type",
        "side",
        "scale_factor",
        "liner_thickness_mm",
        "alignment",
        "shell",
    ] {
        let a = v_base.get(field);
        let b = v_cur.get(field);
        if a != b {
            let mut entry = Map::new();
            entry.insert("from".to_string(), a.cloned().unwrap_or(Value::Null));
            entry.insert("to".to_string(), b.cloned().unwrap_or(Value::Null));
            diff.insert(field.to_string(), Value::Object(entry));
        }
    }

    let map_ops = |ops: &[Value]| -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        for (i, op) in ops.iter().enumerate() {
            let id = op
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("op_{i}"));
            m.insert(id, op.clone());
        }
        m
    };

    let ops_a = map_ops(&baseline.operations);
    let ops_b = map_ops(&current.operations);
    let mut all_ids: Vec<String> = ops_a.keys().chain(ops_b.keys()).cloned().collect();
    all_ids.sort();
    all_ids.dedup();

    let mut op_diffs = Vec::new();
    for id in all_ids {
        match (ops_a.get(&id), ops_b.get(&id)) {
            (None, Some(new_op)) => {
                let mut e = Map::new();
                e.insert("id".to_string(), Value::String(id));
                e.insert("change".to_string(), Value::String("added".to_string()));
                e.insert("to".to_string(), new_op.clone());
                op_diffs.push(Value::Object(e));
            }
            (Some(old_op), None) => {
                let mut e = Map::new();
                e.insert("id".to_string(), Value::String(id));
                e.insert("change".to_string(), Value::String("removed".to_string()));
                e.insert("from".to_string(), old_op.clone());
                op_diffs.push(Value::Object(e));
            }
            (Some(old_op), Some(new_op)) if old_op != new_op => {
                let mut e = Map::new();
                e.insert("id".to_string(), Value::String(id));
                e.insert("change".to_string(), Value::String("modified".to_string()));
                e.insert("from".to_string(), old_op.clone());
                e.insert("to".to_string(), new_op.clone());
                op_diffs.push(Value::Object(e));
            }
            _ => {}
        }
    }
    if !op_diffs.is_empty() {
        diff.insert("operations".to_string(), Value::Array(op_diffs));
    }
    diff
}

/// Builds a clinically sensible default recipe for `tt`, `tf`, or `afo`.
pub fn default_recipe_for_device(device: &str, landmarks_xyz: &BTreeMap<String, Vec3>) -> OpRecipe {
    let lm_val = |key: &str, fallback: Vec3| -> Value {
        let p = landmarks_xyz.get(key).copied().unwrap_or(fallback);
        serde_json::json!({ "xyz": [p.x, p.y, p.z] })
    };

    if device == "afo" {
        let mut landmarks = BTreeMap::new();
        for (k, v) in landmarks_xyz {
            if k.contains("malleolus") {
                landmarks.insert(k.clone(), serde_json::json!({ "xyz": [v.x, v.y, v.z] }));
            }
        }
        let mut shell = Map::new();
        shell.insert("wall_distal_mm".to_string(), Value::from(4.5));
        shell.insert("wall_proximal_mm".to_string(), Value::from(2.5));
        shell.insert("trimline_fillet_mm".to_string(), Value::from(10.0));
        shell.insert("hardware".to_string(), Value::from("afo_coaxial_joint"));
        return OpRecipe {
            device_type: "afo".to_string(),
            side: "left".to_string(),
            units: "mm".to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            kernel_version: KERNEL_VERSION.to_string(),
            scan_sha256: String::new(),
            scale_factor: 1.0,
            liner_thickness_mm: 0.0,
            alignment: default_alignment(),
            landmarks,
            operations: vec![
                serde_json::json!({"op": "afo_posture", "id": "neutralize", "dorsiflexion_deg": 12.0, "eversion_deg": 5.0, "heel_pitch_mm": 10.0}),
                serde_json::json!({"op": "landmark_relief", "id": "med_mal", "landmark": "medial_malleolus", "offset_mm": 4.0, "radius_mm": 22.0}),
                serde_json::json!({"op": "landmark_relief", "id": "lat_mal", "landmark": "lateral_malleolus", "offset_mm": 4.0, "radius_mm": 22.0}),
            ],
            shell,
        };
    }

    if device == "tf" {
        let mut landmarks = BTreeMap::new();
        landmarks.insert(
            "ischial_shelf".to_string(),
            lm_val("ischial_shelf", Vec3::new(0.0, -58.0, 10.0)),
        );
        landmarks.insert(
            "scarpas_triangle".to_string(),
            lm_val("scarpas_triangle", Vec3::new(0.0, 56.0, -10.0)),
        );
        landmarks.insert(
            "trochanter".to_string(),
            lm_val("trochanter", Vec3::new(58.0, 0.0, -15.0)),
        );

        let mut alignment = Map::new();
        alignment.insert("flexion_deg".to_string(), Value::from(5.0));
        alignment.insert("adduction_deg".to_string(), Value::from(8.0));
        alignment.insert("mpt_offset_mm".to_string(), serde_json::json!([0.0, 0.0]));

        let mut shell = Map::new();
        shell.insert("wall_distal_mm".to_string(), Value::from(5.5));
        shell.insert("wall_proximal_mm".to_string(), Value::from(3.0));
        shell.insert("trimline_fillet_mm".to_string(), Value::from(12.0));
        shell.insert("proximal_trim_z_mm".to_string(), Value::from(45.0));
        shell.insert(
            "trimline".to_string(),
            serde_json::json!({
                "preset": "tf_ischial",
                "anterior_z_mm": 32.0,
                "posterior_z_mm": 18.0,
                "medial_z_mm": 10.0,
                "lateral_z_mm": 50.0,
                "points": []
            }),
        );
        shell.insert(
            "hardware".to_string(),
            Value::from("pyramid_50_8_m6_captive"),
        );

        return OpRecipe {
            device_type: "transfemoral_socket".to_string(),
            side: "left".to_string(),
            units: "mm".to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            kernel_version: KERNEL_VERSION.to_string(),
            scan_sha256: String::new(),
            scale_factor: 1.0,
            liner_thickness_mm: 3.0,
            alignment,
            landmarks,
            operations: vec![
                serde_json::json!({"op": "global_compression", "id": "glob", "z_range_mm": [-260.0, 48.0], "percent": -4.0, "transition_mm": 18.0}),
                serde_json::json!({"op": "landmark_relief", "id": "scarpa", "landmark": "scarpas_triangle", "offset_mm": -5.0, "radius_mm": 28.0}),
                serde_json::json!({"op": "landmark_relief", "id": "troch", "landmark": "trochanter", "offset_mm": 4.0, "radius_mm": 26.0}),
            ],
            shell,
        };
    }

    let mut landmarks = BTreeMap::new();
    landmarks.insert(
        "pt_bar".to_string(),
        lm_val("mpt", Vec3::new(0.0, 54.0, 0.0)),
    );
    landmarks.insert(
        "fibular_head".to_string(),
        lm_val("fibular_head", Vec3::new(52.0, 0.0, -25.0)),
    );
    landmarks.insert(
        "tibial_crest".to_string(),
        lm_val("tibial_crest", Vec3::new(0.0, 46.0, -75.0)),
    );
    landmarks.insert(
        "popliteal".to_string(),
        lm_val("popliteal", Vec3::new(0.0, -54.0, -5.0)),
    );

    let mut shell = Map::new();
    shell.insert("wall_distal_mm".to_string(), Value::from(4.5));
    shell.insert("wall_proximal_mm".to_string(), Value::from(2.5));
    shell.insert("trimline_fillet_mm".to_string(), Value::from(10.0));
    shell.insert("proximal_trim_z_mm".to_string(), Value::from(38.0));
    shell.insert(
        "trimline".to_string(),
        serde_json::json!({
            "preset": "ptb",
            "anterior_z_mm": 30.0,
            "posterior_z_mm": 0.0,
            "medial_z_mm": 24.0,
            "lateral_z_mm": 24.0,
            "points": []
        }),
    );
    shell.insert(
        "hardware".to_string(),
        Value::from("pyramid_50_8_m6_captive"),
    );

    OpRecipe {
        device_type: "transtibial_socket".to_string(),
        side: "left".to_string(),
        units: "mm".to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        kernel_version: KERNEL_VERSION.to_string(),
        scan_sha256: String::new(),
        scale_factor: 1.0,
        liner_thickness_mm: 3.0,
        alignment: default_alignment(),
        landmarks,
        operations: vec![
            serde_json::json!({"op": "global_compression", "id": "glob", "z_range_mm": [-220.0, 42.0], "percent": -3.0, "transition_mm": 20.0}),
            serde_json::json!({"op": "zoned_compression", "id": "distal_zone", "z_range_mm": [-220.0, -100.0], "percent": 0.0, "transition_mm": 20.0}),
            serde_json::json!({"op": "landmark_relief", "id": "pt", "landmark": "pt_bar", "offset_mm": -6.0, "radius_mm": 20.0}),
            serde_json::json!({"op": "landmark_relief", "id": "fib", "landmark": "fibular_head", "offset_mm": 4.0, "radius_mm": 24.0}),
            serde_json::json!({"op": "landmark_relief", "id": "crest", "landmark": "tibial_crest", "offset_mm": 3.0, "radius_mm": 22.0}),
            serde_json::json!({"op": "landmark_relief", "id": "pop", "landmark": "popliteal", "offset_mm": -4.0, "radius_mm": 24.0}),
        ],
        shell,
    }
}
