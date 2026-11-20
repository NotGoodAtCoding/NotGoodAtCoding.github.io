//! Stateful in-memory clinical workstation session coordinating Scan QA, rectification, and CAM.

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use crate::carver_cam;
use crate::export_3mf;
use crate::linter;
use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh;
use crate::mesh::TriMesh;
use crate::recipe;
use crate::recipe::OpRecipe;
use crate::rectify;
use crate::scan_qa;
use crate::scan_qa::ScanQaReport;
use crate::shell;
use crate::synthetic;

#[derive(Debug, Clone, Serialize)]
struct MeshBoundsPayload {
    min: [f64; 3],
    max: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
struct MeshPayload {
    vertices: Vec<f64>,
    normals: Vec<f64>,
    indices: Vec<usize>,
    vertex_count: usize,
    face_count: usize,
    center: [f64; 3],
    bounds: MeshBoundsPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    deltas: Option<Vec<f64>>,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() * 0.001
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() * 0.0001
}

fn mesh_payload(m: &TriMesh, deltas: Option<&[f64]>) -> MeshPayload {
    let norms = m.vertex_normals();
    let (bmin, bmax) = m.bounds();
    let center = [
        round3(0.5 * (bmin.x + bmax.x)),
        round3(0.5 * (bmin.y + bmax.y)),
        round3(0.5 * (bmin.z + bmax.z)),
    ];

    let mut vertices = Vec::with_capacity(m.vertices.len() * 3);
    for v in &m.vertices {
        vertices.push(round3(v.x));
        vertices.push(round3(v.y));
        vertices.push(round3(v.z));
    }

    let mut normals = Vec::with_capacity(norms.len() * 3);
    for n in &norms {
        normals.push(round4(n.x));
        normals.push(round4(n.y));
        normals.push(round4(n.z));
    }

    let mut indices = Vec::with_capacity(m.faces.len() * 3);
    for &[a, b, c] in &m.faces {
        indices.push(a);
        indices.push(b);
        indices.push(c);
    }

    MeshPayload {
        vertices,
        normals,
        indices,
        vertex_count: m.vertices.len(),
        face_count: m.faces.len(),
        center,
        bounds: MeshBoundsPayload {
            min: bmin.to_array(),
            max: bmax.to_array(),
        },
        deltas: deltas.map(|ds| ds.iter().map(|&d| round3(d)).collect()),
    }
}

/// Stateful pure-Rust workstation kernel mirroring the `op-core` browser UI endpoints.
#[derive(Debug, Default)]
pub struct WorkstationKernel {
    canonical_mesh: Option<TriMesh>,
    qa_report: Option<ScanQaReport>,
    baseline_recipe: Option<OpRecipe>,
    current_recipe: Option<OpRecipe>,
    artifacts: HashMap<String, Vec<u8>>,
}

impl WorkstationKernel {
    /// Creates an empty workstation session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a procedural clinical limb/foot sample (`"tt"`, `"tf"`, or `"afo"`) and returns JSON.
    pub fn load_sample(&mut self, device: &str) -> Result<String, String> {
        let (aligned, qa_rep, mut rec) = if device == "afo" {
            let case = synthetic::generate_foot_ankle_mesh(12.0, 5.0, 240.0, 230.0, 56, 60);
            let slices = scan_qa::measure_circumferences(&case.mesh, 220.0, 20.0, 140.0);
            let qa = ScanQaReport {
                status: "PASS".to_string(),
                floater_components_removed: 0,
                proximal_cuff_loops: 1,
                internal_tear_loops: 0,
                double_wall_vertices: 0,
                seam_step_score_mm: 0.0,
                detected_scale_ratio: None,
                warnings: Vec::new(),
                errors: Vec::new(),
                slices,
            };
            let r = recipe::default_recipe_for_device("afo", &case.landmark_coords);
            (case.mesh, qa, r)
        } else if device == "tf" {
            let case = synthetic::generate_transfemoral_limb(235.0, 68.0, 95.0, false, 58, 64, 7);
            let (al, qa) = scan_qa::run_scan_qa(
                &case.mesh,
                case.landmark_coords.get("mpt").copied(),
                case.landmark_coords.get("distal_end").copied(),
            );
            let aligned_lms =
                remap_landmarks(&case.landmarks, &case.mesh, &al, &case.landmark_coords);
            let r = recipe::default_recipe_for_device("tf", &aligned_lms);
            (al, qa, r)
        } else {
            let case = synthetic::generate_transtibial_limb(
                180.0, 54.0, 34.0, 35.0, false, true, 44, 64, 0,
            );
            let (al, qa) = scan_qa::run_scan_qa(
                &case.mesh,
                case.landmark_coords.get("mpt").copied(),
                case.landmark_coords.get("distal_end").copied(),
            );
            let aligned_lms =
                remap_landmarks(&case.landmarks, &case.mesh, &al, &case.landmark_coords);
            let r = recipe::default_recipe_for_device("tt", &aligned_lms);
            (al, qa, r)
        };

        rec.scan_sha256 = recipe::mesh_sha256_hex(&aligned);
        let payload = serde_json::json!({
            "session_id": "wasm-local",
            "mesh": mesh_payload(&aligned, None),
            "qa": qa_rep,
            "recipe": rec,
        });

        self.artifacts.clear();
        self.canonical_mesh = Some(aligned);
        self.qa_report = Some(qa_rep);
        self.baseline_recipe = Some(rec.clone());
        self.current_recipe = Some(rec);

        serde_json::to_string(&payload).map_err(|e| e.to_string())
    }

    /// Parses uploaded binary/ASCII STL bytes, runs chairside Scan QA, and initializes a session.
    pub fn upload_stl_bytes(&mut self, stl_bytes: &[u8]) -> Result<String, String> {
        let raw_mesh = mesh::read_stl_bytes(stl_bytes, 1e-5)?;
        let (aligned, qa_rep) = scan_qa::run_scan_qa(&raw_mesh, None, None);
        let mut rec = recipe::default_recipe_for_device("tt", &BTreeMap::new());
        rec.scan_sha256 = recipe::mesh_sha256_hex(&aligned);

        let payload = serde_json::json!({
            "session_id": "wasm-local",
            "mesh": mesh_payload(&aligned, None),
            "qa": qa_rep,
            "recipe": rec,
        });

        self.artifacts.clear();
        self.canonical_mesh = Some(aligned);
        self.qa_report = Some(qa_rep);
        self.baseline_recipe = Some(rec.clone());
        self.current_recipe = Some(rec);

        serde_json::to_string(&payload).map_err(|e| e.to_string())
    }

    /// Applies an `.oprecipe.json` string to the canonical scan and returns rectified mesh + deltas.
    pub fn rectify(&mut self, recipe_json: &str) -> Result<String, String> {
        let canonical = self
            .canonical_mesh
            .as_ref()
            .ok_or_else(|| "No scan loaded in session".to_string())?;
        let rec: OpRecipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
        let res = rectify::apply_recipe(canonical, &rec)?;

        let deltas: Vec<f64> = if res.rectified_mesh.vertices.len() == canonical.vertices.len() {
            let norms = canonical.vertex_normals();
            (0..norms.len())
                .map(|i| {
                    math3d::v_dot(
                        math3d::v_sub(res.rectified_mesh.vertices[i], canonical.vertices[i]),
                        norms[i],
                    )
                })
                .collect()
        } else {
            res.relief_field_mm.clone()
        };

        self.current_recipe = Some(rec);
        let payload = serde_json::json!({
            "rectified": mesh_payload(&res.rectified_mesh, Some(&deltas)),
        });
        serde_json::to_string(&payload).map_err(|e| e.to_string())
    }

    /// Computes structured diff JSON between the intake baseline recipe and `recipe_json`.
    pub fn diff_recipe(&self, recipe_json: &str) -> Result<String, String> {
        let baseline = self
            .baseline_recipe
            .as_ref()
            .ok_or_else(|| "No baseline recipe loaded".to_string())?;
        let cur: OpRecipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
        let diff = recipe::diff_recipes(baseline, &cur);
        serde_json::to_string(&serde_json::json!({ "diff": diff })).map_err(|e| e.to_string())
    }

    /// Compiles watertight shell, `.3mf`, carver positive/G-code, safety lint, and PDF dossier.
    pub fn build(&mut self, recipe_json: &str, target: &str) -> Result<String, String> {
        let canonical = self
            .canonical_mesh
            .as_ref()
            .ok_or_else(|| "No scan loaded in session".to_string())?
            .clone();
        let qa_rep = self
            .qa_report
            .as_ref()
            .ok_or_else(|| "No Scan QA report in session".to_string())?
            .clone();

        let rec: OpRecipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
        let res = rectify::apply_recipe(&canonical, &rec)?;

        let trimline_val = rec.shell.get("trimline");
        let (sh, trim_z) = if rec.device_type == "afo" {
            let s = shell::build_afo_shell(
                &res.rectified_mesh,
                rec.shell_f64("wall_distal_mm", 4.5),
                rec.shell_f64("wall_proximal_mm", 2.5),
                rec.shell_f64("trimline_fillet_mm", 10.0),
            )?;
            (s, None)
        } else if rec.device_type == "transfemoral_socket" {
            let tz = rec.shell_f64("proximal_trim_z_mm", 45.0);
            let s = shell::build_tf_socket_shell(
                &res.rectified_mesh,
                rec.shell_f64("wall_distal_mm", 5.5),
                rec.shell_f64("wall_proximal_mm", 3.0),
                rec.alignment_f64("flexion_deg", 5.0),
                rec.alignment_f64("adduction_deg", 8.0),
                Some(tz),
                trimline_val,
            )?;
            (s, Some(tz))
        } else {
            let tz = rec.shell_f64("proximal_trim_z_mm", 38.0);
            let s = shell::build_tt_socket_shell(
                &res.rectified_mesh,
                rec.shell_f64("wall_distal_mm", 4.5),
                rec.shell_f64("wall_proximal_mm", 2.5),
                rec.alignment_f64("flexion_deg", 5.0),
                rec.alignment_f64("adduction_deg", 5.0),
                Some(tz),
                trimline_val,
            )?;
            (s, Some(tz))
        };

        self.artifacts.clear();
        let mut downloads = Map::new();

        self.artifacts.insert(
            "rectified_positive.stl".to_string(),
            mesh::write_binary_stl_bytes(&res.rectified_mesh, "op-core rectified positive"),
        );
        self.artifacts.insert(
            "recipe.oprecipe.json".to_string(),
            (serde_json::to_string_pretty(&rec).unwrap_or_default() + "\n").into_bytes(),
        );
        self.artifacts.insert(
            "device_shell.stl".to_string(),
            mesh::write_binary_stl_bytes(&sh.shell_mesh, "op-core device shell"),
        );
        let threemf_name = format!("device_{target}.3mf");
        let threemf_bytes =
            export_3mf::export_3mf_bytes(&sh.shell_mesh, &sh.modifier_mesh, target, 100, 6)?;
        self.artifacts.insert(threemf_name.clone(), threemf_bytes);

        downloads.insert(
            "Shell STL (device_shell.stl)".to_string(),
            Value::String("device_shell.stl".to_string()),
        );
        downloads.insert(
            format!("Slicer 3MF ({threemf_name})"),
            Value::String(threemf_name),
        );
        downloads.insert(
            "Rectified Positive STL".to_string(),
            Value::String("rectified_positive.stl".to_string()),
        );
        downloads.insert(
            "Recipe (.oprecipe.json)".to_string(),
            Value::String("recipe.oprecipe.json".to_string()),
        );

        if rec.device_type != "afo" {
            let carver = shell::export_carver_positive(
                &res.rectified_mesh,
                12.0,
                40.0,
                12.7,
                95.0,
                trim_z,
                trimline_val,
            )?;
            self.artifacts.insert(
                "carver_positive.stl".to_string(),
                mesh::write_binary_stl_bytes(&carver.positive_mesh, "op-core carver positive"),
            );
            let cam =
                carver_cam::generate_4axis_gcode_and_simulate(&carver.positive_mesh, 4.0, 2.0, 60);
            self.artifacts
                .insert("carver_4axis.gcode".to_string(), cam.gcode.into_bytes());
            downloads.insert(
                "Carver Positive STL".to_string(),
                Value::String("carver_positive.stl".to_string()),
            );
            downloads.insert(
                "Carver 4-Axis G-Code (.gcode)".to_string(),
                Value::String("carver_4axis.gcode".to_string()),
            );
        }

        let lt = linter::lint_device(
            &sh.shell_mesh,
            2.5,
            8.0,
            None,
            Some(rec.shell_f64("wall_proximal_mm", 2.5)),
        );
        let (dossier_json, dossier_md, dossier_pdf) =
            linter::export_fabrication_dossier_bundle(&rec, &qa_rep, &lt);
        self.artifacts
            .insert("device_dossier.pdf".to_string(), dossier_pdf);
        self.artifacts
            .insert("device_dossier.md".to_string(), dossier_md.into_bytes());
        self.artifacts
            .insert("device_dossier.json".to_string(), dossier_json.into_bytes());

        downloads.insert(
            "Dossier PDF (device_dossier.pdf)".to_string(),
            Value::String("device_dossier.pdf".to_string()),
        );
        downloads.insert(
            "Dossier Markdown".to_string(),
            Value::String("device_dossier.md".to_string()),
        );
        downloads.insert(
            "Dossier JSON".to_string(),
            Value::String("device_dossier.json".to_string()),
        );

        self.current_recipe = Some(rec);

        let payload = serde_json::json!({
            "rectified": mesh_payload(&res.rectified_mesh, None),
            "shell": mesh_payload(&sh.shell_mesh, None),
            "lint": lt,
            "hardware_metrics": sh.hardware_metrics,
            "downloads": Value::Object(downloads),
        });
        serde_json::to_string(&payload).map_err(|e| e.to_string())
    }

    /// Returns generated artifact bytes (STL, 3MF, G-code, JSON, MD, PDF) by filename.
    pub fn artifact_bytes(&self, filename: &str) -> Option<&[u8]> {
        self.artifacts.get(filename).map(Vec::as_slice)
    }
}

fn remap_landmarks(
    indices: &BTreeMap<String, usize>,
    original: &TriMesh,
    aligned: &TriMesh,
    fallback: &BTreeMap<String, Vec3>,
) -> BTreeMap<String, Vec3> {
    if aligned.vertices.len() == original.vertices.len() {
        indices
            .iter()
            .map(|(k, &idx)| (k.clone(), aligned.vertices[idx]))
            .collect()
    } else {
        fallback.clone()
    }
}
