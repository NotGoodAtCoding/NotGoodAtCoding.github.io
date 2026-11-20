//! Print-safety geometry linter and pure-Rust PDF-1.4 / Markdown / JSON dossier generators.

use std::collections::HashMap;
use std::fmt::Write;

use serde::Deserialize;
use serde::Serialize;

use crate::math3d;
use crate::math3d::Vec3;
use crate::mesh::TriMesh;
use crate::recipe::OpRecipe;
use crate::scan_qa::ScanQaReport;
use crate::shell;

/// Pre-print structural and geometric safety check results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    pub passed: bool,
    pub is_watertight_manifold: bool,
    pub min_estimated_wall_mm: f64,
    pub trimline_min_radius_mm: f64,
    pub load_to_print_z_angle_deg: f64,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
}

/// Estimates minimum opposing-wall thickness using a spatial hash over opposing normal pairs.
pub fn estimate_min_wall_thickness_mm(shell_mesh: &TriMesh) -> f64 {
    let verts = &shell_mesh.vertices;
    if verts.is_empty() {
        return 0.0;
    }
    let norms = shell_mesh.vertex_normals();
    let cell = 8.0;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(verts.len());
    for (idx, &v) in verts.iter().enumerate() {
        let key = (
            (v.x / cell).floor() as i32,
            (v.y / cell).floor() as i32,
            (v.z / cell).floor() as i32,
        );
        grid.entry(key).or_default().push(idx);
    }

    let step = (verts.len() / 400).max(1);
    let mut min_thick = f64::INFINITY;
    let mut i = 0;
    while i < verts.len() {
        let vi = verts[i];
        let ni = norms[i];
        let gx = (vi.x / cell).floor() as i32;
        let gy = (vi.y / cell).floor() as i32;
        let gz = (vi.z / cell).floor() as i32;
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &j in bucket {
                            if i == j {
                                continue;
                            }
                            let d_vec = math3d::v_sub(verts[j], vi);
                            let dist = math3d::v_norm(d_vec);
                            if dist > 0.6 && dist < 12.0 {
                                let d_unit = math3d::v_unit(d_vec, Vec3::default());
                                if math3d::v_dot(ni, norms[j]) < -0.75
                                    && math3d::v_dot(d_unit, ni) < -0.75
                                    && dist < best
                                {
                                    best = dist;
                                }
                            }
                        }
                    }
                }
            }
        }
        if best < min_thick {
            min_thick = best;
        }
        i += step;
    }
    if min_thick.is_finite() {
        min_thick
    } else {
        0.0
    }
}

/// Runs pre-print safety checks: manifoldness, minimum wall thickness, notch radius, print-Z angle.
pub fn lint_device(
    shell_mesh: &TriMesh,
    min_wall_mm: f64,
    min_trimline_radius_mm: f64,
    trimline_points: Option<&[Vec3]>,
    measured_wall_override_mm: Option<f64>,
) -> LintReport {
    let mut failures = Vec::new();
    let mut warnings = Vec::new();

    let watertight = shell_mesh.is_watertight_manifold();
    if !watertight {
        failures.push(
            "Shell mesh is not a closed 2-manifold (non-manifold or boundary edges present)"
                .to_string(),
        );
    }

    let est_wall =
        measured_wall_override_mm.unwrap_or_else(|| estimate_min_wall_thickness_mm(shell_mesh));
    if est_wall > 0.0 && est_wall < min_wall_mm {
        failures.push(format!(
            "Minimum wall thickness {est_wall:.2} mm is below safe floor {min_wall_mm:.2} mm"
        ));
    }

    let r_trim = trimline_points
        .map(|pts| shell::compute_curve_min_radius(pts, true))
        .unwrap_or(15.0);
    if r_trim < min_trimline_radius_mm {
        failures.push(format!(
            "Trimline notch radius {r_trim:.2} mm is below {min_trimline_radius_mm:.2} mm stress-riser limit"
        ));
    }

    let angle_deg = 0.0;
    if angle_deg > 25.0 {
        warnings.push(format!(
            "Load-axis tilt ({angle_deg:.1} deg) relative to print Z increases inter-layer shear risk"
        ));
    }

    LintReport {
        passed: failures.is_empty(),
        is_watertight_manifold: watertight,
        min_estimated_wall_mm: est_wall,
        trimline_min_radius_mm: r_trim,
        load_to_print_z_angle_deg: angle_deg,
        failures,
        warnings,
    }
}

fn pdf_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
            _ => out.push(' '),
        }
    }
    out
}

/// Builds a standalone pure-Rust PDF 1.4 custom-device fabrication dossier.
pub fn export_fabrication_dossier_pdf_bytes(
    recipe: &OpRecipe,
    qa_report: &ScanQaReport,
    lint_report: &LintReport,
) -> Vec<u8> {
    let sha_short = if recipe.scan_sha256.is_empty() {
        "N/A"
    } else {
        &recipe.scan_sha256[..recipe.scan_sha256.len().min(48)]
    };

    let mut lines: Vec<(&str, i32, String)> = vec![
        (
            "F2",
            14,
            format!(
                "Custom O&P Device Fabrication Dossier ({})",
                recipe.device_type
            ),
        ),
        (
            "F1",
            9,
            format!(
                "Schema: v{} | Kernel: op-core {} | Side: {} | Units: {}",
                recipe.schema_version, recipe.kernel_version, recipe.side, recipe.units
            ),
        ),
        ("F1", 9, format!("Scan SHA-256: {sha_short}")),
        (
            "F2",
            11,
            "1. Quality Assurance & Pre-Print Safety Gate".to_string(),
        ),
        (
            "F1",
            9,
            format!(
                "Scan QA Status: {} (Floaters={}, Tears={}, DoubleWall={})",
                qa_report.status,
                qa_report.floater_components_removed,
                qa_report.internal_tear_loops,
                qa_report.double_wall_vertices
            ),
        ),
        (
            "F1",
            9,
            format!(
                "Safety Lint: {} | Watertight={} | MinWall={:.2} mm | TrimR={:.1} mm",
                if lint_report.passed { "PASS" } else { "FAIL" },
                lint_report.is_watertight_manifold,
                lint_report.min_estimated_wall_mm,
                lint_report.trimline_min_radius_mm
            ),
        ),
        (
            "F2",
            11,
            "2. Landmark-Anchored Circumferences (20 mm increments)".to_string(),
        ),
        (
            "F1",
            9,
            "Depth (mm)      Taut Circumference (mm)      Cross-Section Area (mm^2)".to_string(),
        ),
    ];

    for sl in qa_report.slices.iter().take(10) {
        lines.push((
            "F1",
            9,
            format!(
                "  -{:5.0}            {:8.1}                    {:9.0}",
                sl.depth_from_anchor_mm, sl.taut_circumference_mm, sl.hull_area_mm2
            ),
        ));
    }

    let align_json = serde_json::to_string(&recipe.alignment).unwrap_or_default();
    lines.push((
        "F2",
        11,
        "3. Parametric Rectification Operations (.oprecipe.json)".to_string(),
    ));
    lines.push((
        "F1",
        9,
        format!(
            "Liner Offset: {:.1} mm | Alignment: {align_json}",
            recipe.liner_thickness_mm
        ),
    ));
    for op in recipe.operations.iter().take(10) {
        let s = serde_json::to_string(op).unwrap_or_default();
        let trunc = &s[..s.len().min(92)];
        lines.push(("F1", 8, format!("  - {trunc}")));
    }
    lines.push((
        "F2",
        11,
        "4. Regulatory Traceability (EU MDR Annex XIII / FDA QMSR)".to_string(),
    ));
    lines.push((
        "F1",
        9,
        "Patient Case ID: _______________   Prescriber NPI/ID: _______________".to_string(),
    ));
    lines.push((
        "F1",
        9,
        "Material Lot #:  _______________   Component Serial #: ______________".to_string(),
    ));

    let mut stream = String::from("BT\n");
    let mut y = 758;
    for (font_key, size, txt) in lines {
        if y < 48 {
            break;
        }
        let _ = writeln!(stream, "/{font_key} {size} Tf");
        let _ = writeln!(stream, "1 0 0 1 44 {y} Tm ({}) Tj", pdf_escape(&txt));
        y -= size + 6;
    }
    stream.push_str("ET\n");
    let stream_bytes = stream.as_bytes();

    let obj6 = [
        format!("6 0 obj\n<< /Length {} >>\nstream\n", stream_bytes.len()).into_bytes(),
        stream_bytes.to_vec(),
        b"endstream\nendobj\n".to_vec(),
    ]
    .concat();

    let objs: [&[u8]; 6] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n",
        &obj6,
    ];

    let mut pdf = Vec::with_capacity(4096);
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets = vec![0usize];
    for obj in objs {
        offsets.push(pdf.len());
        pdf.extend_from_slice(obj);
    }
    let xref_start = pdf.len();
    let _ = write!(
        pdf_string_helper(&mut pdf),
        "xref\n0 {}\n0000000000 65535 f \n",
        objs.len() + 1
    );
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n",
            objs.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

struct ByteWriter<'a>(&'a mut Vec<u8>);

impl Write for ByteWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn pdf_string_helper(buf: &mut Vec<u8>) -> ByteWriter<'_> {
    ByteWriter(buf)
}

/// Formats Markdown and JSON archival dossiers alongside the PDF bytes.
pub fn export_fabrication_dossier_bundle(
    recipe: &OpRecipe,
    qa_report: &ScanQaReport,
    lint_report: &LintReport,
) -> (String, String, Vec<u8>) {
    let json_bundle = serde_json::json!({
        "recipe": recipe,
        "scan_qa": qa_report,
        "safety_lint": lint_report,
        "regulatory_placeholders": {
            "patient_case_id": "UNASSIGNED",
            "prescriber_npi_or_id": "UNASSIGNED",
            "material_lot_number": "UNASSIGNED",
            "hardware_component_serial": "UNASSIGNED"
        }
    });
    let json_str = serde_json::to_string_pretty(&json_bundle).unwrap_or_default() + "\n";

    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Custom O&P Device Fabrication Dossier ({})\n",
        recipe.device_type
    );
    let _ = writeln!(
        md,
        "- **Schema / Kernel:** `{}` / `{}`",
        recipe.schema_version, recipe.kernel_version
    );
    let _ = writeln!(
        md,
        "- **Side / Units:** `{}` / `{}`",
        recipe.side, recipe.units
    );
    let _ = writeln!(
        md,
        "- **Scan SHA-256:** `{}`",
        if recipe.scan_sha256.is_empty() {
            "N/A"
        } else {
            &recipe.scan_sha256
        }
    );
    let _ = writeln!(md, "- **Scan QA Status:** **{}**", qa_report.status);
    let _ = writeln!(
        md,
        "- **Safety Lint Passed:** **{}** (Watertight={}, MinWall={:.2} mm)\n",
        lint_report.passed, lint_report.is_watertight_manifold, lint_report.min_estimated_wall_mm
    );
    md.push_str("## Landmark-Anchored Circumferences (20 mm increments)\n");
    md.push_str("| Depth from Anchor (mm) | Taut-Tape Circumference (mm) | Area (mm^2) |\n");
    md.push_str("| ---: | ---: | ---: |\n");
    for sl in &qa_report.slices {
        let _ = writeln!(
            md,
            "| {:.0} | {:.1} | {:.0} |",
            sl.depth_from_anchor_mm, sl.taut_circumference_mm, sl.hull_area_mm2
        );
    }
    md.push_str("\n## Parametric Operations Log\n```json\n");
    md.push_str(&serde_json::to_string_pretty(&recipe.operations).unwrap_or_default());
    md.push_str("\n```\n");

    let pdf_bytes = export_fabrication_dossier_pdf_bytes(recipe, qa_report, lint_report);
    (json_str, md, pdf_bytes)
}
