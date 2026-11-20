//! Integration tests verifying native Rust parity across TT/TF/AFO pipelines, trimlines, and CAM.

use op_core_wasm::carver_cam;
use op_core_wasm::export_3mf;
use op_core_wasm::linter;
use op_core_wasm::math3d;
use op_core_wasm::mesh;
use op_core_wasm::recipe;
use op_core_wasm::rectify;
use op_core_wasm::scan_qa;
use op_core_wasm::shell;
use op_core_wasm::synthetic;
use op_core_wasm::WorkstationKernel;

#[test]
fn full_tt_limb_knee_patella_qa_and_contoured_trimlines() {
    let tt = synthetic::generate_transtibial_limb(180.0, 54.0, 34.0, 35.0, false, true, 44, 64, 0);
    assert!(tt.mesh.is_watertight_manifold());
    assert!(tt.mesh.signed_volume() > 0.0);

    // Verify broad, low-profile patella above MPT and below suprapatellar dip.
    let pat_y = tt
        .mesh
        .vertices
        .iter()
        .filter(|v| (28.0..=38.0).contains(&v.z) && v.x.abs() < 8.0)
        .map(|v| v.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let pat_side_y = tt
        .mesh
        .vertices
        .iter()
        .filter(|v| (28.0..=38.0).contains(&v.z) && (14.0..=22.0).contains(&v.x.abs()))
        .map(|v| v.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let supra_y = tt
        .mesh
        .vertices
        .iter()
        .filter(|v| (56.0..=63.0).contains(&v.z) && v.x.abs() < 8.0)
        .map(|v| v.y)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(pat_y - supra_y > 5.0);
    assert!(pat_y - pat_side_y < 4.0);

    let (aligned, qa) = scan_qa::run_scan_qa(
        &tt.mesh,
        tt.landmark_coords.get("mpt").copied(),
        tt.landmark_coords.get("distal_end").copied(),
    );
    assert_eq!(qa.status, "PASS", "QA errors: {:?}", qa.errors);

    for preset in ["ptb", "ptb_sc", "ptb_scsp"] {
        let tl = serde_json::json!({ "preset": preset });
        let sh = shell::build_tt_socket_shell(&aligned, 4.5, 2.5, 5.0, 5.0, Some(38.0), Some(&tl))
            .expect("contoured socket shell should build");
        assert!(
            sh.shell_mesh.is_watertight_manifold(),
            "{preset} shell not watertight"
        );
        assert!(sh.shell_mesh.signed_volume() > 0.0);

        let cv =
            shell::export_carver_positive(&aligned, 12.0, 40.0, 12.7, 95.0, Some(38.0), Some(&tl))
                .expect("carver positive should build");
        assert!(
            cv.positive_mesh.is_watertight_manifold(),
            "{preset} carver not watertight"
        );
    }
}

#[test]
fn afo_anatomical_heel_arch_forefoot_and_shell() {
    let afo0 = synthetic::generate_foot_ankle_mesh(0.0, 0.0, 240.0, 230.0, 56, 60);
    assert!(afo0.mesh.is_watertight_manifold());

    let ach_y = afo0
        .mesh
        .vertices
        .iter()
        .filter(|v| (78.0..=90.0).contains(&v.z))
        .map(|v| v.y)
        .fold(f64::INFINITY, f64::min);
    let heel_y = afo0
        .mesh
        .vertices
        .iter()
        .filter(|v| (10.0..=35.0).contains(&v.z))
        .map(|v| v.y)
        .fold(f64::INFINITY, f64::min);
    assert!(ach_y - heel_y > 10.0);

    let afo_pf = synthetic::generate_foot_ankle_mesh(12.0, 5.0, 240.0, 230.0, 56, 60);
    let neutral = rectify::apply_afo_posture(&afo_pf.mesh, 12.0, 5.0, 10.0);
    let sh = shell::build_afo_shell(&neutral, 4.5, 2.5, 10.0).expect("AFO shell should build");
    assert!(sh.shell_mesh.is_watertight_manifold());
    assert!(sh.shell_mesh.signed_volume() > 0.0);
    assert!(sh.trimline_min_radius_mm >= 8.0);
}

#[test]
fn sculpt_strokes_taubin_smoothing_and_dual_cam_exports() {
    let tt = synthetic::generate_transtibial_limb(180.0, 54.0, 34.0, 35.0, false, true, 44, 64, 3);
    let n0 = tt.mesh.vertex_normals();
    let pt_add = tt.landmark_coords["tibial_crest"];
    let pt_rem = tt.landmark_coords["popliteal"];

    let mut rec = recipe::default_recipe_for_device("tt", &tt.landmark_coords);
    rec.liner_thickness_mm = 0.0;
    rec.operations = vec![
        serde_json::json!({
            "op": "sculpt_stroke",
            "id": "add_crest",
            "mode": "add",
            "points": [[pt_add.x, pt_add.y, pt_add.z], [pt_add.x, pt_add.y, pt_add.z - 25.0]],
            "offset_mm": 5.0,
            "radius_mm": 22.0,
            "falloff": "gaussian"
        }),
        serde_json::json!({
            "op": "sculpt_stroke",
            "id": "rem_pop",
            "mode": "remove",
            "points": [[pt_rem.x, pt_rem.y, pt_rem.z]],
            "offset_mm": 4.5,
            "radius_mm": 24.0,
            "falloff": "flat_pad"
        }),
    ];

    let res_raw = rectify::apply_recipe(&tt.mesh, &rec).expect("rectify should succeed");
    let idx_add = tt.landmarks["tibial_crest"];
    let idx_rem = tt.landmarks["popliteal"];
    let d_add = math3d::v_dot(
        math3d::v_sub(
            res_raw.rectified_mesh.vertices[idx_add],
            tt.mesh.vertices[idx_add],
        ),
        n0[idx_add],
    );
    let d_rem = math3d::v_dot(
        math3d::v_sub(
            res_raw.rectified_mesh.vertices[idx_rem],
            tt.mesh.vertices[idx_rem],
        ),
        n0[idx_rem],
    );
    assert!(d_add > 4.0);
    assert!(d_rem < -3.5);

    rec.operations.push(serde_json::json!({
        "op": "smooth",
        "id": "smooth_global",
        "iterations": 4,
        "strength": 0.5
    }));
    let res_sm = rectify::apply_recipe(&tt.mesh, &rec).expect("smooth rectify should succeed");
    assert!(res_sm.rectified_mesh.is_watertight_manifold());
    let vol_ratio =
        (res_sm.rectified_mesh.signed_volume() - res_raw.rectified_mesh.signed_volume()).abs()
            / res_raw.rectified_mesh.signed_volume().max(1.0);
    assert!(vol_ratio < 0.01);

    // Check 3MF, STL roundtrip, G-code, and PDF dossier.
    let sh =
        shell::build_tt_socket_shell(&res_sm.rectified_mesh, 4.5, 2.5, 5.0, 5.0, Some(38.0), None)
            .expect("shell should build");
    let z_orca =
        export_3mf::export_3mf_bytes(&sh.shell_mesh, &sh.modifier_mesh, "orcaslicer", 100, 6)
            .expect("3mf should serialize");
    assert!(z_orca.starts_with(b"PK\x03\x04"));

    let stl = mesh::write_binary_stl_bytes(&sh.shell_mesh, "rust test");
    let reloaded = mesh::read_stl_bytes(&stl, 1e-4).expect("STL should roundtrip");
    assert_eq!(reloaded.faces.len(), sh.shell_mesh.faces.len());

    let cv = shell::export_carver_positive(
        &res_sm.rectified_mesh,
        12.0,
        40.0,
        12.7,
        95.0,
        Some(38.0),
        None,
    )
    .expect("carver positive should build");
    let cam = carver_cam::generate_4axis_gcode_and_simulate(&cv.positive_mesh, 4.0, 2.5, 48);
    assert!(cam.max_gouge_mm < 1e-3);
    assert_eq!(cam.mandrel_collisions, 0);

    let lt = linter::lint_device(&sh.shell_mesh, 2.5, 8.0, None, Some(2.5));
    assert!(lt.passed);
}

#[test]
fn workstation_kernel_end_to_end_tt_tf_afo_and_upload() {
    let mut kernel = WorkstationKernel::new();

    // TT sample -> rectify -> diff -> build -> artifacts
    let sample_str = kernel.load_sample("tt").expect("load TT");
    let sample: serde_json::Value = serde_json::from_str(&sample_str).expect("parse sample JSON");
    assert_eq!(sample["qa"]["status"], "PASS");
    let mut rec = sample["recipe"].clone();
    rec["liner_thickness_mm"] = serde_json::json!(4.0);
    let rec_str = serde_json::to_string(&rec).expect("serialize recipe");

    let rect_str = kernel.rectify(&rec_str).expect("rectify TT");
    let rect: serde_json::Value = serde_json::from_str(&rect_str).expect("parse rect JSON");
    assert_eq!(
        rect["rectified"]["deltas"]
            .as_array()
            .expect("deltas array")
            .len(),
        sample["mesh"]["vertex_count"]
            .as_u64()
            .expect("vertex count") as usize
    );

    let diff_str = kernel.diff_recipe(&rec_str).expect("diff recipe");
    assert!(diff_str.contains("liner_thickness_mm"));

    let build_str = kernel.build(&rec_str, "orcaslicer").expect("build TT");
    let build: serde_json::Value = serde_json::from_str(&build_str).expect("parse build JSON");
    assert_eq!(build["lint"]["passed"], true);
    assert!(kernel
        .artifact_bytes("device_orcaslicer.3mf")
        .expect("3mf")
        .starts_with(b"PK\x03\x04"));
    assert!(kernel
        .artifact_bytes("device_dossier.pdf")
        .expect("pdf")
        .starts_with(b"%PDF-1.4"));
    assert!(kernel.artifact_bytes("carver_4axis.gcode").is_some());

    // TF sample + build
    let tf_str = kernel.load_sample("tf").expect("load TF");
    let tf: serde_json::Value = serde_json::from_str(&tf_str).expect("parse TF");
    assert_eq!(tf["qa"]["status"], "PASS");
    let tf_build_str = kernel
        .build(
            &serde_json::to_string(&tf["recipe"]).expect("tf recipe"),
            "prusaslicer",
        )
        .expect("build TF");
    let tf_build: serde_json::Value = serde_json::from_str(&tf_build_str).expect("parse TF build");
    assert_eq!(tf_build["lint"]["passed"], true);

    // AFO sample + build
    let afo_str = kernel.load_sample("afo").expect("load AFO");
    let afo: serde_json::Value = serde_json::from_str(&afo_str).expect("parse AFO");
    let afo_build_str = kernel
        .build(
            &serde_json::to_string(&afo["recipe"]).expect("afo recipe"),
            "orcaslicer",
        )
        .expect("build AFO");
    let afo_build: serde_json::Value =
        serde_json::from_str(&afo_build_str).expect("parse AFO build");
    assert_eq!(afo_build["lint"]["passed"], true);

    // STL upload roundtrip
    let raw = synthetic::generate_transtibial_limb(180.0, 54.0, 34.0, 35.0, true, false, 44, 64, 2);
    let stl_bytes = mesh::write_binary_stl_bytes(&raw.mesh, "upload test");
    let up_str = kernel.upload_stl_bytes(&stl_bytes).expect("upload STL");
    let up: serde_json::Value = serde_json::from_str(&up_str).expect("parse upload");
    assert_eq!(up["qa"]["status"], "PASS");
}
