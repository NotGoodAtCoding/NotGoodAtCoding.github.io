//! Dual-schema `.3mf` archive exporter (`orcaslicer` | `prusaslicer`) using pure-Rust ZIP.

use std::fmt::Write;

use crate::mesh::TriMesh;

fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn mesh_xml(obj_id: usize, name: &str, mesh: &TriMesh) -> String {
    let mut s = String::with_capacity(mesh.vertices.len() * 48 + mesh.faces.len() * 38 + 256);
    let _ = writeln!(
        s,
        "    <object id=\"{obj_id}\" name=\"{}\" type=\"model\">",
        xml_escape(name)
    );
    s.push_str("      <mesh>\n        <vertices>\n");
    for v in &mesh.vertices {
        let _ = writeln!(
            s,
            "          <vertex x=\"{:.4}\" y=\"{:.4}\" z=\"{:.4}\" />",
            v.x, v.y, v.z
        );
    }
    s.push_str("        </vertices>\n        <triangles>\n");
    for &[a, b, c] in &mesh.faces {
        let _ = writeln!(s, "          <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\" />");
    }
    s.push_str("        </triangles>\n      </mesh>\n    </object>");
    s
}

fn prusaslicer_config(infill_pct: u32, wall_loops: u32) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<config>\n\
  <object id=\"1\" instances_count=\"1\">\n\
    <metadata type=\"object\" key=\"name\" value=\"op_device_assembly\"/>\n\
    <volume firstid=\"0\" lastid=\"0\">\n\
      <metadata type=\"volume\" key=\"name\" value=\"device_shell\"/>\n\
    </volume>\n\
    <volume firstid=\"1\" lastid=\"1\">\n\
      <metadata type=\"volume\" key=\"name\" value=\"distal_reinforcement_modifier\"/>\n\
      <metadata type=\"volume\" key=\"modifier\" value=\"1\"/>\n\
      <metadata type=\"volume\" key=\"fill_density\" value=\"{infill_pct}%\"/>\n\
      <metadata type=\"volume\" key=\"perimeters\" value=\"{wall_loops}\"/>\n\
    </volume>\n\
  </object>\n\
</config>\n"
    )
}

fn orcaslicer_config(infill_pct: u32, wall_loops: u32) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<config>\n\
  <object id=\"1\">\n\
    <metadata key=\"name\" value=\"op_device_assembly\"/>\n\
    <part id=\"1\" subtype=\"normal_part\">\n\
      <metadata key=\"name\" value=\"device_shell\"/>\n\
      <metadata key=\"matrix\" value=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n\
    </part>\n\
    <part id=\"2\" subtype=\"modifier_part\">\n\
      <metadata key=\"name\" value=\"distal_reinforcement_modifier\"/>\n\
      <metadata key=\"sparse_infill_density\" value=\"{infill_pct}%\"/>\n\
      <metadata key=\"wall_loops\" value=\"{wall_loops}\"/>\n\
      <metadata key=\"matrix\" value=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n\
    </part>\n\
  </object>\n\
</config>\n"
    )
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Builds a valid uncompressed (STORED) ZIP archive from `(name, bytes)` entries.
fn build_stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();

    for &(name, data) in entries {
        let name_bytes = name.as_bytes();
        let offset = out.len() as u32;
        let checksum = crc32(data);
        let size = data.len() as u32;

        // Local file header (PK\x03\x04)
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0u16.to_le_bytes()); // mod date
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(data);

        // Central directory header (PK\x01\x02)
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        central.extend_from_slice(&0u16.to_le_bytes()); // mod time
        central.extend_from_slice(&0u16.to_le_bytes()); // mod date
        central.extend_from_slice(&checksum.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra len
        central.extend_from_slice(&0u16.to_le_bytes()); // comment len
        central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attr
        central.extend_from_slice(&0u32.to_le_bytes()); // external attr
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name_bytes);
    }

    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);

    // End of central directory (PK\x05\x06)
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // disk num
    out.extend_from_slice(&0u16.to_le_bytes()); // cd start disk
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

/// Builds slicer-targeted `.3mf` archive bytes containing device shell + modifier volume.
pub fn export_3mf_bytes(
    shell_mesh: &TriMesh,
    modifier_mesh: &TriMesh,
    target: &str,
    infill_pct: u32,
    wall_loops: u32,
) -> Result<Vec<u8>, String> {
    if target != "prusaslicer" && target != "orcaslicer" {
        return Err(format!(
            "Unsupported slicer target {target:?}; choose prusaslicer or orcaslicer"
        ));
    }

    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
  <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n\
  <Default Extension=\"config\" ContentType=\"text/xml\"/>\n\
</Types>\n";

    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
  <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n\
</Relationships>\n";

    let obj_shell = mesh_xml(1, "device_shell", shell_mesh);
    let obj_mod = mesh_xml(2, "distal_reinforcement_modifier", modifier_mesh);
    let model_xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
  <resources>\n\
{obj_shell}\n\
{obj_mod}\n\
  </resources>\n\
  <build>\n\
    <item objectid=\"1\" />\n\
    <item objectid=\"2\" />\n\
  </build>\n\
</model>\n"
    );

    let (cfg_path, cfg_xml) = if target == "prusaslicer" {
        (
            "Metadata/Slic3r_PE_model.config",
            prusaslicer_config(infill_pct, wall_loops),
        )
    } else {
        (
            "Metadata/model_settings.config",
            orcaslicer_config(infill_pct, wall_loops),
        )
    };

    Ok(build_stored_zip(&[
        ("[Content_Types].xml", content_types.as_bytes()),
        ("_rels/.rels", rels.as_bytes()),
        ("3D/3dmodel.model", model_xml.as_bytes()),
        (cfg_path, cfg_xml.as_bytes()),
    ]))
}
