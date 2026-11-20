//! Pure-Rust WebAssembly clinical O&P geometry, rectification, shelling, and dual-CAM kernel.

pub mod carver_cam;
pub mod export_3mf;
pub mod linter;
pub mod math3d;
pub mod mesh;
pub mod recipe;
pub mod rectify;
pub mod repair;
pub mod scan_qa;
pub mod session;
pub mod shell;
pub mod synthetic;
pub mod wasm_api;

pub use crate::session::WorkstationKernel;
pub use crate::wasm_api::OpCoreWasm;
