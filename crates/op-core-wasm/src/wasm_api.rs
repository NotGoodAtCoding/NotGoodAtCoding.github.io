//! Thin `wasm-bindgen` JavaScript/WebGL facade over `WorkstationKernel`.

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

use crate::session::WorkstationKernel;

/// WebAssembly handle for the stateful O&P clinical CAD/CAM kernel.
#[wasm_bindgen]
pub struct OpCoreWasm {
    inner: WorkstationKernel,
}

impl Default for OpCoreWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl OpCoreWasm {
    /// Creates a new in-browser O&P workstation kernel instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WorkstationKernel::new(),
        }
    }

    /// Returns the compiled kernel version string.
    pub fn version(&self) -> String {
        crate::recipe::KERNEL_VERSION.to_string()
    }

    /// Loads a procedural clinical limb/foot sample (`"tt"`, `"tf"`, or `"afo"`).
    pub fn load_sample(&mut self, device: &str) -> Result<String, JsValue> {
        self.inner
            .load_sample(device)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Loads uploaded binary or ASCII STL bytes and runs chairside Scan QA.
    pub fn upload_stl(&mut self, stl_bytes: &[u8]) -> Result<String, JsValue> {
        self.inner
            .upload_stl_bytes(stl_bytes)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Evaluates an `.oprecipe.json` string and returns rectified mesh + radial deltas JSON.
    pub fn rectify(&mut self, recipe_json: &str) -> Result<String, JsValue> {
        self.inner
            .rectify(recipe_json)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Computes structured revision diff JSON vs intake baseline.
    pub fn diff_recipe(&self, recipe_json: &str) -> Result<String, JsValue> {
        self.inner
            .diff_recipe(recipe_json)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Compiles watertight shell, `.3mf`, CNC carver mold/G-code, safety lint, and PDF dossier.
    pub fn build(&mut self, recipe_json: &str, target: &str) -> Result<String, JsValue> {
        self.inner
            .build(recipe_json, target)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Fetches a generated export artifact (`device_shell.stl`, `device_orcaslicer.3mf`, etc.).
    pub fn get_artifact_bytes(&self, filename: &str) -> Option<Vec<u8>> {
        self.inner
            .artifact_bytes(filename)
            .map(|slice| slice.to_vec())
    }
}
