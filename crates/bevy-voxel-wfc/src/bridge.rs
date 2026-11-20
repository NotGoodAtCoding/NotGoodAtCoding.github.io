//! Thread-safe command queue and status channel bridging raw HTML/JS controls
//! to the running Bevy WebGL2 app via `wasm-bindgen`.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

/// Commands pushed from HTML/JS controls and drained each Bevy frame.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Variants are constructed from `#[wasm_bindgen]` exports on `wasm32`.
pub enum BridgeCmd {
    SetSeed { seed: String, animate: bool },
    SetAnimate(bool),
    FinishNow,
    TogglePause,
}

fn cmd_queue() -> &'static Mutex<VecDeque<BridgeCmd>> {
    static QUEUE: OnceLock<Mutex<VecDeque<BridgeCmd>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn status_cell() -> &'static Mutex<String> {
    static STATUS: OnceLock<Mutex<String>> = OnceLock::new();
    STATUS.get_or_init(|| Mutex::new(String::new()))
}

fn seed_cell() -> &'static Mutex<String> {
    static SEED: OnceLock<Mutex<String>> = OnceLock::new();
    SEED.get_or_init(|| Mutex::new(String::new()))
}

/// Enqueues a bridge command for the Bevy update loop to process on the next frame.
#[cfg(any(target_arch = "wasm32", test))]
pub fn push_cmd(cmd: BridgeCmd) {
    if let Ok(mut queue) = cmd_queue().lock() {
        queue.push_back(cmd);
    }
}

/// Drains all pending bridge commands into a `Vec` for Bevy to handle.
pub fn drain_cmds() -> Vec<BridgeCmd> {
    match cmd_queue().lock() {
        Ok(mut queue) => queue.drain(..).collect(),
        Err(_) => Vec::new(),
    }
}

/// Updates the shared status string readable from JS via [`wfc_get_status`].
pub fn set_status(status: String) {
    if let Ok(mut slot) = status_cell().lock() {
        *slot = status;
    }
}

/// Updates the active seed label readable from JS via [`wfc_get_seed`].
pub fn set_seed_label(seed: String) {
    if let Ok(mut slot) = seed_cell().lock() {
        *slot = seed;
    }
}

/// Reads the latest status string published by Bevy.
#[cfg(any(target_arch = "wasm32", test))]
pub fn get_status() -> String {
    match status_cell().lock() {
        Ok(slot) => slot.clone(),
        Err(_) => String::new(),
    }
}

/// Reads the active seed label published by Bevy.
#[cfg(any(target_arch = "wasm32", test))]
pub fn get_seed_label() -> String {
    match seed_cell().lock() {
        Ok(slot) => slot.clone(),
        Err(_) => String::new(),
    }
}

/// Starts generating a new diorama from `seed`, either animated or instant.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_set_seed(seed: String, animate: bool) {
    push_cmd(BridgeCmd::SetSeed { seed, animate });
}

/// Updates whether subsequent generations animate step by step.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_set_animate(animate: bool) {
    push_cmd(BridgeCmd::SetAnimate(animate));
}

/// Finishes the active animated WFC solve immediately.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_finish_now() {
    push_cmd(BridgeCmd::FinishNow);
}

/// Toggles pause/resume of the active animated WFC solve.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_toggle_pause() {
    push_cmd(BridgeCmd::TogglePause);
}

/// Returns the latest plain-text solver status summary.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_get_status() -> String {
    get_status()
}

/// Returns the active seed string so HTML inputs stay in sync with keyboard shortcuts.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn wfc_get_seed() -> String {
    get_seed_label()
}

#[cfg(test)]
mod tests {
    use super::{
        drain_cmds, get_seed_label, get_status, push_cmd, set_seed_label, set_status, BridgeCmd,
    };

    #[test]
    fn bridge_roundtrips_commands_and_status() {
        let _ = drain_cmds();
        push_cmd(BridgeCmd::SetSeed {
            seed: "tarn-99".to_owned(),
            animate: true,
        });
        push_cmd(BridgeCmd::FinishNow);

        let cmds = drain_cmds();
        assert_eq!(
            cmds,
            vec![
                BridgeCmd::SetSeed {
                    seed: "tarn-99".to_owned(),
                    animate: true,
                },
                BridgeCmd::FinishNow,
            ]
        );
        assert!(drain_cmds().is_empty());

        set_status("Seed: tarn-99 | Complete".to_owned());
        assert_eq!(get_status(), "Seed: tarn-99 | Complete");

        set_seed_label("tarn-99".to_owned());
        assert_eq!(get_seed_label(), "tarn-99");
    }
}
