use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as GL, WebGlProgram, WebGlShader,
};

const VERT_SRC: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;

uniform float u_time;
uniform float u_aspect;

out vec3 v_color;

void main() {
    float c = cos(u_time * 0.8);
    float s = sin(u_time * 0.8);
    vec2 rotated = vec2(
        a_pos.x * c - a_pos.y * s,
        a_pos.x * s + a_pos.y * c
    );
    gl_Position = vec4(rotated.x / u_aspect, rotated.y, 0.0, 1.0);
    v_color = a_color;
}
"#;

const FRAG_SRC: &str = r#"#version 300 es
precision highp float;

in vec3 v_color;
uniform float u_time;

out vec4 out_color;

void main() {
    float pulse = 0.85 + 0.15 * sin(u_time * 2.0);
    out_color = vec4(v_color * pulse, 1.0);
}
"#;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no global window")?;
    let document = window.document().ok_or("no document")?;
    let canvas = document
        .get_element_by_id("canvas")
        .ok_or("missing #canvas element")?
        .dyn_into::<HtmlCanvasElement>()?;

    let gl = canvas
        .get_context("webgl2")?
        .ok_or("WebGL2 is not supported in this browser")?
        .dyn_into::<GL>()?;

    let vert_shader = compile_shader(&gl, GL::VERTEX_SHADER, VERT_SRC)?;
    let frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, FRAG_SRC)?;
    let program = link_program(&gl, &vert_shader, &frag_shader)?;
    gl.use_program(Some(&program));

    // Interleaved [x, y, r, g, b] for an equilateral triangle centered at origin.
    #[rustfmt::skip]
    let vertices: [f32; 15] = [
         0.0,    0.65,  0.38, 0.65, 0.98,
        -0.563, -0.325, 0.96, 0.45, 0.71,
         0.563, -0.325, 0.20, 0.83, 0.60,
    ];

    let vao = gl.create_vertex_array().ok_or("failed to create VAO")?;
    gl.bind_vertex_array(Some(&vao));

    let vbo = gl.create_buffer().ok_or("failed to create VBO")?;
    gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vbo));

    // Safe: buffer_data_with_array_buffer_view copies immediately into GPU memory
    // before any WASM memory allocation can occur.
    unsafe {
        let view = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
    }

    let stride = (5 * std::mem::size_of::<f32>()) as i32;
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_with_i32(0, 2, GL::FLOAT, false, stride, 0);

    gl.enable_vertex_attrib_array(1);
    gl.vertex_attrib_pointer_with_i32(
        1,
        3,
        GL::FLOAT,
        false,
        stride,
        (2 * std::mem::size_of::<f32>()) as i32,
    );

    let u_time = gl.get_uniform_location(&program, "u_time");
    let u_aspect = gl.get_uniform_location(&program, "u_aspect");

    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let win_clone = window.clone();

    *g.borrow_mut() = Some(Closure::new(move |timestamp_ms: f64| {
        let dpr = win_clone.device_pixel_ratio();
        let width = ((canvas.client_width() as f64) * dpr).max(1.0) as u32;
        let height = ((canvas.client_height() as f64) * dpr).max(1.0) as u32;
        if canvas.width() != width || canvas.height() != height {
            canvas.set_width(width);
            canvas.set_height(height);
        }

        gl.viewport(0, 0, width as i32, height as i32);
        gl.clear_color(0.06, 0.07, 0.09, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);

        let t = (timestamp_ms * 0.001) as f32;
        let aspect = (width as f32) / (height as f32);
        gl.uniform1f(u_time.as_ref(), t);
        gl.uniform1f(u_aspect.as_ref(), aspect);

        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        request_animation_frame(&win_clone, f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(&window, g.borrow().as_ref().unwrap());
    Ok(())
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut(f64)>) {
    window
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register requestAnimationFrame");
}

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or_else(|| String::from("unable to create shader object"))?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl
        .get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("unknown shader compile error")))
    }
}

fn link_program(
    gl: &GL,
    vert: &WebGlShader,
    frag: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = gl
        .create_program()
        .ok_or_else(|| String::from("unable to create shader program"))?;
    gl.attach_shader(&program, vert);
    gl.attach_shader(&program, frag);
    gl.link_program(&program);

    if gl
        .get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("unknown program link error")))
    }
}
