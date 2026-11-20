//! 3D vector, rotation, Jacobi eigensolver, and 2D convex-hull primitives.

use serde::Deserialize;
use serde::Serialize;

/// 3D Cartesian point or vector in millimeters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Creates a new 3D vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Converts this vector into a `[x, y, z]` array.
    pub const fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Creates a vector from a `[x, y, z]` array.
    pub const fn from_array(arr: [f64; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

/// 2D point for cross-sectional convex hull and girth measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    /// Creates a new 2D point.
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

pub fn v_add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

pub fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

pub fn v_scale(a: Vec3, s: f64) -> Vec3 {
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

pub fn v_dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn v_cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

pub fn v_norm(a: Vec3) -> f64 {
    (a.x * a.x + a.y * a.y + a.z * a.z).sqrt()
}

pub fn v_dist(a: Vec3, b: Vec3) -> f64 {
    v_norm(v_sub(a, b))
}

pub fn v_unit(a: Vec3, fallback: Vec3) -> Vec3 {
    let n = v_norm(a);
    if n < 1e-12 {
        return fallback;
    }
    let inv = 1.0 / n;
    Vec3::new(a.x * inv, a.y * inv, a.z * inv)
}

/// Rotates `v` around unit `axis` passing through `origin` by `angle_rad`.
pub fn rodrigues_rotate(v: Vec3, axis: Vec3, angle_rad: f64, origin: Vec3) -> Vec3 {
    let k = v_unit(axis, Vec3::new(0.0, 0.0, 1.0));
    let p = v_sub(v, origin);
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    let k_dot_p = v_dot(k, p);
    let k_cross_p = v_cross(k, p);
    let one_minus_c = 1.0 - c;
    let rotated = Vec3::new(
        p.x * c + k_cross_p.x * s + k.x * k_dot_p * one_minus_c,
        p.y * c + k_cross_p.y * s + k.y * k_dot_p * one_minus_c,
        p.z * c + k_cross_p.z * s + k.z * k_dot_p * one_minus_c,
    );
    v_add(rotated, origin)
}

/// Computes eigenvalues (descending) and unit eigenvectors of a 3x3 symmetric matrix via Jacobi.
pub fn jacobi_eigen_3x3(cov: [[f64; 3]; 3], max_iter: usize) -> ([f64; 3], [Vec3; 3]) {
    let mut a = cov;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..max_iter {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0;
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p])
        };
        let c = theta.cos();
        let s = theta.sin();

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                let nrp = c * arp - s * arq;
                let nrq = s * arp + c * arq;
                a[r][p] = nrp;
                a[p][r] = nrp;
                a[r][q] = nrq;
                a[q][r] = nrq;
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    let mut pairs: [(f64, Vec3); 3] = [
        (
            a[0][0],
            v_unit(
                Vec3::new(v[0][0], v[1][0], v[2][0]),
                Vec3::new(1.0, 0.0, 0.0),
            ),
        ),
        (
            a[1][1],
            v_unit(
                Vec3::new(v[0][1], v[1][1], v[2][1]),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ),
        (
            a[2][2],
            v_unit(
                Vec3::new(v[0][2], v[1][2], v[2][2]),
                Vec3::new(0.0, 0.0, 1.0),
            ),
        ),
    ];
    pairs.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
    (
        [pairs[0].0, pairs[1].0, pairs[2].0],
        [pairs[0].1, pairs[1].1, pairs[2].1],
    )
}

/// C^2-continuous quintic smoothstep clamped to `[0.0, 1.0]`.
pub fn smoothstep5(t: f64) -> f64 {
    let c = t.clamp(0.0, 1.0);
    c * c * c * (c * (c * 6.0 - 15.0) + 10.0)
}

/// Returns the 2D convex hull in CCW order via Andrew's monotone chain.
pub fn convex_hull_2d(points: &[Vec2]) -> Vec<Vec2> {
    let mut pts: Vec<(i64, i64)> = points
        .iter()
        .map(|p| ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64))
        .collect();
    pts.sort_unstable();
    pts.dedup();

    let clean: Vec<Vec2> = pts
        .into_iter()
        .map(|(x, y)| Vec2::new(x as f64 * 1e-6, y as f64 * 1e-6))
        .collect();
    if clean.len() <= 2 {
        return clean;
    }

    let cross = |o: Vec2, a: Vec2, b: Vec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    let mut lower: Vec<Vec2> = Vec::with_capacity(clean.len());
    for &p in &clean {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<Vec2> = Vec::with_capacity(clean.len());
    for &p in clean.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Computes the perimeter of a 2D polygon.
pub fn polygon_perimeter_2d(poly: &[Vec2], closed: bool) -> f64 {
    if poly.len() < 2 {
        return 0.0;
    }
    let limit = if closed { poly.len() } else { poly.len() - 1 };
    let mut total = 0.0;
    for i in 0..limit {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % poly.len()];
        total += (p1.x - p0.x).hypot(p1.y - p0.y);
    }
    total
}

/// Computes the unsigned area of a closed 2D polygon via the shoelace formula.
pub fn polygon_area_2d(poly: &[Vec2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..poly.len() {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % poly.len()];
        sum += p0.x * p1.y - p1.x * p0.y;
    }
    0.5 * sum.abs()
}
