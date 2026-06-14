use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Object};

/// Thin wasm-bindgen wrapper around turbovec::TurboQuantIndex
/// Exposes a JS-friendly API: constructor, add_vectors, query

#[wasm_bindgen]
pub struct TurboVecWasm {
    inner: turbovec::TurboQuantIndex,
}

#[wasm_bindgen]
impl TurboVecWasm {
    /// Construct a new index with known dimension and bit width.
    /// dim: vector dimensionality (must be multiple of 8)
    /// bit_width: 2,3 or 4
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, bit_width: usize) -> Result<TurboVecWasm, JsValue> {
        match turbovec::TurboQuantIndex::new(dim, bit_width) {
            Ok(idx) => Ok(TurboVecWasm { inner: idx }),
            Err(e) => Err(JsValue::from_str(&format!("construct error: {:?}", e))),
        }
    }

    /// Add a flat Float32Array of length n * dim. Panics / errors propagate as exceptions.
    #[wasm_bindgen]
    pub fn add_vectors(&mut self, data: &Float32Array) -> Result<(), JsValue> {
        let len = data.length() as usize;
        // copy into Rust Vec<f32>
        let mut buf = vec![0f32; len];
        data.copy_to(&mut buf[..]);
        // call add (assumes dim was provided at construction)
        self.inner.add(&buf);
        Ok(())
    }

    /// Query with a Float32Array of length nq * dim. Returns an Array where each
    /// element is an Array of result objects { index: i64, score: f32 } for that query.
    #[wasm_bindgen]
    pub fn query(&self, q: &Float32Array, k: usize) -> Result<Array, JsValue> {
        let len = q.length() as usize;
        let mut buf = vec![0f32; len];
        q.copy_to(&mut buf[..]);
        let results = self.inner.search(&buf, k);

        let out = Array::new();
        let nq = results.nq;
        let k_eff = results.k;
        for qi in 0..nq {
            let row = Array::new();
            for j in 0..k_eff {
                let idx = results.indices[qi * k_eff + j];
                let score = results.scores[qi * k_eff + j];
                let obj = Object::new();
                js_sys::Reflect::set(&obj, &JsValue::from_str("index"), &JsValue::from_f64(idx as f64))?;
                js_sys::Reflect::set(&obj, &JsValue::from_str("score"), &JsValue::from_f64(score as f64))?;
                row.push(&obj);
            }
            out.push(&row);
        }
        Ok(out)
    }

    /// Return the number of vectors in the index
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Prepare caches (optional): rotation/centroids/blocked layout
    #[wasm_bindgen]
    pub fn prepare(&self) {
        self.inner.prepare();
    }
}
