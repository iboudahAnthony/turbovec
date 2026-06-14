"""Python shim for running turbovec in Pyodide.

This module provides a thin Python wrapper that calls the turbovec-wasm
JavaScript exports (wasm-bindgen) via Pyodide's JS bridge. It preserves
a small, Pythonic API: TurboVec (new, add, query, len).

Usage (in-browser with Pyodide):

# In JS before running Python:
#  - load and initialize the turbovec-wasm pkg (init function)
#  - expose `TurboVecWasm` on the global `window` object

# In Python (Pyodide):
# from turbovec_python_pyodide import TurboVec
# tv = TurboVec(128, 4)
# tv.add(numpy_array)
# res = tv.query(query_array, k=10)
"""

from js import TurboVecWasm, Float32Array, Object  # provided by main.js demo loader
import pyodide
import numpy as np


class TurboVec:
    """Python-facing wrapper around the JS TurboVecWasm class.

    Methods:
    - __init__(dim, bit_width)
    - add(array_like)
    - query(array_like, k)
    - len()
    """

    def __init__(self, dim: int, bit_width: int):
        # Assumes the JS `TurboVecWasm` constructor is available globally.
        try:
            self._inner = TurboVecWasm.new(dim, bit_width)
        except Exception as e:
            raise RuntimeError(f"failed to construct TurboVecWasm: {e}")

    def add(self, arr):
        """Add vectors. Accepts a 1D or 2D array-like. For 2D array, it must be
        shape (n, dim) and will be flattened to a float32 buffer.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            flat = a.ravel()
        elif a.ndim == 1:
            flat = a
        else:
            raise ValueError("array must be 1D or 2D")
        # Transfer to JS Float32Array without intermediate copy when possible
        js_array = pyodide.to_js(flat, dict(target=Float32Array))
        # call JS method
        self._inner.add_vectors(js_array)

    def query(self, arr, k: int):
        """Query with a 1D or 2D array-like of queries. Returns a list of
        lists of dicts: [{"index": int, "score": float}, ...]
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            flat = a.ravel()
        elif a.ndim == 1:
            flat = a
        else:
            raise ValueError("query array must be 1D or 2D")
        js_q = pyodide.to_js(flat, dict(target=Float32Array))
        res = self._inner.query(js_q, k)
        # res is a JS Array of Arrays of JS Objects. Convert to Python list.
        py_res = []
        for i in range(res.length):
            row = res[i]
            py_row = []
            for j in range(row.length):
                obj = row[j]
                idx = int(obj.index) if hasattr(obj, 'index') else int(obj.get('index'))
                score = float(obj.score) if hasattr(obj, 'score') else float(obj.get('score'))
                py_row.append({"index": idx, "score": score})
            py_res.append(py_row)
        return py_res

    def len(self):
        return int(self._inner.len)
