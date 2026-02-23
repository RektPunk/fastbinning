use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod core;
use crate::core::numerical::NumBin;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyNumBin {
    #[pyo3(get)]
    pub bin_id: usize,
    #[pyo3(get)]
    pub range: (f64, f64),
    #[pyo3(get)]
    pub pos: i32,
    #[pyo3(get)]
    pub neg: i32,
    #[pyo3(get)]
    pub woe: f64,
    #[pyo3(get)]
    pub iv: f64,
    #[pyo3(get)]
    pub is_missing: bool,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyCatBin {
    #[pyo3(get)]
    pub bin_id: usize,
    #[pyo3(get)]
    pub categories: Vec<String>,
    #[pyo3(get)]
    pub pos: i32,
    #[pyo3(get)]
    pub neg: i32,
    #[pyo3(get)]
    pub woe: f64,
    #[pyo3(get)]
    pub iv: f64,
    #[pyo3(get)]
    pub is_missing: bool,
}

#[pyclass]
pub struct NumericalBinning {
    pub max_bins: usize,
    pub min_bin_pct: f64,
    pub bins: Option<Vec<NumBin>>,
}

#[pymethods]
impl NumericalBinning {
    #[new]
    pub fn pynew(max_bins: usize, min_bin_pct: f64) -> Self {
        Self::new(max_bins, min_bin_pct)
    }

    pub fn fit(
        &mut self,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<i32>,
    ) -> PyResult<Vec<PyNumBin>> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let results = self.execute_fit(x_view, y_view);
        let py_results: Vec<PyNumBin> = results
            .iter()
            .map(|b| PyNumBin {
                bin_id: b.bin_id,
                range: b.range,
                pos: b.pos,
                neg: b.neg,
                woe: b.woe,
                iv: b.iv,
                is_missing: b.is_missing,
            })
            .collect();
        self.bins = Some(results);
        Ok(py_results)
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let bins = self.bins.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NotFittedError: Call fit() before transform()",
            )
        })?;
        let x_view = x.as_array();
        let output: Vec<f64> = self.execute_transform(x_view, bins);
        Ok(output.into_pyarray(py))
    }
}

#[pyclass]
pub struct CategoricalBinning {
    pub max_bins: usize,
    pub min_bin_pct: f64,
}

#[pymethods]
impl CategoricalBinning {
    #[new]
    pub fn pynew(max_bins: usize, min_bin_pct: f64) -> Self {
        Self::new(max_bins, min_bin_pct)
    }

    pub fn fit(
        &self,
        x: PyReadonlyArray1<i32>,
        y: PyReadonlyArray1<i32>,
        category_names: Vec<String>,
    ) -> PyResult<Vec<PyCatBin>> {
        let x_slice = x.as_slice().expect("x array must be contiguous");
        let y_slice = y.as_slice().expect("y array must be contiguous");
        let results = self.execute_fit(x_slice, y_slice, category_names);
        let py_results = results
            .into_iter()
            .map(|b| PyCatBin {
                bin_id: b.bin_id,
                categories: b.categories,
                pos: b.pos,
                neg: b.neg,
                woe: b.woe,
                iv: b.iv,
                is_missing: b.is_missing,
            })
            .collect();
        Ok(py_results)
    }
}

#[pymodule]
fn fastbinning(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CategoricalBinning>()?;
    m.add_class::<NumericalBinning>()?;
    m.add_class::<PyCatBin>()?;
    m.add_class::<PyNumBin>()?;
    Ok(())
}
