use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

mod core;
use crate::core::categorical::CategoricalBinning;
use crate::core::numerical::NumericalBinning;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyNumBin {
    #[pyo3(get)]
    pub bin_id: usize,
    #[pyo3(get)]
    pub range: (f64, f64),
    #[pyo3(get)]
    pub pos: f64,
    #[pyo3(get)]
    pub neg: f64,
    #[pyo3(get)]
    pub woe: f64,
    #[pyo3(get)]
    pub iv: f64,
    #[pyo3(get)]
    pub is_missing: bool,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyCatBin {
    #[pyo3(get)]
    pub bin_id: usize,
    #[pyo3(get)]
    pub categories: Vec<String>,
    #[pyo3(get)]
    pub pos: f64,
    #[pyo3(get)]
    pub neg: f64,
    #[pyo3(get)]
    pub woe: f64,
    #[pyo3(get)]
    pub iv: f64,
    #[pyo3(get)]
    pub is_missing: bool,
}

#[pymethods]
impl NumericalBinning {
    #[new]
    pub fn pynew(max_bins: usize, initial_bins_count: usize) -> Self {
        Self::new(max_bins, initial_bins_count)
    }

    pub fn fit(
        &self,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<i32>,
    ) -> PyResult<Vec<PyNumBin>> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let results = self.execute_fit(x_view, y_view);
        let py_results = results
            .into_iter()
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
        Ok(py_results)
    }
}

#[pymethods]
impl CategoricalBinning {
    #[new]
    pub fn pynew(max_bins: usize) -> Self {
        Self::new(max_bins)
    }

    pub fn fit(&self, x: Vec<String>, y: Vec<i32>) -> PyResult<Vec<PyCatBin>> {
        let results = self.execute_fit(x, y);
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
