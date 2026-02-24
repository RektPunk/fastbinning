use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod core;
use crate::core::categorical::CatBin;
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
    pub indices: Vec<i32>,
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
    pub _bins: Option<Vec<NumBin>>,
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
        self._bins = Some(results);
        self.bins()
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let _bins = self._bins.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NotFittedError: Call fit() before transform()",
            )
        })?;
        let x_view = x.as_array();
        let output: Vec<f64> = self.execute_transform(x_view, _bins);
        Ok(output.into_pyarray(py))
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.fit(x.clone(), y)?;
        self.transform(py, x)
    }

    #[getter]
    pub fn bins(&self) -> PyResult<Vec<PyNumBin>> {
        let bins = self._bins.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NotFittedError: Call fit() before accessing 'bins'",
            )
        })?;
        let py_results = bins
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
        Ok(py_results)
    }
}

#[pyclass]
pub struct CategoricalBinning {
    pub max_bins: usize,
    pub min_bin_pct: f64,
    pub _bins: Option<Vec<CatBin>>,
}

#[pymethods]
impl CategoricalBinning {
    #[new]
    pub fn pynew(max_bins: usize, min_bin_pct: f64) -> Self {
        Self::new(max_bins, min_bin_pct)
    }

    pub fn fit(
        &mut self,
        x: PyReadonlyArray1<i32>,
        y: PyReadonlyArray1<i32>,
    ) -> PyResult<Vec<PyCatBin>> {
        let x_slice = x.as_slice().expect("x array must be contiguous");
        let y_slice = y.as_slice().expect("y array must be contiguous");
        let results = self.execute_fit(x_slice, y_slice);
        self._bins = Some(results);
        self.bins()
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let _bins = self._bins.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NotFittedError: Call fit() before transform()",
            )
        })?;
        let x_view = x.as_slice().expect("x array must be contiguous");
        let output: Vec<f64> = self.execute_transform(x_view, _bins);
        Ok(output.into_pyarray(py))
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i32>,
        y: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.fit(x.clone(), y)?;
        self.transform(py, x)
    }

    #[getter]
    pub fn bins(&self) -> PyResult<Vec<PyCatBin>> {
        let bins = self._bins.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "NotFittedError: Call fit() before accessing 'bins'",
            )
        })?;

        let py_results = bins
            .iter()
            .map(|b| PyCatBin {
                bin_id: b.bin_id,
                indices: b.indices.clone(),
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
