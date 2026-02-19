use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bin {
    pub left: f64,
    pub right: f64,
    pub pos: f64,
    pub neg: f64,
    pub count: f64,
    pub woe: f64,
    pub categories: Vec<String>,
}
