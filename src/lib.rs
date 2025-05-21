use wasm_bindgen::prelude::*;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
use smartcore::linalg::basic::matrix::DenseMatrix;
use serde::{Serialize, Deserialize};
use serde_wasm_bindgen;
use std::convert::TryInto;
use js_sys::Uint32Array;

// For better debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

#[wasm_bindgen]
pub struct RandomForestModel {
    model: Option<RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>>,
}

#[derive(Serialize, Deserialize)]
pub struct PredictionResult {
    prediction: u32,
    // Without probabilities since they're not available in this version
    predicted_class: u32,
}

#[wasm_bindgen]
impl RandomForestModel {
    #[wasm_bindgen(constructor)]
    pub fn new(_n_trees: usize, _max_depth: usize, _min_samples_leaf: usize) -> Self {
        console_log!("Creating new RandomForestModel instance");
        
        // Start with no model - we'll create it when training
        RandomForestModel { model: None }
    }
    
    #[wasm_bindgen]
    pub fn train(&mut self, x_data: Vec<f64>, y_data: Vec<u32>, n_features: usize, n_trees: usize) -> bool {
        console_log!("Training model with {} samples, {} features", x_data.len() / n_features, n_features);
        
        if x_data.is_empty() || x_data.len() % n_features != 0 {
            console_log!("Invalid input dimensions: data length = {}, features = {}", x_data.len(), n_features);
            return false;
        }
        
        let n_samples = x_data.len() / n_features;
        
        if n_samples != y_data.len() {
            console_log!("Sample count mismatch: x samples = {}, y samples = {}", n_samples, y_data.len());
            return false;
        }
        
        // Print first few samples for debugging
        for i in 0..std::cmp::min(3, n_samples) {
            let mut sample = String::new();
            for j in 0..n_features {
                sample.push_str(&format!("{:.2} ", x_data[i * n_features + j]));
            }
            console_log!("Sample {}: {} -> class {}", i, sample, y_data[i]);
        }
        
        // Convert flat array to 2D array for from_2d_array
        let mut x_2d: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut row: Vec<f64> = Vec::with_capacity(n_features);
            for j in 0..n_features {
                row.push(x_data[i * n_features + j]);
            }
            x_2d.push(row);
        }
        
        // Convert to expected format for from_2d_array
        let x_refs: Vec<&[f64]> = x_2d.iter().map(|v| v.as_slice()).collect();
        
        // Create a matrix from the samples
        let x = DenseMatrix::from_2d_array(&x_refs);
        
        // Make sure tree parameters are within valid ranges
        let n_trees_u16: u16 = match n_trees.try_into() {
            Ok(val) => val,
            Err(_) => {
                console_log!("n_trees too large, using default of 10");
                10
            }
        };
        
        // Default values for max_depth and min_samples_leaf
        let max_depth_u16: u16 = 5;
        
        // We'll use sqrt(n_features) for the m parameter (standard practice)
        let m_value = (f64::sqrt(n_features as f64) as usize).max(1);
        console_log!("Using m_value = {}", m_value);
        
        // Create parameters with safe defaults
        let parameters = RandomForestClassifierParameters::default()
            .with_n_trees(n_trees_u16)
            .with_max_depth(max_depth_u16)
            .with_min_samples_leaf(1)
            .with_m(m_value);
            
        // Fit the model
        match RandomForestClassifier::fit(&x, &y_data, parameters) {
            Ok(trained_model) => {
                console_log!("Model trained successfully");
                self.model = Some(trained_model);
                true
            },
            Err(e) => {
                console_log!("Error training model: {:?}", e);
                false
            }
        }
    }
    
    #[wasm_bindgen]
    pub fn predict(&self, x_data: Vec<f64>, n_features: usize) -> Result<JsValue, JsError> {
        if self.model.is_none() {
            return Err(JsError::new("Model not trained yet"));
        }
        
        if x_data.is_empty() || x_data.len() % n_features != 0 {
            return Err(JsError::new("Invalid input dimensions"));
        }
        
        let n_samples = x_data.len() / n_features;
        console_log!("Predicting for {} samples", n_samples);
        
        // Convert flat array to 2D array for from_2d_array
        let mut x_2d: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut row: Vec<f64> = Vec::with_capacity(n_features);
            for j in 0..n_features {
                row.push(x_data[i * n_features + j]);
            }
            x_2d.push(row);
        }
        
        // Print first sample for debugging
        if !x_2d.is_empty() {
            let sample = x_2d[0].iter().map(|x| format!("{:.2}", x)).collect::<Vec<String>>().join(", ");
            console_log!("First sample to predict: [{}]", sample);
        }
        
        // Convert to expected format for from_2d_array
        let x_refs: Vec<&[f64]> = x_2d.iter().map(|v| v.as_slice()).collect();
        
        let x = DenseMatrix::from_2d_array(&x_refs);
        
        // Get predictions
        match self.model.as_ref().unwrap().predict(&x) {
            Ok(predictions) => {
                // Debug first few predictions
                for i in 0..std::cmp::min(predictions.len(), 3) {
                    console_log!("Prediction {}: class {}", i, predictions[i]);
                }
                
                // Since we're returning a Vec<u32>, we can convert it to a Uint32Array for JavaScript
                let result = Uint32Array::new_with_length(predictions.len() as u32);
                for (i, &pred) in predictions.iter().enumerate() {
                    result.set_index(i as u32, pred);
                }
                
                Ok(result.into())
            },
            Err(e) => {
                console_log!("Prediction error: {:?}", e);
                Err(JsError::new(&format!("Prediction error: {:?}", e)))
            },
        }
    }
    
    #[wasm_bindgen]
    pub fn predict_probabilities(&self, x_data: Vec<f64>, n_features: usize) -> Result<JsValue, JsError> {
        if self.model.is_none() {
            return Err(JsError::new("Model not trained yet"));
        }
        
        if x_data.is_empty() || x_data.len() % n_features != 0 {
            return Err(JsError::new("Invalid input dimensions"));
        }
        
        let n_samples = x_data.len() / n_features;
        console_log!("Predicting probabilities for {} samples", n_samples);
        
        // Convert flat array to 2D array for from_2d_array
        let mut x_2d: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut row: Vec<f64> = Vec::with_capacity(n_features);
            for j in 0..n_features {
                row.push(x_data[i * n_features + j]);
            }
            x_2d.push(row);
        }
        
        // Convert to expected format for from_2d_array
        let x_refs: Vec<&[f64]> = x_2d.iter().map(|v| v.as_slice()).collect();
        
        let x = DenseMatrix::from_2d_array(&x_refs);
        
        // Since predict_with_probabilities is not available in this version,
        // we'll just use predict and create dummy probability results
        match self.model.as_ref().unwrap().predict(&x) {
            Ok(predictions) => {
                // Create simplified result objects with predictions
                let results: Vec<PredictionResult> = predictions.iter()
                    .map(|&pred| {
                        PredictionResult {
                            prediction: pred,
                            predicted_class: pred,
                        }
                    })
                    .collect();
                
                Ok(serde_wasm_bindgen::to_value(&results)?)
            },
            Err(e) => {
                console_log!("Prediction error: {:?}", e);
                Err(JsError::new(&format!("Prediction error: {:?}", e)))
            },
        }
    }
    
    // Test if the model is trained
    #[wasm_bindgen]
    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }
}

// Helper function for testing
#[wasm_bindgen]
pub fn train_iris_example() -> Result<RandomForestModel, JsError> {
    console_log!("Creating iris example model");
    
    // Simulated Iris dataset (just for testing)
    let x_values = vec![
        5.1, 3.5, 1.4, 0.2, 
        4.9, 3.0, 1.4, 0.2, 
        4.7, 3.2, 1.3, 0.2, 
        4.6, 3.1, 1.5, 0.2,
        5.0, 3.6, 1.4, 0.2, 
        5.4, 3.9, 1.7, 0.4, 
        4.6, 3.4, 1.4, 0.3, 
        5.0, 3.4, 1.5, 0.2,
        4.4, 2.9, 1.4, 0.2, 
        4.9, 3.1, 1.5, 0.1, 
        7.0, 3.2, 4.7, 1.4, 
        6.4, 3.2, 4.5, 1.5
    ];
    
    let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    
    let mut model = RandomForestModel::new(10, 5, 1);
    if !model.train(x_values, y, 4, 10) {
        return Err(JsError::new("Failed to train iris example model"));
    }
    
    Ok(model)
}