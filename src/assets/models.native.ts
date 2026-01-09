/**
 * ONNX Model Assets Registry (React Native only)
 * 
 * This file is only loaded on React Native platforms.
 * Web uses public/models/ folder directly.
 */

export const MODEL_ASSETS = {
  preprocessing: require('../../assets/models/cough_preprocessing.onnx'),
  detector: require('../../assets/models/cough_detector_int8.onnx'),
};


