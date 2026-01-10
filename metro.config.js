// Metro bundler configuration
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add .onnx, .wasm, and .db files as assets
config.resolver.assetExts.push('onnx', 'wasm', 'db');

// Ensure wasm is NOT in sourceExts
config.resolver.sourceExts = config.resolver.sourceExts.filter(ext => ext !== 'wasm');

module.exports = config;
