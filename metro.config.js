// Metro bundler configuration - optimized for Node.js file watcher
const { getDefaultConfig } = require('expo/metro-config');

// Force disable watchman
process.env.WATCHMAN_DISABLE = '1';
delete process.env.WATCHMAN_SOCK;

const config = getDefaultConfig(__dirname);

// Optimize file watching - only watch source files
config.watchFolders = [__dirname];
config.resolver = {
  ...config.resolver,
  // Add .onnx files as assets so they can be bundled
  assetExts: [...(config.resolver?.assetExts || []), 'onnx'],
  blockList: [
    // Exclude most of node_modules from watching
    /node_modules\/.*\/node_modules\/.*/,
    /\.git\/.*/,
    /uploaded_audio\/.*/,
    /__pycache__\/.*/,
    /\.expo\/.*/,
  ],
};

// Configure watcher (removed invalid usePolling option)
config.watcher = {
  ...config.watcher,
  healthCheck: {
    enabled: true,
  },
};

module.exports = config;
