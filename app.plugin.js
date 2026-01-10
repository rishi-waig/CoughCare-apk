const { withDangerousMod } = require('@expo/config-plugins');
const fs = require('fs');
const path = require('path');

/**
 * Expo config plugin to add OnnxruntimePackage to MainApplication.kt
 * This fixes the "Cannot read property 'install' of null" error on Android with Expo 54
 * 
 * Based on: https://github.com/microsoft/onnxruntime/issues/26796
 * Solution from: Reddit - onnxruntime-react-native with Expo 54
 */
function withOnnxRuntimePlugin(config) {
  // Use withDangerousMod to modify MainApplication.kt after prebuild
  return withDangerousMod(config, [
    'android',
    async (config) => {
      // Get the path to MainApplication.kt
      const packageName = config.android?.package || 'com.coughcare.app';
      const packagePath = packageName.split('.').join('/');
      const mainApplicationPath = path.join(
        config.modRequest.platformProjectRoot,
        'app/src/main/java',
        packagePath,
        'MainApplication.kt'
      );
      
      console.log('[ONNX Plugin] Looking for MainApplication.kt at:', mainApplicationPath);
      
      if (fs.existsSync(mainApplicationPath)) {
        let mainApplication = fs.readFileSync(mainApplicationPath, 'utf8');
        const originalContent = mainApplication;
        mainApplication = modifyMainApplicationContent(mainApplication);
        
        if (mainApplication !== originalContent) {
          fs.writeFileSync(mainApplicationPath, mainApplication);
          console.log('[ONNX Plugin] Successfully added OnnxruntimePackage to MainApplication.kt');
        } else {
          console.log('[ONNX Plugin] OnnxruntimePackage already present in MainApplication.kt');
        }
      } else {
        console.warn('[ONNX Plugin] MainApplication.kt not found at:', mainApplicationPath);
        console.warn('[ONNX Plugin] This is normal if prebuild hasn\'t run yet. Run: npx expo prebuild -p android');
      }
      
      return config;
    },
  ]);
}

function modifyMainApplication(config) {
  if (config.modResults && config.modResults.contents) {
    config.modResults.contents = modifyMainApplicationContent(config.modResults.contents);
  }
  return config;
}

function modifyMainApplicationContent(mainApplication) {
    // Add import if not present
    if (!mainApplication.includes('import ai.onnxruntime.reactnative.OnnxruntimePackage')) {
      // Find the last import statement (Kotlin uses semicolons, but some may not)
      const importRegex = /(import\s+[^\n]+)/g;
      const imports = [...mainApplication.matchAll(importRegex)];
      
      if (imports.length > 0) {
        const lastImport = imports[imports.length - 1];
        const insertIndex = lastImport.index + lastImport[0].length;
        // Add after the last import, with proper newline
        mainApplication = 
          mainApplication.slice(0, insertIndex) +
          '\nimport ai.onnxruntime.reactnative.OnnxruntimePackage' +
          mainApplication.slice(insertIndex);
      } else {
        // Fallback: add after expo imports
        const expoImportMatch = mainApplication.match(/import expo\.modules\.ReactNativeHostWrapper/);
        if (expoImportMatch) {
          const insertIndex = expoImportMatch.index + expoImportMatch[0].length;
          mainApplication = 
            mainApplication.slice(0, insertIndex) +
            '\n\nimport ai.onnxruntime.reactnative.OnnxruntimePackage' +
            mainApplication.slice(insertIndex);
        }
      }
    }
    
    // Add OnnxruntimePackage() to packages list if not present
    if (!mainApplication.includes('add(OnnxruntimePackage())')) {
      // Find the PackageList(this).packages.apply { block
      const packagesRegex = /(PackageList\(this\)\.packages\.apply\s*\{)/;
      const match = mainApplication.match(packagesRegex);
      
      if (match) {
        // Find the closing brace of the apply block
        let startIndex = match.index + match[0].length;
        let braceCount = 1;
        let endIndex = startIndex;
        
        while (endIndex < mainApplication.length && braceCount > 0) {
          if (mainApplication[endIndex] === '{') braceCount++;
          if (mainApplication[endIndex] === '}') braceCount--;
          endIndex++;
        }
        
        if (endIndex > startIndex) {
          // Insert before the closing brace
          const beforeBrace = mainApplication.slice(0, endIndex - 1);
          const afterBrace = mainApplication.slice(endIndex - 1);
          
          // Check if we need a newline
          const needsNewline = !beforeBrace.trim().endsWith('{') && 
                               !beforeBrace.trim().endsWith('//') &&
                               !beforeBrace.trim().endsWith('(');
          
          const indent = '              '; // 14 spaces to match Expo's style
          
          mainApplication = 
            beforeBrace +
            (needsNewline ? '\n' : '') +
            indent + 'add(OnnxruntimePackage())\n' +
            afterBrace;
        }
      }
    }
    
    return mainApplication;
}

module.exports = withOnnxRuntimePlugin;

