#!/bin/bash
# Quick setup script for Android development build with ONNX Runtime

echo "🚀 Setting up ONNX Runtime for Android..."
echo ""

# Step 1: Prebuild
echo "📦 Step 1: Running prebuild to generate Android project..."
npx expo prebuild --clean -p android

if [ $? -ne 0 ]; then
    echo "❌ Prebuild failed!"
    exit 1
fi

echo "✅ Prebuild complete"
echo ""

# Step 2: Verify MainApplication.kt
echo "🔍 Step 2: Verifying MainApplication.kt..."
MAIN_APP="android/app/src/main/java/com/coughcare/app/MainApplication.kt"

if [ -f "$MAIN_APP" ]; then
    if grep -q "OnnxruntimePackage" "$MAIN_APP"; then
        echo "✅ OnnxruntimePackage found in MainApplication.kt"
    else
        echo "⚠️  OnnxruntimePackage NOT found - plugin may have failed"
        echo "   You may need to add it manually"
    fi
else
    echo "⚠️  MainApplication.kt not found - prebuild may have failed"
fi

echo ""
echo "📱 Step 3: Building development build..."
echo "   Run: npx expo run:android"
echo ""
echo "🎯 Step 4: Start dev client"
echo "   Run: npx expo start --dev-client"
echo ""
echo "✅ Setup complete! Build and test your app."


