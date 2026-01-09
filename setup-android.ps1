# Quick setup script for Android development build with ONNX Runtime (PowerShell)

Write-Host "🚀 Setting up ONNX Runtime for Android..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Prebuild
Write-Host "📦 Step 1: Running prebuild to generate Android project..." -ForegroundColor Yellow
npx expo prebuild --clean -p android

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Prebuild failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Prebuild complete" -ForegroundColor Green
Write-Host ""

# Step 2: Verify MainApplication.kt
Write-Host "🔍 Step 2: Verifying MainApplication.kt..." -ForegroundColor Yellow
$MAIN_APP = "android\app\src\main\java\com\coughcare\app\MainApplication.kt"

if (Test-Path $MAIN_APP) {
    $content = Get-Content $MAIN_APP -Raw
    if ($content -match "OnnxruntimePackage") {
        Write-Host "✅ OnnxruntimePackage found in MainApplication.kt" -ForegroundColor Green
    } else {
        Write-Host "⚠️  OnnxruntimePackage NOT found - plugin may have failed" -ForegroundColor Yellow
        Write-Host "   You may need to add it manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  MainApplication.kt not found - prebuild may have failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📱 Step 3: Building development build..." -ForegroundColor Cyan
Write-Host "   Run: npx expo run:android" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Step 4: Start dev client" -ForegroundColor Cyan
Write-Host "   Run: npx expo start --dev-client" -ForegroundColor White
Write-Host ""
Write-Host "✅ Setup complete! Build and test your app." -ForegroundColor Green


