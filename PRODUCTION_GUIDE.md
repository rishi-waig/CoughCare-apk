# 🚀 CoughCare Production Deployment Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Running the Application](#running-the-application)
6. [Important Files](#important-files)
7. [Files to Delete](#files-to-delete)

---

## Prerequisites

### Required Software
- **Docker** (version 20.10+) and **Docker Compose** (version 2.0+)
- **Node.js** (version 18+) and **npm** (version 9+)
- **Expo CLI** (`npm install -g expo-cli`)

### Verify Installation
```bash
docker --version
docker-compose --version
node --version
npm --version
expo --version
```

---

## Project Structure

```
coughcare_waig 3/
├── backend_api_actual_model.py    # Flask API server
├── backup_best_model_*.pth        # Trained ML model
├── train_cough_detector_attention.py  # Model architecture (required by backend)
├── precompute_spectrograms.py      # Audio processor (required by backend)
├── requirements.txt                 # Python dependencies
├── Dockerfile.backend              # Backend Docker image
├── docker-compose.yml              # Docker orchestration
├── .dockerignore                   # Docker ignore file
│
├── src/                            # React Native/Expo frontend
│   ├── App.tsx                     # Main app component
│   ├── main.tsx                    # Entry point
│   ├── index.css                   # Global styles
│   │
│   ├── screens/                    # ✅ KEEP - All screen components
│   │   ├── HomeScreen.tsx
│   │   ├── ConsentScreen.tsx
│   │   ├── QuestionsScreen.tsx
│   │   ├── CoughRecorderScreen.tsx
│   │   ├── AnalyzingScreen.tsx
│   │   ├── ResultScreen.tsx
│   │   ├── TbResultScreen.tsx
│   │   └── ChatbotScreen.tsx
│   │
│   ├── components/                 # ✅ KEEP - Reusable components
│   │   └── AppHeader.tsx
│   │
│   ├── navigation/                 # ✅ KEEP - Navigation setup
│   │   └── AppNavigator.tsx
│   │
│   └── utils/                      # ✅ KEEP - Utility functions
│       ├── api.ts                  # Backend API client
│       ├── audioRecorder.ts        # Audio recording (RN/Web)
│       └── storage.ts              # AsyncStorage wrapper
│
├── public/                         # ✅ KEEP - Static assets
│   ├── logo.png
│   ├── lungs.png
│   └── samples/                    # Sample audio files
│
├── uploaded_audio/                # ✅ KEEP - Audio upload directory (empty)
│
├── package.json                    # Frontend dependencies
├── app.json                        # Expo configuration
├── babel.config.js                 # Babel config
├── metro.config.js                 # Metro bundler config
├── tsconfig.json                   # TypeScript config
└── tsconfig.app.json               # TypeScript app config
```

---

## Backend Setup

### Step 1: Prepare Backend Files

Ensure these files exist in the root directory:
- ✅ `backend_api_actual_model.py` - Flask API
- ✅ `backup_best_model_20251015_170801.pth` - Trained model
- ✅ `train_cough_detector_attention.py` - Model architecture (REQUIRED)
- ✅ `precompute_spectrograms.py` - Audio processor (REQUIRED)
- ✅ `requirements.txt` - Python dependencies
- ✅ `Dockerfile.backend` - Docker configuration
- ✅ `docker-compose.yml` - Docker orchestration

### Step 2: Build and Start Backend

```bash
# Navigate to project directory
cd "coughcare_waig 3"

# Build the Docker image
docker-compose build backend

# Start the backend service
docker-compose up -d backend

# Check if backend is running
docker-compose ps

# View backend logs
docker-compose logs -f backend

# Test backend health
curl http://localhost:5001/health
```

### Step 3: Verify Backend

Expected response from `/health`:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "mode": "ACTUAL_MODEL_INFERENCE"
}
```

### Backend Management Commands

```bash
# Stop backend
docker-compose stop backend

# Restart backend
docker-compose restart backend

# Stop and remove containers
docker-compose down

# Stop, remove containers, and volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build backend
```

---

## Frontend Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd "coughcare_waig 3"

# Install npm dependencies
npm install

# If you encounter peer dependency issues
npm install --legacy-peer-deps
```

### Step 2: Configure API Endpoint

Edit `src/utils/api.ts` and ensure the API base URL is correct:

```typescript
// For development (localhost)
const API_BASE_URL = 'http://localhost:5001';

// For production, update to your backend URL
// const API_BASE_URL = 'https://your-backend-domain.com';
```

### Step 3: Start Frontend

```bash
# Start Expo development server
npx expo start

# Or for web only
npx expo start --web

# Or for Android only
npx expo start --android
```

### Step 4: Access Frontend

- **Web**: Open `http://localhost:19006` (or the URL shown in terminal)
- **Android**: Scan QR code with Expo Go app
- **iOS**: Scan QR code with Camera app (if on same network)

---

## Running the Application

### Complete Setup (Backend + Frontend)

#### Terminal 1 - Backend:
```bash
cd "coughcare_waig 3"
docker-compose up backend
```

#### Terminal 2 - Frontend:
```bash
cd "coughcare_waig 3"
npx expo start
```

### Production Deployment

#### Backend (Docker):
```bash
# Build and start
docker-compose up -d --build backend

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

#### Frontend (Expo):
```bash
# Build for production
npx expo build:web        # Web build
npx expo build:android    # Android APK/AAB
npx expo build:ios        # iOS build (requires Apple Developer account)

# Or use EAS Build (recommended)
npm install -g eas-cli
eas build --platform android
eas build --platform ios
```

---

## Important Files

### ✅ MUST KEEP - Backend

| File | Purpose | Required |
|------|---------|----------|
| `backend_api_actual_model.py` | Flask API server | ✅ Yes |
| `backup_best_model_*.pth` | Trained ML model weights | ✅ Yes |
| `train_cough_detector_attention.py` | Model architecture (imported by API) | ✅ Yes |
| `precompute_spectrograms.py` | Audio processor (imported by API) | ✅ Yes |
| `requirements.txt` | Python dependencies | ✅ Yes |
| `Dockerfile.backend` | Docker image definition | ✅ Yes |
| `docker-compose.yml` | Docker orchestration | ✅ Yes |
| `.dockerignore` | Docker build optimization | ✅ Yes |

### ✅ MUST KEEP - Frontend

| File/Directory | Purpose | Required |
|----------------|---------|----------|
| `src/App.tsx` | Main app component | ✅ Yes |
| `src/main.tsx` | Entry point | ✅ Yes |
| `src/index.css` | Global styles | ✅ Yes |
| `src/screens/` | All 8 screen components | ✅ Yes |
| `src/components/` | Reusable components | ✅ Yes |
| `src/navigation/` | Navigation setup | ✅ Yes |
| `src/utils/api.ts` | Backend API client | ✅ Yes |
| `src/utils/audioRecorder.ts` | Audio recording | ✅ Yes |
| `src/utils/storage.ts` | Storage utilities | ✅ Yes |
| `package.json` | Frontend dependencies | ✅ Yes |
| `app.json` | Expo configuration | ✅ Yes |
| `babel.config.js` | Babel configuration | ✅ Yes |
| `metro.config.js` | Metro bundler config | ✅ Yes |
| `tsconfig.json` | TypeScript config | ✅ Yes |
| `tsconfig.app.json` | TypeScript app config | ✅ Yes |
| `public/` | Static assets | ✅ Yes |
| `uploaded_audio/` | Audio upload directory | ✅ Yes (empty) |

---

## Files to Delete

### ❌ DELETE - Development Files

These files are **NOT needed** for production:

```bash
# Documentation files (already deleted)
*.md (except this guide)

# Shell scripts (already deleted)
*.sh

# Old web code (already deleted)
src/pages/              # Old React web pages
src/utils/wavRecorder.ts  # Web-only audio recorder
App.tsx (root)          # Old web App.tsx
index.html              # Web-only HTML

# Web build configs (already deleted)
vite.config.ts
tailwind.config.js
postcss.config.js
src/vite-env.d.ts
tsconfig.node.json

# Training scripts (already deleted)
train_cough_detector_attention.py  # ⚠️ WAIT - Backend needs this!
evaluate_model.py
precompute_spectrograms.py  # ⚠️ WAIT - Backend needs this!

# Other development files (already deleted)
disable-watchman-wrapper.js
lungs.png (root duplicate)
```

### ⚠️ IMPORTANT NOTE

**DO NOT DELETE** these files (even though they seem like training scripts):
- ✅ `train_cough_detector_attention.py` - **REQUIRED** by backend API
- ✅ `precompute_spectrograms.py` - **REQUIRED** by backend API

The backend imports classes from these files:
```python
from train_cough_detector_attention import Config, AttnMILResNet, get_device
from precompute_spectrograms import CoughAudioProcessor
```

---

## Troubleshooting

### Backend Issues

**Backend won't start:**
```bash
# Check Docker logs
docker-compose logs backend

# Rebuild image
docker-compose build --no-cache backend

# Check if port 5001 is available
lsof -i :5001
```

**Model not loading:**
- Verify `backup_best_model_*.pth` exists in root directory
- Check `MODEL_PATH` environment variable in `docker-compose.yml`

**Import errors:**
- Ensure `train_cough_detector_attention.py` and `precompute_spectrograms.py` exist
- Check that these files are copied in `Dockerfile.backend`

### Frontend Issues

**Expo won't start:**
```bash
# Clear cache
npx expo start -c

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**API connection errors:**
- Verify backend is running: `curl http://localhost:5001/health`
- Check API URL in `src/utils/api.ts`
- For Android emulator, use `http://10.0.2.2:5001`
- For iOS simulator, use `http://localhost:5001`

**Build errors:**
```bash
# Clear all caches
rm -rf node_modules .expo
npm install
npx expo start -c
```

---

## Production Checklist

Before deploying to production:

- [ ] Backend Docker image builds successfully
- [ ] Backend health check passes
- [ ] Frontend dependencies installed
- [ ] API endpoint configured correctly
- [ ] Model file exists and is accessible
- [ ] All required Python files present
- [ ] All required React Native screens present
- [ ] Static assets in `public/` directory
- [ ] `uploaded_audio/` directory exists (can be empty)
- [ ] Environment variables configured
- [ ] Docker volumes configured for persistence
- [ ] Health checks working
- [ ] CORS configured correctly

---

## Quick Start Commands

```bash
# 1. Start backend
docker-compose up -d backend

# 2. Verify backend
curl http://localhost:5001/health

# 3. Start frontend
npx expo start

# 4. Access application
# - Web: http://localhost:19006
# - Mobile: Scan QR code
```

---

## Support

For issues or questions:
1. Check Docker logs: `docker-compose logs backend`
2. Check Expo logs: View terminal output
3. Verify all required files are present
4. Check network connectivity between frontend and backend

---

**Last Updated:** January 2025
**Version:** 1.0.0

