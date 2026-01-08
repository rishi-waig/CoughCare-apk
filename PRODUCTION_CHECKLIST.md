# 🚀 Production Deployment Checklist

## ⚠️ CRITICAL: Code is NOT production-ready yet!

This checklist outlines what needs to be fixed before deploying to production.

---

## ✅ What's Already Good

- ✅ Backend Docker setup is production-ready
- ✅ Environment variables for backend (MODEL_PATH, PORT, AUDIO_SAVE_DIR)
- ✅ Health checks configured
- ✅ Error handling in place
- ✅ CORS enabled (but needs restriction - see below)

---

## ❌ Critical Issues to Fix

### 1. **API URL Configuration** ⚠️ CRITICAL

**Problem:**
- Hardcoded development IP addresses (`10.100.32.31:5001`)
- Placeholder production URL (`https://your-api-domain.com`)
- `app.json` has `localhost:5001` hardcoded

**Fix Required:**
```bash
# Option 1: Set environment variable (RECOMMENDED)
export EXPO_PUBLIC_API_BASE_URL=https://api.yourdomain.com

# Option 2: Update app.json
# Change "apiBaseUrl" in app.json to your production URL
```

**Files to Update:**
- ✅ `src/utils/api.ts` - Already updated to support env vars
- ✅ `src/screens/CoughRecorderScreen.tsx` - Already updated
- ⚠️ `app.json` - Update `extra.apiBaseUrl` to production URL

---

### 2. **CORS Configuration** ⚠️ SECURITY

**Problem:**
```python
CORS(app)  # Allows ALL origins - SECURITY RISK!
```

**Fix Required:**
```python
# In backend_api_actual_model.py
import os

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '').split(',')
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == ['']:
    # Development: allow all
    CORS(app)
else:
    # Production: restrict to specific origins
    CORS(app, origins=ALLOWED_ORIGINS)
```

**Environment Variable:**
```bash
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

---

### 3. **Console Logging** ⚠️ PERFORMANCE

**Problem:**
- 99+ console.log statements throughout the codebase
- Logs sensitive data (API URLs, response data)
- Performance impact in production

**Status:**
- ✅ Partially fixed in `api.ts` (wrapped in `__DEV__`)
- ⚠️ Still need to fix in other files

**Files with Console Logs:**
- `src/screens/CoughRecorderScreen.tsx` (22 instances)
- `src/screens/ResultScreen.tsx` (16 instances)
- `src/screens/AnalyzingScreen.tsx` (30 instances)
- `src/screens/TbResultScreen.tsx` (14 instances)
- `src/screens/ChatbotScreen.tsx` (3 instances)
- `src/utils/storage.ts` (4 instances)
- `src/utils/audioRecorder.ts` (4 instances)

**Fix:**
Wrap all console.log/error/warn in `__DEV__` checks:
```typescript
if (__DEV__) {
  console.log('Debug info:', data);
}
```

---

### 4. **Backend Security** ⚠️

**Missing:**
- Rate limiting
- Request size limits (partially done - 10MB in nginx)
- Authentication/Authorization (if needed)
- HTTPS enforcement
- Security headers

**Recommended:**
```python
# Add to backend_api_actual_model.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/detect-cough', methods=['POST'])
@limiter.limit("10 per minute")  # Limit cough detection requests
def detect_cough():
    # ... existing code
```

---

### 5. **Environment Variables** ⚠️

**Backend (✅ Good):**
- `MODEL_PATH` - ✅ Configured
- `PORT` - ✅ Configured
- `AUDIO_SAVE_DIR` - ✅ Configured
- `ALLOWED_ORIGINS` - ⚠️ Missing (for CORS)

**Frontend (⚠️ Needs Setup):**
- `EXPO_PUBLIC_API_BASE_URL` - ⚠️ Must be set for production

---

### 6. **Error Handling** ⚠️

**Status:**
- ✅ Basic error handling exists
- ⚠️ Need user-friendly error messages
- ⚠️ Need error reporting/monitoring (Sentry, etc.)

---

### 7. **Build Configuration** ⚠️

**Required:**
- [ ] Update `app.json` with production API URL
- [ ] Set `EXPO_PUBLIC_API_BASE_URL` environment variable
- [ ] Configure CORS allowed origins
- [ ] Remove/guard all console.log statements
- [ ] Test production build locally
- [ ] Set up error monitoring (Sentry, etc.)

---

## 📋 Pre-Deployment Steps

### Step 1: Configure API URL

```bash
# Create .env file (for Expo)
echo "EXPO_PUBLIC_API_BASE_URL=https://api.yourdomain.com" > .env

# Or update app.json
# Change: "apiBaseUrl": "http://localhost:5001"
# To:     "apiBaseUrl": "https://api.yourdomain.com"
```

### Step 2: Fix CORS

```bash
# Update docker-compose.yml
environment:
  - ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Step 3: Remove Console Logs

Run a script to wrap all console statements:
```bash
# Manual review recommended, but can use find/replace:
# Find: console.log(
# Replace: if (__DEV__) { console.log(
# Then add closing brace }
```

### Step 4: Test Production Build

```bash
# Web
npx expo build:web

# Android
eas build --platform android --profile production

# iOS
eas build --platform ios --profile production
```

### Step 5: Deploy Backend

```bash
# Update docker-compose.yml with production settings
# Deploy to your server (AWS, GCP, Azure, etc.)
docker-compose up -d --build backend
```

---

## 🔒 Security Checklist

- [ ] CORS restricted to specific origins
- [ ] HTTPS enforced
- [ ] Rate limiting implemented
- [ ] Request size limits configured
- [ ] Environment variables secured (not in git)
- [ ] API keys/secrets in environment variables
- [ ] Error messages don't leak sensitive info
- [ ] Input validation on all endpoints
- [ ] File upload size limits
- [ ] Security headers configured (nginx)

---

## 📊 Monitoring & Logging

**Recommended:**
- [ ] Set up error tracking (Sentry, Rollbar, etc.)
- [ ] Set up application monitoring (New Relic, Datadog, etc.)
- [ ] Configure log aggregation
- [ ] Set up alerts for errors/uptime
- [ ] Monitor API response times
- [ ] Track model inference performance

---

## 🚀 Deployment Checklist

### Backend
- [ ] Docker image built and tested
- [ ] Environment variables configured
- [ ] CORS origins set
- [ ] Health check endpoint working
- [ ] Model file accessible
- [ ] Volume mounts configured
- [ ] Logs accessible
- [ ] Backup strategy in place

### Frontend
- [ ] API URL configured (env var or app.json)
- [ ] Production build tested
- [ ] Console logs removed/guarded
- [ ] Error handling tested
- [ ] App icons and splash screens set
- [ ] Permissions configured correctly
- [ ] Bundle size optimized

---

## 📝 Notes

- The code structure is good, but needs configuration for production
- Most issues are configuration-related, not code quality issues
- Backend is more production-ready than frontend
- Frontend needs environment variable setup

---

**Last Updated:** January 2025
**Status:** ⚠️ NOT PRODUCTION READY - Configuration needed


