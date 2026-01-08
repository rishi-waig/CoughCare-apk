# CoughCare - Quick Start

AI-powered cough detection and TB screening application.

## Prerequisites

- **Node.js** (v18+)
- **Docker** & **Docker Compose**

## Quick Start

### Backend (CPU-based, Docker)

```bash
docker-compose up --build -d backend
```

Backend: `http://localhost:5001`

**Useful commands:**
```bash
docker-compose logs -f backend    # View logs
docker-compose stop backend       # Stop
docker-compose down              # Stop & remove
```

### Frontend

```bash
npm install    # First time only
npm start      # Start Expo dev server
```

**Platform-specific:**
```bash
npm run web      # Web browser
npm run android # Android
npm run ios     # iOS
```

## Backend Info

- **CPU-only** PyTorch (no GPU needed)
- Model: `backup_best_model_20251015_170801.pth`
- API: `/api/detect-cough` | Health: `/health`
- Audio saved to: `./uploaded_audio/`

## Frontend Info

- **React Native** + **Expo**
- Platforms: Web, Android, iOS
- API URL: Auto-detected or `EXPO_PUBLIC_API_BASE_URL`

## Troubleshooting

**Backend issues:**
```bash
docker-compose down && docker-compose up --build -d backend
docker-compose logs backend  # Check logs
```

**Frontend can't connect:**
- Verify backend: `docker-compose ps`
- Check API URL in browser console
- Ensure port 5001 is available

**Port conflicts:**
- Backend: Edit `5001:5000` in `docker-compose.yml`
- Frontend: Expo auto-assigns port

