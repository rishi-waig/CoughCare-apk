// API utility for backend communication with client-side ONNX fallback
import Constants from 'expo-constants';
import { Platform } from 'react-native';

// Enable client-side ONNX inference (set to false to always use backend)
const USE_CLIENT_SIDE_ONNX = true;

// Lazy import for ONNX inference to avoid bundling issues
let OnnxInference: typeof import('./onnxInference') | null = null;
async function getOnnxInference() {
  if (!OnnxInference) {
    OnnxInference = await import('./onnxInference');
  }
  return OnnxInference;
}

// API Base URL Configuration
// Priority: 1. Environment variable (EXPO_PUBLIC_API_BASE_URL)
//           2. app.json extra.apiBaseUrl
//           3. Development defaults (localhost/10.0.2.2)
//           4. Production placeholder (must be configured)
const getApiBaseUrl = () => {
  // Check environment variable first (highest priority for production)
  const envUrl = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (envUrl) {
    return envUrl;
  }

  // Check app.json configuration
  const configUrl = Constants.expoConfig?.extra?.apiBaseUrl;
  if (configUrl && !configUrl.includes('localhost') && !configUrl.includes('127.0.0.1')) {
    return configUrl; // Use if it's not a localhost URL (production config)
  }

  // Development defaults
  if (__DEV__) {
    if (Platform.OS === 'android') {
      // For Android emulator, use actual IP for Windows Docker compatibility
      // TODO: Replace with your development machine's IP or use 10.0.2.2
      return 'http://10.100.32.31:5001';
    }
    // iOS simulator and web use localhost in development
    return 'http://localhost:5001';
  }

  // Production: Must be configured via EXPO_PUBLIC_API_BASE_URL or app.json
  // This will throw an error if not configured, preventing accidental deployment
  const prodUrl = configUrl || 'https://your-api-domain.com';
  if (prodUrl.includes('your-api-domain.com') || prodUrl.includes('localhost')) {
    console.error('⚠️ PRODUCTION API URL NOT CONFIGURED! Set EXPO_PUBLIC_API_BASE_URL or update app.json');
  }
  return prodUrl;
};

const API_BASE_URL = getApiBaseUrl();

export const api = {
  /**
   * Detect cough using client-side ONNX (web) or backend API (native)
   * On web: Uses ONNX Runtime Web for 100% client-side inference
   * On native: Falls back to backend API
   */
  async detectCough(audioBlobOrUri: Blob | string, filename: string = 'cough.wav'): Promise<any> {
    // Try client-side ONNX inference (works on both web and React Native)
    if (USE_CLIENT_SIDE_ONNX) {
      try {
        console.log('[API] Attempting client-side ONNX inference (web)...');

        // Get audio data as ArrayBuffer
        let audioData: ArrayBuffer;

        if (typeof audioBlobOrUri === 'string') {
          // For React Native file URIs, read the file
          if (Platform.OS !== 'web' && audioBlobOrUri.startsWith('file://')) {
            const FileSystem = await import('expo-file-system/legacy');
            const base64 = await FileSystem.readAsStringAsync(audioBlobOrUri, {
              encoding: FileSystem.EncodingType.Base64,
            });
            // Convert base64 to ArrayBuffer
            const binaryString = atob(base64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i);
            }
            audioData = bytes.buffer;
          } else {
            // Fetch from URL/blob URL (web)
            const response = await fetch(audioBlobOrUri);
            if (!response.ok) {
              throw new Error(`Failed to fetch audio: ${response.status}`);
            }
            audioData = await response.arrayBuffer();
          }
        } else {
          // Blob to ArrayBuffer (web)
          audioData = await audioBlobOrUri.arrayBuffer();
        }

        console.log(`[API] Audio data: ${audioData.byteLength} bytes`);

        // Run ONNX inference
        const onnx = await getOnnxInference();
        const result = await onnx.detectCough(audioData);

        // Hardcode TB detection to false for client-side inference
        const finalResult = {
          ...result,
          tbDetected: false,
          tbConfidence: 0.0
        };

        console.log('[API] Client-side ONNX result:', finalResult);
        return finalResult;

      } catch (onnxError: any) {
        console.warn('[API] Client-side ONNX failed:', onnxError.message);
        // Don't fall back to backend - throw error instead
        throw new Error(`ONNX inference failed: ${onnxError.message}. Please ensure models are available and audio is in WAV format.`);
      }
    }

    // If we reach here, ONNX should have worked
    // This should not happen if USE_CLIENT_SIDE_ONNX is true
    throw new Error('ONNX inference not available. Please enable USE_CLIENT_SIDE_ONNX or ensure models are loaded.');
  },

  async analyzeTB(audioBlobOrUri: Blob | string, filename: string = 'cough.wav'): Promise<any> {
    console.log('[API] TB Analysis hardcoded to FALSE (Offline Mode)');
    // Return mock response immediately without hitting server
    return {
      tbDetected: false,
      confidence: 0.0,
      message: "TB Analysis is disabled in offline mode.",
      error: null
    };
  },

  async healthCheck(): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  },

  /**
   * Initialize client-side ONNX models (web and React Native)
   * Call this early to pre-load models for faster inference
   */
  async initClientONNX(): Promise<void> {
    if (USE_CLIENT_SIDE_ONNX) {
      try {
        console.log('[API] Pre-loading ONNX models...');
        const onnx = await getOnnxInference();
        await onnx.initONNX();
        console.log('[API] ONNX models ready');
      } catch (error) {
        console.warn('[API] Failed to pre-load ONNX models:', error);
      }
    }
  },

  /**
   * Check if client-side ONNX is available
   */
  isClientONNXReady(): boolean {
    if (!USE_CLIENT_SIDE_ONNX || !OnnxInference) {
      return false;
    }
    return OnnxInference.isONNXReady();
  },

  /**
   * Get ONNX model info
   */
  async getONNXInfo() {
    const onnx = await getOnnxInference();
    return onnx.getModelInfo();
  },
};

