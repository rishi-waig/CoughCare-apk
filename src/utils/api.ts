// API utility for client-side ONNX inference (Offline Mode)
import { Platform } from 'react-native';

// Enable client-side ONNX inference (Always true for offline app)
const USE_CLIENT_SIDE_ONNX = true;

// Lazy import for ONNX inference to avoid bundling issues
let OnnxInference: typeof import('./onnxInference') | null = null;
async function getOnnxInference() {
  if (!OnnxInference) {
    OnnxInference = await import('./onnxInference');
  }
  return OnnxInference;
}

export const api = {
  /**
   * Detect cough using client-side ONNX
   * Works on both web and native (offline)
   */
  async detectCough(audioBlobOrUri: Blob | string, _filename: string = 'cough.wav'): Promise<any> {
    try {
      console.log('[API] Attempting client-side ONNX inference...');

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
      throw new Error(`ONNX inference failed: ${onnxError.message}. Please ensure models are available and audio is in WAV format.`);
    }
  },

  async analyzeTB(audioBlobOrUri: Blob | string, _filename: string = 'cough.wav'): Promise<any> {
    console.log('[API] TB Analysis hardcoded to FALSE (Offline Mode)');
    // Return mock response immediately without hitting server
    return {
      tbDetected: false,
      confidence: 0.0,
      message: "TB Analysis is disabled in offline mode.",
      error: null
    };
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

