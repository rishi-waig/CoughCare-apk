// API utility for backend communication
import Constants from 'expo-constants';
import { Platform } from 'react-native';

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
  async detectCough(audioBlobOrUri: Blob | string, filename: string = 'cough.wav'): Promise<any> {
    try {
      const formData = new FormData();
      
      // For React Native, use URI directly. For web, use Blob
      if (typeof audioBlobOrUri === 'string') {
        // React Native: use file URI
        formData.append('audio', {
          uri: audioBlobOrUri,
          type: audioBlobOrUri.endsWith('.wav') ? 'audio/wav' : 
                audioBlobOrUri.endsWith('.mp3') ? 'audio/mp3' : 'audio/m4a',
          name: filename,
        } as any);
      } else {
        // Web: use Blob
        formData.append('audio', audioBlobOrUri as any, filename);
      }

      const apiUrl = `${API_BASE_URL}/api/detect-cough`;
      if (__DEV__) {
        console.log('Sending request to:', apiUrl);
        console.log('FormData audio:', typeof audioBlobOrUri === 'string' ? audioBlobOrUri : 'Blob');
      }

      // For React Native, don't set Content-Type - let FormData set it automatically with boundary
      // Setting headers manually can break multipart/form-data boundary
      const fetchOptions: RequestInit = {
        method: 'POST',
        body: formData,
      };

      // Only set Accept header, not Content-Type (FormData needs to set boundary automatically)
      if (Platform.OS === 'web') {
        fetchOptions.headers = {
          'Accept': 'application/json',
        };
      }
      // For React Native, don't set any headers - FormData will handle it

      const response = await fetch(apiUrl, fetchOptions);

      if (__DEV__) {
        console.log('Response received:', response.status, response.statusText);
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData?.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      if (__DEV__) {
        console.log('Response data:', data);
      }
      return data;
    } catch (error) {
      console.error('Error detecting cough:', error);
      throw error;
    }
  },

  async analyzeTB(audioBlobOrUri: Blob | string, filename: string = 'cough.wav'): Promise<any> {
    try {
      const formData = new FormData();
      
      // For React Native, use URI directly. For web, use Blob
      if (typeof audioBlobOrUri === 'string') {
        // React Native: use file URI
        formData.append('audio', {
          uri: audioBlobOrUri,
          type: audioBlobOrUri.endsWith('.wav') ? 'audio/wav' : 
                audioBlobOrUri.endsWith('.mp3') ? 'audio/mp3' : 'audio/m4a',
          name: filename,
        } as any);
      } else {
        // Web: use Blob
        formData.append('audio', audioBlobOrUri as any, filename);
      }

      const apiUrl = `${API_BASE_URL}/api/analyze-tb`;
      
      // For React Native, don't set Content-Type - let FormData set it automatically
      const fetchOptions: RequestInit = {
        method: 'POST',
        body: formData,
      };

      if (Platform.OS === 'web') {
        fetchOptions.headers = {
          'Accept': 'application/json',
        };
      }

      const response = await fetch(apiUrl, fetchOptions);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData?.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error analyzing TB:', error);
      throw error;
    }
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
};

