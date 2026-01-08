// Audio Recorder utility using expo-av instead of Web Audio API
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';
import { Platform } from 'react-native';

export class AudioRecorder {
  private recording: Audio.Recording | null = null;
  private isRecording: boolean = false;
  private recordingUri: string | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];

  async start(): Promise<void> {
    try {
      if (Platform.OS === 'web') {
        // Use Web Audio API for web platform
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);
        this.audioChunks = [];
        
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data);
          }
        };
        
        this.mediaRecorder.onstop = () => {
          const blob = new Blob(this.audioChunks, { type: 'audio/webm' });
          this.recordingUri = URL.createObjectURL(blob);
        };
        
        this.mediaRecorder.start();
        this.isRecording = true;
      } else {
        // Use expo-av for React Native
        // Request permissions
        const { status } = await Audio.requestPermissionsAsync();
        if (status !== 'granted') {
          throw new Error('Audio recording permission not granted');
        }

        // Set audio mode for recording
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
        });

        // Create new recording
        const { recording } = await Audio.Recording.createAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY,
          (status) => {
            // Recording status updates
            if (status.isDoneRecording) {
              this.isRecording = false;
            }
          }
        );

        this.recording = recording;
        this.isRecording = true;
      }
    } catch (error: any) {
      console.error('Error starting recording:', error);
      throw error;
    }
  }

  async stop(): Promise<string> {
    if (Platform.OS === 'web') {
      if (!this.mediaRecorder || !this.isRecording) {
        throw new Error('No active recording');
      }

      try {
        return new Promise((resolve, reject) => {
          if (!this.mediaRecorder) {
            reject(new Error('No media recorder'));
            return;
          }

          this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const uri = URL.createObjectURL(blob);
            this.recordingUri = uri;
            this.isRecording = false;
            
            // Stop all tracks
            if (this.mediaRecorder.stream) {
              this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            resolve(uri);
          };

          this.mediaRecorder.stop();
        });
      } catch (error: any) {
        console.error('Error stopping recording:', error);
        throw error;
      }
    } else {
      if (!this.recording || !this.isRecording) {
        throw new Error('No active recording');
      }

      try {
        await this.recording.stopAndUnloadAsync();
        const uri = this.recording.getURI();
        
        if (!uri) {
          throw new Error('Failed to get recording URI');
        }

        this.recordingUri = uri;
        this.isRecording = false;
        this.recording = null;

        // Reset audio mode
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
          playsInSilentModeIOS: false,
        });

        return uri;
      } catch (error: any) {
        console.error('Error stopping recording:', error);
        throw error;
      }
    }
  }

  async getBlob(): Promise<Blob> {
    if (!this.recordingUri) {
      throw new Error('No recording available');
    }

    try {
      // Read file as base64
      const base64 = await FileSystem.readAsStringAsync(this.recordingUri, {
        encoding: ((FileSystem as any).EncodingType?.Base64 || 'base64') as any,
      });

      // Convert base64 to blob
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      
      // Determine MIME type from file extension
      const mimeType = this.recordingUri.endsWith('.m4a') 
        ? 'audio/m4a' 
        : this.recordingUri.endsWith('.wav')
        ? 'audio/wav'
        : 'audio/mp4';

      return new Blob([byteArray], { type: mimeType });
    } catch (error) {
      console.error('Error getting blob:', error);
      throw error;
    }
  }

  async getUri(): Promise<string | null> {
    return this.recordingUri;
  }

  getIsRecording(): boolean {
    return this.isRecording;
  }

  async cleanup(): Promise<void> {
    if (Platform.OS === 'web') {
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        try {
          this.mediaRecorder.stop();
        } catch (error) {
          // Already stopped
        }
      }
      if (this.mediaRecorder?.stream) {
        this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
      this.mediaRecorder = null;
      this.audioChunks = [];
    } else {
      if (this.recording) {
        try {
          await this.recording.stopAndUnloadAsync();
        } catch (error) {
          // Already stopped
        }
        this.recording = null;
      }
    }
    
    this.isRecording = false;
    
    if (this.recordingUri) {
      try {
        if (Platform.OS === 'web') {
          URL.revokeObjectURL(this.recordingUri);
        } else {
          await FileSystem.deleteAsync(this.recordingUri, { idempotent: true });
        }
      } catch (error) {
        // File might not exist
      }
      this.recordingUri = null;
    }
  }
}

