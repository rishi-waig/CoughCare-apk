// Audio Recorder utility
// - Web: Uses MediaRecorder (Web Audio API)
// - Native: Uses react-native-audio-record (guarantees 16-bit PCM WAV)
import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import AudioRecord from 'react-native-audio-record';
import { Buffer } from 'buffer';

export class AudioRecorder {
  private isRecording: boolean = false;
  private recordingUri: string | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];

  async start(): Promise<void> {
    try {
      if (Platform.OS === 'web') {
        // --- WEB IMPLEMENTATION ---
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
        // --- NATIVE IMPLEMENTATION (Android/iOS) ---
        // Configure for 16-bit PCM WAV @ 16kHz (Required by ONNX Model)
        const options = {
          sampleRate: 16000,  // default 44100
          channels: 1,        // 1 or 2, default 1
          bitsPerSample: 16,  // 8 or 16, default 16
          audioSource: 6,     // android only (VOICE_RECOGNITION)
          wavFile: 'cough_recording.wav' // default 'audio.wav'
        };

        AudioRecord.init(options);
        AudioRecord.start();
        this.isRecording = true;
      }
    } catch (error: any) {
      console.error('Error starting recording:', error);
      throw error;
    }
  }

  async stop(): Promise<string> {
    if (!this.isRecording) {
      throw new Error('No active recording');
    }

    if (Platform.OS === 'web') {
      // --- WEB STOP ---
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
    } else {
      // --- NATIVE STOP ---
      try {
        const audioFile = await AudioRecord.stop();
        this.isRecording = false;
        this.recordingUri = `file://${audioFile}`;
        console.log('Recording saved to:', this.recordingUri);
        return this.recordingUri;
      } catch (error) {
        console.error('Error stopping native recording:', error);
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

      // Determine MIME type
      const mimeType = this.recordingUri.endsWith('.wav') ? 'audio/wav' : 'audio/mp4';

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
        } catch (error) { }
      }
      if (this.mediaRecorder?.stream) {
        this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
      this.mediaRecorder = null;
      this.audioChunks = [];
    } else {
      if (this.isRecording) {
        try {
          await AudioRecord.stop();
        } catch (error) { }
      }
    }

    this.isRecording = false;
    this.recordingUri = null;
  }
}

