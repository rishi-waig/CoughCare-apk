import { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Alert, Platform } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import { Audio } from 'expo-av';
import { Asset } from 'expo-asset';
import Animated, {
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';
import { AudioRecorder } from '../utils/audioRecorder';
import * as FileSystem from 'expo-file-system/legacy';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);
const MIN_RECORD_SECONDS = 5;

export default function CoughRecorderScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [minStopReached, setMinStopReached] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const pulseScale = useSharedValue(1);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (sound) {
        sound.unloadAsync();
      }
      if (recorderRef.current) {
        recorderRef.current.cleanup();
      }
    };
  }, [sound]);

  const startRecording = async () => {
    try {
      const recorder = new AudioRecorder();
      await recorder.start();
      recorderRef.current = recorder;

      setIsRecording(true);
      setRecordingTime(0);
      setMinStopReached(false);
      setIsUploaded(false);
      setAudioUri(null);

      pulseScale.value = withRepeat(
        withSequence(
          withTiming(1.08, { duration: 900 }),
          withTiming(1, { duration: 900 })
        ),
        -1,
        false
      );

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          const next = prev + 1;
          if (next >= MIN_RECORD_SECONDS) {
            setMinStopReached(true);
          }
          return next;
        });
      }, 1000);
    } catch (error: any) {
      console.error('Error accessing microphone:', error);
      Alert.alert('Permission Error', 'Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = async () => {
    if (recorderRef.current && isRecording) {
      try {
        const uri = await recorderRef.current.stop();
        setAudioUri(uri);
        setIsRecording(false);
        pulseScale.value = 1;

        if (timerRef.current) {
          clearInterval(timerRef.current);
        }

        // Get duration
        const { sound: audioSound } = await Audio.Sound.createAsync(
          { uri },
          { shouldPlay: false }
        );
        const status = await audioSound.getStatusAsync();
        if (status.isLoaded) {
          setRecordingTime(Math.round(status.durationMillis! / 1000));
        }
        await audioSound.unloadAsync();
      } catch (error) {
        console.error('Error stopping recording:', error);
        Alert.alert('Error', 'Failed to stop recording');
      }
    }
  };

  const resetRecording = async () => {
    try {
      console.log('resetRecording - Starting reset...');

      // Stop any playing sound
      if (sound) {
        try {
          await sound.unloadAsync();
        } catch (error) {
          console.warn('Error unloading sound:', error);
        }
        setSound(null);
      }

      // Stop timer if running
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Stop recording if active
      if (recorderRef.current) {
        try {
          await recorderRef.current.cleanup();
        } catch (error) {
          console.warn('Error cleaning up recorder:', error);
        }
        recorderRef.current = null;
      }

      // Clean up audio URI
      if (audioUri) {
        try {
          if (Platform.OS === 'web') {
            // For web, revoke blob URL
            if (audioUri.startsWith('blob:')) {
              URL.revokeObjectURL(audioUri);
            }
          } else {
            // For React Native, delete file
            await FileSystem.deleteAsync(audioUri, { idempotent: true });
          }
        } catch (error) {
          console.warn('Error deleting audio file:', error);
        }
      }

      // Reset all state
      setAudioUri(null);
      setRecordingTime(0);
      setMinStopReached(false);
      setIsUploaded(false);
      setIsRecording(false);
      pulseScale.value = 1;

      console.log('resetRecording - Reset complete');
    } catch (error: any) {
      console.error('Error resetting recording:', error);
      // Still reset state even if cleanup fails
      setAudioUri(null);
      setRecordingTime(0);
      setMinStopReached(false);
      setIsUploaded(false);
      setIsRecording(false);
    }
  };

  const loadSampleAudio = async () => {
    // Prevent multiple simultaneous loads
    if (isLoadingSample) {
      console.log('Sample loading already in progress, skipping...');
      return;
    }

    if (isUploaded && audioUri) {
      console.log('Sample already loaded, skipping...');
      return;
    }

    setIsLoadingSample(true);

    try {
      // Hardcoded sample file - loads directly from public folder (no backend needed)
      const sampleFile = '20251104_150725_454926_cough.wav';

      let sampleUrl: string;
      let uri: string;

      if (Platform.OS === 'web') {
        // For web, fetch directly from public folder
        sampleUrl = `/samples/${sampleFile}`;
        console.log('Loading sample from:', sampleUrl);
        const response = await fetch(sampleUrl);

        if (!response.ok) {
          throw new Error(`Failed to fetch sample: ${response.statusText}`);
        }

        const blob = await response.blob();
        uri = URL.createObjectURL(blob);
      } else {
        // For React Native, load from bundled assets
        console.log('Loading sample from bundled assets...');
        try {
          // Lazy require - only executed on React Native (using .native.ts extension)
          let sampleAsset: any;
          try {
            sampleAsset = require('../assets/audio.native').SAMPLE_AUDIO;
          } catch (requireError) {
            throw new Error('Sample audio asset not found. Make sure the file is in assets/audio/ folder.');
          }
          
          // Load from bundled assets using expo-asset
          const asset = Asset.fromModule(sampleAsset);
          await asset.downloadAsync();
          
          if (!asset.localUri) {
            throw new Error('Failed to load sample audio asset - localUri is null');
          }
          
          // Copy to cache for easier access
          const cacheDir = FileSystem.cacheDirectory;
          const localUri = `${cacheDir}${sampleFile}`;
          await FileSystem.copyAsync({
            from: asset.localUri,
            to: localUri,
          });
          uri = localUri;
          console.log('Sample audio loaded from assets and copied to cache:', uri);
        } catch (assetError: any) {
          console.error('Failed to load sample from assets:', assetError);
          throw new Error(`Sample audio is not available: ${assetError.message}\n\nYou can still record your own cough audio using the microphone button.`);
        }
      }

      setAudioUri(uri);
      setIsUploaded(true);
      setIsRecording(false);
      setIsLoadingSample(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      // Get duration - try multiple methods for web compatibility
      try {
        if (Platform.OS === 'web') {
          // For web, use HTML5 Audio API
          const audio = new (window as any).Audio(uri);
          await new Promise((resolve, reject) => {
            audio.addEventListener('loadedmetadata', () => {
              const duration = Math.round(audio.duration);
              if (isFinite(duration)) {
                setRecordingTime(duration || 1);
              } else {
                setRecordingTime(10); // Fallback for infinite/unknown duration
              }
              resolve(null);
            });
            audio.addEventListener('error', reject);
            audio.load();
          });
        } else {
          // For React Native, use expo-av
          const { sound: audioSound } = await Audio.Sound.createAsync(
            { uri },
            { shouldPlay: false }
          );
          const status = await audioSound.getStatusAsync();
          if (status.isLoaded && status.durationMillis) {
            setRecordingTime(Math.round(status.durationMillis / 1000));
          } else {
            setRecordingTime(1); // Default to 1 second if duration unavailable
          }
          await audioSound.unloadAsync();
        }
      } catch (durationError) {
        console.warn('Could not get duration, setting default:', durationError);
        setRecordingTime(10); // Set to 10 seconds as fallback for sample files
      }

      Alert.alert('Success', 'Sample audio loaded successfully!');
    } catch (error: any) {
      console.error('Error loading sample audio:', error);
      setIsLoadingSample(false);
      const errorMessage = error.message || 'Unknown error';

      // Show a more helpful error message
      Alert.alert(
        'Sample Audio Unavailable',
        `${errorMessage}\n\nNote: Sample audio is optional. You can still record your own cough audio using the microphone.`,
        [
          { text: 'OK', style: 'default' },
        ]
      );
    }
  };

  const pickAudioFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/*',
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        const uri = result.assets[0].uri;
        setAudioUri(uri);
        setIsUploaded(true);
        setIsRecording(false);

        if (timerRef.current) {
          clearInterval(timerRef.current);
        }

        // Get duration - try multiple methods for web compatibility
        try {
          if (Platform.OS === 'web') {
            // For web, use HTML5 Audio API
            const audio = new (window as any).Audio(uri);
            await new Promise((resolve, reject) => {
              audio.addEventListener('loadedmetadata', () => {
                const duration = Math.round(audio.duration);
                if (isFinite(duration)) {
                  setRecordingTime(duration || 1);
                } else {
                  setRecordingTime(10); // Fallback for infinite/unknown duration
                }
                resolve(null);
              });
              audio.addEventListener('error', reject);
              audio.load();
            });
          } else {
            // For React Native, use expo-av
            const { sound: audioSound } = await Audio.Sound.createAsync(
              { uri },
              { shouldPlay: false }
            );
            const status = await audioSound.getStatusAsync();
            if (status.isLoaded && status.durationMillis) {
              setRecordingTime(Math.round(status.durationMillis / 1000));
            } else {
              setRecordingTime(1); // Default to 1 second if duration unavailable
            }
            await audioSound.unloadAsync();
          }
        } catch (durationError) {
          console.warn('Could not get duration, setting default:', durationError);
          setRecordingTime(10); // Set to 10 seconds as fallback for uploaded files
        }
      }
    } catch (error) {
      console.error('Error picking file:', error);
      Alert.alert('Error', 'Failed to pick audio file.');
    }
  };

  const playAudio = async () => {
    if (!audioUri) return;

    try {
      if (sound) {
        await sound.unloadAsync();
      }

      const { sound: audioSound } = await Audio.Sound.createAsync(
        { uri: audioUri },
        { shouldPlay: true }
      );
      setSound(audioSound);

      audioSound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          audioSound.unloadAsync();
          setSound(null);
        }
      });
    } catch (error) {
      console.error('Error playing audio:', error);
      Alert.alert('Error', 'Failed to play audio.');
    }
  };

  const handleSubmit = async () => {
    if (!audioUri) {
      Alert.alert('Error', 'No audio file available. Please record or upload an audio file.');
      return;
    }

    // For uploaded/sample files, allow submission even if duration is less than 5 seconds
    // Only enforce minimum for live recordings
    if (!isUploaded && recordingTime < MIN_RECORD_SECONDS) {
      Alert.alert('Recording Too Short', `Please record for at least ${MIN_RECORD_SECONDS} seconds.`);
      return;
    }

    // For uploaded files, ensure we have at least some duration
    if (isUploaded && recordingTime === 0) {
      Alert.alert('Error', 'Could not determine audio duration. Please try again.');
      return;
    }

    try {
      console.log('handleSubmit - audioUri:', audioUri);
      console.log('handleSubmit - isUploaded:', isUploaded);
      console.log('handleSubmit - recordingTime:', recordingTime);

      // Simply navigate to Analyzing screen with audioUri
      // The AnalyzingScreen will handle blob creation
      navigation.navigate('Analyzing', { audioUri });
    } catch (error: any) {
      console.error('Error submitting audio:', error);
      Alert.alert('Error', `Failed to submit audio: ${error.message || 'Please try again.'}`);
    }
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds) || isNaN(seconds) || seconds < 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60); // Ensure integer
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const micAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  return (
    <LinearGradient
      colors={['#158B95', '#0f6b73', '#e0f2f1']}
      style={styles.container}
    >
      <AppHeader
        variant="solid"
        rightSlot={
          <TouchableOpacity
            style={styles.homeButton}
            onPress={() => navigation.navigate('Home')}
            activeOpacity={0.8}
          >
            <Ionicons name="home" size={20} color="#158B95" />
            <Text style={styles.homeButtonText}>Home</Text>
          </TouchableOpacity>
        }
      />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          entering={FadeInDown.delay(150).duration(400).springify()}
          style={styles.card}
        >
          <View style={styles.header}>
            <Text style={styles.title}>Record Your Cough</Text>
            <Text style={styles.subtitle}>
              Please record a clear cough sound for accurate analysis.
              Click the microphone button below to start recording.
            </Text>
          </View>

          <View style={styles.content}>
            {!isRecording && !audioUri && (
              <>
                <AnimatedTouchableOpacity
                  style={[styles.recordButton, micAnimatedStyle]}
                  onPress={startRecording}
                  activeOpacity={0.9}
                >
                  <Ionicons name="mic" size={48} color="#FFFFFF" />
                </AnimatedTouchableOpacity>

                <View style={styles.divider}>
                  <Text style={styles.dividerText}>or</Text>
                </View>

                <TouchableOpacity
                  style={styles.uploadButton}
                  onPress={pickAudioFile}
                  activeOpacity={0.8}
                >
                  <Ionicons name="document-attach" size={24} color="#158B95" />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.uploadButtonText}>Upload an audio file</Text>
                    <Text style={styles.uploadButtonSubtext}>WAV, WEBM, MP3, OGG, or M4A</Text>
                  </View>
                </TouchableOpacity>

                <View style={styles.divider}>
                  <Text style={styles.dividerText}>or</Text>
                </View>

                <TouchableOpacity
                  style={[styles.sampleButton, isLoadingSample && styles.sampleButtonDisabled]}
                  onPress={loadSampleAudio}
                  activeOpacity={0.8}
                  disabled={isLoadingSample}
                >
                  <Ionicons name="musical-notes" size={24} color={isLoadingSample ? "#9CA3AF" : "#158B95"} />
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.sampleButtonText, isLoadingSample && styles.sampleButtonTextDisabled]}>
                      {isLoadingSample ? 'Loading Sample...' : 'Load Sample Audio'}
                    </Text>
                    <Text style={styles.sampleButtonSubtext}>Try with a pre-loaded cough sample</Text>
                  </View>
                </TouchableOpacity>
              </>
            )}

            {isRecording && (
              <View style={styles.recordingContainer}>
                <Animated.View style={[styles.recordingButton, micAnimatedStyle]}>
                  <Ionicons name="mic" size={48} color="#FFFFFF" />
                </Animated.View>
                <Text style={styles.recordingTime}>{formatTime(recordingTime)}</Text>
                <Text style={styles.recordingText}>Recording in progress...</Text>
                <TouchableOpacity
                  style={[
                    styles.stopButton,
                    !minStopReached && styles.stopButtonDisabled,
                  ]}
                  onPress={stopRecording}
                  disabled={!minStopReached}
                  activeOpacity={0.8}
                >
                  <Ionicons name="stop-circle" size={24} color="#FFFFFF" />
                  <Text style={styles.stopButtonText}>Stop Recording</Text>
                </TouchableOpacity>
                {!minStopReached && (
                  <Text style={styles.minTimeText}>
                    Stop will be enabled after {MIN_RECORD_SECONDS} seconds.
                  </Text>
                )}
              </View>
            )}

            {audioUri && !isRecording && (
              <View style={styles.playbackContainer}>
                <View style={styles.successIcon}>
                  <Ionicons name="checkmark-circle" size={72} color="#10B981" />
                </View>
                <Text style={styles.successText}>✓ Recording Complete!</Text>
                <Text style={styles.durationText}>
                  Duration: {formatTime(recordingTime)}{isUploaded ? ' (uploaded)' : ''}
                </Text>

                <TouchableOpacity
                  style={styles.playButton}
                  onPress={playAudio}
                  activeOpacity={0.8}
                >
                  <Ionicons name="play" size={24} color="#158B95" />
                  <Text style={styles.playButtonText}>Play Recording</Text>
                </TouchableOpacity>

                {!isUploaded && recordingTime < MIN_RECORD_SECONDS && (
                  <Text style={styles.warningText}>
                    Please record at least {MIN_RECORD_SECONDS} seconds before submitting.
                  </Text>
                )}
                {isUploaded && recordingTime === 0 && (
                  <Text style={styles.warningText}>
                    Loading audio duration...
                  </Text>
                )}

                <View style={styles.actionButtons}>
                  <TouchableOpacity
                    style={styles.resetButton}
                    onPress={resetRecording}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="refresh" size={20} color="#334155" />
                    <Text style={styles.resetButtonText}>Record Again</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[
                      styles.submitButton,
                      (!isUploaded && recordingTime < MIN_RECORD_SECONDS) && styles.submitButtonDisabled,
                      (isUploaded && recordingTime === 0) && styles.submitButtonDisabled,
                    ]}
                    onPress={handleSubmit}
                    disabled={(!isUploaded && recordingTime < MIN_RECORD_SECONDS) || (isUploaded && recordingTime === 0)}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="send" size={20} color="#FFFFFF" />
                    <Text style={styles.submitButtonText}>Analyze Cough</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>

          <View style={styles.tipsContainer}>
            <Text style={styles.tipsTitle}>📝 Recording Tips:</Text>
            <View style={styles.tipsList}>
              <Text style={styles.tipItem}>• Find a quiet environment</Text>
              <Text style={styles.tipItem}>• Hold your device close to your mouth</Text>
              <Text style={styles.tipItem}>• Record for at least {MIN_RECORD_SECONDS} seconds</Text>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
}

import { FONTS, COLORS } from '../theme';

// ... (imports remain the same)

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 32,
  },
  homeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 50,
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  homeButtonText: {
    color: COLORS.primary,
    fontSize: 14,
    fontFamily: FONTS.medium,
  },
  card: {
    backgroundColor: COLORS.background,
    borderRadius: 24,
    padding: 36,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 10,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontFamily: FONTS.bold,
    color: '#1F2937',
    marginBottom: 18,
    textAlign: 'center',
    letterSpacing: -0.4,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    textAlign: 'center',
    lineHeight: 26,
    letterSpacing: 0.1,
  },
  content: {
    alignItems: 'center',
    gap: 32,
  },
  recordButton: {
    width: 128,
    height: 128,
    borderRadius: 64,
    backgroundColor: '#14B8A6',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#14B8A6',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 8,
  },
  divider: {
    width: '100%',
    alignItems: 'center',
    marginVertical: 16,
  },
  dividerText: {
    color: '#6B7280',
    fontSize: 16,
    fontFamily: FONTS.regular,
  },
  uploadButton: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    padding: 16,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#D1D5DB',
    borderRadius: 16,
    backgroundColor: '#F9FAFB',
  },
  uploadButtonText: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: '#374151',
    letterSpacing: 0.1,
  },
  uploadButtonSubtext: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#6B7280',
    letterSpacing: 0.05,
  },
  sampleButton: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    padding: 16,
    borderWidth: 2,
    borderColor: '#14B8A6',
    borderRadius: 16,
    backgroundColor: '#ECFDF5',
  },
  sampleButtonText: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: '#065F46',
    letterSpacing: 0.1,
  },
  sampleButtonSubtext: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#065F46',
    letterSpacing: 0.05,
  },
  sampleButtonDisabled: {
    opacity: 0.6,
    borderColor: '#9CA3AF',
    backgroundColor: '#F3F4F6',
  },
  sampleButtonTextDisabled: {
    color: '#9CA3AF',
  },
  recordingContainer: {
    alignItems: 'center',
    gap: 16,
  },
  recordingButton: {
    width: 128,
    height: 128,
    borderRadius: 64,
    backgroundColor: '#EF4444',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordingTime: {
    fontSize: 32,
    fontFamily: FONTS.bold,
    color: '#1F2937',
    letterSpacing: -0.5,
  },
  recordingText: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    letterSpacing: 0.1,
  },
  stopButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: '#EF4444',
  },
  stopButtonDisabled: {
    backgroundColor: '#D1D5DB',
  },
  stopButtonText: {
    color: COLORS.background,
    fontSize: 16,
    fontFamily: FONTS.semiBold,
  },
  minTimeText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#6B7280',
  },
  playbackContainer: {
    width: '100%',
    alignItems: 'center',
    gap: 24,
  },
  successIcon: {
    marginBottom: 8,
  },
  successText: {
    fontSize: 24,
    fontFamily: FONTS.semiBold,
    color: '#1F2937',
    letterSpacing: -0.3,
  },
  durationText: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    letterSpacing: 0.1,
  },
  playButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    borderWidth: 2,
    borderColor: COLORS.primary,
    backgroundColor: COLORS.background,
  },
  playButtonText: {
    color: COLORS.primary,
    fontSize: 16,
    fontFamily: FONTS.semiBold,
  },
  warningText: {
    fontSize: 14,
    color: '#EF4444',
    fontFamily: FONTS.medium,
    textAlign: 'center',
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: '#E5E7EB',
  },
  resetButtonText: {
    color: '#1F2937',
    fontSize: 16,
    fontFamily: FONTS.semiBold,
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: '#14B8A6',
    shadowColor: '#14B8A6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  submitButtonDisabled: {
    backgroundColor: '#D1D5DB',
    shadowOpacity: 0,
    elevation: 0,
  },
  submitButtonText: {
    color: COLORS.background,
    fontSize: 16,
    fontFamily: FONTS.semiBold,
  },
  tipsContainer: {
    marginTop: 32,
    padding: 24,
    backgroundColor: '#ECFDF5',
    borderRadius: 16,
  },
  tipsTitle: {
    fontSize: 18,
    fontFamily: FONTS.semiBold,
    color: '#1F2937',
    marginBottom: 14,
    letterSpacing: -0.2,
  },
  tipsList: {
    gap: 10,
  },
  tipItem: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#374151',
    lineHeight: 24,
    letterSpacing: 0.1,
  },
});

