import { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Platform, Alert } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  FadeInDown,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';
import { api } from '../utils/api';
// FileSystem no longer needed - using URI directly for React Native
// Client-side ONNX inference is handled automatically by api.ts

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, 'Analyzing'>;

export default function AnalyzingScreen() {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<AnalyzingRouteProp>();
  const audioUri = route.params?.audioUri;
  const hasAnalyzed = useRef(false);

  const rotate = useSharedValue(0);
  const opacity = useSharedValue(0.5);

  useEffect(() => {
    rotate.value = withRepeat(
      withTiming(360, { duration: 1800 }),
      -1,
      false
    );
    opacity.value = withRepeat(
      withTiming(1, { duration: 1500 }),
      -1,
      true
    );
  }, []);

  useEffect(() => {
    console.log('AnalyzingScreen - Mounted');
    console.log('AnalyzingScreen - audioUri:', audioUri);
    console.log('AnalyzingScreen - hasAnalyzed:', hasAnalyzed.current);

    if (!audioUri) {
      console.log('AnalyzingScreen - No audioUri, redirecting to RecordCough');
      Alert.alert('Error', 'No audio file provided. Please record or upload an audio file.');
      setTimeout(() => {
        navigation.replace('RecordCough');
      }, 2000);
      return;
    }

    if (hasAnalyzed.current) {
      console.log('AnalyzingScreen - Already analyzed, skipping');
      return;
    }
    hasAnalyzed.current = true;

    const analyzeCough = async () => {
      try {
        console.log('AnalyzingScreen - Starting analysis...');
        console.log('AnalyzingScreen - audioUri:', audioUri);
        console.log('AnalyzingScreen - audioUri type:', typeof audioUri);

        // Validate audioUri is a string
        if (!audioUri || typeof audioUri !== 'string') {
          throw new Error(`Invalid audioUri: expected string, got ${typeof audioUri} (${audioUri})`);
        }

        console.log('AnalyzingScreen - audioUri starts with:', audioUri.substring(0, 20));

        let blob: Blob;

        if (Platform.OS === 'web') {
          console.log('AnalyzingScreen - Web platform, fetching blob...');
          // For web, fetch the blob URL
          if (audioUri.startsWith('blob:') || audioUri.startsWith('http')) {
            console.log('AnalyzingScreen - Fetching from blob/http URL');
            const response = await fetch(audioUri);
            if (!response.ok) {
              throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
            }
            blob = await response.blob();
            console.log('AnalyzingScreen - Blob created, size:', blob.size, 'type:', blob.type);
          } else {
            // If it's a file path, try to fetch it
            console.log('AnalyzingScreen - Fetching from file path');
            const response = await fetch(audioUri);
            if (!response.ok) {
              throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
            }
            blob = await response.blob();
            console.log('AnalyzingScreen - Blob created, size:', blob.size, 'type:', blob.type);
          }
        } else {
          console.log('AnalyzingScreen - React Native platform, using URI directly...');
          // For React Native, pass the URI directly to the API
          // The API will handle it using FormData with file URI
          console.log('AnalyzingScreen - Calling API with URI...');
          const data = await api.detectCough(audioUri, 'cough.wav');

          console.log('AnalyzingScreen - API Response:', JSON.stringify(data, null, 2));

          // Ensure data is properly formatted
          const resultData = {
            coughDetected: data?.coughDetected === true || data?.coughDetected === 'true' || data?.coughDetected === 1,
            tbDetected: data?.tbDetected === true || data?.tbDetected === 'true' || data?.tbDetected === 1,
            error: data?.error || undefined,
            message: data?.message || data?.error || undefined,
            confidence: data?.confidence || data?.file_probability || undefined,
            ...data, // Include all other fields
          };

          console.log('AnalyzingScreen - Formatted result:', JSON.stringify(resultData, null, 2));
          console.log('AnalyzingScreen - coughDetected value:', resultData.coughDetected);
          console.log('AnalyzingScreen - tbDetected value:', resultData.tbDetected);
          console.log('AnalyzingScreen - Navigating to Result...');

          // Use navigation.navigate instead of replace to allow back navigation
          navigation.navigate('Result', { result: resultData });
          return; // Exit early for React Native
        }

        if (!blob || blob.size === 0) {
          throw new Error('Failed to create valid audio blob');
        }

        console.log('AnalyzingScreen - Calling API...');
        const data = await api.detectCough(blob, 'cough.wav');

        console.log('AnalyzingScreen - API Response:', JSON.stringify(data, null, 2));

        // Ensure data is properly formatted
        const resultData = {
          coughDetected: data?.coughDetected === true || data?.coughDetected === 'true' || data?.coughDetected === 1,
          tbDetected: data?.tbDetected === true || data?.tbDetected === 'true' || data?.tbDetected === 1,
          error: data?.error || undefined,
          message: data?.message || data?.error || undefined,
          confidence: data?.confidence || data?.file_probability || undefined,
          ...data, // Include all other fields
        };

        console.log('AnalyzingScreen - Formatted result:', JSON.stringify(resultData, null, 2));
        console.log('AnalyzingScreen - coughDetected value:', resultData.coughDetected);
        console.log('AnalyzingScreen - tbDetected value:', resultData.tbDetected);
        console.log('AnalyzingScreen - Navigating to Result...');

        // Use navigation.navigate instead of replace to allow back navigation
        navigation.navigate('Result', { result: resultData });
      } catch (error: any) {
        console.error('AnalyzingScreen - Error analyzing cough:', error);
        console.error('AnalyzingScreen - Error details:', {
          message: error?.message,
          stack: error?.stack,
          name: error?.name,
        });

        const message = error?.message || 'Unable to analyze cough audio.';

        console.log('AnalyzingScreen - Navigating to Result with error...');
        navigation.navigate('Result', {
          result: {
            coughDetected: false,
            error: message,
            message: `Error: ${message}`,
          },
        });
      }
    };

    // Small delay to ensure screen is mounted and visible
    const timer = setTimeout(() => {
      analyzeCough();
    }, 500);

    return () => {
      clearTimeout(timer);
    };
  }, [audioUri, navigation]);

  const spinnerAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${rotate.value}deg` }],
  }));

  const textAnimatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  return (
    <LinearGradient
      colors={['#158B95', '#0f6b73', '#e0f2f1']}
      style={styles.container}
    >
      <AppHeader variant="translucent" />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          entering={FadeInDown.delay(150).duration(400).springify()}
          style={styles.card}
        >
          <Animated.View style={[styles.spinner, spinnerAnimatedStyle]}>
            <View style={styles.spinnerCircle} />
          </Animated.View>

          <Text style={styles.title}>Analyzing Your Cough</Text>

          <Animated.Text style={[styles.subtitle, textAnimatedStyle]}>
            Processing audio sample...
          </Animated.Text>

          <View style={styles.progressBar}>
            <Animated.View style={[styles.progressFill, { width: '100%' }]} />
          </View>

          <Text style={styles.infoText}>
            {Platform.OS === 'web' && api.isClientONNXReady()
              ? 'Running AI analysis locally in your browser...'
              : 'Our AI model is analyzing your cough pattern...'}
          </Text>

          <View style={styles.steps}>
            {[
              { label: 'Detection', icon: '🔍' },
              { label: 'Analysis', icon: '🧠' },
              { label: 'Results', icon: '📊' },
            ].map((step) => (
              <View key={step.label} style={styles.step}>
                <Text style={styles.stepIcon}>{step.icon}</Text>
                <Text style={styles.stepLabel}>{step.label}</Text>
              </View>
            ))}
          </View>
        </Animated.View>

        <Text style={styles.footerText}>
          {Platform.OS === 'web' && api.isClientONNXReady()
            ? 'Processing audio locally using ONNX Runtime - no data leaves your device!'
            : 'Please wait while we analyze your cough using our trained model...'}
        </Text>
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
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 24,
    padding: 48,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 10,
    maxWidth: 500,
    width: '100%',
    alignSelf: 'center',
  },
  spinner: {
    width: 96,
    height: 96,
    marginBottom: 32,
  },
  spinnerCircle: {
    width: '100%',
    height: '100%',
    borderRadius: 48,
    borderWidth: 8,
    borderColor: COLORS.primary,
    borderTopColor: 'transparent',
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
    marginBottom: 24,
    textAlign: 'center',
    letterSpacing: 0.1,
  },
  progressBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#D1FAE5',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 16,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: 4,
  },
  infoText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#6B7280',
    textAlign: 'center',
    marginTop: 16,
    letterSpacing: 0.1,
  },
  steps: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginTop: 32,
  },
  step: {
    alignItems: 'center',
  },
  stepIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  stepLabel: {
    fontSize: 12,
    fontFamily: FONTS.medium,
    color: '#4B5563',
    letterSpacing: 0.2,
  },
  footerText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    marginTop: 40,
    maxWidth: 500,
    alignSelf: 'center',
  },
});

