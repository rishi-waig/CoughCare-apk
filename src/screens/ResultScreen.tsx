import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Platform } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;
type ResultRouteProp = RouteProp<RootStackParamList, 'Result'>;

interface ResultData {
  coughDetected: boolean | string | number;
  tbDetected?: boolean | string | number;
  error?: string;
  message?: string;
  confidence?: number;
}

export default function ResultScreen() {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<ResultRouteProp>();
  const result = route.params?.result as ResultData | undefined;
  const [showDebug, setShowDebug] = React.useState(__DEV__);

  const scale = useSharedValue(0);

  useEffect(() => {
    scale.value = withSpring(1, { stiffness: 200 });
  }, []);

  useEffect(() => {
    console.log('ResultScreen - Component mounted');
    console.log('ResultScreen - Route params:', JSON.stringify(route.params, null, 2));
    console.log('ResultScreen - Received result:', JSON.stringify(result, null, 2));
    console.log('ResultScreen - Result type:', typeof result);
    console.log('ResultScreen - Result keys:', result ? Object.keys(result) : 'null');
    console.log('ResultScreen - coughDetected value:', result?.coughDetected);
    console.log('ResultScreen - coughDetected type:', typeof result?.coughDetected);

    const isCoughDetected = result?.coughDetected === true || result?.coughDetected === 'true' || result?.coughDetected === 1 || result?.coughDetected === '1';
    console.log('ResultScreen - coughDetected === true:', result?.coughDetected === true);
    console.log('ResultScreen - coughDetected === "true":', result?.coughDetected === 'true');
    console.log('ResultScreen - isCoughDetected (combined):', isCoughDetected);

    if (!result) {
      console.log('ResultScreen - No result, will show loading state');
      // Don't redirect immediately - show loading state first
      return;
    }

    console.log('ResultScreen - Result has data, will render');
    console.log('ResultScreen - Will show cough detected screen:', isCoughDetected);
  }, [navigation, result, route.params]);

  if (!result) {
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
            entering={FadeInDown.delay(200)}
            style={styles.errorCard}
          >
            <Text style={styles.errorTitle}>Loading Results...</Text>
            <Text style={styles.errorMessage}>
              Processing your analysis. Please wait...
            </Text>
            <TouchableOpacity
              style={styles.homeButton}
              onPress={() => navigation.navigate('Home')}
              activeOpacity={0.8}
            >
              <Ionicons name="home" size={20} color="#1F2937" />
              <Text style={styles.homeButtonText}>Back to Home</Text>
            </TouchableOpacity>
          </Animated.View>
        </ScrollView>
      </LinearGradient>
    );
  }

  const { error, confidence, message } = result;
  const coughDetected = result.coughDetected;

  // Normalize coughDetected to boolean
  const isCoughDetected = coughDetected === true || coughDetected === 'true' || coughDetected === 1 || coughDetected === '1';

  console.log('ResultScreen - Rendering with:', {
    coughDetected,
    isCoughDetected,
    error,
    confidence,
    message,
    resultType: typeof coughDetected,
    coughDetectedValue: coughDetected
  });

  // Show success screen if cough detected
  console.log('ResultScreen - Checking coughDetected condition:', {
    coughDetected,
    isTrue: coughDetected === true,
    isStringTrue: coughDetected === 'true',
    isNumberOne: coughDetected === 1,
    willShow: isCoughDetected
  });

  if (isCoughDetected) {
    console.log('ResultScreen - Rendering cough detected screen');
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
            style={[styles.errorCard, { borderColor: '#D1FAE5' }]}
          >
            <Animated.View style={[styles.errorIcon, { backgroundColor: '#D1FAE5' }]}>
              <Ionicons name="checkmark-circle" size={80} color="#10B981" />
            </Animated.View>
            <Text style={styles.errorTitle}>Cough Detected!</Text>
            <Text style={styles.errorMessage}>
              {message || 'Cough detected successfully. The analysis indicates a cough pattern was found in your audio.'}
            </Text>
            {confidence !== undefined && (
              <View style={styles.confidenceContainer}>
                <Text style={styles.confidenceLabel}>Confidence:</Text>
                <Text style={styles.confidenceValue}>
                  {(Number(confidence) * 100).toFixed(1)}%
                </Text>
              </View>
            )}
            <View style={styles.buttons}>
              <TouchableOpacity
                style={styles.tryAgainButton}
                onPress={() => navigation.navigate('TbResult', { result })}
                activeOpacity={0.8}
              >
                <Ionicons name="arrow-forward" size={20} color="#FFFFFF" />
                <Text style={styles.tryAgainButtonText}>View TB Analysis</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.homeButton}
                onPress={() => navigation.navigate('Home')}
                activeOpacity={0.8}
              >
                <Ionicons name="home" size={20} color="#1F2937" />
                <Text style={styles.homeButtonText}>Back to Home</Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        </ScrollView>
      </LinearGradient>
    );
  }

  // Error case - no cough detected OR show any result
  // Always show something - don't return null
  if (!isCoughDetected || error) {
    const iconScale = useSharedValue(0);

    React.useEffect(() => {
      iconScale.value = withSpring(1, { stiffness: 200 });
    }, []);

    const iconAnimatedStyle = useAnimatedStyle(() => ({
      transform: [{ scale: iconScale.value }],
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
            style={styles.errorCard}
          >
            <Animated.View style={[styles.errorIcon, iconAnimatedStyle]}>
              <Ionicons name="close-circle" size={80} color="#EF4444" />
            </Animated.View>

            <Text style={styles.errorTitle}>No Cough Detected</Text>

            <Text style={styles.errorMessage}>
              {error || message || 'We could not detect a clear cough sound in your recording. Please try again and make sure to cough clearly into the microphone.'}
            </Text>

            {result.confidence !== undefined && (
              <View style={styles.confidenceContainer}>
                <Text style={styles.confidenceLabel}>Confidence:</Text>
                <Text style={styles.confidenceValue}>
                  {(result.confidence * 100).toFixed(1)}%
                </Text>
              </View>
            )}

            <View style={styles.buttons}>
              <TouchableOpacity
                style={styles.tryAgainButton}
                onPress={() => navigation.navigate('RecordCough')}
                activeOpacity={0.8}
              >
                <Ionicons name="refresh" size={20} color="#FFFFFF" />
                <Text style={styles.tryAgainButtonText}>Try Again</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.homeButton}
                onPress={() => navigation.navigate('Home')}
                activeOpacity={0.8}
              >
                <Ionicons name="home" size={20} color="#1F2937" />
                <Text style={styles.homeButtonText}>Back to Home</Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        </ScrollView>
      </LinearGradient>
    );
  }

  // Fallback - show result data even if format is unexpected
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
          style={styles.errorCard}
        >
          <Text style={styles.errorTitle}>Analysis Complete</Text>
          <Text style={styles.errorMessage}>
            {result ? `Result received. coughDetected: ${result.coughDetected}, error: ${result.error || 'none'}` : 'No result data'}
          </Text>

          {showDebug && result && (
            <View style={styles.debugContainer}>
              <Text style={styles.debugTitle}>Debug Info:</Text>
              <Text style={styles.debugText} selectable>
                {JSON.stringify(result, null, 2)}
              </Text>
              <TouchableOpacity
                style={styles.debugButton}
                onPress={() => setShowDebug(false)}
              >
                <Text style={styles.debugButtonText}>Hide Debug</Text>
              </TouchableOpacity>
            </View>
          )}

          {!showDebug && (
            <TouchableOpacity
              style={styles.debugButton}
              onPress={() => setShowDebug(true)}
            >
              <Text style={styles.debugButtonText}>Show Debug Info</Text>
            </TouchableOpacity>
          )}

          <View style={styles.buttons}>
            <TouchableOpacity
              style={styles.homeButton}
              onPress={() => navigation.navigate('Home')}
              activeOpacity={0.8}
            >
              <Ionicons name="home" size={20} color="#1F2937" />
              <Text style={styles.homeButtonText}>Back to Home</Text>
            </TouchableOpacity>
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
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
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
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 10,
    maxWidth: 600,
    width: '100%',
  },
  title: {
    fontSize: 28,
    fontFamily: FONTS.semiBold,
    color: '#1F2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    textAlign: 'center',
  },
  errorCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 24,
    padding: 48,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 10,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
    borderWidth: 1,
    borderColor: '#FEE2E2',
  },
  errorIcon: {
    width: 128,
    height: 128,
    borderRadius: 64,
    backgroundColor: '#FEE2E2',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  errorTitle: {
    fontSize: 28,
    fontFamily: FONTS.bold,
    color: '#1F2937',
    marginBottom: 18,
    textAlign: 'center',
    letterSpacing: -0.4,
  },
  errorMessage: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    marginBottom: 32,
    textAlign: 'center',
    lineHeight: 26,
    letterSpacing: 0.1,
  },
  buttons: {
    flexDirection: 'row',
    gap: 16,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  tryAgainButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: COLORS.primary,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  tryAgainButtonText: {
    color: COLORS.background,
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    letterSpacing: 0.2,
  },
  homeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: '#E5E7EB',
  },
  homeButtonText: {
    color: '#1F2937',
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    letterSpacing: 0.2,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    padding: 12,
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
  },
  confidenceLabel: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#6B7280',
    marginRight: 10,
    letterSpacing: 0.1,
  },
  confidenceValue: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: '#1F2937',
    letterSpacing: 0.2,
  },
  debugContainer: {
    marginTop: 16,
    padding: 16,
    backgroundColor: '#F9FAFB',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    maxWidth: '100%',
  },
  debugTitle: {
    fontSize: 14,
    fontFamily: FONTS.semiBold,
    color: '#374151',
    marginBottom: 8,
  },
  debugText: {
    fontSize: 12,
    color: '#6B7280',
    fontFamily: Platform.OS === 'web' ? 'monospace' : 'monospace',
    marginBottom: 8,
  },
  debugButton: {
    padding: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 4,
    alignSelf: 'flex-start',
  },
  debugButtonText: {
    fontSize: 12,
    fontFamily: FONTS.regular,
    color: '#374151',
  },
});

