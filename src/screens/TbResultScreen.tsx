import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Linking, ActivityIndicator } from 'react-native';
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
type TbResultRouteProp = RouteProp<RootStackParamList, 'TbResult'>;

interface ResultData {
  coughDetected: boolean | string | number;
  tbDetected?: boolean | string | number;
  error?: string;
  message?: string;
  confidence?: number;
  audioUri?: string;
}

export default function TbResultScreen() {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<TbResultRouteProp>();
  const result = route.params?.result as ResultData | undefined;
  const [tbResult, setTbResult] = useState<ResultData | null>(result || null);

  const scale = useSharedValue(0);

  useEffect(() => {
    scale.value = withSpring(1, { stiffness: 200 });
  }, []);

  const scaleAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  useEffect(() => {
    console.log('TbResultScreen - Mounted');
    console.log('TbResultScreen - Route params:', JSON.stringify(route.params, null, 2));
    console.log('TbResultScreen - Result:', JSON.stringify(result, null, 2));
    console.log('TbResultScreen - Result type:', typeof result);
    console.log('TbResultScreen - Result keys:', result ? Object.keys(result) : 'null');

    if (!result) {
      console.log('TbResultScreen - No result, will show loading state');
      // Don't redirect immediately - show loading state first
      return;
    }

    // Normalize coughDetected check
    const isCoughDetected = result.coughDetected === true || result.coughDetected === 'true' || result.coughDetected === 1 || result.coughDetected === '1';
    console.log('TbResultScreen - coughDetected value:', result.coughDetected);
    console.log('TbResultScreen - isCoughDetected:', isCoughDetected);

    if (!isCoughDetected) {
      console.log('TbResultScreen - No cough detected, will show error state');
      // Set result anyway so we can show error message
      setTbResult({
        ...result,
        coughDetected: false,
      });
      return;
    }

    // Use existing result - tbDetected should already be in result from detectCough API
    console.log('TbResultScreen - Using existing result with tbDetected:', result.tbDetected);

    // Normalize tbDetected
    const normalizedTbDetected = result.tbDetected === true || result.tbDetected === 'true' || result.tbDetected === 1 || result.tbDetected === '1' ? true : false;

    setTbResult({
      ...result,
      coughDetected: true,
      tbDetected: normalizedTbDetected,
    });

    console.log('TbResultScreen - Set tbResult with:', {
      coughDetected: true,
      tbDetected: normalizedTbDetected,
    });
  }, [navigation, result, route.params]);


  // Show loading state
  if (!tbResult) {
    return (
      <LinearGradient
        colors={['#158B95', '#0f6b73', '#e0f2f1']}
        style={styles.container}
      >
        <AppHeader variant="translucent" />
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#14B8A6" />
          <Text style={styles.loadingText}>Loading results...</Text>
        </View>
      </LinearGradient>
    );
  }

  // Normalize coughDetected check
  const isCoughDetected = tbResult.coughDetected === true || tbResult.coughDetected === 'true' || tbResult.coughDetected === 1 || tbResult.coughDetected === '1';

  // Show error if no cough detected
  if (!isCoughDetected) {
    console.log('TbResultScreen - Rendering error state (no cough detected)');
    return (
      <LinearGradient
        colors={['#158B95', '#0f6b73', '#e0f2f1']}
        style={styles.container}
      >
        <AppHeader variant="translucent" />
        <View style={styles.centerContent}>
          <Text style={styles.errorTitle}>No Cough Detected</Text>
          <Text style={styles.errorMessage}>
            Please record a cough first before analyzing for TB.
          </Text>
          <TouchableOpacity
            style={styles.homeButton}
            onPress={() => navigation.navigate('Home')}
            activeOpacity={0.8}
          >
            <Ionicons name="home" size={20} color="#FFFFFF" />
            <Text style={styles.homeButtonText}>Back to Home</Text>
          </TouchableOpacity>
        </View>
      </LinearGradient>
    );
  }

  const { tbDetected } = tbResult;
  const isTBDetected = tbDetected === true || tbDetected === 'true' || tbDetected === 1 || tbDetected === '1';

  console.log('TbResultScreen - Rendering main content with:', {
    tbDetected,
    isTBDetected,
    coughDetected: tbResult.coughDetected,
    isCoughDetected,
    willShowContent: true,
    tbResultKeys: Object.keys(tbResult),
  });

  const iconScale = useSharedValue(0);

  React.useEffect(() => {
    iconScale.value = withSpring(1, { stiffness: 200 });
  }, []);

  const iconAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: iconScale.value }],
  }));

  const handleOpenResources = () => {
    Linking.openURL('https://www.who.int/health-topics/tuberculosis');
  };

  console.log('TbResultScreen - About to render main content');
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
          style={[
            styles.card,
            isTBDetected ? styles.cardWarning : styles.cardSuccess,
            scaleAnimatedStyle,
          ]}
        >
          <Animated.View style={[
            styles.iconContainer,
            isTBDetected ? styles.iconContainerWarning : styles.iconContainerSuccess,
            iconAnimatedStyle,
          ]}>
            {isTBDetected ? (
              <Ionicons name="warning" size={80} color="#F97316" />
            ) : (
              <Ionicons name="checkmark-circle" size={80} color="#10B981" />
            )}
          </Animated.View>

          <Text style={styles.title}>
            {isTBDetected ? 'TB Indicators Detected' : 'No TB Detected'}
          </Text>

          <Text style={styles.message}>
            {isTBDetected
              ? 'Based on the analysis of your cough sound, our AI model has detected indicators that may be associated with tuberculosis.'
              : 'Based on your cough recording and assessment, our AI analysis did not find any signs of tuberculosis.'}
          </Text>

          <View style={[
            styles.recommendations,
            isTBDetected ? styles.recommendationsWarning : styles.recommendationsSuccess,
          ]}>
            <Text style={styles.recommendationsTitle}>
              {isTBDetected ? '⚠️ Important Recommendations:' : '✅ What This Means:'}
            </Text>
            <View style={styles.recommendationsList}>
              {isTBDetected ? (
                <>
                  <Text style={styles.recommendationItem}>
                    • <Text style={styles.bold}>Consult a healthcare professional immediately</Text> for proper diagnosis
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • Get a chest X-ray and sputum test for confirmation
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • Avoid close contact with others until diagnosed
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • This is a preliminary screening, not a final diagnosis
                  </Text>
                </>
              ) : (
                <>
                  <Text style={styles.recommendationItem}>
                    • Your cough pattern does not show TB indicators
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • This is a preliminary screening result
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • If symptoms persist, consult a healthcare professional
                  </Text>
                  <Text style={styles.recommendationItem}>
                    • Regular health checkups are always recommended
                  </Text>
                </>
              )}
            </View>
          </View>

          {isTBDetected && (
            <Animated.View entering={FadeInDown.delay(400)} style={styles.referralSection}>
              <View style={styles.referralHeader}>
                <Ionicons name="person-add" size={24} color="#1E40AF" />
                <Text style={styles.referralTitle}>Need Medical Consultation?</Text>
              </View>
              <Text style={styles.referralText}>
                We strongly recommend consulting with a healthcare professional for proper diagnosis and treatment.
              </Text>
              <TouchableOpacity
                style={styles.resourcesButton}
                onPress={handleOpenResources}
                activeOpacity={0.8}
              >
                <Text style={styles.resourcesButtonText}>Find TB Healthcare Resources</Text>
              </TouchableOpacity>
            </Animated.View>
          )}

          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={styles.homeButton}
              onPress={() => navigation.navigate('Home')}
              activeOpacity={0.8}
            >
              <Ionicons name="home" size={20} color="#FFFFFF" />
              <Text style={styles.homeButtonText}>Back to Home</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.newAssessmentButton}
              onPress={() => navigation.navigate('Assessment', { reset: true })}
              activeOpacity={0.8}
            >
              <Ionicons name="refresh" size={20} color="#1F2937" />
              <Text style={styles.newAssessmentButtonText}>New Assessment</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.disclaimer}>
            <Text style={styles.disclaimerTitle}>⚠️ DISCLAIMER</Text>
            <Text style={styles.disclaimerText}>
              This tool's results are for demonstration purposes only, not for clinical use.
              Always consult qualified healthcare professionals for medical diagnosis and treatment.
            </Text>
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
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  cardWarning: {
    borderWidth: 1,
    borderColor: '#FED7AA',
  },
  cardSuccess: {
    borderWidth: 1,
    borderColor: '#BBF7D0',
  },
  iconContainer: {
    width: 128,
    height: 128,
    borderRadius: 64,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  iconContainerWarning: {
    backgroundColor: '#FED7AA',
  },
  iconContainerSuccess: {
    backgroundColor: '#D1FAE5',
  },
  title: {
    fontSize: 32,
    fontFamily: FONTS.bold,
    color: '#1F2937',
    marginBottom: 18,
    textAlign: 'center',
    letterSpacing: -0.4,
  },
  message: {
    fontSize: 17,
    fontFamily: FONTS.regular,
    color: '#4B5563',
    marginBottom: 28,
    textAlign: 'center',
    lineHeight: 26,
    letterSpacing: 0.1,
  },
  recommendations: {
    width: '100%',
    padding: 24,
    borderRadius: 16,
    marginBottom: 24,
  },
  recommendationsWarning: {
    backgroundColor: '#FED7AA',
  },
  recommendationsSuccess: {
    backgroundColor: '#D1FAE5',
  },
  recommendationsTitle: {
    fontSize: 18,
    fontFamily: FONTS.semiBold,
    color: '#1F2937',
    marginBottom: 14,
    letterSpacing: -0.2,
  },
  recommendationsList: {
    gap: 10,
  },
  recommendationItem: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#374151',
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  bold: {
    fontFamily: FONTS.bold,
  },
  referralSection: {
    width: '100%',
    padding: 24,
    backgroundColor: '#DBEAFE',
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#93C5FD',
    marginBottom: 24,
  },
  referralHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  referralTitle: {
    fontSize: 18,
    fontFamily: FONTS.bold,
    color: '#1E3A8A',
    letterSpacing: -0.2,
  },
  referralText: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#1E40AF',
    marginBottom: 16,
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  resourcesButton: {
    width: '100%',
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#2563EB',
    borderRadius: 50,
    alignItems: 'center',
  },
  resourcesButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    letterSpacing: 0.2,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 16,
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginBottom: 24,
  },
  homeButton: {
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
  homeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    letterSpacing: 0.2,
  },
  newAssessmentButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 50,
    backgroundColor: '#E5E7EB',
  },
  newAssessmentButtonText: {
    color: '#1F2937',
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    letterSpacing: 0.2,
  },
  disclaimer: {
    width: '100%',
    padding: 16,
    backgroundColor: '#FEF3C7',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FCD34D',
  },
  disclaimerTitle: {
    fontSize: 14,
    fontFamily: FONTS.semiBold,
    color: '#92400E',
    textAlign: 'center',
    marginBottom: 10,
    letterSpacing: 0.3,
  },
  disclaimerText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: '#78350F',
    textAlign: 'center',
    lineHeight: 22,
    letterSpacing: 0.1,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 18,
    color: '#FFFFFF',
    fontFamily: FONTS.semiBold,
  },
  errorTitle: {
    fontSize: 24,
    fontFamily: FONTS.bold,
    color: '#FFFFFF',
    marginBottom: 16,
    textAlign: 'center',
  },
  errorMessage: {
    fontSize: 16,
    color: '#E0F2F1',
    fontFamily: FONTS.regular,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 24,
  },
});

