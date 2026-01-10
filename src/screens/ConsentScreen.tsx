import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  FadeInDown,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

const AnimatedView = Animated.createAnimatedComponent(View);
const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

export default function ConsentScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [consentGiven, setConsentGiven] = useState(false);

  const handleContinue = () => {
    if (consentGiven) {
      navigation.navigate('Assessment');
    }
  };

  const handleDecline = () => {
    navigation.navigate('Home');
  };

  return (
    <LinearGradient
      colors={['#f0fbfc', '#FFFFFF', '#d8f1f3']}
      style={styles.container}
    >
      <AppHeader variant="translucent" />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <AnimatedView
          entering={FadeInDown.duration(400).springify()}
          style={styles.card}
        >
          <AnimatedView entering={FadeInDown.delay(80).duration(400).springify()} style={styles.header}>
            <View style={styles.iconContainer}>
              <Ionicons name="shield-checkmark" size={28} color="#158B95" />
            </View>
            <View>
              <Text style={styles.title}>Informed Consent</Text>
              <Text style={styles.subtitle}>Please read and accept before proceeding</Text>
            </View>
          </AnimatedView>

          <AnimatedView entering={FadeInDown.delay(150).duration(400).springify()} style={styles.content}>
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Data Collection and Usage</Text>
              <View style={styles.list}>
                {[
                  'We will collect information about your cough symptoms through a brief questionnaire and audio recording',
                  'Your voice recording will be analyzed using AI to assess cough characteristics',
                  'All data will be used solely for providing you with health assessment results',
                  'Your data will be stored securely and kept confidential',
                  'You may withdraw from the assessment at any time',
                ].map((item, index) => (
                  <View key={index} style={styles.listItem}>
                    <View style={styles.bullet} />
                    <Text style={styles.listText}>{item}</Text>
                  </View>
                ))}
              </View>
            </View>

            <View style={styles.disclaimerSection}>
              <Text style={styles.disclaimerTitle}>Important Disclaimer</Text>
              <Text style={styles.disclaimerText}>
                This assessment is for informational purposes only and does not constitute medical advice, diagnosis, or treatment.
                Always consult with a qualified healthcare professional for medical concerns.
              </Text>
            </View>
          </AnimatedView>

          <AnimatedView entering={FadeInDown.delay(220).duration(400).springify()} style={styles.checkboxContainer}>
            <TouchableOpacity
              style={styles.checkboxTouchable}
              onPress={() => setConsentGiven(!consentGiven)}
              activeOpacity={0.7}
            >
              <View style={[styles.checkbox, consentGiven && styles.checkboxChecked]}>
                {consentGiven && <Ionicons name="checkmark" size={16} color="#FFFFFF" />}
              </View>
              <Text style={styles.checkboxLabel}>
                I have read and understood the information above, and I consent to the collection and use of my data
                for this health assessment.
              </Text>
            </TouchableOpacity>
          </AnimatedView>

          <AnimatedView entering={FadeInDown.delay(280).duration(400).springify()} style={styles.buttons}>
            <AnimatedTouchableOpacity
              style={[styles.button, styles.declineButton]}
              onPress={handleDecline}
              activeOpacity={0.8}
            >
              <Ionicons name="arrow-back" size={20} color="#334155" />
              <Text style={styles.declineButtonText}>Decline & Go Back</Text>
            </AnimatedTouchableOpacity>

            <AnimatedTouchableOpacity
              style={[
                styles.button,
                styles.continueButton,
                !consentGiven && styles.disabledButton,
              ]}
              onPress={handleContinue}
              disabled={!consentGiven}
              activeOpacity={0.8}
            >
              <Text style={[styles.continueButtonText, !consentGiven && styles.disabledText]}>
                Accept & Continue
              </Text>
              <Ionicons
                name="arrow-forward"
                size={20}
                color={consentGiven ? '#FFFFFF' : '#94A3B8'}
              />
            </AnimatedTouchableOpacity>
          </AnimatedView>

          {!consentGiven && (
            <AnimatedView entering={FadeInDown.delay(500)}>
              <Text style={styles.helperText}>
                Please check the consent box above to continue
              </Text>
            </AnimatedView>
          )}
        </AnimatedView>
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
    backgroundColor: COLORS.background,
    borderRadius: 24,
    padding: 28,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 8,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  iconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(21, 139, 149, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  title: {
    fontSize: 24,
    fontFamily: FONTS.bold,
    color: COLORS.text,
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: COLORS.textSecondary,
    marginTop: 6,
    letterSpacing: 0.1,
  },
  content: {
    marginBottom: 24,
  },
  section: {
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    padding: 24,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  sectionTitle: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: '#1E293B',
    marginBottom: 16,
    letterSpacing: -0.2,
  },
  list: {
    gap: 14,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.primary,
    marginTop: 9,
    marginRight: 14,
  },
  listText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: COLORS.text,
    flex: 1,
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  disclaimerSection: {
    backgroundColor: '#FEF3C7',
    borderRadius: 12,
    padding: 24,
    borderWidth: 1,
    borderColor: '#FCD34D',
  },
  disclaimerTitle: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: '#92400E',
    marginBottom: 14,
    letterSpacing: -0.2,
  },
  disclaimerText: {
    fontSize: 13,
    fontFamily: FONTS.regular,
    color: '#78350F',
    lineHeight: 22,
    letterSpacing: 0.1,
  },
  checkboxContainer: {
    marginBottom: 24,
  },
  checkboxTouchable: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 4,
    borderWidth: 2,
    borderColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
    marginTop: 2,
  },
  checkboxChecked: {
    backgroundColor: COLORS.primary,
  },
  checkboxLabel: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: COLORS.text,
    flex: 1,
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  buttons: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  button: {
    flex: 1,
    minWidth: 150,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 50,
    gap: 8,
  },
  declineButton: {
    backgroundColor: COLORS.background,
    borderWidth: 2,
    borderColor: '#CBD5E1',
  },
  declineButtonText: {
    color: COLORS.text,
    fontSize: 15,
    fontFamily: FONTS.medium,
  },
  continueButton: {
    backgroundColor: COLORS.primary,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 25,
    elevation: 8,
  },
  continueButtonText: {
    color: COLORS.background,
    fontSize: 15,
    fontFamily: FONTS.semiBold,
  },
  disabledButton: {
    backgroundColor: COLORS.border,
    shadowOpacity: 0,
    elevation: 0,
  },
  disabledText: {
    color: '#94A3B8',
  },
  helperText: {
    fontSize: 13,
    fontFamily: FONTS.regular,
    color: '#64748B',
    textAlign: 'center',
    marginTop: 16,
  },
});

