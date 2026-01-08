import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Image } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withRepeat,
  withTiming,
  withSequence,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';
import { FONTS, COLORS } from '../theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

export default function HomeScreen() {
  const navigation = useNavigation<NavigationProp>();
  const scale = useSharedValue(1);
  const lungScale = useSharedValue(1);
  const opacity = useSharedValue(0);

  React.useEffect(() => {
    opacity.value = withSpring(1, { stiffness: 120, damping: 18 });
    lungScale.value = withRepeat(
      withSequence(
        withTiming(1.08, { duration: 2000 }),
        withTiming(1, { duration: 2000 })
      ),
      -1,
      false
    );
  }, []);

  const handleStartAssessment = () => {
    navigation.navigate('Consent');
  };

  const containerAnimatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  const lungAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: lungScale.value }],
  }));

  const buttonAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handleButtonPress = () => {
    scale.value = withSequence(
      withTiming(0.96, { duration: 80 }),
      withSpring(1.04, { stiffness: 250, damping: 15 }),
      withSpring(1, { stiffness: 250, damping: 15 })
    );
    setTimeout(handleStartAssessment, 120);
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
        <Animated.View style={[styles.content, containerAnimatedStyle]}>
          <View style={styles.textSection}>
            <Animated.Text style={styles.title}>
              Get Your Cough Symptoms Assessed
            </Animated.Text>

            <Animated.Text style={styles.description}>
              Our free assessment tool helps you understand your cough symptoms better.
              Answer a few simple questions and get personalized guidance on next steps
              for your respiratory health.
            </Animated.Text>

            <View style={styles.featuresList}>
              {['Quick 5-question assessment', 'Voice-based symptom analysis', 'Free and confidential'].map((item, index) => (
                <Animated.View
                  key={index}
                  style={styles.featureItem}
                >
                  <View style={styles.bullet} />
                  <Text style={styles.featureText}>{item}</Text>
                </Animated.View>
              ))}
            </View>

            <AnimatedTouchableOpacity
              style={[styles.startButton, buttonAnimatedStyle]}
              onPress={handleButtonPress}
              activeOpacity={0.9}
            >
              <Text style={styles.buttonText}>Start Assessment</Text>
              <Animated.View style={styles.arrowContainer}>
                <Ionicons name="arrow-forward" size={20} color="#FFFFFF" />
              </Animated.View>
            </AnimatedTouchableOpacity>
          </View>

          <Animated.View style={[styles.imageSection, lungAnimatedStyle]}>
            <Image
              source={require('../../public/lungs.png')}
              style={styles.lungsImage}
              resizeMode="contain"
            />
          </Animated.View>
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 48,
  },
  content: {
    maxWidth: 1200,
    width: '100%',
    alignSelf: 'center',
  },
  textSection: {
    marginBottom: 36,
  },
  title: {
    fontSize: 32,
    fontFamily: FONTS.bold,
    color: COLORS.text,
    marginBottom: 16,
    lineHeight: 40,
    letterSpacing: -0.5,
  },
  description: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: COLORS.textSecondary,
    marginBottom: 28,
    lineHeight: 26,
    letterSpacing: 0.2,
  },
  featuresList: {
    marginTop: 12,
    marginBottom: 40,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 14,
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.primary,
    marginTop: 10,
    marginRight: 14,
  },
  featureText: {
    fontSize: 15,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    flex: 1,
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  startButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.primary,
    paddingHorizontal: 40,
    paddingVertical: 18,
    borderRadius: 50,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 6,
  },
  buttonText: {
    color: COLORS.background,
    fontSize: 17,
    fontFamily: FONTS.semiBold,
    marginRight: 10,
    letterSpacing: 0.3,
  },
  arrowContainer: {
    marginLeft: 4,
  },
  imageSection: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 32,
  },
  lungsImage: {
    width: '100%',
    maxWidth: 400,
    height: 400,
  },
});

