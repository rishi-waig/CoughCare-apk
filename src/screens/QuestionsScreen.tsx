import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  FadeInDown,
  FadeOutUp,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';
import { storage } from '../utils/storage';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

const questions = [
  {
    id: 1,
    question: 'Are you experiencing a fever?',
    options: ['Yes', 'No'],
  },
  {
    id: 2,
    question: 'Do you have a cough?',
    options: ['Yes', 'No'],
  },
  {
    id: 3,
    question: 'Have you coughed up blood with sputum anytime in the last 6 months?',
    options: ['Yes', 'No'],
  },
  {
    id: 4,
    question: 'Have you lost weight without trying in the last 6 months?',
    options: ['Yes', 'No'],
  },
  {
    id: 5,
    question: 'Have you experienced night sweats in the last month?',
    options: ['Yes', 'No'],
  },
];

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

export default function QuestionsScreen() {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<RouteProp<RootStackParamList, 'Assessment'>>();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<{ [key: number]: string }>({});

  React.useEffect(() => {
    if (route.params?.reset) {
      setCurrentQuestionIndex(0);
      setAnswers({});
      navigation.setParams({ reset: undefined });
    }
  }, [route.params?.reset]);

  const currentQuestion = questions[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / questions.length) * 100;

  const handleAnswer = (answer: string) => {
    const newAnswers = { ...answers, [currentQuestionIndex]: answer };
    setAnswers(newAnswers);

    setTimeout(() => {
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
      } else {
        storage.setItem('assessmentAnswers', JSON.stringify(newAnswers));
        navigation.navigate('RecordCough');
      }
    }, 250);
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      storage.setItem('assessmentAnswers', JSON.stringify(answers));
      navigation.navigate('RecordCough');
    }
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const progressAnimatedStyle = useAnimatedStyle(() => ({
    width: `${progress}%`,
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
        <View style={styles.progressContainer}>
          <View style={styles.progressHeader}>
            <Text style={styles.progressText}>
              Question {currentQuestionIndex + 1} of {questions.length}
            </Text>
            <Text style={styles.progressPercent}>{Math.round(progress)}%</Text>
          </View>
          <View style={styles.progressBar}>
            <Animated.View style={[styles.progressFill, progressAnimatedStyle]} />
          </View>
        </View>

        <Animated.View
          key={currentQuestionIndex}
          entering={FadeInDown.duration(350).springify()}
          exiting={FadeOutUp.duration(250)}
          style={styles.card}
        >
          <Text style={styles.question}>{currentQuestion.question}</Text>

          <View style={styles.options}>
            {currentQuestion.options.map((option, index) => {
              const isSelected = answers[currentQuestionIndex] === option;
              return (
                <AnimatedTouchableOpacity
                  key={`${currentQuestionIndex}-${index}`}
                  entering={FadeInDown.delay(index * 80).duration(350).springify()}
                  style={[
                    styles.option,
                    isSelected && styles.optionSelected,
                  ]}
                  onPress={() => handleAnswer(option)}
                  activeOpacity={0.8}
                >
                  <Text style={[
                    styles.optionText,
                    isSelected && styles.optionTextSelected,
                  ]}>
                    {option}
                  </Text>
                  <Ionicons
                    name="chevron-forward"
                    size={20}
                    color={isSelected ? '#FFFFFF' : '#94A3B8'}
                  />
                </AnimatedTouchableOpacity>
              );
            })}
          </View>

          <View style={styles.navigation}>
            <TouchableOpacity
              style={[
                styles.navButton,
                currentQuestionIndex === 0 && styles.navButtonDisabled,
              ]}
              onPress={handleBack}
              disabled={currentQuestionIndex === 0}
              activeOpacity={0.8}
            >
              <Ionicons name="chevron-back" size={20} color={currentQuestionIndex === 0 ? '#94A3B8' : '#158B95'} />
              <Text style={[
                styles.navButtonText,
                currentQuestionIndex === 0 && styles.navButtonTextDisabled,
              ]}>
                Back
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.navButtonNext}
              onPress={handleNext}
              activeOpacity={0.8}
            >
              <Text style={styles.navButtonNextText}>
                {currentQuestionIndex < questions.length - 1 ? 'Next' : 'Finish'}
              </Text>
              <Ionicons name="chevron-forward" size={20} color="#FFFFFF" />
            </TouchableOpacity>
          </View>
        </Animated.View>

        <Text style={styles.footerText}>
          Your responses are confidential and will help us provide better guidance
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
  progressContainer: {
    marginBottom: 24,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressText: {
    fontSize: 14,
    fontFamily: FONTS.semiBold,
    color: 'rgba(255, 255, 255, 0.9)',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  progressPercent: {
    fontSize: 14,
    fontFamily: FONTS.semiBold,
    color: COLORS.background,
    letterSpacing: 0.5,
  },
  progressBar: {
    width: '100%',
    height: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    borderRadius: 5,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: 5,
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 28,
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
  question: {
    fontSize: 24,
    fontFamily: FONTS.bold,
    color: COLORS.text,
    marginBottom: 36,
    lineHeight: 38,
    letterSpacing: -0.4,
  },
  options: {
    gap: 14,
    marginBottom: 36,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    borderWidth: 1.5,
    borderColor: 'rgba(21, 139, 149, 0.15)',
  },
  optionSelected: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  optionText: {
    fontSize: 17,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    letterSpacing: 0.1,
  },
  optionTextSelected: {
    color: COLORS.background,
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 16,
  },
  navButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  navButtonDisabled: {
    backgroundColor: COLORS.border,
  },
  navButtonText: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: COLORS.primary,
    letterSpacing: 0.2,
  },
  navButtonTextDisabled: {
    color: '#94A3B8',
  },
  navButtonNext: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 12,
    backgroundColor: COLORS.primary,
  },
  navButtonNextText: {
    fontSize: 16,
    fontFamily: FONTS.semiBold,
    color: COLORS.background,
    letterSpacing: 0.2,
  },
  footerText: {
    fontSize: 14,
    fontFamily: FONTS.regular,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    marginTop: 24,
  },
});

