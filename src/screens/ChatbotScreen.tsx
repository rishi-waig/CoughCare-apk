import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput, KeyboardAvoidingView, Platform, Alert } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import Animated, {
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import AppHeader from '../components/AppHeader';
import { storage } from '../utils/storage';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export default function ChatbotScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m here to help analyze your cough symptoms. Please use the microphone button below to describe your symptoms, or type your message.',
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);

  const pulseScale = useSharedValue(1);

  useEffect(() => {
    const assessmentAnswers = storage.getItem('assessmentAnswers');
    if (assessmentAnswers) {
      assessmentAnswers.then((data) => {
        if (data) {
          const answers = JSON.parse(data);
          console.log('Assessment answers:', answers);
        }
      });
    }
  }, []);

  useEffect(() => {
    scrollViewRef.current?.scrollToEnd({ animated: true });
  }, [messages]);

  const startRecording = async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Error', 'Microphone permission not granted');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      recordingRef.current = recording;
      setIsRecording(true);

      pulseScale.value = withRepeat(
        withSequence(
          withTiming(1.15, { duration: 600 }),
          withTiming(1, { duration: 600 })
        ),
        -1,
        false
      );
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = async () => {
    if (recordingRef.current && isRecording) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
        setIsRecording(false);
        pulseScale.value = 1;

        handleVoiceInput();
      } catch (error) {
        console.error('Error stopping recording:', error);
      }
    }
  };

  const handleVoiceInput = async () => {
    setIsProcessing(true);

    const userMessage: Message = {
      id: Date.now().toString(),
      text: 'Voice message recorded',
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Thank you for sharing your symptoms. Based on your voice analysis and the information provided, I recommend consulting with a healthcare professional for a proper diagnosis. Your symptoms have been recorded for review.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
      setIsProcessing(false);
    }, 2000);
  };

  const handleSendMessage = () => {
    if (inputText.trim() === '') return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsProcessing(true);

    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Thank you for describing your symptoms. Based on the information you\'ve provided throughout the assessment, I recommend scheduling an appointment with a healthcare provider for a thorough evaluation. Would you like to record a voice sample of your cough for additional analysis?',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
      setIsProcessing(false);
    }, 1500);
  };

  const handleGoHome = () => {
    navigation.navigate('Home');
  };

  const micAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  return (
    <View style={styles.container}>
      <AppHeader
        variant="solid"
        rightSlot={
          <TouchableOpacity
            style={styles.homeButton}
            onPress={handleGoHome}
            activeOpacity={0.8}
          >
            <Ionicons name="home" size={16} color="#158B95" />
            <Text style={styles.homeButtonText}>Start New Assessment</Text>
          </TouchableOpacity>
        }
      />

      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
      >
        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map((message, index) => (
            <Animated.View
              key={message.id}
              entering={FadeInDown.delay(index * 40).duration(350).springify()}
              style={[
                styles.messageContainer,
                message.sender === 'user' ? styles.userMessage : styles.botMessage,
              ]}
            >
              <Text style={[
                styles.messageText,
                message.sender === 'user' && styles.userMessageText,
              ]}>
                {message.text}
              </Text>
              <Text style={[
                styles.timestamp,
                message.sender === 'user' && styles.userTimestamp,
              ]}>
                {message.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </Text>
            </Animated.View>
          ))}

          {isProcessing && (
            <View style={styles.botMessage}>
              <Ionicons name="ellipse" size={20} color="#158B95" />
            </View>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          <View style={styles.inputContent}>
            <AnimatedTouchableOpacity
              style={[
                styles.micButton,
                isRecording && styles.micButtonRecording,
                micAnimatedStyle,
              ]}
              onPress={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
              activeOpacity={0.8}
            >
              {isRecording ? (
                <Ionicons name="mic-off" size={24} color="#FFFFFF" />
              ) : (
                <Ionicons name="mic" size={24} color={isRecording ? '#FFFFFF' : '#158B95'} />
              )}
            </AnimatedTouchableOpacity>

            <TextInput
              style={styles.textInput}
              value={inputText}
              onChangeText={setInputText}
              placeholder="Type your message or use voice..."
              multiline
              editable={!isProcessing}
              onSubmitEditing={handleSendMessage}
            />

            <TouchableOpacity
              style={[
                styles.sendButton,
                (inputText.trim() === '' || isProcessing) && styles.sendButtonDisabled,
              ]}
              onPress={handleSendMessage}
              disabled={inputText.trim() === '' || isProcessing}
              activeOpacity={0.8}
            >
              <Ionicons
                name="send"
                size={24}
                color={inputText.trim() === '' || isProcessing ? '#94A3B8' : '#FFFFFF'}
              />
            </TouchableOpacity>
          </View>
        </View>

        {isRecording && (
          <Text style={styles.recordingText}>
            Recording... Click the microphone to stop
          </Text>
        )}

        <Text style={styles.disclaimer}>
          This is not a substitute for professional medical advice. Please consult a healthcare
          provider for diagnosis and treatment.
        </Text>
      </KeyboardAvoidingView>
    </View>
  );
}

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

import { FONTS, COLORS } from '../theme';

// ... (imports remain the same)

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  keyboardView: {
    flex: 1,
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
    letterSpacing: 0.2,
  },
  messagesContainer: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  messagesContent: {
    padding: 24,
    gap: 16,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  messageContainer: {
    maxWidth: '85%', // Increased slightly for better readability
    borderRadius: 20,
    padding: 18,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: COLORS.primary,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#F3F4F6',
  },
  messageText: {
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#1F2937',
    lineHeight: 24,
    letterSpacing: 0.1,
  },
  userMessageText: {
    color: '#FFFFFF',
  },
  timestamp: {
    fontSize: 12,
    fontFamily: FONTS.regular,
    color: '#6B7280',
    marginTop: 6,
    letterSpacing: 0.1,
  },
  userTimestamp: {
    color: 'rgba(255, 255, 255, 0.7)',
  },
  inputContainer: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  inputContent: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 16,
    gap: 12,
    maxWidth: 800,
    width: '100%',
    alignSelf: 'center',
  },
  micButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  micButtonRecording: {
    backgroundColor: COLORS.primary,
  },
  textInput: {
    flex: 1,
    minHeight: 48,
    maxHeight: 120,
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderWidth: 1.5,
    borderColor: '#D1D5DB',
    borderRadius: 24,
    fontSize: 16,
    fontFamily: FONTS.regular,
    color: '#1F2937',
    letterSpacing: 0.1,
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#D1D5DB',
  },
  recordingText: {
    fontSize: 14,
    fontFamily: FONTS.medium,
    color: COLORS.primary,
    textAlign: 'center',
    paddingVertical: 10,
    letterSpacing: 0.1,
  },
  disclaimer: {
    fontSize: 12,
    fontFamily: FONTS.regular,
    color: '#6B7280',
    textAlign: 'center',
    paddingHorizontal: 16,
    paddingBottom: 16,
    lineHeight: 18,
    letterSpacing: 0.1,
  },
});

