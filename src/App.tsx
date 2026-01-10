import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import AppNavigator from './navigation/AppNavigator';
import { useAppFonts } from './hooks/useAppFonts';
import { View, ActivityIndicator, Platform } from 'react-native';
import { api } from './utils/api';

import { initDatabase } from './services/DatabaseService';

export default function App() {
  const fontsLoaded = useAppFonts();

  // Initialize DB and pre-load ONNX models
  useEffect(() => {
    initDatabase().catch(console.error);

    if (Platform.OS === 'web') {
      api.initClientONNX().catch(console.warn);
    }
  }, []);

  if (!fontsLoaded) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#158B95" />
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <StatusBar style="auto" />
        <AppNavigator />
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}
