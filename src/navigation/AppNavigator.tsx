import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import ConsentScreen from '../screens/ConsentScreen';
import QuestionsScreen from '../screens/QuestionsScreen';
import CoughRecorderScreen from '../screens/CoughRecorderScreen';
import AnalyzingScreen from '../screens/AnalyzingScreen';
import ResultScreen from '../screens/ResultScreen';
import TbResultScreen from '../screens/TbResultScreen';
import ChatbotScreen from '../screens/ChatbotScreen';

export type RootStackParamList = {
  Home: undefined;
  Consent: undefined;
  Assessment: { reset?: boolean } | undefined;
  RecordCough: undefined;
  Analyzing: { audioUri: string };
  Result: { result: any };
  TbResult: { result: any };
  Analysis: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen name="Consent" component={ConsentScreen} />
      <Stack.Screen name="Assessment" component={QuestionsScreen} />
      <Stack.Screen name="RecordCough" component={CoughRecorderScreen} />
      <Stack.Screen name="Analyzing" component={AnalyzingScreen} />
      <Stack.Screen name="Result" component={ResultScreen} />
      <Stack.Screen name="TbResult" component={TbResultScreen} />
      <Stack.Screen name="Analysis" component={ChatbotScreen} />
    </Stack.Navigator>
  );
}

