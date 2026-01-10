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

import DashboardScreen from '../screens/DashboardScreen';
import NewParticipantScreen from '../screens/NewParticipantScreen';
import ViewRecordScreen from '../screens/ViewRecordScreen';
import PendingResultsScreen from '../screens/PendingResultsScreen';
import ViewDraftsScreen from '../screens/ViewDraftsScreen';
import AddTestResultsScreen from '../screens/AddTestResultsScreen';

export type RootStackParamList = {
  Dashboard: undefined;
  NewParticipant: undefined;
  ViewRecord: { participantId: string };
  PendingResults: undefined;
  ViewDrafts: undefined;
  AddTestResults: { participantId: string };
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
      initialRouteName="Dashboard"
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen name="Dashboard" component={DashboardScreen} />
      <Stack.Screen name="NewParticipant" component={NewParticipantScreen} />
      <Stack.Screen name="ViewRecord" component={ViewRecordScreen} />
      <Stack.Screen name="PendingResults" component={PendingResultsScreen} />
      <Stack.Screen name="ViewDrafts" component={ViewDraftsScreen} />
      <Stack.Screen name="AddTestResults" component={AddTestResultsScreen} />
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

