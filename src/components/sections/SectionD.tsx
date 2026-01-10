/**
 * Section D: Cough & Audio Recording
 */

import React from 'react';
import { View, Text, StyleSheet, Platform, Alert } from 'react-native';
import { Asset } from 'expo-asset';
import { RecordingCard } from '../forms/RecordingCard';
import { ParticipantFormData, AnalysisResult } from '../../types/participantForm';

interface SectionDProps {
    formData: ParticipantFormData;
    updateField: <K extends keyof ParticipantFormData>(field: K, value: ParticipantFormData[K]) => void;
    activeRecordingKey: string | null;
    recordingDuration: number;
    recordedDurations: Record<string, number>;
    analysisResults: Record<string, AnalysisResult>;
    onStartRecording: (key: string) => void;
    onStopRecording: (key: string) => Promise<string | null>;
    onClearRecording: (key: string) => void;
    onUseSample: (key: string) => Promise<void>;
}

export const SectionD: React.FC<SectionDProps> = ({
    formData,
    updateField,
    activeRecordingKey,
    recordingDuration,
    recordedDurations,
    analysisResults,
    onStartRecording,
    onStopRecording,
    onClearRecording,
    onUseSample,
}) => {
    const handleUseSample = async (key: string) => {
        await onUseSample(key);
    };

    const handleStopRecording = async (key: string) => {
        // Get minimum seconds for this recording
        const minSeconds = key === 'recordingBackground' ? 10 : 5;
        // Get current duration - if actively recording this key, use recordingDuration, otherwise use recorded duration
        const currentDuration = activeRecordingKey === key ? recordingDuration : (recordedDurations[key] || 0);
        
        // Validate duration before stopping (safety check - button should already be disabled)
        if (activeRecordingKey === key && currentDuration < minSeconds) {
            Alert.alert(
                'Recording Too Short',
                `Please record for at least ${minSeconds} seconds. Current duration: ${currentDuration} seconds.`,
                [{ text: 'OK' }]
            );
            return;
        }
        
        const uri = await onStopRecording(key);
        if (uri) {
            updateField(key as keyof ParticipantFormData, uri);
        }
    };

    const handleReRecord = (key: string) => {
        updateField(key as keyof ParticipantFormData, null);
        onClearRecording(key);
    };

    return (
        <>
            <View style={styles.protocolBox}>
                <Text style={styles.protocolText}>
                    <Text style={{ fontWeight: '700' }}>Protocol: </Text>
                    Record 3 cough samples (min 5 sec each) and 1 ambient sound recording (min 10 sec). Ask participant to cough naturally into the phone.
                </Text>
            </View>

            <RecordingCard
                title="Cough Recording 1"
                subtitle="Min: 5 seconds"
                recordingKey="recording1"
                isRecorded={!!formData.recording1}
                isRecording={activeRecordingKey === 'recording1'}
                currentDuration={activeRecordingKey === 'recording1' ? recordingDuration : (recordedDurations['recording1'] || 0)}
                minSeconds={5}
                analysis={analysisResults['recording1']}
                onStartRecording={() => onStartRecording('recording1')}
                onStopRecording={() => handleStopRecording('recording1')}
                onReRecord={() => handleReRecord('recording1')}
                onUseSample={() => handleUseSample('recording1')}
            />

            <RecordingCard
                title="Cough Recording 2"
                subtitle="Min: 5 seconds"
                recordingKey="recording2"
                isRecorded={!!formData.recording2}
                isRecording={activeRecordingKey === 'recording2'}
                currentDuration={activeRecordingKey === 'recording2' ? recordingDuration : (recordedDurations['recording2'] || 0)}
                minSeconds={5}
                analysis={analysisResults['recording2']}
                onStartRecording={() => onStartRecording('recording2')}
                onStopRecording={() => handleStopRecording('recording2')}
                onReRecord={() => handleReRecord('recording2')}
                onUseSample={() => handleUseSample('recording2')}
            />

            <RecordingCard
                title="Cough Recording 3"
                subtitle="Min: 5 seconds"
                recordingKey="recording3"
                isRecorded={!!formData.recording3}
                isRecording={activeRecordingKey === 'recording3'}
                currentDuration={activeRecordingKey === 'recording3' ? recordingDuration : (recordedDurations['recording3'] || 0)}
                minSeconds={5}
                analysis={analysisResults['recording3']}
                onStartRecording={() => onStartRecording('recording3')}
                onStopRecording={() => handleStopRecording('recording3')}
                onReRecord={() => handleReRecord('recording3')}
                onUseSample={() => handleUseSample('recording3')}
            />

            <RecordingCard
                title="Ambient Sound Recording"
                subtitle="Min: 10 seconds"
                recordingKey="recordingBackground"
                isRecorded={!!formData.recordingBackground}
                isRecording={activeRecordingKey === 'recordingBackground'}
                currentDuration={activeRecordingKey === 'recordingBackground' ? recordingDuration : (recordedDurations['recordingBackground'] || 0)}
                minSeconds={10}
                analysis={analysisResults['recordingBackground']}
                onStartRecording={() => onStartRecording('recordingBackground')}
                onStopRecording={() => handleStopRecording('recordingBackground')}
                onReRecord={() => handleReRecord('recordingBackground')}
                onUseSample={() => handleUseSample('recordingBackground')}
            />
        </>
    );
};

const styles = StyleSheet.create({
    protocolBox: {
        backgroundColor: '#EFF6FF',
        padding: 16,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#BFDBFE',
        marginBottom: 24,
    },
    protocolText: {
        color: '#1E40AF',
        fontSize: 14,
        lineHeight: 20,
    },
});

