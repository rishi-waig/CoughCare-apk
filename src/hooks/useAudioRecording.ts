/**
 * Custom hook for managing audio recording functionality
 */

import { useState, useRef, useEffect } from 'react';
import { Platform, Alert } from 'react-native';
import { Audio } from 'expo-av';
import { detectCoughFromUrl } from '../utils/onnxInference';
import { AnalysisResult } from '../types/participantForm';
import { AudioRecorder } from '../utils/audioRecorder';

export const useAudioRecording = () => {
    const [activeRecordingKey, setActiveRecordingKey] = useState<string | null>(null);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const [recordedDurations, setRecordedDurations] = useState<Record<string, number>>({});
    const [analysisResults, setAnalysisResults] = useState<Record<string, AnalysisResult>>({});

    const recorderRef = useRef<AudioRecorder | null>(null);
    const recordingInterval = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        (async () => {
            const { status } = await Audio.requestPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert('Permission Required', 'Microphone permission is required to record audio.');
            }
        })();

        // Initialize recorder
        recorderRef.current = new AudioRecorder();

        return () => {
            // Cleanup
            if (recorderRef.current) {
                recorderRef.current.cleanup();
            }
            if (recordingInterval.current) {
                clearInterval(recordingInterval.current);
            }
        };
    }, []);

    const analyzeAudio = async (key: string, uri: string) => {
        setAnalysisResults(prev => ({ ...prev, [key]: { loading: true } }));
        try {
            let audioUri = uri;
            if (Platform.OS !== 'web' && !audioUri.startsWith('http') && !audioUri.startsWith('file://') && !audioUri.startsWith('/')) {
                audioUri = 'file://' + audioUri;
            }

            const result = await detectCoughFromUrl(audioUri);
            console.log(`Analysis Result for ${key}:`, result);
            setAnalysisResults(prev => ({ ...prev, [key]: { loading: false, result } }));
        } catch (error) {
            console.error(`Analysis failed for ${key}:`, error);
            const errorMessage = error instanceof Error ? error.message : String(error);
            setAnalysisResults(prev => ({ ...prev, [key]: { loading: false, error: errorMessage } }));
        }
    };

    const startRecording = async (key: string) => {
        if (activeRecordingKey || !recorderRef.current) return;

        try {
            // Ensure audio mode is set for recording
            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
                staysActiveInBackground: true,
            });

            await recorderRef.current.start();

            setActiveRecordingKey(key);
            setRecordingDuration(0);

            recordingInterval.current = setInterval(() => {
                setRecordingDuration(prev => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Failed to start recording', err);
            Alert.alert('Error', 'Failed to start recording.');
        }
    };

    const stopRecording = async (key: string): Promise<string | null> => {
        if (!recorderRef.current) return null;

        try {
            if (recordingInterval.current) {
                clearInterval(recordingInterval.current);
                recordingInterval.current = null;
            }

            const uri = await recorderRef.current.stop();

            const finalDuration = recordingDuration;
            setActiveRecordingKey(null);
            setRecordingDuration(0);

            if (uri) {
                setRecordedDurations(prev => ({ ...prev, [key]: finalDuration }));
                await analyzeAudio(key, uri);
                return uri;
            }

            return null;
        } catch (error) {
            console.error('Failed to stop recording', error);
            Alert.alert('Error', 'Failed to stop recording.');
            return null;
        }
    };

    const clearRecording = (key: string) => {
        setRecordedDurations(prev => {
            const newDurations = { ...prev };
            delete newDurations[key];
            return newDurations;
        });
        setAnalysisResults(prev => {
            const newResults = { ...prev };
            delete newResults[key];
            return newResults;
        });
    };

    // Get audio duration from file
    const getAudioDuration = async (uri: string): Promise<number> => {
        try {
            if (Platform.OS === 'web') {
                // For web, use HTML5 Audio API
                return new Promise((resolve, reject) => {
                    const audio = new (window as any).Audio(uri);
                    audio.addEventListener('loadedmetadata', () => {
                        const duration = Math.round(audio.duration);
                        if (isFinite(duration) && duration > 0) {
                            resolve(duration);
                        } else {
                            resolve(10); // Fallback
                        }
                    });
                    audio.addEventListener('error', () => {
                        resolve(10); // Fallback on error
                    });
                    audio.load();
                });
            } else {
                // For React Native, use expo-av
                const { sound } = await Audio.Sound.createAsync(
                    { uri },
                    { shouldPlay: false }
                );
                const status = await sound.getStatusAsync();
                await sound.unloadAsync();
                if (status.isLoaded && status.durationMillis) {
                    return Math.round(status.durationMillis / 1000);
                }
                return 10; // Fallback
            }
        } catch (error) {
            console.warn('Could not get audio duration:', error);
            return 10; // Fallback
        }
    };

    // Expose analyzeAudio for manual calls (e.g., sample audio)
    const analyzeAudioManually = async (key: string, uri: string) => {
        // Get actual duration from audio file
        const duration = await getAudioDuration(uri);
        setRecordedDurations(prev => ({ ...prev, [key]: duration }));
        // Then analyze
        await analyzeAudio(key, uri);
    };

    return {
        activeRecordingKey,
        recordingDuration,
        recordedDurations,
        analysisResults,
        startRecording,
        stopRecording,
        clearRecording,
        analyzeAudioManually,
    };
};
