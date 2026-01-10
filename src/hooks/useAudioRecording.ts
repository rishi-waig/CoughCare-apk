/**
 * Custom hook for managing audio recording functionality
 */

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

