/**
 * Recording Card Component for audio recording
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AnalysisResult } from '../../types/participantForm';

interface RecordingCardProps {
    title: string;
    subtitle: string;
    recordingKey: string;
    isRecorded: boolean;
    isRecording: boolean;
    currentDuration: number;
    minSeconds: number;
    analysis?: AnalysisResult;
    onStartRecording: () => void;
    onStopRecording: () => void;
    onReRecord: () => void;
    onUseSample?: () => void;
}

export const RecordingCard: React.FC<RecordingCardProps> = ({
    title,
    subtitle,
    isRecorded,
    isRecording,
    currentDuration,
    minSeconds,
    analysis,
    onStartRecording,
    onStopRecording,
    onReRecord,
    onUseSample,
}) => {
    const progress = Math.min(currentDuration / minSeconds, 1);
    const remaining = Math.max(minSeconds - currentDuration, 0);
    const meetsMinimum = currentDuration >= minSeconds;
    const isTooShort = isRecorded && !isRecording && currentDuration < minSeconds;
    // For recording in progress, check if we can stop
    const canStop = isRecording && meetsMinimum;

    let cardStyle = styles.recordingCard;
    if (isRecording) cardStyle = styles.recordingCardRecording;
    if (isRecorded && meetsMinimum) cardStyle = styles.recordingCardDone;
    if (isTooShort) cardStyle = styles.recordingCardError;

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    };

    return (
        <View style={cardStyle}>
            <View style={styles.recordingHeader}>
                <View>
                    <Text style={styles.recordingTitle}>{title}</Text>
                    <Text style={styles.recordingSubtitle}>{subtitle}</Text>
                </View>
                {isRecorded ? (
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <Ionicons name="checkmark" size={20} color="#22C55E" />
                        <Text style={{ color: '#22C55E', fontWeight: '600', marginLeft: 4 }}>Done</Text>
                    </View>
                ) : (
                    !isRecording && onUseSample && (
                        <TouchableOpacity
                            onPress={onUseSample}
                            style={{ padding: 4 }}
                        >
                            <Ionicons name="flask-outline" size={24} color="#475569" />
                        </TouchableOpacity>
                    )
                )}
            </View>

            {/* Progress Bar */}
            <View style={styles.progressBarBackground}>
                <View style={[
                    styles.progressBarFill,
                    { width: `${progress * 100}%` },
                    isRecording && remaining > 0 ? { backgroundColor: '#EF4444' } : { backgroundColor: '#22C55E' }
                ]} />
            </View>

            <Text style={styles.timerText}>
                {formatTime(isRecorded && !isRecording ? Math.max(currentDuration, 1) : currentDuration)}
            </Text>

            {/* Status Text for Recording */}
            {isRecording && (
                <Text style={[
                    styles.statusText,
                    remaining > 0 ? { color: '#EF4444' } : { color: '#22C55E' }
                ]}>
                    {remaining > 0
                        ? `Record at least ${remaining} more seconds`
                        : "✓ Minimum duration reached"}
                </Text>
            )}

            {/* Analysis Result Display - Only show when not loading */}
            {isRecorded && !isRecording && !analysis?.loading && (
                <View style={{ marginTop: 8, marginBottom: 16, alignItems: 'center' }}>
                    {analysis?.error ? (
                        <View style={{
                            flexDirection: 'row',
                            alignItems: 'center',
                            backgroundColor: '#FEF2F2',
                            paddingHorizontal: 12,
                            paddingVertical: 6,
                            borderRadius: 16,
                            borderWidth: 1,
                            borderColor: '#FECACA'
                        }}>
                            <Ionicons name="alert-circle" size={18} color="#EF4444" style={{ marginRight: 6 }} />
                            <Text style={{ color: '#B91C1C', fontSize: 14, fontWeight: '600' }}>
                                Analysis Error
                            </Text>
                        </View>
                    ) : analysis?.result ? (
                        <View style={{
                            flexDirection: 'row',
                            alignItems: 'center',
                            backgroundColor: analysis.result.coughDetected ? '#FEF2F2' : '#F0FDF4',
                            paddingHorizontal: 12,
                            paddingVertical: 6,
                            borderRadius: 16,
                            borderWidth: 1,
                            borderColor: analysis.result.coughDetected ? '#FECACA' : '#BBF7D0'
                        }}>
                            <Ionicons
                                name={analysis.result.coughDetected ? "alert-circle" : "checkmark-circle"}
                                size={18}
                                color={analysis.result.coughDetected ? "#EF4444" : "#22C55E"}
                                style={{ marginRight: 6 }}
                            />
                            <Text style={{
                                color: analysis.result.coughDetected ? "#B91C1C" : "#15803D",
                                fontWeight: '600',
                                fontSize: 14
                            }}>
                                {analysis.result.coughDetected ? "Cough Detected" : "No Cough Detected"}
                                {analysis.result.confidence && ` (${(analysis.result.confidence * 100).toFixed(0)}%)`}
                            </Text>
                        </View>
                    ) : null}
                </View>
            )}

            {/* Error Message for Too Short Recording */}
            {isTooShort && (
                <View style={styles.errorBanner}>
                    <Ionicons name="warning" size={18} color="#F59E0B" style={{ marginRight: 8 }} />
                    <Text style={styles.errorBannerText}>
                        Recording too short. Please record again (minimum {minSeconds} seconds).
                    </Text>
                </View>
            )}

            {/* Buttons - Hide during analysis, show loader */}
            {analysis?.loading ? (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#2563EB" />
                    <Text style={styles.loadingText}>Analyzing audio...</Text>
                </View>
            ) : isRecording ? (
                <TouchableOpacity
                    style={[
                        styles.stopBtn,
                        !canStop && styles.stopBtnDisabled
                    ]}
                    onPress={canStop ? onStopRecording : undefined}
                    disabled={!canStop}
                >
                    <View style={[
                        styles.stopIcon,
                        !canStop && styles.stopIconDisabled
                    ]} />
                    <Text style={[
                        styles.stopBtnText,
                        !canStop && styles.stopBtnTextDisabled
                    ]}>
                        {canStop ? 'Stop' : `Record ${remaining} more second${remaining !== 1 ? 's' : ''}`}
                    </Text>
                </TouchableOpacity>
            ) : isRecorded ? (
                <TouchableOpacity
                    style={styles.reRecordBtn}
                    onPress={onReRecord}
                >
                    <Ionicons name="refresh" size={20} color="white" style={{ marginRight: 8 }} />
                    <Text style={styles.reRecordBtnText}>Re-record</Text>
                </TouchableOpacity>
            ) : (
                <TouchableOpacity
                    style={styles.startRecordingBtn}
                    onPress={onStartRecording}
                >
                    <Ionicons name="mic" size={20} color="white" style={{ marginRight: 8 }} />
                    <Text style={styles.startRecordingText}>Start Recording</Text>
                </TouchableOpacity>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    recordingCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    recordingCardRecording: {
        backgroundColor: '#FEF2F2',
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#FECACA',
    },
    recordingCardDone: {
        backgroundColor: '#F0FDF4',
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#BBF7D0',
    },
    recordingCardError: {
        backgroundColor: '#FEF2F2',
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#FECACA',
    },
    recordingHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 16,
    },
    recordingTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1E293B',
        marginBottom: 4,
    },
    recordingSubtitle: {
        fontSize: 12,
        color: '#64748B',
    },
    progressBarBackground: {
        height: 8,
        backgroundColor: '#E2E8F0',
        borderRadius: 4,
        marginBottom: 8,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        borderRadius: 4,
        backgroundColor: '#22C55E',
    },
    timerText: {
        fontSize: 24,
        fontWeight: '700',
        color: '#1E293B',
        textAlign: 'center',
        marginBottom: 8,
        fontVariant: ['tabular-nums'],
    },
    statusText: {
        fontSize: 14,
        textAlign: 'center',
        marginBottom: 16,
        fontWeight: '500',
    },
    startRecordingBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#EF4444',
        padding: 12,
        borderRadius: 8,
    },
    startRecordingText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 14,
    },
    stopBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'white',
        padding: 12,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#EF4444',
    },
    stopIcon: {
        width: 12,
        height: 12,
        backgroundColor: '#EF4444',
        borderRadius: 2,
        marginRight: 8,
    },
    stopBtnText: {
        color: '#EF4444',
        fontWeight: '600',
        fontSize: 14,
    },
    stopBtnDisabled: {
        backgroundColor: '#F3F4F6',
        borderColor: '#D1D5DB',
        opacity: 0.6,
    },
    stopIconDisabled: {
        backgroundColor: '#9CA3AF',
    },
    stopBtnTextDisabled: {
        color: '#6B7280',
    },
    errorBanner: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FEF3C7',
        padding: 12,
        borderRadius: 8,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#FDE68A',
    },
    errorBannerText: {
        flex: 1,
        color: '#92400E',
        fontSize: 14,
        fontWeight: '500',
    },
    reRecordBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#64748B',
        padding: 12,
        borderRadius: 8,
    },
    reRecordBtnText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 14,
    },
    loadingContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
    },
    loadingText: {
        marginTop: 12,
        color: '#2563EB',
        fontSize: 14,
        fontWeight: '500',
    },
});

