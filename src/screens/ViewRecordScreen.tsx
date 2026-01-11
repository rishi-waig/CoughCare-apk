/**
 * View Record Screen - Detailed view of a participant record
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    SafeAreaView,
    StatusBar,
    TouchableOpacity,
    ActivityIndicator,
    TextInput,
    Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { getParticipantById, getRecordingsByParticipantId, saveParticipant, Participant, Recording } from '../services/DatabaseService';
import { Dropdown } from '../components/forms/Dropdown';

type ViewRecordScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'ViewRecord'>;
type ViewRecordScreenRouteProp = RouteProp<RootStackParamList, 'ViewRecord'>;

const ViewRecordScreen = () => {
    const navigation = useNavigation<ViewRecordScreenNavigationProp>();
    const route = useRoute<ViewRecordScreenRouteProp>();
    const participantId = route.params?.participantId;

    const [participant, setParticipant] = useState<Participant | null>(null);
    const [recordings, setRecordings] = useState<Recording[]>([]);
    const [loading, setLoading] = useState(true);
    const [isEditingTestResults, setIsEditingTestResults] = useState(false);
    const [saving, setSaving] = useState(false);
    const [expandedDropdown, setExpandedDropdown] = useState<string | null>(null);

    // Form data for test results
    const [testFormData, setTestFormData] = useState({
        testDone: null as string | null,
        testType: null as string | null,
        dateCollection: '',
        dateResult: '',
        testResult: null as string | null,
        testSite: null as string | null,
        testNotes: '',
    });

    useEffect(() => {
        loadData();
    }, [participantId]);

    // Reload data when screen comes into focus (e.g., after adding test results)
    useFocusEffect(
        React.useCallback(() => {
            if (participantId) {
                loadData();
            }
        }, [participantId])
    );

    const loadData = async () => {
        if (!participantId) {
            setLoading(false);
            return;
        }

        try {
            setLoading(true);
            const [participantData, recordingsData] = await Promise.all([
                getParticipantById(participantId),
                getRecordingsByParticipantId(participantId)
            ]);
            setParticipant(participantData);
            setRecordings(recordingsData);
        } catch (error) {
            console.error('Error loading record:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatValue = (value: any): string => {
        if (value === null || value === undefined || value === '') {
            return 'N/A';
        }
        if (typeof value === 'boolean') {
            return value ? 'Yes' : 'No';
        }
        if (typeof value === 'number') {
            if (value === 0) return 'No';
            if (value === 1) return 'Yes';
            return value.toString();
        }
        return String(value);
    };

    const getStatusBadgeColor = (status?: string) => {
        switch (status) {
            case 'synced':
                return { bg: '#DCFCE7', text: '#16A34A' };
            case 'pending':
                return { bg: '#FEF3C7', text: '#D97706' };
            case 'draft':
                return { bg: '#E0E7FF', text: '#6366F1' };
            default:
                return { bg: '#FEF3C7', text: '#D97706' };
        }
    };

    const parseSymptoms = (symptomsJson?: string) => {
        if (!symptomsJson) return {};
        try {
            return JSON.parse(symptomsJson);
        } catch {
            return {};
        }
    };

    const getRecordingLabel = (type: string): string => {
        const labels: Record<string, string> = {
            'cough_1': 'Cough Recording 1',
            'cough_2': 'Cough Recording 2',
            'cough_3': 'Cough Recording 3',
            'background': 'Ambient Sound',
        };
        return labels[type] || type;
    };

    const formatDateForInput = (dateString?: string | null): string => {
        if (!dateString) return '';
        try {
            const date = new Date(dateString);
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const year = date.getFullYear();
            return `${month}/${day}/${year}`;
        } catch {
            return '';
        }
    };

    const parseDateInput = (dateString: string): string | null => {
        if (!dateString || dateString.trim() === '') return null;
        try {
            // Parse MM/DD/YYYY format
            const parts = dateString.split('/');
            if (parts.length === 3) {
                const month = parseInt(parts[0]) - 1;
                const day = parseInt(parts[1]);
                const year = parseInt(parts[2]);
                const date = new Date(year, month, day);
                if (!isNaN(date.getTime())) {
                    return date.toISOString().split('T')[0]; // Return YYYY-MM-DD
                }
            }
        } catch {
            // Invalid date format
        }
        return null;
    };

    const handleSaveTestResults = async () => {
        if (!participant) return;

        // Validation
        if (!testFormData.testDone) {
            Alert.alert('Validation Error', 'Please select "Was a test done?"');
            return;
        }

        if (testFormData.testDone === 'Yes') {
            if (!testFormData.testType) {
                Alert.alert('Validation Error', 'Test Type is required when test is done');
                return;
            }
            if (!testFormData.testResult) {
                Alert.alert('Validation Error', 'TB Diagnosis Result is required when test is done');
                return;
            }
        }

        try {
            setSaving(true);

            // Parse dates
            const dateCollection = parseDateInput(testFormData.dateCollection);
            const dateResult = parseDateInput(testFormData.dateResult);

            // Map test result values
            const mapTestResult = (result: string | null): string | null => {
                if (!result) return null;
                const mapping: Record<string, string> = {
                    'TB Positive': 'Positive',
                    'TB Negative': 'Negative',
                    'RIF Resistant': 'Inconclusive',
                    'Indeterminate': 'Inconclusive',
                    'Pending': 'Pending'
                };
                return mapping[result] || result;
            };

            // Update participant
            await saveParticipant({
                ...participant,
                test_done: testFormData.testDone,
                test_type: testFormData.testType,
                test_date_collection: dateCollection,
                test_date_result: dateResult,
                test_result: mapTestResult(testFormData.testResult),
                test_site: testFormData.testSite,
                test_notes: testFormData.testNotes || null,
            });

            // Reload data
            await loadData();
            setIsEditingTestResults(false);
            setExpandedDropdown(null);

            Alert.alert('Success', 'Test results saved successfully!');
        } catch (error) {
            console.error('Error saving test results:', error);
            Alert.alert('Error', 'Failed to save test results. Please try again.');
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <SafeAreaView style={styles.container}>
                <StatusBar barStyle="light-content" backgroundColor="#2563EB" />
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#2563EB" />
                    <Text style={styles.loadingText}>Loading record...</Text>
                </View>
            </SafeAreaView>
        );
    }

    if (!participant) {
        return (
            <SafeAreaView style={styles.container}>
                <StatusBar barStyle="light-content" backgroundColor="#2563EB" />
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                        <Ionicons name="arrow-back" size={24} color="white" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>View Record</Text>
                </View>
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>Record not found</Text>
                </View>
            </SafeAreaView>
        );
    }

    const symptoms = parseSymptoms(participant.symptoms);
    const statusBadge = getStatusBadgeColor(participant.status);
    const analysisResult = participant.analysis_result ? JSON.parse(participant.analysis_result) : null;

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#2563EB" />

            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="white" />
                </TouchableOpacity>
                <View style={{ flex: 1 }}>
                    <Text style={styles.headerTitle}>View Record</Text>
                    {participant.status !== 'draft' && (
                        <Text style={styles.headerSubtitle}>Can add test results</Text>
                    )}
                </View>
                <View style={[styles.statusBadge, { backgroundColor: statusBadge.bg }]}>
                    <Text style={[styles.statusBadgeText, { color: statusBadge.text }]}>
                        {participant.status === 'synced' ? 'Synced' : participant.status === 'pending' ? 'Unsynced' : 'Draft'}
                    </Text>
                </View>
            </View>

            <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 300 }}>
                {/* Individual Details */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="person" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                        <Text style={styles.sectionTitle}>Individual Details</Text>
                    </View>
                    <View style={styles.sectionContent}>
                        <InfoRow label="Full Name" value={formatValue(participant.full_name)} />
                        <InfoRow label="Mobile Number" value={formatValue(participant.mobile_number)} />
                        <InfoRow label="Age" value={participant.age ? `${participant.age} years` : 'N/A'} />
                        <InfoRow label="Gender" value={formatValue(participant.gender)} />
                        <InfoRow label="Screening Date" value={formatValue(participant.date_of_screening)} />
                        <InfoRow
                            label="Consent"
                            value={participant.consent_obtained === 1 ? 'Yes' : 'No'}
                            highlight={participant.consent_obtained === 1}
                        />
                    </View>
                </View>

                {/* Location */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="location" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                        <Text style={styles.sectionTitle}>Location</Text>
                    </View>
                    <View style={styles.sectionContent}>
                        <InfoRow label="Region" value={formatValue(participant.region)} />
                        <InfoRow label="District" value={formatValue(participant.district)} />
                        <InfoRow label="Facility" value={formatValue(participant.facility)} />
                        {participant.community && (
                            <InfoRow label="Community" value={formatValue(participant.community)} />
                        )}
                    </View>
                </View>

                {/* Health History */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="pulse" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                        <Text style={styles.sectionTitle}>Health History</Text>
                    </View>
                    <View style={styles.sectionContent}>
                        <InfoRow label="Diabetes" value={formatValue(participant.diabetes_status)} />
                        <InfoRow label="HIV Status" value={formatValue(participant.hiv_status)} />
                        <InfoRow label="COVID-19" value={formatValue(participant.covid_status)} />
                        <InfoRow label="Tobacco Use" value={formatValue(participant.tobacco_use)} />
                        {participant.tobacco_use === 1 && participant.tobacco_duration && (
                            <InfoRow label="Tobacco Duration" value={formatValue(participant.tobacco_duration)} />
                        )}
                        <InfoRow label="Alcohol Use" value={formatValue(participant.alcohol_use)} />
                        {participant.alcohol_use === 1 && participant.alcohol_duration && (
                            <InfoRow label="Alcohol Duration" value={formatValue(participant.alcohol_duration)} />
                        )}
                        <InfoRow label="Previous TB" value={formatValue(participant.previous_tb)} />
                        {participant.previous_tb === 1 && participant.last_tb_year && (
                            <InfoRow label="Last TB Year" value={formatValue(participant.last_tb_year)} />
                        )}
                        {participant.previous_tb === 1 && participant.tb_treatment_completed && (
                            <InfoRow label="TB Treatment Completed" value={formatValue(participant.tb_treatment_completed)} />
                        )}
                    </View>
                </View>

                {/* Symptoms */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="pulse" size={20} color="#EF4444" style={{ marginRight: 8 }} />
                        <Text style={styles.sectionTitle}>Symptoms</Text>
                    </View>
                    <View style={styles.sectionContent}>
                        {Object.keys(symptoms).length === 0 ? (
                            <Text style={styles.emptyText}>No symptoms reported</Text>
                        ) : (
                            <>
                                {Object.entries(symptoms).map(([key, value]: [string, any]) => {
                                    if (value?.present === true) {
                                        return (
                                            <InfoRow
                                                key={key}
                                                label={key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                                                value={value.duration ? `${value.duration} days` : 'Yes'}
                                            />
                                        );
                                    }
                                    return null;
                                })}
                            </>
                        )}
                    </View>
                </View>

                {/* Audio Recordings */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="mic" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                        <Text style={styles.sectionTitle}>Audio Recordings</Text>
                    </View>
                    <View style={styles.sectionContent}>
                        {(() => {
                            // Create a map to ensure we only show one recording per type (latest if duplicates exist)
                            const recordingMap = new Map<string, Recording>();
                            recordings.forEach(rec => {
                                const existing = recordingMap.get(rec.recording_type);
                                if (!existing || (rec.id && existing.id && rec.id > existing.id)) {
                                    recordingMap.set(rec.recording_type, rec);
                                }
                            });

                            // Show all 4 recording types, with actual data if available
                            const recordingTypes = ['cough_1', 'cough_2', 'cough_3', 'background'];
                            return recordingTypes.map(type => {
                                const recording = recordingMap.get(type);
                                return (
                                    <InfoRow
                                        key={type}
                                        label={getRecordingLabel(type)}
                                        value={recording && recording.duration
                                            ? `${recording.duration} seconds`
                                            : 'Not recorded'}
                                    />
                                );
                            });
                        })()}
                    </View>
                </View>

                {/* Analysis Results */}
                {analysisResult && (
                    <View style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <Ionicons name="analytics" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                            <Text style={styles.sectionTitle}>Analysis Results</Text>
                        </View>
                        <View style={styles.sectionContent}>
                            <InfoRow
                                label="Cough Detected"
                                value={analysisResult.coughDetected ? 'Yes' : 'No'}
                                highlight={analysisResult.coughDetected}
                            />
                            {analysisResult.confidence && (
                                <InfoRow
                                    label="Confidence"
                                    value={`${(analysisResult.confidence * 100).toFixed(1)}%`}
                                />
                            )}
                        </View>
                    </View>
                )}

                {/* Diagnostic Testing */}
                <View style={styles.section}>
                    <View style={styles.sectionHeader}>
                        <Ionicons name="document-text" size={20} color="#2563EB" style={{ marginRight: 8 }} />
                        <View style={{ flex: 1, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Text style={styles.sectionTitle}>Diagnostic Testing</Text>
                            {/* Only show Add Results button if pending and not editing */}
                            {participant.status === 'pending' && !isEditingTestResults && (
                                <TouchableOpacity
                                    style={styles.addButton}
                                    onPress={() => {
                                        // Initialize form with existing data
                                        setTestFormData({
                                            testDone: participant.test_done || null,
                                            testType: participant.test_type || null,
                                            dateCollection: participant.test_date_collection ? formatDateForInput(participant.test_date_collection) : '',
                                            dateResult: participant.test_date_result ? formatDateForInput(participant.test_date_result) : '',
                                            testResult: participant.test_result || null,
                                            testSite: participant.test_site || null,
                                            testNotes: participant.test_notes || '',
                                        });
                                        setIsEditingTestResults(true);
                                    }}
                                >
                                    <Ionicons name="pencil" size={16} color="#2563EB" style={{ marginRight: 4 }} />
                                    <Text style={styles.addButtonText}>Add Results</Text>
                                </TouchableOpacity>
                            )}
                        </View>
                    </View>
                    <View style={styles.sectionContent}>
                        {isEditingTestResults ? (
                            // Edit Form
                            <View style={styles.testFormContainer}>
                                <Dropdown
                                    label="Was a test done?"
                                    value={testFormData.testDone}
                                    options={['Yes', 'No', 'Not yet']}
                                    onSelect={(val) => {
                                        const value = val === 'Select' ? null : val;
                                        if (val === 'No' || val === 'Not yet') {
                                            setTestFormData({
                                                ...testFormData,
                                                testDone: value,
                                                testType: null,
                                                testResult: null,
                                                testSite: null,
                                            });
                                        } else {
                                            setTestFormData({ ...testFormData, testDone: value });
                                        }
                                    }}
                                    isExpanded={expandedDropdown === 'testDone'}
                                    onToggle={() => setExpandedDropdown(expandedDropdown === 'testDone' ? null : 'testDone')}
                                    placeholder="Select"
                                />

                                {testFormData.testDone === 'Yes' && (
                                    <>
                                        <Dropdown
                                            label="Test Type"
                                            value={testFormData.testType}
                                            options={[
                                                'Select test type',
                                                'GeneXpert',
                                                'Smear Microscopy',
                                                'Culture',
                                                'Chest X-Ray (CXR)',
                                                'Other'
                                            ]}
                                            onSelect={(val) => {
                                                setTestFormData({ ...testFormData, testType: val === 'Select test type' ? null : val });
                                            }}
                                            isExpanded={expandedDropdown === 'testType'}
                                            onToggle={() => setExpandedDropdown(expandedDropdown === 'testType' ? null : 'testType')}
                                            placeholder="Select test type"
                                        />

                                        <View style={styles.formGroup}>
                                            <Text style={styles.formLabel}>Date of Specimen Collection</Text>
                                            <TextInput
                                                style={styles.formInput}
                                                placeholder="DD/MM/YYYY"
                                                value={testFormData.dateCollection}
                                                onChangeText={(text) => {
                                                    let cleaned = text.replace(/\D/g, '');
                                                    let formatted = cleaned;
                                                    if (cleaned.length > 2) {
                                                        formatted = cleaned.slice(0, 2) + '/' + cleaned.slice(2);
                                                    }
                                                    if (cleaned.length > 4) {
                                                        formatted = formatted.slice(0, 5) + '/' + cleaned.slice(4, 8);
                                                    }
                                                    setTestFormData({ ...testFormData, dateCollection: formatted });
                                                }}
                                                keyboardType="numeric"
                                                maxLength={10}
                                            />
                                        </View>

                                        <View style={styles.formGroup}>
                                            <Text style={styles.formLabel}>Date of Result</Text>
                                            <TextInput
                                                style={styles.formInput}
                                                placeholder="DD/MM/YYYY"
                                                value={testFormData.dateResult}
                                                onChangeText={(text) => {
                                                    let cleaned = text.replace(/\D/g, '');
                                                    let formatted = cleaned;
                                                    if (cleaned.length > 2) {
                                                        formatted = cleaned.slice(0, 2) + '/' + cleaned.slice(2);
                                                    }
                                                    if (cleaned.length > 4) {
                                                        formatted = formatted.slice(0, 5) + '/' + cleaned.slice(4, 8);
                                                    }
                                                    setTestFormData({ ...testFormData, dateResult: formatted });
                                                }}
                                                keyboardType="numeric"
                                                maxLength={10}
                                            />
                                        </View>

                                        <Dropdown
                                            label="TB Diagnosis Result"
                                            value={testFormData.testResult}
                                            options={[
                                                'Select result',
                                                'TB Positive',
                                                'TB Negative',
                                                'RIF Resistant',
                                                'Indeterminate',
                                                'Pending'
                                            ]}
                                            onSelect={(val) => {
                                                setTestFormData({ ...testFormData, testResult: val === 'Select result' ? null : val });
                                            }}
                                            isExpanded={expandedDropdown === 'testResult'}
                                            onToggle={() => setExpandedDropdown(expandedDropdown === 'testResult' ? null : 'testResult')}
                                            placeholder="Select result"
                                        />

                                        <Dropdown
                                            label="Site of Disease"
                                            value={testFormData.testSite}
                                            options={[
                                                'Select site',
                                                'Pulmonary',
                                                'Extra-pulmonary',
                                                'Both',
                                                'Unknown'
                                            ]}
                                            onSelect={(val) => {
                                                setTestFormData({ ...testFormData, testSite: val === 'Select site' ? null : val });
                                            }}
                                            isExpanded={expandedDropdown === 'testSite'}
                                            onToggle={() => setExpandedDropdown(expandedDropdown === 'testSite' ? null : 'testSite')}
                                            placeholder="Select site"
                                        />

                                        <View style={styles.formGroup}>
                                            <Text style={styles.formLabel}>Additional Notes (Optional)</Text>
                                            <TextInput
                                                style={[styles.formInput, styles.textArea]}
                                                placeholder="Enter any additional notes"
                                                value={testFormData.testNotes}
                                                onChangeText={(text) => {
                                                    setTestFormData({ ...testFormData, testNotes: text });
                                                }}
                                                multiline
                                                numberOfLines={4}
                                                textAlignVertical="top"
                                            />
                                        </View>
                                    </>
                                )}
                            </View>
                        ) : (
                            // View Mode
                            <>
                                {!participant.test_done || participant.test_done === 'Not yet' ? (
                                    <Text style={styles.emptyText}>No diagnostic testing results added yet</Text>
                                ) : (
                                    <>
                                        <InfoRow label="Test Done" value={formatValue(participant.test_done)} />
                                        {participant.test_type && (
                                            <InfoRow label="Test Type" value={formatValue(participant.test_type)} />
                                        )}
                                        {participant.test_date_collection && (
                                            <InfoRow label="Collection Date" value={formatValue(participant.test_date_collection)} />
                                        )}
                                        {participant.test_date_result && (
                                            <InfoRow label="Result Date" value={formatValue(participant.test_date_result)} />
                                        )}
                                        {participant.test_result && (
                                            <InfoRow
                                                label="Result"
                                                value={formatValue(participant.test_result)}
                                                highlight={participant.test_result === 'Positive' || participant.test_result === 'TB Positive'}
                                            />
                                        )}
                                        {participant.test_site && (
                                            <InfoRow label="Site" value={formatValue(participant.test_site)} />
                                        )}
                                        {participant.test_notes && (
                                            <InfoRow label="Notes" value={formatValue(participant.test_notes)} />
                                        )}
                                    </>
                                )}
                            </>
                        )}
                    </View>
                </View>

            </ScrollView>

            {/* Cancel and Save Results Buttons - Only show when editing */}
            {
                isEditingTestResults && (
                    <View style={styles.footer}>
                        <TouchableOpacity
                            style={styles.cancelButton}
                            onPress={() => {
                                setIsEditingTestResults(false);
                                setExpandedDropdown(null);
                            }}
                            disabled={saving}
                        >
                            <Text style={styles.cancelButtonText}>Cancel</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.saveButton, saving && styles.saveButtonDisabled]}
                            onPress={handleSaveTestResults}
                            disabled={saving}
                        >
                            {saving ? (
                                <Text style={styles.saveButtonText}>Saving...</Text>
                            ) : (
                                <>
                                    <Ionicons name="document-text" size={18} color="white" style={{ marginRight: 6 }} />
                                    <Text style={styles.saveButtonText}>Save Results</Text>
                                </>
                            )}
                        </TouchableOpacity>
                    </View>
                )
            }
        </SafeAreaView >
    );
};

// Info Row Component
const InfoRow: React.FC<{ label: string; value: string; highlight?: boolean }> = ({ label, value, highlight }) => (
    <View style={styles.infoRow}>
        <Text style={styles.infoLabel}>{label}</Text>
        <View style={styles.infoValueContainer}>
            {highlight && value === 'Yes' ? (
                <View style={styles.highlightBadge}>
                    <Text style={styles.highlightText}>{value}</Text>
                </View>
            ) : (
                <Text style={[styles.infoValue, highlight && value === 'Positive' && styles.positiveValue]}>
                    {value}
                </Text>
            )}
        </View>
    </View>
);

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#F8FAFC',
    },
    header: {
        backgroundColor: '#2563EB',
        padding: 16,
        paddingTop: 40,
        flexDirection: 'row',
        alignItems: 'center',
    },
    backButton: {
        marginRight: 16,
    },
    headerTitle: {
        color: 'white',
        fontSize: 18,
        fontWeight: '600',
    },
    headerSubtitle: {
        color: '#BFDBFE',
        fontSize: 12,
        marginTop: 2,
    },
    statusBadge: {
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 12,
    },
    statusBadgeText: {
        fontSize: 12,
        fontWeight: '600',
    },
    content: {
        flex: 1,
        padding: 16,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 12,
        color: '#64748B',
        fontSize: 14,
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    errorText: {
        color: '#EF4444',
        fontSize: 16,
    },
    section: {
        backgroundColor: 'white',
        borderRadius: 12,
        marginBottom: 16,
    },
    sectionHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#E2E8F0',
    },
    sectionTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1E293B',
    },
    sectionContent: {
        padding: 16,
    },
    infoRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    infoLabel: {
        fontSize: 14,
        color: '#64748B',
        flex: 1,
    },
    infoValueContainer: {
        flex: 1,
        alignItems: 'flex-end',
    },
    infoValue: {
        fontSize: 14,
        color: '#1E293B',
        fontWeight: '500',
        textAlign: 'right',
    },
    highlightBadge: {
        backgroundColor: '#DCFCE7',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 6,
    },
    highlightText: {
        color: '#16A34A',
        fontSize: 12,
        fontWeight: '600',
    },
    positiveValue: {
        color: '#EF4444',
        fontWeight: '600',
    },
    emptyText: {
        fontSize: 14,
        color: '#94A3B8',
        fontStyle: 'italic',
    },
    addButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 6,
        backgroundColor: '#EFF6FF',
    },
    addButtonText: {
        color: '#2563EB',
        fontSize: 12,
        fontWeight: '600',
    },
    addTestButton: {
        backgroundColor: '#2563EB',
        padding: 16,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 8,
    },
    addTestButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
    testFormContainer: {
        paddingVertical: 8,
    },
    formGroup: {
        marginBottom: 16,
    },
    formLabel: {
        fontSize: 14,
        fontWeight: '600',
        color: '#1E293B',
        marginBottom: 8,
    },
    formInput: {
        borderWidth: 1,
        borderColor: '#E2E8F0',
        borderRadius: 8,
        padding: 12,
        fontSize: 14,
        color: '#1E293B',
        backgroundColor: '#FFFFFF',
    },
    textArea: {
        minHeight: 100,
        textAlignVertical: 'top',
    },
    footer: {
        flexDirection: 'row',
        padding: 16,
        backgroundColor: 'white',
        borderTopWidth: 1,
        borderTopColor: '#E2E8F0',
        gap: 12,
    },
    cancelButton: {
        flex: 1,
        padding: 14,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
        alignItems: 'center',
    },
    cancelButtonText: {
        color: '#64748B',
        fontSize: 16,
        fontWeight: '600',
    },
    saveButton: {
        flex: 1,
        padding: 14,
        borderRadius: 8,
        backgroundColor: '#22C55E',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
    saveButtonDisabled: {
        backgroundColor: '#94A3B8',
    },
    saveButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
});

export default ViewRecordScreen;

