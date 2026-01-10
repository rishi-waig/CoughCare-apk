/**
 * Add Test Results Screen - Form to add/edit diagnostic test results
 */

import { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    SafeAreaView,
    StatusBar,
    TouchableOpacity,
    TextInput,
    Platform,
    Alert,
    Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { getParticipantById, saveParticipant, Participant } from '../services/DatabaseService';
import { Dropdown } from '../components/forms/Dropdown';

type AddTestResultsScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'AddTestResults'>;
type AddTestResultsScreenRouteProp = RouteProp<RootStackParamList, 'AddTestResults'>;

interface TestResultsFormData {
    testDone: string | null;
    testType: string | null;
    dateCollection: string;
    dateResult: string;
    testResult: string | null;
    testSite: string | null;
    testNotes: string;
}

const AddTestResultsScreen = () => {
    const navigation = useNavigation<AddTestResultsScreenNavigationProp>();
    const route = useRoute<AddTestResultsScreenRouteProp>();
    const participantId = route.params?.participantId;

    const [participant, setParticipant] = useState<Participant | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [formData, setFormData] = useState<TestResultsFormData>({
        testDone: null,
        testType: null,
        dateCollection: '',
        dateResult: '',
        testResult: null,
        testSite: null,
        testNotes: '',
    });
    const [showValidationModal, setShowValidationModal] = useState(false);
    const [validationErrors, setValidationErrors] = useState<string[]>([]);
    const [expandedDropdown, setExpandedDropdown] = useState<string | null>(null);

    useEffect(() => {
        loadParticipant();
    }, [participantId]);

    const loadParticipant = async () => {
        if (!participantId) {
            setLoading(false);
            return;
        }

        try {
            setLoading(true);
            const data = await getParticipantById(participantId);
            if (data) {
                setParticipant(data);
                // Pre-fill form with existing data
                setFormData({
                    testDone: data.test_done || null,
                    testType: data.test_type || null,
                    dateCollection: data.test_date_collection || '',
                    dateResult: data.test_date_result || '',
                    testResult: data.test_result || null,
                    testSite: data.test_site || null,
                    testNotes: data.test_notes || '',
                });
            }
        } catch (error) {
            console.error('Error loading participant:', error);
            Alert.alert('Error', 'Failed to load participant data.');
        } finally {
            setLoading(false);
        }
    };

    const validateForm = (): boolean => {
        const errors: string[] = [];

        if (!formData.testDone) {
            errors.push('"Was a test done?" is required');
        }

        if (formData.testDone === 'Yes') {
            if (!formData.testType) {
                errors.push('Test Type is required when test is done');
            }
            if (!formData.testResult) {
                errors.push('TB Diagnosis Result is required when test is done');
            }
        }

        setValidationErrors(errors);
        return errors.length === 0;
    };

    const handleSave = async () => {
        if (!validateForm()) {
            if (Platform.OS === 'web') {
                setShowValidationModal(true);
            } else {
                Alert.alert(
                    'Validation Error',
                    `Please fix the following errors:\n\n${validationErrors.join('\n')}`
                );
            }
            return;
        }

        if (!participant) {
            Alert.alert('Error', 'Participant data not found.');
            return;
        }

        try {
            setSaving(true);

            // Parse dates before saving
            const dateCollection = parseDateInput(formData.dateCollection);
            const dateResult = parseDateInput(formData.dateResult);

            // Map test result values to match PRD requirements
            const mapTestResult = (result: string | null): string | null => {
                if (!result) return null;
                const mapping: Record<string, string> = {
                    'TB Positive': 'Positive',
                    'TB Negative': 'Negative',
                    'RIF Resistant': 'Inconclusive', // Map to closest match
                    'Indeterminate': 'Inconclusive',
                    'Pending': 'Pending'
                };
                return mapping[result] || result;
            };

            // Update participant with test results
            await saveParticipant({
                ...participant,
                test_done: formData.testDone,
                test_type: formData.testType,
                test_date_collection: dateCollection,
                test_date_result: dateResult,
                test_result: mapTestResult(formData.testResult),
                test_site: formData.testSite,
                test_notes: formData.testNotes || null,
            });

            console.log('Test results saved successfully!');

            if (Platform.OS === 'web') {
                Alert.alert('Success', 'Test results saved successfully!', [
                    { text: 'OK', onPress: () => navigation.goBack() }
                ]);
            } else {
                Alert.alert('Success', 'Test results saved successfully!', [
                    { text: 'OK', onPress: () => navigation.goBack() }
                ]);
            }
        } catch (error) {
            console.error('Error saving test results:', error);
            Alert.alert('Error', 'Failed to save test results. Please try again.');
        } finally {
            setSaving(false);
        }
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

    if (loading) {
        return (
            <SafeAreaView style={styles.container}>
                <StatusBar barStyle="light-content" backgroundColor="#2563EB" />
                <View style={styles.loadingContainer}>
                    <Text style={styles.loadingText}>Loading...</Text>
                </View>
            </SafeAreaView>
        );
    }

    if (!participant) {
        return (
            <SafeAreaView style={styles.container}>
                <StatusBar barStyle="light-content" backgroundColor="#2563EB" />
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>Participant not found</Text>
                    <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
                        <Text style={styles.backButtonText}>Go Back</Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#2563EB" />

            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="white" />
                </TouchableOpacity>
                <View style={{ flex: 1 }}>
                    <Text style={styles.headerTitle}>Diagnostic Testing</Text>
                    <Text style={styles.headerSubtitle}>Add Test Results</Text>
                </View>
            </View>

            <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 300 }}>
                <View style={styles.formCard}>
                    {/* Was a test done? */}
                    <View style={styles.formGroup}>
                        <Dropdown
                            label="Was a test done?"
                            value={formData.testDone}
                            options={['Yes', 'No', 'Not yet']}
                            onSelect={(val) => {
                                const value = val === 'Select' ? null : val;
                                // Clear dependent fields if "No" or "Not yet"
                                if (val === 'No' || val === 'Not yet') {
                                    setFormData({
                                        ...formData,
                                        testDone: value,
                                        testType: null,
                                        testResult: null,
                                        testSite: null,
                                    });
                                } else {
                                    setFormData({ ...formData, testDone: value });
                                }
                            }}
                            isExpanded={expandedDropdown === 'testDone'}
                            onToggle={() => setExpandedDropdown(expandedDropdown === 'testDone' ? null : 'testDone')}
                            placeholder="Select"
                        />
                    </View>

                    {/* Test Type - Only show if test is done */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Dropdown
                                label="Test Type"
                                value={formData.testType}
                                options={[
                                    'Select test type',
                                    'GeneXpert',
                                    'Smear Microscopy',
                                    'Culture',
                                    'Chest X-Ray (CXR)',
                                    'Other'
                                ]}
                                onSelect={(val) => {
                                    setFormData({ ...formData, testType: val === 'Select test type' ? null : val });
                                }}
                                isExpanded={expandedDropdown === 'testType'}
                                onToggle={() => setExpandedDropdown(expandedDropdown === 'testType' ? null : 'testType')}
                                placeholder="Select test type"
                            />
                        </View>
                    )}

                    {/* Date of Specimen Collection */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Text style={styles.label}>Date of Specimen Collection</Text>
                            <TextInput
                                style={styles.input}
                                placeholder="mm/dd/yyyy"
                                value={formatDateForInput(formData.dateCollection)}
                                onChangeText={(text) => {
                                    setFormData({ ...formData, dateCollection: text });
                                }}
                                keyboardType="numeric"
                            />
                        </View>
                    )}

                    {/* Date of Result */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Text style={styles.label}>Date of Result</Text>
                            <TextInput
                                style={styles.input}
                                placeholder="mm/dd/yyyy"
                                value={formatDateForInput(formData.dateResult)}
                                onChangeText={(text) => {
                                    setFormData({ ...formData, dateResult: text });
                                }}
                                keyboardType="numeric"
                            />
                        </View>
                    )}

                    {/* TB Diagnosis Result - Only show if test is done */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Dropdown
                                label="TB Diagnosis Result"
                                value={formData.testResult}
                                options={[
                                    'Select result',
                                    'TB Positive',
                                    'TB Negative',
                                    'RIF Resistant',
                                    'Indeterminate',
                                    'Pending'
                                ]}
                                onSelect={(val) => {
                                    setFormData({ ...formData, testResult: val === 'Select result' ? null : val });
                                }}
                                isExpanded={expandedDropdown === 'testResult'}
                                onToggle={() => setExpandedDropdown(expandedDropdown === 'testResult' ? null : 'testResult')}
                                placeholder="Select result"
                            />
                        </View>
                    )}

                    {/* Site of Disease - Only show if test is done */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Dropdown
                                label="Site of Disease"
                                value={formData.testSite}
                                options={[
                                    'Select site',
                                    'Pulmonary',
                                    'Extra-pulmonary',
                                    'Both',
                                    'Unknown'
                                ]}
                                onSelect={(val) => {
                                    setFormData({ ...formData, testSite: val === 'Select site' ? null : val });
                                }}
                                isExpanded={expandedDropdown === 'testSite'}
                                onToggle={() => setExpandedDropdown(expandedDropdown === 'testSite' ? null : 'testSite')}
                                placeholder="Select site"
                            />
                        </View>
                    )}

                    {/* Additional Notes */}
                    {formData.testDone === 'Yes' && (
                        <View style={styles.formGroup}>
                            <Text style={styles.label}>Additional Notes (Optional)</Text>
                            <TextInput
                                style={[styles.input, styles.textArea]}
                                placeholder="Enter any additional notes"
                                value={formData.testNotes}
                                onChangeText={(text) => {
                                    setFormData({ ...formData, testNotes: text });
                                }}
                                multiline
                                numberOfLines={4}
                                textAlignVertical="top"
                            />
                        </View>
                    )}
                </View>
            </ScrollView>

            {/* Footer Buttons */}
            < View style={styles.footer} >
                <TouchableOpacity
                    style={styles.cancelButton}
                    onPress={() => navigation.goBack()}
                    disabled={saving}
                >
                    <Text style={styles.cancelButtonText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[styles.saveButton, saving && styles.saveButtonDisabled]}
                    onPress={handleSave}
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
            </View >

            {/* Validation Modal for Web */}
            {
                Platform.OS === 'web' && (
                    <Modal
                        visible={showValidationModal}
                        transparent={true}
                        animationType="fade"
                        onRequestClose={() => setShowValidationModal(false)}
                    >
                        <View style={styles.modalOverlay}>
                            <View style={styles.modalContent}>
                                <View style={styles.modalHeader}>
                                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                                        <Ionicons name="alert-circle" size={24} color="#EF4444" style={{ marginRight: 12 }} />
                                        <Text style={styles.modalTitle}>Validation Error</Text>
                                    </View>
                                    <TouchableOpacity
                                        onPress={() => setShowValidationModal(false)}
                                        style={styles.modalCloseButton}
                                    >
                                        <Ionicons name="close" size={24} color="#64748B" />
                                    </TouchableOpacity>
                                </View>
                                <ScrollView style={styles.modalScrollView}>
                                    <Text style={styles.modalText}>
                                        Please fix the following errors:
                                    </Text>
                                    {validationErrors.map((error, index) => (
                                        <Text key={index} style={styles.modalErrorItem}>
                                            • {error}
                                        </Text>
                                    ))}
                                </ScrollView>
                                <TouchableOpacity
                                    style={styles.modalButton}
                                    onPress={() => setShowValidationModal(false)}
                                >
                                    <Text style={styles.modalButtonText}>OK</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </Modal>
                )
            }
        </SafeAreaView >
    );
};

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
        color: '#64748B',
        fontSize: 16,
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
        marginBottom: 20,
    },
    backButtonText: {
        color: '#2563EB',
        fontSize: 16,
        fontWeight: '600',
    },
    formCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    formGroup: {
        marginBottom: 20,
    },
    label: {
        fontSize: 14,
        fontWeight: '600',
        color: '#1E293B',
        marginBottom: 8,
    },
    input: {
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
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    modalContent: {
        backgroundColor: 'white',
        borderRadius: 12,
        width: '90%',
        maxWidth: 500,
        maxHeight: '80%',
    },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#E2E8F0',
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: '#1E293B',
    },
    modalCloseButton: {
        padding: 4,
    },
    modalScrollView: {
        maxHeight: 300,
        padding: 20,
    },
    modalText: {
        fontSize: 14,
        color: '#64748B',
        marginBottom: 12,
    },
    modalErrorItem: {
        fontSize: 14,
        color: '#EF4444',
        marginBottom: 8,
    },
    modalButton: {
        backgroundColor: '#2563EB',
        padding: 16,
        borderBottomLeftRadius: 12,
        borderBottomRightRadius: 12,
        alignItems: 'center',
    },
    modalButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
});

export default AddTestResultsScreen;

