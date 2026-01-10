import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    SafeAreaView,
    StatusBar,
    LayoutAnimation,
    Platform,
    UIManager,
    ActivityIndicator,
    Alert,
    Modal
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { Asset } from 'expo-asset';
import { saveParticipant, saveRecording, getDB } from '../services/DatabaseService';
import { useParticipantForm } from '../hooks/useParticipantForm';
import { useAudioRecording } from '../hooks/useAudioRecording';
import { validateForm, formatValidationErrors } from '../utils/formValidation';
import { AccordionSection } from '../components/forms/AccordionSection';
import { SectionA } from '../components/sections/SectionA';
import { SectionB } from '../components/sections/SectionB';
import { SectionC } from '../components/sections/SectionC';
import { SectionD } from '../components/sections/SectionD';

if (Platform.OS === 'android') {
    if (UIManager.setLayoutAnimationEnabledExperimental) {
        UIManager.setLayoutAnimationEnabledExperimental(true);
    }
}

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'NewParticipant'>;

const NewParticipantScreen = () => {
    const navigation = useNavigation<NavigationProp>();
    const [expandedSection, setExpandedSection] = useState<string | null>('A');
    const [expandedDropdown, setExpandedDropdown] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [showValidationModal, setShowValidationModal] = useState(false);
    const [validationErrors, setValidationErrors] = useState<string>('');
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [successMessage, setSuccessMessage] = useState('Participant saved successfully!');

    // Use custom hooks
    const { formData, updateField, updateSymptom } = useParticipantForm();
    const {
        activeRecordingKey,
        recordingDuration,
        recordedDurations,
        analysisResults,
        startRecording,
        stopRecording,
        clearRecording,
        analyzeAudioManually,
    } = useAudioRecording();

    const toggleSection = (section: string) => {
        LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
        setExpandedSection(expandedSection === section ? null : section);
    };

    // Close sections B, C, D if consent is revoked
    React.useEffect(() => {
        if (formData.consentObtained !== true && expandedSection && expandedSection !== 'A') {
            setExpandedSection('A');
        }
    }, [formData.consentObtained]);

    const handleUseSample = async (key: string) => {
        try {
            let uri;
            if (Platform.OS === 'web') {
                uri = '/samples/sample-cough.webm';
            } else {
                const asset = Asset.fromModule(require('../../assets/audio/20251104_150725_454926_cough.wav'));
                await asset.downloadAsync();
                uri = asset.localUri || asset.uri;
            }

            if (uri) {
                updateField(key as keyof typeof formData, uri);
                // Trigger analysis for sample audio using the hook's method
                await analyzeAudioManually(key, uri);
            }
        } catch (err) {
            console.error('Error loading sample', err);
            Alert.alert('Error', 'Failed to load sample audio.');
        }
    };

    const handleSaveDraft = async () => {
        try {
            console.log('Save Draft button clicked');
            setIsSubmitting(true);

            // Save current form data as draft (no validation required)
            await saveParticipant({
                participant_id: formData.participantId || '',
                mobile_number: formData.mobileNumber || '',
                full_name: formData.fullName || '',
                age: parseInt(formData.age) || 0,
                gender: formData.gender || '',
                address: formData.address || null,
                date_of_screening: formData.dateOfScreening || new Date().toISOString().split('T')[0],
                region: formData.region || '',
                district: formData.district || '',
                facility: formData.facility || '',
                community: formData.community || null,
                data_collector_name: formData.dataCollectorName || '',
                consent_obtained: formData.consentObtained ? 1 : 0,
                diabetes_status: formData.diabetesStatus || '',
                hiv_status: formData.hivStatus || '',
                covid_status: formData.covidStatus || '',
                tobacco_use: formData.tobaccoUse ? 1 : 0,
                tobacco_duration: formData.tobaccoDuration || null,
                alcohol_use: formData.alcoholUse ? 1 : 0,
                alcohol_duration: formData.alcoholDuration || null,
                previous_tb: formData.previousTb ? 1 : 0,
                last_tb_year: formData.tbYear || null,
                tb_treatment_completed: formData.tbTreatmentStatus || null,
                symptoms: JSON.stringify(formData.symptoms || {}),
                test_done: null,
                test_type: null,
                test_date_collection: null,
                test_date_result: null,
                test_result: null,
                test_site: null,
                test_notes: null,
                status: 'draft', // Save as draft
                analysis_result: null
            });

            // Save any recordings that exist (even if incomplete)
            const database = await getDB();
            try {
                // Delete existing recordings for this participant
                await database.runAsync(
                    `DELETE FROM recordings WHERE participant_id = ?`,
                    [formData.participantId]
                );
            } catch (error) {
                console.warn('Error deleting existing recordings:', error);
            }

            // Save recordings that exist
            const recordings = [
                { key: 'recording1', type: 'cough_1' },
                { key: 'recording2', type: 'cough_2' },
                { key: 'recording3', type: 'cough_3' },
                { key: 'recordingBackground', type: 'background' }
            ];

            for (const rec of recordings) {
                const uri = formData[rec.key as keyof typeof formData] as string;
                if (uri) {
                    await saveRecording({
                        participant_id: formData.participantId,
                        file_path: uri,
                        recording_type: rec.type,
                        duration: recordedDurations[rec.key] || 0
                    });
                }
            }

            console.log('Draft saved successfully!');
            
            // Show success message
            setSuccessMessage('Draft saved successfully!');
            if (Platform.OS === 'web') {
                setShowSuccessModal(true);
            } else {
                Alert.alert("Draft Saved", "Your draft has been saved. You can continue editing later.", [
                    { text: "OK", onPress: () => navigation.navigate('Dashboard') }
                ]);
            }
        } catch (error) {
            console.error('Error saving draft:', error);
            if (Platform.OS === 'web') {
                alert('Error saving draft. Please try again.');
            } else {
                Alert.alert("Error", "Failed to save draft. Please try again.");
            }
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleSubmit = async () => {
        console.log('Submit button clicked');
        console.log('Form data:', formData);
        console.log('Recorded durations:', recordedDurations);
        
        // Validate form with recorded durations
        const errors = validateForm(formData, recordedDurations);
        console.log('Validation errors:', errors);
        
        if (errors.length > 0) {
            const errorMessage = formatValidationErrors(errors);
            console.log('Showing validation error alert:', errorMessage);
            
            // Use Modal for web, Alert for native
            if (Platform.OS === 'web') {
                setValidationErrors(errorMessage);
                setShowValidationModal(true);
                // Auto-expand first error section
                const firstErrorSection = errors[0]?.section;
                if (firstErrorSection) {
                    setExpandedSection(firstErrorSection);
                }
            } else {
                Alert.alert(
                    "Missing Required Fields",
                    errorMessage,
                    [
                        {
                            text: "OK",
                            style: "default",
                            onPress: () => {
                                console.log('User acknowledged validation errors');
                                // Optionally scroll to first error section
                                const firstErrorSection = errors[0]?.section;
                                if (firstErrorSection) {
                                    setExpandedSection(firstErrorSection);
                                }
                            }
                        }
                    ],
                    { cancelable: true }
                );
            }
            return;
        }

        setIsSubmitting(true);
        try {
            console.log('Starting submission...');
            // Aggregate results
            const keys = ['recording1', 'recording2', 'recording3'];
            let maxConfidence = 0;
            let primaryResult = null;

            for (const key of keys) {
                const res = analysisResults[key]?.result;
                if (res) {
                    const confidence = res.confidence ?? 0;
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        primaryResult = res;
                    }
                }
            }

            // Fallback if no result found (shouldn't happen if validation passes and analysis runs)
            if (!primaryResult && analysisResults['recording1']?.result) {
                primaryResult = analysisResults['recording1'].result;
            }

            // Save to DB - Ensure all fields are properly stored (null for missing, not undefined)
            console.log('Saving participant to database...');
            await saveParticipant({
                participant_id: formData.participantId || '',
                mobile_number: formData.mobileNumber || '',
                full_name: formData.fullName || '',
                age: parseInt(formData.age) || 0,
                gender: formData.gender || '',
                address: formData.address || null,
                date_of_screening: formData.dateOfScreening || new Date().toISOString().split('T')[0],
                region: formData.region || '',
                district: formData.district || '',
                facility: formData.facility || '',
                community: formData.community || null,
                data_collector_name: formData.dataCollectorName || '',
                consent_obtained: formData.consentObtained ? 1 : 0,
                diabetes_status: formData.diabetesStatus || '',
                hiv_status: formData.hivStatus || '',
                covid_status: formData.covidStatus || '',
                tobacco_use: formData.tobaccoUse ? 1 : 0,
                tobacco_duration: formData.tobaccoDuration || null,
                alcohol_use: formData.alcoholUse ? 1 : 0,
                alcohol_duration: formData.alcoholDuration || null,
                previous_tb: formData.previousTb ? 1 : 0,
                last_tb_year: formData.tbYear || null,
                tb_treatment_completed: formData.tbTreatmentStatus || null,
                symptoms: JSON.stringify(formData.symptoms || {}),
                test_done: null,
                test_type: null,
                test_date_collection: null,
                test_date_result: null,
                test_result: null,
                test_site: null,
                test_notes: null,
                status: 'pending', // Ready for sync
                analysis_result: primaryResult ? JSON.stringify(primaryResult) : null
            });

            // Save Recordings - Delete existing recordings for this participant first to prevent duplicates
            // Then save new ones
            const recordings = [
                { key: 'recording1', type: 'cough_1' },
                { key: 'recording2', type: 'cough_2' },
                { key: 'recording3', type: 'cough_3' },
                { key: 'recordingBackground', type: 'background' }
            ];

            console.log('Saving recordings...');
            
            // Delete existing recordings for this participant to prevent duplicates
            const database = await getDB();
            try {
                await database.runAsync(
                    `DELETE FROM recordings WHERE participant_id = ?`,
                    [formData.participantId]
                );
            } catch (error) {
                console.warn('Error deleting existing recordings:', error);
            }

            // Save new recordings
            for (const rec of recordings) {
                const uri = formData[rec.key as keyof typeof formData] as string;
                if (uri) {
                    console.log(`Saving ${rec.type}:`, uri);
                    await saveRecording({
                        participant_id: formData.participantId,
                        file_path: uri,
                        recording_type: rec.type,
                        duration: recordedDurations[rec.key] || 0
                    });
                }
            }

            console.log('Submission successful!');
            
            // Use Modal for web, Alert for native
            setSuccessMessage('Participant saved successfully!');
            if (Platform.OS === 'web') {
                setShowSuccessModal(true);
            } else {
                Alert.alert("Success", "Participant saved successfully!", [
                    {
                        text: "OK",
                        onPress: () => {
                            // Navigate back to Dashboard
                            navigation.navigate('Dashboard' as never);
                        }
                    }
                ]);
            }

        } catch (error) {
            console.error("Submission failed:", error);
            const errorMessage = error instanceof Error ? error.message : String(error);
            Alert.alert(
                "Error", 
                `Could not submit case. ${errorMessage}\n\nPlease check the console for more details.`,
                [{ text: "OK" }]
            );
        } finally {
            setIsSubmitting(false);
        }
    };


    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#2563EB" />

            {/* App Bar */}
            <View style={styles.appBar}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="white" />
                </TouchableOpacity>
                <View>
                    <Text style={styles.appBarTitle}>New Participant</Text>
                    <Text style={styles.appBarSubtitle}>Complete all sections</Text>
                </View>
            </View>

            <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 100 }}>
                {/* Section A */}
                <AccordionSection
                    title="A. Individual & Location Details"
                    section="A"
                    isExpanded={expandedSection === 'A'}
                    onToggle={() => toggleSection('A')}
                >
                    <SectionA
                        formData={formData}
                        updateField={updateField}
                        expandedDropdown={expandedDropdown}
                        setExpandedDropdown={setExpandedDropdown}
                    />
                </AccordionSection>

                {/* Section B: Comorbidities & Vulnerability */}
                <AccordionSection
                    title="B. Comorbidities & Vulnerability"
                    section="B"
                    isExpanded={expandedSection === 'B'}
                    onToggle={() => toggleSection('B')}
                    disabled={formData.consentObtained !== true}
                >
                    <SectionB
                        formData={formData}
                        updateField={updateField}
                        expandedDropdown={expandedDropdown}
                        setExpandedDropdown={setExpandedDropdown}
                    />
                </AccordionSection>

                {/* Section C: Symptoms */}
                <AccordionSection
                    title="C. Symptoms"
                    section="C"
                    isExpanded={expandedSection === 'C'}
                    onToggle={() => toggleSection('C')}
                    disabled={formData.consentObtained !== true}
                >
                    <SectionC
                        formData={formData}
                        updateSymptom={updateSymptom}
                    />
                </AccordionSection>

                {/* Section D: Cough & Audio Recording */}
                <AccordionSection
                    title="D. Cough & Audio Recording"
                    section="D"
                    isExpanded={expandedSection === 'D'}
                    onToggle={() => toggleSection('D')}
                    disabled={formData.consentObtained !== true}
                >
                    <SectionD
                        formData={formData}
                        updateField={updateField}
                        activeRecordingKey={activeRecordingKey}
                        recordingDuration={recordingDuration}
                        recordedDurations={recordedDurations}
                        analysisResults={analysisResults}
                        onStartRecording={startRecording}
                        onStopRecording={stopRecording}
                        onClearRecording={clearRecording}
                        onUseSample={handleUseSample}
                    />
                </AccordionSection>
            </ScrollView>

            {/* Validation Error Modal for Web */}
            <Modal
                visible={showValidationModal}
                transparent={true}
                animationType="fade"
                onRequestClose={() => setShowValidationModal(false)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <View style={styles.modalHeader}>
                            <Text style={styles.modalTitle}>Missing Required Fields</Text>
                            <TouchableOpacity
                                onPress={() => setShowValidationModal(false)}
                                style={styles.modalCloseButton}
                            >
                                <Ionicons name="close" size={24} color="#64748B" />
                            </TouchableOpacity>
                        </View>
                        <ScrollView style={styles.modalScrollView}>
                            <Text style={styles.modalText}>{validationErrors}</Text>
                        </ScrollView>
                        <TouchableOpacity
                            style={styles.modalButton}
                            onPress={() => {
                                setShowValidationModal(false);
                                console.log('User acknowledged validation errors');
                            }}
                        >
                            <Text style={styles.modalButtonText}>OK</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>

            {/* Success Modal for Web */}
            <Modal
                visible={showSuccessModal}
                transparent={true}
                animationType="fade"
                onRequestClose={() => {
                    setShowSuccessModal(false);
                    navigation.navigate('Dashboard' as never);
                }}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <View style={styles.modalHeader}>
                            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                                <Ionicons name="checkmark-circle" size={24} color="#22C55E" style={{ marginRight: 12 }} />
                                <Text style={styles.modalTitle}>Success</Text>
                            </View>
                            <TouchableOpacity
                                onPress={() => {
                                    setShowSuccessModal(false);
                                    navigation.navigate('Dashboard');
                                }}
                                style={styles.modalCloseButton}
                            >
                                <Ionicons name="close" size={24} color="#64748B" />
                            </TouchableOpacity>
                        </View>
                        <View style={styles.modalScrollView}>
                            <Text style={styles.modalText}>
                                {successMessage}
                            </Text>
                            <Text style={[styles.modalText, { marginTop: 12, fontSize: 12, color: '#64748B' }]}>
                                {successMessage.includes('Draft') 
                                    ? 'You can continue editing this form later from the Drafts section.'
                                    : 'The participant data has been saved and will appear on the dashboard.'}
                            </Text>
                        </View>
                        <TouchableOpacity
                            style={styles.modalButton}
                            onPress={() => {
                                setShowSuccessModal(false);
                                navigation.navigate('Dashboard');
                            }}
                        >
                            <Text style={styles.modalButtonText}>Go to Dashboard</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>

            {/* Footer */}
            <View style={styles.footer}>
                <TouchableOpacity 
                    style={styles.draftBtn}
                    onPress={handleSaveDraft}
                    disabled={isSubmitting}
                >
                    <Ionicons name="save-outline" size={20} color="#334155" style={{ marginRight: 8 }} />
                    <Text style={styles.draftBtnText}>Save Draft</Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[
                        styles.submitBtn,
                        formData.consentObtained === true ? styles.submitBtnActive : {}
                    ]}
                    disabled={formData.consentObtained !== true || isSubmitting}
                    onPress={() => {
                        console.log('Submit button pressed');
                        console.log('Consent obtained:', formData.consentObtained);
                        console.log('Is submitting:', isSubmitting);
                        handleSubmit();
                    }}
                >
                    {isSubmitting ? (
                        <ActivityIndicator size="small" color="white" style={{ marginRight: 8 }} />
                    ) : null}
                    <Text style={styles.submitBtnText}>
                        {isSubmitting ? "Analyzing..." : "Submit"}
                    </Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#F1F5F9',
    },
    appBar: {
        backgroundColor: '#2563EB',
        padding: 16,
        paddingTop: 40,
        flexDirection: 'row',
        alignItems: 'center',
    },
    backButton: {
        marginRight: 16,
    },
    appBarTitle: {
        color: 'white',
        fontSize: 18,
        fontWeight: '600',
    },
    appBarSubtitle: {
        color: '#BFDBFE',
        fontSize: 12,
    },
    content: {
        flex: 1,
    },
    footer: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: 'white',
        padding: 16,
        flexDirection: 'row',
        borderTopWidth: 1,
        borderTopColor: '#E2E8F0',
        gap: 12,
    },
    draftBtn: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#E2E8F0',
        padding: 16,
        borderRadius: 8,
    },
    draftBtnText: {
        color: '#334155',
        fontWeight: '600',
        fontSize: 16,
    },
    submitBtn: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#CBD5E1', // Disabled look for now
        padding: 16,
        borderRadius: 8,
        flexDirection: 'row',
    },
    submitBtnActive: {
        backgroundColor: '#22C55E', // Green color to match images
    },
    submitBtnText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 16,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    modalContent: {
        backgroundColor: 'white',
        borderRadius: 16,
        width: '100%',
        maxWidth: 500,
        maxHeight: '80%',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
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
        fontSize: 20,
        fontWeight: '700',
        color: '#1E293B',
    },
    modalCloseButton: {
        padding: 4,
    },
    modalScrollView: {
        maxHeight: 400,
        padding: 20,
    },
    modalText: {
        fontSize: 14,
        color: '#475569',
        lineHeight: 22,
    },
    modalButton: {
        backgroundColor: '#2563EB',
        padding: 16,
        borderBottomLeftRadius: 16,
        borderBottomRightRadius: 16,
        alignItems: 'center',
    },
    modalButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
});

export default NewParticipantScreen;
