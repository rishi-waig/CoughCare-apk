/**
 * Section C: Symptoms
 */

import React from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import { ParticipantFormData } from '../../types/participantForm';

interface SectionCProps {
    formData: ParticipantFormData;
    updateSymptom: (key: string, updates: { present?: boolean | null; duration?: string }) => void;
}

const SYMPTOMS = [
    { key: 'fever', label: 'Fever' },
    { key: 'cough', label: 'Cough' },
    { key: 'weightLoss', label: 'Weight loss (past 6 months)' },
    { key: 'bloodInSputum', label: 'Blood in sputum' },
    { key: 'chestPain', label: 'Chest pain' },
    { key: 'lossOfAppetite', label: 'Loss of appetite' },
    { key: 'shortnessOfBreath', label: 'Shortness of breath' },
    { key: 'nightSweats', label: 'Night sweats (past 1 month)' },
];

export const SectionC: React.FC<SectionCProps> = ({
    formData,
    updateSymptom,
}) => {
    return (
        <>
            <Text style={styles.helperText}>Select all that apply and specify duration (days)</Text>

            {SYMPTOMS.map((symptom) => {
                const current = formData.symptoms[symptom.key] || {};
                return (
                    <View key={symptom.key} style={styles.symptomContainer}>
                        <View style={styles.symptomHeader}>
                            <Text style={styles.symptomLabel}>{symptom.label}</Text>
                            <View style={styles.symptomRadioGroup}>
                                <TouchableOpacity
                                    style={[
                                        styles.symptomBtn,
                                        current.present === true && styles.symptomBtnYes
                                    ]}
                                    onPress={() => updateSymptom(symptom.key, { present: true })}
                                >
                                    <Text style={[
                                        styles.symptomBtnText,
                                        current.present === true && styles.symptomBtnTextActive
                                    ]}>Yes</Text>
                                </TouchableOpacity>
                                <TouchableOpacity
                                    style={[
                                        styles.symptomBtn,
                                        current.present === false && styles.symptomBtnNo
                                    ]}
                                    onPress={() => updateSymptom(symptom.key, { present: false, duration: '' })}
                                >
                                    <Text style={[
                                        styles.symptomBtnText,
                                        current.present === false && styles.symptomBtnTextActive
                                    ]}>No</Text>
                                </TouchableOpacity>
                            </View>
                        </View>

                        {current.present === true && (
                            <TextInput
                                style={styles.durationInputFull}
                                placeholder="Duration in days"
                                keyboardType="numeric"
                                value={current.duration}
                                onChangeText={(text) => updateSymptom(symptom.key, { duration: text })}
                            />
                        )}
                    </View>
                );
            })}
        </>
    );
};

const styles = StyleSheet.create({
    helperText: {
        fontSize: 12,
        color: '#64748B',
        marginBottom: 16,
        fontStyle: 'italic',
    },
    symptomContainer: {
        marginBottom: 16,
        backgroundColor: '#F8FAFC',
        padding: 12,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    symptomHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    symptomLabel: {
        fontSize: 16,
        color: '#1E293B',
    },
    symptomRadioGroup: {
        flexDirection: 'row',
        gap: 8,
    },
    symptomBtn: {
        paddingVertical: 6,
        paddingHorizontal: 12,
        borderRadius: 6,
        borderWidth: 1,
        borderColor: '#CBD5E1',
        backgroundColor: 'white',
    },
    symptomBtnYes: {
        backgroundColor: '#2563EB',
        borderColor: '#2563EB',
    },
    symptomBtnNo: {
        backgroundColor: '#64748B',
        borderColor: '#64748B',
    },
    symptomBtnText: {
        fontSize: 14,
        color: '#64748B',
        fontWeight: '500',
    },
    symptomBtnTextActive: {
        color: 'white',
    },
    durationInputFull: {
        borderWidth: 1,
        borderColor: '#CBD5E1',
        borderRadius: 6,
        padding: 8,
        backgroundColor: 'white',
        fontSize: 14,
    },
});

