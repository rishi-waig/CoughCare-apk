/**
 * Section A: Individual & Location Details
 */

import React from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Dropdown } from '../forms/Dropdown';
import { RadioButtonGroup } from '../forms/RadioButtonGroup';
import { ParticipantFormData } from '../../types/participantForm';

interface SectionAProps {
    formData: ParticipantFormData;
    updateField: <K extends keyof ParticipantFormData>(field: K, value: ParticipantFormData[K]) => void;
    expandedDropdown: string | null;
    setExpandedDropdown: (key: string | null) => void;
}

export const SectionA: React.FC<SectionAProps> = ({
    formData,
    updateField,
    expandedDropdown,
    setExpandedDropdown,
}) => {
    return (
        <>
            <Text style={styles.label}>Mobile Number *</Text>
            <TextInput
                style={styles.input}
                placeholder="0XX XXX XXXX"
                keyboardType="phone-pad"
                value={formData.mobileNumber}
                onChangeText={(text) => updateField('mobileNumber', text)}
            />

            <Text style={styles.label}>Full Name *</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter full name"
                value={formData.fullName}
                onChangeText={(text) => updateField('fullName', text)}
            />

            <View style={styles.row}>
                <View style={[styles.col, { marginRight: 8 }]}>
                    <Text style={styles.label}>Age *</Text>
                    <TextInput
                        style={styles.input}
                        placeholder="Years"
                        keyboardType="numeric"
                        value={formData.age}
                        onChangeText={(text) => updateField('age', text)}
                    />
                </View>
                <View style={[styles.col, { marginLeft: 8 }]}>
                    <Dropdown
                        label="Gender *"
                        value={formData.gender}
                        options={['Male', 'Female', 'Transgender']}
                        onSelect={(val) => updateField('gender', val)}
                        isExpanded={expandedDropdown === 'gender'}
                        onToggle={() => setExpandedDropdown(expandedDropdown === 'gender' ? null : 'gender')}
                    />
                </View>
            </View>

            <Text style={styles.label}>Address (Optional)</Text>
            <TextInput
                style={[styles.input, { height: 80, textAlignVertical: 'top' }]}
                placeholder="Enter address"
                multiline
                numberOfLines={3}
                value={formData.address}
                onChangeText={(text) => updateField('address', text)}
            />

            <Text style={styles.label}>Date of Screening *</Text>
            <TextInput
                style={styles.input}
                placeholder="DD/MM/YYYY"
                keyboardType="numeric"
                maxLength={10}
                value={formData.dateOfScreening}
                onChangeText={(text) => {
                    // Simple date mask: DD/MM/YYYY
                    let cleaned = text.replace(/\D/g, '');
                    let formatted = cleaned;
                    if (cleaned.length > 2) {
                        formatted = cleaned.slice(0, 2) + '/' + cleaned.slice(2);
                    }
                    if (cleaned.length > 4) {
                        formatted = formatted.slice(0, 5) + '/' + cleaned.slice(4, 8);
                    }
                    updateField('dateOfScreening', formatted);
                }}
            />

            <Dropdown
                label="Region *"
                value={formData.region}
                options={[
                    'Ashanti', 'Bono', 'Bono East', 'Ahafo', 'Central', 'Eastern',
                    'Greater Accra', 'Northern', 'North East', 'Savannah', 'Oti',
                    'Upper East', 'Upper West', 'Volta', 'Western', 'Western North'
                ]}
                onSelect={(val) => updateField('region', val)}
                isExpanded={expandedDropdown === 'region'}
                onToggle={() => setExpandedDropdown(expandedDropdown === 'region' ? null : 'region')}
                placeholder="Select Region"
            />

            <Text style={styles.label}>District *</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter district"
                value={formData.district}
                onChangeText={(text) => updateField('district', text)}
            />

            <Text style={styles.label}>Facility / Site *</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter facility name"
                value={formData.facility}
                onChangeText={(text) => updateField('facility', text)}
            />

            <Text style={styles.label}>Community Name (Optional)</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter community"
                value={formData.community}
                onChangeText={(text) => updateField('community', text)}
            />

            <Text style={styles.label}>Data Collector Name *</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter your name"
                value={formData.dataCollectorName}
                onChangeText={(text) => updateField('dataCollectorName', text)}
            />

            <View style={[styles.consentContainer, formData.consentObtained === false && styles.consentContainerError]}>
                <Text style={styles.label}>Consent Obtained *</Text>
                <View style={styles.radioGroup}>
                    <RadioButtonGroup
                        value={formData.consentObtained}
                        onSelect={(val) => updateField('consentObtained', val)}
                        variant={formData.consentObtained === true ? 'success' : formData.consentObtained === false ? 'error' : 'default'}
                    />
                </View>
            </View>

            {formData.consentObtained === false && (
                <View style={styles.errorBox}>
                    <Ionicons name="warning" size={20} color="#F59E0B" style={{ marginRight: 8 }} />
                    <Text style={styles.errorText}>Cannot proceed without consent. Please obtain consent to continue.</Text>
                </View>
            )}
        </>
    );
};

const styles = StyleSheet.create({
    label: {
        fontSize: 14,
        color: '#475569',
        marginBottom: 8,
        marginTop: 16,
        fontWeight: '500',
    },
    input: {
        borderWidth: 1,
        borderColor: '#CBD5E1',
        borderRadius: 8,
        padding: 12,
        fontSize: 16,
        color: '#1E293B',
        backgroundColor: 'white',
    },
    row: {
        flexDirection: 'row',
    },
    col: {
        flex: 1,
    },
    consentContainer: {
        marginTop: 16,
        backgroundColor: '#EFF6FF',
        padding: 16,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#BFDBFE',
    },
    consentContainerError: {
        borderColor: '#93C5FD',
    },
    radioGroup: {
        flexDirection: 'row',
        gap: 16,
    },
    errorBox: {
        flexDirection: 'row',
        backgroundColor: '#FEF2F2',
        padding: 12,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#FECACA',
        marginTop: 16,
        alignItems: 'center',
    },
    errorText: {
        color: '#B91C1C',
        fontSize: 14,
    },
});

