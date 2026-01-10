/**
 * Section B: Comorbidities & Vulnerability
 */

import React from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { Dropdown } from '../forms/Dropdown';
import { RadioButtonGroup } from '../forms/RadioButtonGroup';
import { ParticipantFormData } from '../../types/participantForm';

interface SectionBProps {
    formData: ParticipantFormData;
    updateField: <K extends keyof ParticipantFormData>(field: K, value: ParticipantFormData[K]) => void;
    expandedDropdown: string | null;
    setExpandedDropdown: (key: string | null) => void;
}

export const SectionB: React.FC<SectionBProps> = ({
    formData,
    updateField,
    expandedDropdown,
    setExpandedDropdown,
}) => {
    return (
        <>
            <Dropdown
                label="Diabetes Status"
                value={formData.diabetesStatus}
                options={['Yes, diagnosed', 'No', 'Unknown']}
                onSelect={(val) => updateField('diabetesStatus', val)}
                isExpanded={expandedDropdown === 'diabetes'}
                onToggle={() => setExpandedDropdown(expandedDropdown === 'diabetes' ? null : 'diabetes')}
            />

            <Dropdown
                label="HIV Status"
                value={formData.hivStatus}
                options={['Positive', 'Negative', 'Unknown', 'Prefer not to say']}
                onSelect={(val) => updateField('hivStatus', val)}
                isExpanded={expandedDropdown === 'hiv'}
                onToggle={() => setExpandedDropdown(expandedDropdown === 'hiv' ? null : 'hiv')}
            />

            <Dropdown
                label="COVID-19 Status"
                value={formData.covidStatus}
                options={['Current infection', 'Previously infected', 'No', 'Unknown']}
                onSelect={(val) => updateField('covidStatus', val)}
                isExpanded={expandedDropdown === 'covid'}
                onToggle={() => setExpandedDropdown(expandedDropdown === 'covid' ? null : 'covid')}
            />

            <RadioButtonGroup
                label="Tobacco Use"
                value={formData.tobaccoUse}
                onSelect={(val) => updateField('tobaccoUse', val)}
            />

            {formData.tobaccoUse === true && (
                <Dropdown
                    label="Duration"
                    value={formData.tobaccoDuration}
                    options={['< 1 year', '1-5 years', '5-10 years', '> 10 years']}
                    onSelect={(val) => updateField('tobaccoDuration', val)}
                    isExpanded={expandedDropdown === 'tobaccoDuration'}
                    onToggle={() => setExpandedDropdown(expandedDropdown === 'tobaccoDuration' ? null : 'tobaccoDuration')}
                />
            )}

            <RadioButtonGroup
                label="Alcohol Use"
                value={formData.alcoholUse}
                onSelect={(val) => updateField('alcoholUse', val)}
            />

            {formData.alcoholUse === true && (
                <Dropdown
                    label="Duration"
                    value={formData.alcoholDuration}
                    options={['< 1 year', '1-5 years', '5-10 years', '> 10 years']}
                    onSelect={(val) => updateField('alcoholDuration', val)}
                    isExpanded={expandedDropdown === 'alcoholDuration'}
                    onToggle={() => setExpandedDropdown(expandedDropdown === 'alcoholDuration' ? null : 'alcoholDuration')}
                />
            )}

            <RadioButtonGroup
                label="Previous TB Diagnosis"
                value={formData.previousTb}
                onSelect={(val) => updateField('previousTb', val)}
            />

            {formData.previousTb === true && (
                <>
                    <Text style={styles.label}>Year of treatment</Text>
                    <TextInput
                        style={styles.input}
                        placeholder="Enter year"
                        keyboardType="numeric"
                        maxLength={4}
                        value={formData.tbYear}
                        onChangeText={(text) => updateField('tbYear', text.replace(/[^0-9]/g, ''))}
                    />

                    <Dropdown
                        label="Completed treatment?"
                        value={formData.tbTreatmentStatus}
                        options={['Yes', 'No', "Don't remember"]}
                        onSelect={(val) => updateField('tbTreatmentStatus', val)}
                        isExpanded={expandedDropdown === 'tbTreatment'}
                        onToggle={() => setExpandedDropdown(expandedDropdown === 'tbTreatment' ? null : 'tbTreatment')}
                    />
                </>
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
});

