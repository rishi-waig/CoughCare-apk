/**
 * Custom hook for managing participant form state
 */

import { useState } from 'react';
import { ParticipantFormData } from '../types/participantForm';

export const useParticipantForm = () => {
    const [formData, setFormData] = useState<ParticipantFormData>({
        participantId: Math.floor(Math.random() * 100000).toString(),
        dataCollectorName: '',
        mobileNumber: '',
        fullName: '',
        age: '',
        gender: null,
        address: '',
        dateOfScreening: new Date().toLocaleDateString(),
        region: null,
        district: '',
        facility: '',
        community: '',
        consentObtained: null,
        diabetesStatus: null,
        hivStatus: null,
        covidStatus: null,
        tobaccoUse: null,
        tobaccoDuration: null,
        alcoholUse: null,
        alcoholDuration: null,
        previousTb: null,
        tbYear: '',
        tbTreatmentStatus: null,
        symptoms: {},
        recording1: null,
        recording2: null,
        recording3: null,
        recordingBackground: null,
    });

    const updateFormData = (updates: Partial<ParticipantFormData>) => {
        setFormData(prev => ({ ...prev, ...updates }));
    };

    const updateField = <K extends keyof ParticipantFormData>(
        field: K,
        value: ParticipantFormData[K]
    ) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    const updateSymptom = (key: string, updates: { present?: boolean | null; duration?: string }) => {
        setFormData(prev => ({
            ...prev,
            symptoms: {
                ...prev.symptoms,
                [key]: {
                    ...prev.symptoms[key],
                    ...updates
                }
            }
        }));
    };

    return {
        formData,
        setFormData,
        updateFormData,
        updateField,
        updateSymptom,
    };
};

