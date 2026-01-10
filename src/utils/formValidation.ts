/**
 * Form validation utilities for New Participant Form
 * Based on PRD: Cough Against TB – Data Collection App (Ghana) v1.0
 */

import { ParticipantFormData } from '../types/participantForm';

export interface ValidationError {
    field: string;
    message: string;
    section: 'A' | 'B' | 'C' | 'D';
}

export const validateForm = (formData: ParticipantFormData, recordedDurations?: Record<string, number>): ValidationError[] => {
    const errors: ValidationError[] = [];

    // Section A: Individual & Location Details
    if (!formData.mobileNumber || formData.mobileNumber.trim() === '') {
        errors.push({ field: 'mobileNumber', message: 'Mobile Number is required', section: 'A' });
    }
    if (!formData.participantId || formData.participantId.trim() === '') {
        errors.push({ field: 'participantId', message: 'Participant ID is required', section: 'A' });
    }
    if (!formData.fullName || formData.fullName.trim() === '') {
        errors.push({ field: 'fullName', message: 'Full Name is required', section: 'A' });
    }
    if (!formData.age || formData.age.trim() === '' || isNaN(parseInt(formData.age))) {
        errors.push({ field: 'age', message: 'Age is required and must be a number', section: 'A' });
    }
    if (!formData.gender) {
        errors.push({ field: 'gender', message: 'Gender is required', section: 'A' });
    }
    if (!formData.dateOfScreening || formData.dateOfScreening.trim() === '') {
        errors.push({ field: 'dateOfScreening', message: 'Date of Screening is required', section: 'A' });
    }
    if (!formData.region) {
        errors.push({ field: 'region', message: 'Region is required', section: 'A' });
    }
    if (!formData.district || formData.district.trim() === '') {
        errors.push({ field: 'district', message: 'District is required', section: 'A' });
    }
    if (!formData.facility || formData.facility.trim() === '') {
        errors.push({ field: 'facility', message: 'Facility / Site is required', section: 'A' });
    }
    if (!formData.dataCollectorName || formData.dataCollectorName.trim() === '') {
        errors.push({ field: 'dataCollectorName', message: 'Data Collector Name is required', section: 'A' });
    }
    if (formData.consentObtained !== true) {
        errors.push({ field: 'consentObtained', message: 'Consent must be obtained (Yes) to proceed', section: 'A' });
    }

    // Section B: Comorbidities & Vulnerability
    if (!formData.diabetesStatus) {
        errors.push({ field: 'diabetesStatus', message: 'Diabetes Status is required', section: 'B' });
    }
    if (!formData.hivStatus) {
        errors.push({ field: 'hivStatus', message: 'HIV Status is required', section: 'B' });
    }
    if (!formData.covidStatus) {
        errors.push({ field: 'covidStatus', message: 'COVID-19 Status is required', section: 'B' });
    }
    if (formData.tobaccoUse === null || formData.tobaccoUse === undefined) {
        errors.push({ field: 'tobaccoUse', message: 'Tobacco Use is required', section: 'B' });
    } else if (formData.tobaccoUse === true && (!formData.tobaccoDuration || formData.tobaccoDuration.trim() === '')) {
        errors.push({ field: 'tobaccoDuration', message: 'Tobacco Use Duration is required when Tobacco Use is Yes', section: 'B' });
    }
    if (formData.alcoholUse === null || formData.alcoholUse === undefined) {
        errors.push({ field: 'alcoholUse', message: 'Alcohol Use is required', section: 'B' });
    } else if (formData.alcoholUse === true && (!formData.alcoholDuration || formData.alcoholDuration.trim() === '')) {
        errors.push({ field: 'alcoholDuration', message: 'Alcohol Use Duration is required when Alcohol Use is Yes', section: 'B' });
    }
    if (formData.previousTb === null || formData.previousTb === undefined) {
        errors.push({ field: 'previousTb', message: 'Previous TB Diagnosis/Treatment is required', section: 'B' });
    }

    // Section C: Symptoms - All symptoms are mandatory (Yes/No), duration required if Yes
    const symptomMap: Record<string, string> = {
        fever: 'Fever',
        cough: 'Cough',
        weightLoss: 'Weight Loss (last 6 months)',
        bloodInSputum: 'Blood in Sputum',
        chestPain: 'Chest Pain',
        lossOfAppetite: 'Loss of Appetite',
        shortnessOfBreath: 'Shortness of Breath',
        nightSweats: 'Night Sweats (last 1 month)'
    };

    const symptomKeys = Object.keys(symptomMap);
    for (const key of symptomKeys) {
        const symptom = formData.symptoms[key];
        // Check if symptom exists and has a present value
        if (!symptom || symptom.present === null || symptom.present === undefined) {
            errors.push({ field: `symptoms.${key}`, message: `${symptomMap[key]} is required (Yes/No)`, section: 'C' });
        } else if (symptom.present === true) {
            // If Yes, duration is required
            if (!symptom.duration || symptom.duration.trim() === '' || isNaN(parseInt(symptom.duration))) {
                errors.push({ field: `symptoms.${key}.duration`, message: `Duration (in days) is required for ${symptomMap[key]}`, section: 'C' });
            }
        }
    }

    // Section D: Cough & Audio Recording
    // Check if recordings exist and meet minimum duration requirements
    if (!formData.recording1) {
        errors.push({ field: 'recording1', message: 'Cough Recording 1 is required (minimum 5 seconds)', section: 'D' });
    } else if (recordedDurations && recordedDurations['recording1'] < 5) {
        errors.push({ field: 'recording1', message: 'Cough Recording 1 must be at least 5 seconds', section: 'D' });
    }

    if (!formData.recording2) {
        errors.push({ field: 'recording2', message: 'Cough Recording 2 is required (minimum 5 seconds)', section: 'D' });
    } else if (recordedDurations && recordedDurations['recording2'] < 5) {
        errors.push({ field: 'recording2', message: 'Cough Recording 2 must be at least 5 seconds', section: 'D' });
    }

    if (!formData.recording3) {
        errors.push({ field: 'recording3', message: 'Cough Recording 3 is required (minimum 5 seconds)', section: 'D' });
    } else if (recordedDurations && recordedDurations['recording3'] < 5) {
        errors.push({ field: 'recording3', message: 'Cough Recording 3 must be at least 5 seconds', section: 'D' });
    }

    if (!formData.recordingBackground) {
        errors.push({ field: 'recordingBackground', message: 'Ambient Sound Recording is required (minimum 10 seconds)', section: 'D' });
    } else if (recordedDurations && recordedDurations['recordingBackground'] < 10) {
        errors.push({ field: 'recordingBackground', message: 'Ambient Sound Recording must be at least 10 seconds', section: 'D' });
    }

    return errors;
};

// Helper function to format validation errors for display
export const formatValidationErrors = (errors: ValidationError[]): string => {
    if (errors.length === 0) return '';
    
    const sectionErrors: Record<string, ValidationError[]> = {
        'A': [],
        'B': [],
        'C': [],
        'D': []
    };

    errors.forEach(error => {
        sectionErrors[error.section].push(error);
    });

    // Create a summary first
    let message = `Please complete ${errors.length} required field(s):\n\n`;
    
    const sectionNames: Record<string, string> = {
        'A': 'Section A - Individual & Location Details',
        'B': 'Section B - Comorbidities & Vulnerability',
        'C': 'Section C - Symptoms',
        'D': 'Section D - Cough & Audio Recording'
    };

    // Show summary by section
    ['A', 'B', 'C', 'D'].forEach(section => {
        if (sectionErrors[section].length > 0) {
            message += `${sectionNames[section]}: ${sectionErrors[section].length} error(s)\n`;
        }
    });

    message += '\n---\n\n';

    // Show detailed errors (limit to first 10 to avoid message being too long)
    const maxErrors = 10;
    const errorsToShow = errors.slice(0, maxErrors);
    const remainingCount = errors.length - maxErrors;

    errorsToShow.forEach((err, index) => {
        message += `${index + 1}. ${err.message}\n`;
    });

    if (remainingCount > 0) {
        message += `\n... and ${remainingCount} more error(s).\n`;
    }

    message += '\nPlease fill all required fields before submitting.';

    return message;
};

