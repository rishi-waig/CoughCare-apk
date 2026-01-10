/**
 * Type definitions for the New Participant Form
 */

export interface ParticipantFormData {
    participantId: string;
    dataCollectorName: string;
    mobileNumber: string;
    fullName: string;
    age: string;
    gender: string | null;
    address: string;
    dateOfScreening: string;
    region: string | null;
    district: string;
    facility: string;
    community: string;
    consentObtained: boolean | null;
    diabetesStatus: string | null;
    hivStatus: string | null;
    covidStatus: string | null;
    tobaccoUse: boolean | null;
    tobaccoDuration: string | null;
    alcoholUse: boolean | null;
    alcoholDuration: string | null;
    previousTb: boolean | null;
    tbYear: string;
    tbTreatmentStatus: string | null;
    symptoms: Record<string, { present: boolean | null; duration: string }>;
    recording1: string | null;
    recording2: string | null;
    recording3: string | null;
    recordingBackground: string | null;
}

export interface AnalysisResult {
    loading: boolean;
    result?: {
        coughDetected: boolean;
        confidence?: number;
        [key: string]: any;
    };
    error?: string;
}

export interface AnalysisResults {
    [key: string]: AnalysisResult;
}

