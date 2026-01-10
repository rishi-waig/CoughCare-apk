import * as SQLite from 'expo-sqlite';

let db: SQLite.SQLiteDatabase | null = null;
let initPromise: Promise<void> | null = null;

export interface Participant {
    id?: number;
    participant_id: string;
    mobile_number: string;
    full_name: string;
    age: number;
    gender: string;
    address?: string | null;
    date_of_screening: string;
    region: string;
    district: string;
    facility: string;
    community?: string | null;
    data_collector_name: string;
    consent_obtained: number; // 0 or 1
    diabetes_status: string;
    hiv_status: string;
    covid_status: string;
    tobacco_use: number; // 0 or 1
    tobacco_duration?: string | null;
    alcohol_use: number; // 0 or 1
    alcohol_duration?: string | null;
    previous_tb: number; // 0 or 1
    last_tb_year?: string | null;
    tb_treatment_completed?: string | null;
    symptoms: string; // JSON string
    test_done?: string | null; // Yes, No, Not yet
    test_type?: string | null;
    test_date_collection?: string | null;
    test_date_result?: string | null;
    test_result?: string | null;
    test_site?: string | null;
    test_notes?: string | null;
    created_at?: string;
    synced?: number;
    status?: string; // draft, pending, synced
    analysis_result?: string | null; // JSON string for ONNX result
}

export interface Recording {
    id?: number;
    participant_id: string;
    file_path: string;
    recording_type: string; // cough_1, cough_2, cough_3, background
    duration: number;
    created_at?: string;
    synced?: number;
}

export const initDatabase = async () => {
    // If already initialized, return
    if (db) {
        return;
    }
    
    // If initialization is in progress, wait for it
    if (initPromise) {
        return initPromise;
    }

    // Start initialization
    initPromise = (async () => {
        try {
            db = await SQLite.openDatabaseAsync('cough_against_tb_v2.db');
            await db.execAsync(`
      PRAGMA journal_mode = WAL;
      
      CREATE TABLE IF NOT EXISTS participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT UNIQUE NOT NULL,
        mobile_number TEXT NOT NULL,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        address TEXT,
        date_of_screening TEXT NOT NULL,
        region TEXT NOT NULL,
        district TEXT NOT NULL,
        facility TEXT NOT NULL,
        community TEXT,
        data_collector_name TEXT NOT NULL,
        consent_obtained INTEGER NOT NULL,
        diabetes_status TEXT NOT NULL,
        hiv_status TEXT NOT NULL,
        covid_status TEXT NOT NULL,
        tobacco_use INTEGER NOT NULL,
        tobacco_duration TEXT,
        alcohol_use INTEGER NOT NULL,
        alcohol_duration TEXT,
        previous_tb INTEGER NOT NULL,
        last_tb_year TEXT,
        tb_treatment_completed TEXT,
        symptoms TEXT NOT NULL, -- JSON
        test_done TEXT,
        test_type TEXT,
        test_date_collection TEXT,
        test_date_result TEXT,
        test_result TEXT,
        test_site TEXT,
        test_notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        synced INTEGER DEFAULT 0,
        status TEXT DEFAULT 'draft',
        analysis_result TEXT -- JSON
      );

      CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        recording_type TEXT NOT NULL,
        duration INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        synced INTEGER DEFAULT 0,
        FOREIGN KEY (participant_id) REFERENCES participants (participant_id),
        UNIQUE(participant_id, recording_type)
      );
    `);
            console.log('Database initialized successfully (v2)');
        } catch (error) {
            console.error('Error initializing database:', error);
            initPromise = null; // Reset on error so it can be retried
            throw error;
        }
    })();

    return initPromise;
};

export const getDB = async () => {
    // Wait for initialization if not ready
    if (!db) {
        if (!initPromise) {
            // If no initialization has started, start it now
            await initDatabase();
        } else {
            // Wait for existing initialization
            await initPromise;
        }
    }
    
    if (!db) {
        throw new Error('Database initialization failed');
    }
    
    return db;
};

// Helper function to normalize values (ensure null instead of undefined, empty strings for required fields)
const normalizeValue = (value: any, defaultValue: any = null): any => {
    if (value === undefined || value === '') {
        return defaultValue;
    }
    return value;
};

export const saveParticipant = async (participant: Participant) => {
    const database = await getDB();
    try {
        // Normalize all values to ensure proper storage (null for missing, not undefined)
        const normalizedParticipant = {
            participant_id: normalizeValue(participant.participant_id, ''),
            mobile_number: normalizeValue(participant.mobile_number, ''),
            full_name: normalizeValue(participant.full_name, ''),
            age: normalizeValue(participant.age, 0),
            gender: normalizeValue(participant.gender, ''),
            address: normalizeValue(participant.address, null),
            date_of_screening: normalizeValue(participant.date_of_screening, new Date().toISOString().split('T')[0]),
            region: normalizeValue(participant.region, ''),
            district: normalizeValue(participant.district, ''),
            facility: normalizeValue(participant.facility, ''),
            community: normalizeValue(participant.community, null),
            data_collector_name: normalizeValue(participant.data_collector_name, ''),
            consent_obtained: normalizeValue(participant.consent_obtained, 0),
            diabetes_status: normalizeValue(participant.diabetes_status, ''),
            hiv_status: normalizeValue(participant.hiv_status, ''),
            covid_status: normalizeValue(participant.covid_status, ''),
            tobacco_use: normalizeValue(participant.tobacco_use, 0),
            tobacco_duration: normalizeValue(participant.tobacco_duration, null),
            alcohol_use: normalizeValue(participant.alcohol_use, 0),
            alcohol_duration: normalizeValue(participant.alcohol_duration, null),
            previous_tb: normalizeValue(participant.previous_tb, 0),
            last_tb_year: normalizeValue(participant.last_tb_year, null),
            tb_treatment_completed: normalizeValue(participant.tb_treatment_completed, null),
            symptoms: normalizeValue(participant.symptoms, '{}'),
            test_done: normalizeValue(participant.test_done, null),
            test_type: normalizeValue(participant.test_type, null),
            test_date_collection: normalizeValue(participant.test_date_collection, null),
            test_date_result: normalizeValue(participant.test_date_result, null),
            test_result: normalizeValue(participant.test_result, null),
            test_site: normalizeValue(participant.test_site, null),
            test_notes: normalizeValue(participant.test_notes, null),
            status: normalizeValue(participant.status, 'draft'),
            analysis_result: normalizeValue(participant.analysis_result, null)
        };

        const result = await database.runAsync(
            `INSERT OR REPLACE INTO participants (
                participant_id, mobile_number, full_name, age, gender, address, date_of_screening,
                region, district, facility, community, data_collector_name, consent_obtained,
                diabetes_status, hiv_status, covid_status, tobacco_use, tobacco_duration,
                alcohol_use, alcohol_duration, previous_tb, last_tb_year, tb_treatment_completed,
                symptoms, test_done, test_type, test_date_collection, test_date_result,
                test_result, test_site, test_notes, status, analysis_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
                normalizedParticipant.participant_id,
                normalizedParticipant.mobile_number,
                normalizedParticipant.full_name,
                normalizedParticipant.age,
                normalizedParticipant.gender,
                normalizedParticipant.address,
                normalizedParticipant.date_of_screening,
                normalizedParticipant.region,
                normalizedParticipant.district,
                normalizedParticipant.facility,
                normalizedParticipant.community,
                normalizedParticipant.data_collector_name,
                normalizedParticipant.consent_obtained,
                normalizedParticipant.diabetes_status,
                normalizedParticipant.hiv_status,
                normalizedParticipant.covid_status,
                normalizedParticipant.tobacco_use,
                normalizedParticipant.tobacco_duration,
                normalizedParticipant.alcohol_use,
                normalizedParticipant.alcohol_duration,
                normalizedParticipant.previous_tb,
                normalizedParticipant.last_tb_year,
                normalizedParticipant.tb_treatment_completed,
                normalizedParticipant.symptoms,
                normalizedParticipant.test_done,
                normalizedParticipant.test_type,
                normalizedParticipant.test_date_collection,
                normalizedParticipant.test_date_result,
                normalizedParticipant.test_result,
                normalizedParticipant.test_site,
                normalizedParticipant.test_notes,
                normalizedParticipant.status,
                normalizedParticipant.analysis_result
            ]
        );
        return result.lastInsertRowId;
    } catch (error) {
        console.error("Error saving participant:", error);
        throw error;
    }
};

export const saveRecording = async (recording: Recording) => {
    const database = await getDB();
    try {
        // Use INSERT OR REPLACE to prevent duplicates based on (participant_id, recording_type)
        await database.runAsync(
            `INSERT OR REPLACE INTO recordings (participant_id, file_path, recording_type, duration) 
             VALUES (?, ?, ?, ?)`,
            [recording.participant_id, recording.file_path, recording.recording_type, recording.duration]
        );
    } catch (error) {
        console.error("Error saving recording:", error);
        throw error;
    }
};

export const getParticipants = async (): Promise<Participant[]> => {
    const database = await getDB();
    try {
        const rows = await database.getAllAsync<Participant>(`SELECT * FROM participants ORDER BY created_at DESC`);
        return rows;
    } catch (error) {
        console.error("Error fetching participants:", error);
        return [];
    }
};

export const getParticipantById = async (participantId: string): Promise<Participant | null> => {
    const database = await getDB();
    try {
        const row = await database.getFirstAsync<Participant>(
            `SELECT * FROM participants WHERE participant_id = ?`,
            [participantId]
        );
        return row || null;
    } catch (error) {
        console.error("Error fetching participant:", error);
        return null;
    }
};

export const getRecordingsByParticipantId = async (participantId: string): Promise<Recording[]> => {
    const database = await getDB();
    try {
        // Get the latest recording for each type (to handle any existing duplicates)
        const rows = await database.getAllAsync<Recording>(
            `SELECT * FROM recordings 
             WHERE participant_id = ? 
             AND id IN (
                 SELECT MAX(id) 
                 FROM recordings 
                 WHERE participant_id = ? 
                 GROUP BY recording_type
             )
             ORDER BY 
                 CASE recording_type 
                     WHEN 'cough_1' THEN 1
                     WHEN 'cough_2' THEN 2
                     WHEN 'cough_3' THEN 3
                     WHEN 'background' THEN 4
                     ELSE 5
                 END`,
            [participantId, participantId]
        );
        return rows;
    } catch (error) {
        console.error("Error fetching recordings:", error);
        return [];
    }
};

// Clean up duplicate recordings (keep only the latest for each participant_id + recording_type)
export const cleanupDuplicateRecordings = async () => {
    const database = await getDB();
    try {
        await database.runAsync(`
            DELETE FROM recordings 
            WHERE id NOT IN (
                SELECT MAX(id) 
                FROM recordings 
                GROUP BY participant_id, recording_type
            )
        `);
        console.log('Cleaned up duplicate recordings');
    } catch (error) {
        console.error("Error cleaning up duplicates:", error);
    }
};

// Helper to format value for display (show "N/A" for null/undefined)
const formatForDisplay = (value: any): string => {
    if (value === null || value === undefined || value === '') {
        return 'N/A';
    }
    return String(value);
};

// Debug function to view all database contents
export const viewDatabaseContents = async () => {
    const database = await getDB();
    try {
        console.log('\n========== DATABASE CONTENTS ==========');
        
        // Get all participants
        const participants = await database.getAllAsync<Participant>(
            `SELECT * FROM participants ORDER BY created_at DESC`
        );
        console.log(`\n📋 PARTICIPANTS (${participants.length} total):`);
        participants.forEach((p, index) => {
            console.log(`\n${index + 1}. Participant ID: ${formatForDisplay(p.participant_id)}`);
            console.log(`   Name: ${formatForDisplay(p.full_name)}`);
            console.log(`   Mobile: ${formatForDisplay(p.mobile_number)}`);
            console.log(`   Age: ${formatForDisplay(p.age)}, Gender: ${formatForDisplay(p.gender)}`);
            console.log(`   Address: ${formatForDisplay(p.address)}`);
            console.log(`   Date of Screening: ${formatForDisplay(p.date_of_screening)}`);
            console.log(`   Region: ${formatForDisplay(p.region)}, District: ${formatForDisplay(p.district)}`);
            console.log(`   Facility: ${formatForDisplay(p.facility)}, Community: ${formatForDisplay(p.community)}`);
            console.log(`   Data Collector: ${formatForDisplay(p.data_collector_name)}`);
            console.log(`   Consent: ${p.consent_obtained ? 'Yes' : 'No'}`);
            console.log(`   Diabetes: ${formatForDisplay(p.diabetes_status)}`);
            console.log(`   HIV: ${formatForDisplay(p.hiv_status)}`);
            console.log(`   COVID: ${formatForDisplay(p.covid_status)}`);
            console.log(`   Tobacco Use: ${p.tobacco_use ? 'Yes' : 'No'}, Duration: ${formatForDisplay(p.tobacco_duration)}`);
            console.log(`   Alcohol Use: ${p.alcohol_use ? 'Yes' : 'No'}, Duration: ${formatForDisplay(p.alcohol_duration)}`);
            console.log(`   Previous TB: ${p.previous_tb ? 'Yes' : 'No'}, Year: ${formatForDisplay(p.last_tb_year)}, Completed: ${formatForDisplay(p.tb_treatment_completed)}`);
            console.log(`   Symptoms: ${formatForDisplay(p.symptoms)}`);
            console.log(`   Test Done: ${formatForDisplay(p.test_done)}, Type: ${formatForDisplay(p.test_type)}`);
            console.log(`   Test Result: ${formatForDisplay(p.test_result)}, Site: ${formatForDisplay(p.test_site)}`);
            console.log(`   Test Notes: ${formatForDisplay(p.test_notes)}`);
            console.log(`   Status: ${formatForDisplay(p.status)}`);
            console.log(`   Created: ${formatForDisplay(p.created_at)}`);
            console.log(`   Synced: ${p.synced ? 'Yes' : 'No'}`);
            if (p.analysis_result) {
                try {
                    const analysis = JSON.parse(p.analysis_result);
                    console.log(`   Analysis: Cough Detected: ${analysis.coughDetected}, Confidence: ${analysis.confidence ? (analysis.confidence * 100).toFixed(1) + '%' : 'N/A'}`);
                } catch (e) {
                    console.log(`   Analysis: ${formatForDisplay(p.analysis_result)}`);
                }
            }
        });

        // Get all recordings
        const recordings = await database.getAllAsync<Recording>(
            `SELECT * FROM recordings ORDER BY created_at DESC`
        );
        console.log(`\n🎤 RECORDINGS (${recordings.length} total):`);
        recordings.forEach((r, index) => {
            console.log(`\n${index + 1}. Recording ID: ${formatForDisplay(r.id)}`);
            console.log(`   Participant ID: ${formatForDisplay(r.participant_id)}`);
            console.log(`   Type: ${formatForDisplay(r.recording_type)}`);
            console.log(`   Duration: ${formatForDisplay(r.duration)} seconds`);
            console.log(`   File Path: ${formatForDisplay(r.file_path)}`);
            console.log(`   Created: ${formatForDisplay(r.created_at)}`);
            console.log(`   Synced: ${r.synced ? 'Yes' : 'No'}`);
        });

        // Get stats
        const stats = await getStats();
        console.log(`\n📊 STATISTICS:`);
        console.log(`   Pending: ${stats.pending}`);
        console.log(`   Drafts: ${stats.drafts}`);
        console.log(`   Total: ${stats.total}`);

        console.log('\n========================================\n');
        
        return {
            participants,
            recordings,
            stats
        };
    } catch (error) {
        console.error("Error viewing database contents:", error);
        throw error;
    }
};

export const getStats = async () => {
    const database = await getDB();
    try {
        const pending = await database.getFirstAsync<{ count: number }>(`SELECT COUNT(*) as count FROM participants WHERE status = 'pending'`);
        const drafts = await database.getFirstAsync<{ count: number }>(`SELECT COUNT(*) as count FROM participants WHERE status = 'draft'`);
        const total = await database.getFirstAsync<{ count: number }>(`SELECT COUNT(*) as count FROM participants`);
        
        return {
            pending: pending?.count || 0,
            drafts: drafts?.count || 0,
            total: total?.count || 0
        };
    } catch (error) {
        console.error("Error fetching stats:", error);
        return { pending: 0, drafts: 0, total: 0 };
    }
};

export const getPendingParticipants = async (): Promise<Participant[]> => {
    const database = await getDB();
    try {
        const rows = await database.getAllAsync<Participant>(
            `SELECT * FROM participants WHERE status = 'pending' ORDER BY created_at DESC`
        );
        return rows;
    } catch (error) {
        console.error("Error fetching pending participants:", error);
        return [];
    }
};

export const getDraftParticipants = async (): Promise<Participant[]> => {
    const database = await getDB();
    try {
        const rows = await database.getAllAsync<Participant>(
            `SELECT * FROM participants WHERE status = 'draft' ORDER BY created_at DESC`
        );
        return rows;
    } catch (error) {
        console.error("Error fetching draft participants:", error);
        return [];
    }
};
