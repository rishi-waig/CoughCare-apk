import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, StatusBar, SafeAreaView, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';

type DashboardScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Dashboard'>;

import { RouteProp, useRoute } from '@react-navigation/native';

type DashboardScreenRouteProp = RouteProp<RootStackParamList, 'Dashboard'>;

import { useFocusEffect } from '@react-navigation/native';
import { getParticipants, getStats, viewDatabaseContents } from '../services/DatabaseService';

const DashboardScreen = () => {
    const navigation = useNavigation<DashboardScreenNavigationProp>();
    const route = useRoute<DashboardScreenRouteProp>();

    const [stats, setStats] = useState({ pending: 0, drafts: 0, total: 0 });
    const [isOnline, setIsOnline] = useState(true);
    const [recentCases, setRecentCases] = useState<any[]>([]);

    const loadData = async () => {
        try {
            const fetchedStats = await getStats();
            setStats(fetchedStats);

            const participants = await getParticipants();
            const formattedCases = participants.map(p => ({
                id: p.participant_id,
                name: p.full_name || 'Unnamed Participant',
                phone: p.mobile_number || 'No phone',
                date: p.created_at || new Date().toISOString(),
                status: p.analysis_result ? (JSON.parse(p.analysis_result).coughDetected ? "Cough Detected" : "No Cough") : "Pending",
                region: p.region,
                recordStatus: p.status || 'pending',
                analysis: p.analysis_result ? JSON.parse(p.analysis_result) : null
            }));
            setRecentCases(formattedCases);
        } catch (error) {
            console.error("Failed to load dashboard data", error);
        }
    };

    // Log database contents on mount (for debugging)
    useEffect(() => {
        viewDatabaseContents().catch(console.error);
    }, []);

    // Network connectivity detection
    useEffect(() => {
        // For web, use navigator.onLine
        if (Platform.OS === 'web') {
            const updateOnlineStatus = () => {
                setIsOnline(typeof navigator !== 'undefined' ? navigator.onLine : true);
            };

            // Set initial status
            updateOnlineStatus();

            // Listen for online/offline events
            if (typeof window !== 'undefined') {
                window.addEventListener('online', updateOnlineStatus);
                window.addEventListener('offline', updateOnlineStatus);
            }

            return () => {
                if (typeof window !== 'undefined') {
                    window.removeEventListener('online', updateOnlineStatus);
                    window.removeEventListener('offline', updateOnlineStatus);
                }
            };
        } else {
            // For native, dynamically import NetInfo to avoid web bundling issues
            let unsubscribe: (() => void) | null = null;
            let isMounted = true;

            const setupNetInfo = async () => {
                try {
                    // Dynamic import with error handling
                    const netInfoModule = await import('@react-native-community/netinfo');
                    const NetInfo = netInfoModule.default || netInfoModule;
                    
                    if (!isMounted) return;
                    
                    // Set up listener
                    unsubscribe = NetInfo.addEventListener(state => {
                        if (isMounted) {
                            setIsOnline(state.isConnected ?? false);
                        }
                    });

                    // Get initial state
                    const state = await NetInfo.fetch();
                    if (isMounted) {
                        setIsOnline(state.isConnected ?? false);
                    }
                } catch (error) {
                    // NetInfo might not be available (e.g., during web bundling)
                    // Default to online for native platforms
                    if (isMounted) {
                        console.warn('NetInfo not available, defaulting to online:', error);
                        setIsOnline(true);
                    }
                }
            };

            setupNetInfo();

            return () => {
                isMounted = false;
                if (unsubscribe) {
                    unsubscribe();
                }
            };
        }
    }, []);

    useFocusEffect(
        React.useCallback(() => {
            loadData();
        }, [])
    );

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#2563EB" />

            {/* Header */}
            <View style={styles.header}>
                <View>
                    <Text style={styles.headerTitle}>TB Screening</Text>
                    <Text style={styles.headerSubtitle}>Ghana Data Collection</Text>
                </View>
                <View style={[
                    styles.onlineBadge,
                    !isOnline && { backgroundColor: '#64748B' }
                ]}>
                    <Ionicons 
                        name={isOnline ? "wifi" : "wifi-outline"} 
                        size={16} 
                        color="white" 
                        style={{ marginRight: 4 }} 
                    />
                    <Text style={styles.onlineText}>{isOnline ? 'Online' : 'Offline'}</Text>
                </View>
            </View>

            <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 20 }}>
                {/* Stats Cards */}
                <View style={styles.statsContainer}>
                    <View style={styles.statCard}>
                        <Text style={styles.statLabel}>Pending</Text>
                        <Text style={[styles.statValue, { color: '#EA580C' }]}>{stats.pending}</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Text style={styles.statLabel}>Drafts</Text>
                        <Text style={[styles.statValue, { color: '#64748B' }]}>{stats.drafts}</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Text style={styles.statLabel}>Total</Text>
                        <Text style={[styles.statValue, { color: '#2563EB' }]}>{stats.total}</Text>
                    </View>
                </View>

                {/* New Participant Button */}
                <TouchableOpacity
                    style={styles.newParticipantBtn}
                    onPress={() => {
                        // Navigate to new participant flow
                        navigation.navigate('NewParticipant');
                    }}
                >
                    <Ionicons name="add" size={24} color="white" style={{ marginRight: 8 }} />
                    <Text style={styles.newParticipantText}>New Participant</Text>
                </TouchableOpacity>

                {/* Sync Section */}
                <View style={styles.sectionCard}>
                    <View style={styles.syncHeader}>
                        <View>
                            <Text style={styles.sectionLabel}>Last Sync</Text>
                            <Text style={styles.syncTime}>—</Text>
                        </View>
                        <TouchableOpacity style={styles.syncBtn}>
                            <Ionicons name="refresh" size={18} color="white" style={{ marginRight: 6 }} />
                            <Text style={styles.syncBtnText}>Sync</Text>
                        </TouchableOpacity>
                    </View>
                    <View style={styles.syncStatusContainer}>
                        <Text style={styles.syncStatusText}>{stats.pending} record(s) ready to sync</Text>
                    </View>
                </View>

                {/* Recent Cases */}
                <Text style={styles.sectionTitle}>Recent Cases</Text>

                {recentCases.length === 0 ? (
                    <View style={{ padding: 20, alignItems: 'center' }}>
                        <Text style={{ color: '#64748B' }}>No recent cases found.</Text>
                    </View>
                ) : (
                    recentCases.map((item, index) => {
                        return (
                            <TouchableOpacity
                                key={index}
                                style={styles.caseCard}
                                onPress={() => {
                                    navigation.navigate('ViewRecord', { participantId: item.id });
                                }}
                            >
                                <View style={styles.caseHeader}>
                                    <View>
                                        <Text style={styles.caseName}>{item.name}</Text>
                                        <Text style={styles.caseId}>{item.phone}</Text>
                                    </View>
                                    <View style={[
                                        styles.statusBadge,
                                        item.recordStatus === 'pending' ? { backgroundColor: '#FEF3C7' } :
                                            item.recordStatus === 'synced' ? { backgroundColor: '#DCFCE7' } :
                                                { backgroundColor: '#E0E7FF' }
                                    ]}>
                                        <Text style={[
                                            styles.statusText,
                                            item.recordStatus === 'pending' ? { color: '#D97706' } :
                                                item.recordStatus === 'synced' ? { color: '#16A34A' } :
                                                    { color: '#6366F1' }
                                        ]}>
                                            {item.recordStatus === 'pending' ? 'Pending' :
                                                item.recordStatus === 'synced' ? 'Synced' : 'Draft'}
                                        </Text>
                                    </View>
                                </View>
                                <View style={styles.caseFooter}>
                                    <View style={styles.caseInfoRow}>
                                        <Ionicons name="time-outline" size={16} color="#64748B" style={{ marginRight: 4 }} />
                                        <Text style={styles.caseInfoText}>
                                            {new Date(item.date).toLocaleDateString('en-GB', {
                                                day: 'numeric',
                                                month: 'short',
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </Text>
                                    </View>
                                    {item.region && (
                                        <View style={styles.caseInfoRow}>
                                            <Ionicons name="location-outline" size={16} color="#E11D48" style={{ marginRight: 4 }} />
                                            <Text style={styles.caseInfoText}>{item.region}</Text>
                                        </View>
                                    )}
                                </View>
                                {item.analysis && (
                                    <View style={{ marginTop: 8, paddingTop: 8, borderTopWidth: 1, borderTopColor: '#F1F5F9' }}>
                                        <Text style={{ fontSize: 12, color: '#64748B' }}>
                                            Confidence: {(item.analysis.confidence * 100).toFixed(1)}%
                                        </Text>
                                    </View>
                                )}
                            </TouchableOpacity>
                        );
                    })
                )}

                {/* Footer Actions */}
                <View style={styles.footerActions}>
                    <TouchableOpacity
                        style={styles.footerBtn}
                        onPress={() => navigation.navigate('PendingResults')}
                    >
                        <Text style={styles.footerBtnText}>View Pending</Text>
                        {stats.pending > 0 && (
                            <Text style={styles.footerBtnSubtext}>{stats.pending} items</Text>
                        )}
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.footerBtn}
                        onPress={() => navigation.navigate('ViewDrafts')}
                    >
                        <Text style={styles.footerBtnText}>View Drafts</Text>
                        {stats.drafts > 0 && (
                            <Text style={styles.footerBtnSubtext}>{stats.drafts} items</Text>
                        )}
                    </TouchableOpacity>
                </View>

            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#F8FAFC',
    },
    header: {
        backgroundColor: '#2563EB',
        padding: 20,
        paddingTop: 40,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
    headerSubtitle: {
        fontSize: 14,
        color: '#BFDBFE',
        marginTop: 4,
    },
    onlineBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 20,
    },
    onlineText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 12,
    },
    content: {
        flex: 1,
        padding: 16,
    },
    statsContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 20,
    },
    statCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 16,
        width: '31%',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    statLabel: {
        fontSize: 12,
        color: '#64748B',
        marginBottom: 8,
    },
    statValue: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    newParticipantBtn: {
        backgroundColor: '#059669',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        borderRadius: 12,
        marginBottom: 20,
        shadowColor: '#059669',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 3,
    },
    newParticipantText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
    sectionCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 16,
        marginBottom: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    syncHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
    },
    sectionLabel: {
        fontSize: 12,
        color: '#64748B',
    },
    syncTime: {
        fontSize: 16,
        color: '#334155',
        fontWeight: '500',
    },
    syncBtn: {
        backgroundColor: '#2563EB',
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 8,
    },
    syncBtnText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 14,
    },
    syncStatusContainer: {
        backgroundColor: '#FFF7ED',
        padding: 12,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#FFEDD5',
        alignItems: 'center',
    },
    syncStatusText: {
        color: '#C2410C',
        fontSize: 14,
        fontWeight: '500',
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#1E293B',
        marginBottom: 12,
    },
    caseCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 16,
        marginBottom: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    caseHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    caseName: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1E293B',
    },
    caseId: {
        fontSize: 14,
        color: '#64748B',
        marginTop: 2,
    },
    statusBadge: {
        backgroundColor: '#FFEDD5',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
    },
    statusText: {
        color: '#C2410C',
        fontSize: 12,
        fontWeight: '600',
    },
    caseFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        borderTopWidth: 1,
        borderTopColor: '#F1F5F9',
        paddingTop: 12,
    },
    caseInfoRow: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    caseInfoText: {
        fontSize: 12,
        color: '#64748B',
    },
    footerActions: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        gap: 12,
    },
    footerBtn: {
        flex: 1,
        backgroundColor: 'white',
        padding: 16,
        borderRadius: 12,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    footerBtnText: {
        color: '#334155',
        fontWeight: '600',
        marginBottom: 4,
    },
    footerBtnSubtext: {
        fontSize: 12,
        color: '#EA580C',
    },
});

export default DashboardScreen;
