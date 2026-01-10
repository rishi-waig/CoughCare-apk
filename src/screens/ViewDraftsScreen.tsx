/**
 * View Drafts Screen - Shows all draft records
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    SafeAreaView,
    StatusBar,
    TouchableOpacity,
    ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { getDraftParticipants, Participant } from '../services/DatabaseService';

type ViewDraftsScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'ViewDrafts'>;

const ViewDraftsScreen = () => {
    const navigation = useNavigation<ViewDraftsScreenNavigationProp>();
    const [participants, setParticipants] = useState<Participant[]>([]);
    const [loading, setLoading] = useState(true);

    useFocusEffect(
        React.useCallback(() => {
            loadData();
        }, [])
    );

    const loadData = async () => {
        try {
            setLoading(true);
            const data = await getDraftParticipants();
            setParticipants(data);
        } catch (error) {
            console.error('Error loading drafts:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString?: string) => {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            const day = date.getDate();
            const month = date.toLocaleDateString('en-GB', { month: 'short' });
            const year = date.getFullYear();
            const hours = date.getHours().toString().padStart(2, '0');
            const minutes = date.getMinutes().toString().padStart(2, '0');
            return `${day} ${month} ${year}, ${hours}:${minutes}`;
        } catch {
            return 'N/A';
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#2563EB" />

            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="white" />
                </TouchableOpacity>
                <View style={{ flex: 1 }}>
                    <Text style={styles.headerTitle}>View Drafts</Text>
                    <Text style={styles.headerSubtitle}>
                        {participants.length} {participants.length === 1 ? 'item' : 'items'}
                    </Text>
                </View>
            </View>

            {loading ? (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#2563EB" />
                    <Text style={styles.loadingText}>Loading...</Text>
                </View>
            ) : (
                <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 20 }}>
                    {participants.length === 0 ? (
                        <View style={styles.emptyContainer}>
                            <Ionicons name="document-text-outline" size={64} color="#94A3B8" />
                            <Text style={styles.emptyText}>No draft records</Text>
                            <Text style={styles.emptySubtext}>All records have been completed</Text>
                        </View>
                    ) : (
                        participants.map((participant) => (
                            <TouchableOpacity
                                key={participant.participant_id}
                                style={styles.recordCard}
                                onPress={() => {
                                    navigation.navigate('ViewRecord', { participantId: participant.participant_id });
                                }}
                            >
                                <View style={styles.cardHeader}>
                                    <View style={{ flex: 1 }}>
                                        <Text style={styles.participantName}>
                                            {participant.full_name || 'Unnamed Participant'}
                                        </Text>
                                        <Text style={styles.mobileText}>
                                            Mobile: {participant.mobile_number || 'N/A'}
                                        </Text>
                                    </View>
                                    <View style={styles.draftBadge}>
                                        <Text style={styles.draftText}>Draft</Text>
                                    </View>
                                </View>
                                <View style={styles.cardFooter}>
                                    <Ionicons name="time-outline" size={16} color="#64748B" style={{ marginRight: 4 }} />
                                    <Text style={styles.dateText}>
                                        {formatDate(participant.created_at)}
                                    </Text>
                                </View>
                            </TouchableOpacity>
                        ))
                    )}
                </ScrollView>
            )}
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
        marginTop: 12,
        color: '#64748B',
        fontSize: 14,
    },
    emptyContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40,
    },
    emptyText: {
        fontSize: 18,
        fontWeight: '600',
        color: '#64748B',
        marginTop: 16,
    },
    emptySubtext: {
        fontSize: 14,
        color: '#94A3B8',
        marginTop: 8,
    },
    recordCard: {
        backgroundColor: 'white',
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    participantName: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1E293B',
        marginBottom: 4,
    },
    mobileText: {
        fontSize: 14,
        color: '#64748B',
    },
    draftBadge: {
        backgroundColor: '#6366F1',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 12,
    },
    draftText: {
        color: 'white',
        fontSize: 12,
        fontWeight: '600',
    },
    cardFooter: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: '#F1F5F9',
    },
    dateText: {
        fontSize: 12,
        color: '#64748B',
    },
});

export default ViewDraftsScreen;

