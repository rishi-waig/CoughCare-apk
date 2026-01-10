/**
 * Accordion Section Component
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface AccordionSectionProps {
    title: string;
    section: string;
    isExpanded: boolean;
    onToggle: () => void;
    children: React.ReactNode;
    disabled?: boolean;
    disabledMessage?: string;
}

export const AccordionSection: React.FC<AccordionSectionProps> = ({
    title,
    section,
    isExpanded,
    onToggle,
    children,
    disabled = false,
    disabledMessage
}) => {
    const handleToggle = () => {
        if (!disabled) {
            onToggle();
        }
    };

    return (
        <View style={[styles.sectionContainer, isExpanded && { zIndex: 20 }, disabled && styles.sectionDisabled]}>
            <TouchableOpacity 
                style={[styles.accordionHeader, disabled && styles.accordionHeaderDisabled]} 
                onPress={handleToggle}
                disabled={disabled}
            >
                <Text style={[styles.accordionTitle, disabled && styles.accordionTitleDisabled]}>{title}</Text>
                <Ionicons
                    name={isExpanded ? "chevron-up" : "chevron-down"}
                    size={24}
                    color={disabled ? "#94A3B8" : "#64748B"}
                />
            </TouchableOpacity>
            {isExpanded && !disabled && (
                <View style={styles.sectionContent}>
                    {children}
                </View>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    sectionContainer: {
        backgroundColor: 'white',
        marginTop: 1,
        borderBottomWidth: 1,
        borderBottomColor: '#E2E8F0',
    },
    accordionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: 16,
        backgroundColor: 'white',
    },
    accordionTitle: {
        fontSize: 16,
        color: '#334155',
        fontWeight: '500',
    },
    sectionContent: {
        padding: 16,
        paddingTop: 0,
    },
    sectionDisabled: {
        backgroundColor: 'white',
    },
    accordionHeaderDisabled: {
        backgroundColor: 'white',
    },
    accordionTitleDisabled: {
        color: '#94A3B8',
        fontWeight: '400',
    },
    disabledMessage: {
        padding: 12,
        paddingHorizontal: 16,
        backgroundColor: '#FEF2F2',
        borderTopWidth: 1,
        borderTopColor: '#FECACA',
    },
    disabledMessageText: {
        color: '#B91C1C',
        fontSize: 13,
        fontStyle: 'italic',
    },
});

