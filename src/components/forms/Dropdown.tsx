/**
 * Reusable Dropdown component
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface DropdownProps {
    label: string;
    value: string | null;
    options: string[];
    onSelect: (val: string) => void;
    isExpanded: boolean;
    onToggle: () => void;
    placeholder?: string;
}

export const Dropdown: React.FC<DropdownProps> = ({
    label,
    value,
    options,
    onSelect,
    isExpanded,
    onToggle,
    placeholder = "Select"
}) => {
    return (
        <View style={{ marginBottom: 16, zIndex: isExpanded ? 1000 : 1 }}>
            <Text style={styles.label}>{label}</Text>
            <TouchableOpacity
                style={styles.dropdown}
                onPress={onToggle}
            >
                <Text style={value ? styles.inputText : styles.placeholderText}>
                    {value || placeholder}
                </Text>
                <Ionicons name="chevron-down" size={20} color="#64748B" />
            </TouchableOpacity>
            {isExpanded && (
                <View style={styles.dropdownList}>
                    <ScrollView
                        nestedScrollEnabled
                        style={{ maxHeight: 250 }}
                        persistentScrollbar={true}
                        keyboardShouldPersistTaps="handled"
                    >
                        {options.map((item) => (
                            <TouchableOpacity
                                key={item}
                                style={styles.dropdownItem}
                                onPress={() => {
                                    onSelect(item);
                                    onToggle();
                                }}
                            >
                                <Text style={styles.dropdownItemText}>{item}</Text>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                </View>
            )}
        </View>
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
    dropdown: {
        borderWidth: 1,
        borderColor: '#CBD5E1',
        borderRadius: 8,
        padding: 12,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: 'white',
    },
    placeholderText: {
        color: '#94A3B8',
        fontSize: 16,
    },
    inputText: {
        color: '#1E293B',
        fontSize: 16,
    },
    dropdownList: {
        position: 'absolute',
        top: '100%',
        left: 0,
        right: 0,
        backgroundColor: 'white',
        borderWidth: 1,
        borderColor: '#CBD5E1',
        borderRadius: 8,
        marginTop: 4,
        zIndex: 1000,
        elevation: 5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        maxHeight: 200,
    },
    dropdownItem: {
        padding: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#F1F5F9',
    },
    dropdownItemText: {
        fontSize: 16,
        color: '#1E293B',
    },
});

