/**
 * Reusable Radio Button Group component
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface RadioButtonGroupProps {
    label?: string;
    value: boolean | null;
    onSelect: (val: boolean) => void;
    variant?: 'default' | 'success' | 'error';
}

export const RadioButtonGroup: React.FC<RadioButtonGroupProps> = ({
    label,
    value,
    onSelect,
    variant = 'default'
}) => {
    const getButtonStyle = (selected: boolean) => {
        if (variant === 'success' && selected) {
            return [styles.radioButton, styles.radioButtonSuccess];
        }
        if (variant === 'error' && selected) {
            return [styles.radioButton, styles.radioButtonError];
        }
        if (selected) {
            return [styles.radioButton, styles.radioButtonActive];
        }
        return styles.radioButton;
    };

    const getTextStyle = (selected: boolean) => {
        if (variant === 'success' && selected) {
            return [styles.radioText, styles.radioTextWhite];
        }
        if (variant === 'error' && selected) {
            return [styles.radioText, styles.radioTextWhite];
        }
        if (selected) {
            return [styles.radioText, styles.radioTextActive];
        }
        return styles.radioText;
    };

    return (
        <View style={{ marginBottom: 16 }}>
            {label && <Text style={styles.label}>{label}</Text>}
            <View style={styles.radioGroup}>
                <TouchableOpacity
                    style={getButtonStyle(value === true)}
                    onPress={() => onSelect(true)}
                >
                    <Text style={getTextStyle(value === true)}>Yes</Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={getButtonStyle(value === false)}
                    onPress={() => onSelect(false)}
                >
                    <Text style={getTextStyle(value === false)}>No</Text>
                </TouchableOpacity>
            </View>
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
    radioGroup: {
        flexDirection: 'row',
        gap: 16,
    },
    radioButton: {
        flex: 1,
        backgroundColor: '#F8FAFC', // Light gray background for unselected
        padding: 12,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#CBD5E1',
        alignItems: 'center',
        justifyContent: 'center',
    },
    radioButtonActive: {
        borderColor: '#2563EB',
        backgroundColor: '#EFF6FF',
        borderWidth: 2, // Thicker border for active
    },
    radioButtonSuccess: {
        backgroundColor: '#22C55E',
        borderColor: '#22C55E',
    },
    radioButtonError: {
        backgroundColor: '#EF4444',
        borderColor: '#EF4444',
    },
    radioText: {
        color: '#000000', // Pure black for maximum contrast
        fontWeight: '600',
        fontSize: 16,
    },
    radioTextActive: {
        color: '#2563EB',
        fontWeight: '600',
    },
    radioTextWhite: {
        color: 'white',
        fontWeight: '600',
    },
});

