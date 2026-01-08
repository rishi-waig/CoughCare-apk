import React, { ReactNode } from 'react';
import { View, Image, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import Constants from 'expo-constants';
import { COLORS } from '../theme';

interface AppHeaderProps {
  rightSlot?: ReactNode;
  variant?: 'solid' | 'translucent';
}

const LOGO_ALT = Constants.expoConfig?.extra?.logoAlt || 'AI Cough Screening Assistant';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

export function AppHeader({ rightSlot, variant = 'solid' }: AppHeaderProps) {
  const navigation = useNavigation<NavigationProp>();
  const translateY = useSharedValue(-80);
  const opacity = useSharedValue(0);

  React.useEffect(() => {
    translateY.value = withSpring(0, { stiffness: 100, damping: 15 });
    opacity.value = withSpring(1, { stiffness: 100, damping: 15 });
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
    opacity: opacity.value,
  }));

  return (
    <Animated.View style={[styles.header, variant === 'translucent' && styles.translucent, animatedStyle]}>
      <View style={styles.container}>
        <TouchableOpacity
          onPress={() => navigation.navigate('Home')}
          style={styles.logoButton}
          activeOpacity={0.7}
        >
          <Image
            source={require('../../public/logo.png')}
            style={styles.logo}
            resizeMode="contain"
            accessibilityLabel={LOGO_ALT}
          />
        </TouchableOpacity>

        {rightSlot && (
          <View style={styles.rightSlot}>
            {rightSlot}
          </View>
        )}
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: COLORS.background,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 4,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    paddingHorizontal: 20,
    paddingVertical: 12,
    zIndex: 30,
    paddingTop: Platform.OS === 'android' ? 40 : 12, // Adjust for status bar on Android if not handled by SafeAreaView
  },
  translucent: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderBottomColor: 'rgba(255, 255, 255, 0.4)',
  },
  container: {
    flexDirection: 'row',
    width: '100%',
    maxWidth: 1200,
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  logoButton: {
    alignItems: 'flex-start',
  },
  logo: {
    height: 40, // Slightly smaller for better proportion
    width: 120, // Ensure enough width for aspect ratio
  },
  rightSlot: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
});

export default AppHeader;
