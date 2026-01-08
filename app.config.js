export default {
  expo: {
    name: "CoughCare",
    slug: "coughcare",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./public/logo.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./public/logo.png",
      resizeMode: "contain",
      backgroundColor: "#158B95"
    },
    assetBundlePatterns: [
      "**/*"
    ],
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.coughcare.app"
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./public/logo.png",
        backgroundColor: "#158B95"
      },
      package: "com.coughcare.app",
      permissions: [
        "RECORD_AUDIO"
      ]
    },
    web: {
      favicon: "./public/logo.png"
    },
    plugins: [
      [
        "expo-av",
        {
          microphonePermission: "Allow CoughCare to access your microphone to record cough sounds."
        }
      ]
    ],
    extra: {
      eas: {
        projectId: "1259b35e-7195-465f-bc01-dc86fcf7f4b1"
      },
      apiBaseUrl: process.env.EXPO_PUBLIC_API_BASE_URL || "",
      logoVersion: process.env.EXPO_PUBLIC_LOGO_VERSION || "1",
      logoAlt: process.env.EXPO_PUBLIC_LOGO_ALT || "AI Cough Screening Assistant"
    }
  }
};

