# CoughCare App (Deployment)

This is the deployment repository for the CoughCare application.

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

Clone the **PROD** branch of the repository:

```bash
git clone -b PROD https://github.com/rishi-waig/CoughCare-apk.git
cd CoughCare-apk
```

### 2. Install Dependencies

Install the necessary packages using npm:

```bash
npm install
```

### 3. Run the Application

Start the development server:

```bash
npx expo start
```


#### Run on Web (Localhost)

*   Or run directly with:
    ```bash
    npx expo start
    ```
try on 8081/8082
### 4. Build APK (Optional)

To build the Android APK using EAS:

```bash
eas build -p android --profile preview
```

## 📋 Prerequisites

*   [Node.js](https://nodejs.org/) (LTS version recommended)
*   [Git](https://git-scm.com/)
*   [Expo CLI](https://docs.expo.dev/get-started/installation/) (optional, can use `npx expo`)

## 🛠️ Troubleshooting

*   **Audio Recording Issues:** Ensure you are testing on a real device or a simulator that supports audio input. The app uses `react-native-audio-record` which requires native code (development build or APK). It may not work fully in Expo Go.
*   **Build Errors:** Make sure you have the latest version of EAS CLI installed: `npm install -g eas-cli`.
