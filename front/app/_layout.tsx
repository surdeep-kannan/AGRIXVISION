import { Stack } from "expo-router";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { LanguageProvider } from "../src/LanguageContext";

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <LanguageProvider>
        <Stack screenOptions={{ headerShown: false }}>
          {/* Tabs layout */}
          <Stack.Screen name="tabs" />

          {/* Standalone process screen */}
          <Stack.Screen name="process" />
        </Stack>
      </LanguageProvider>
    </SafeAreaProvider>
  );
}
