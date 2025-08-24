import React from "react";
import { Button, ScrollView, StyleSheet, Text, View, useColorScheme } from "react-native";
import { Locale, translations } from "../../src/i18n";
import { useLanguage } from "../../src/LanguageContext";
import { ThemeColors } from "../../utils/colors";

export default function SettingsScreen() {
  const colorScheme = useColorScheme() || "light";
  const colors = ThemeColors[colorScheme];
  const { locale, setLocale } = useLanguage();
  const i18n = translations[locale] || translations["en"];
  const locales = Object.keys(translations) as Locale[];

  const settingsTitle = i18n.settings.title;
  const languageSelection = (i18n.settings as any).languageSelection || "Select your preferred language";
  const currentLanguageLabel = (i18n.settings as any).currentLanguage || "Current Language:";

  const styles = StyleSheet.create({
    container: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: colors.background, padding: 20 },
    text: { fontSize: 20, color: colors.text, marginBottom: 20 },
    buttonContainer: { flexDirection: "row", flexWrap: "wrap", justifyContent: "space-around" },
    buttonWrapper: { margin: 5, minWidth: 100 },
  });

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.text}>{settingsTitle}</Text>
      <Text style={styles.text}>{languageSelection}</Text>
      <Text style={styles.text}>{currentLanguageLabel} {locale.toUpperCase()}</Text>
      <View style={styles.buttonContainer}>
        {locales.map((l) => (
          <View key={l} style={styles.buttonWrapper}>
            <Button title={l.toUpperCase()} onPress={() => setLocale(l)} color={colors.button} disabled={locale === l} />
          </View>
        ))}
      </View>
    </ScrollView>
  );
}
