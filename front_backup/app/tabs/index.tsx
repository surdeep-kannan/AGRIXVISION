import { useRouter } from "expo-router";
import React from "react";
import { Button, ScrollView, StyleSheet, Text, View, useColorScheme } from "react-native";
import { useLanguage } from "../../src/LanguageContext";
import { translations } from "../../src/i18n";
import { ThemeColors } from "../../utils/colors";

export default function HomeScreen() {
  const colorScheme = useColorScheme() || "light";
  const colors = ThemeColors[colorScheme];
  const { locale } = useLanguage();
  const i18n = translations[locale];
  const router = useRouter();

  const styles = StyleSheet.create({
    container: { flexGrow: 1, justifyContent: "center", alignItems: "center", backgroundColor: colors.background, padding: 20 },
    card: { backgroundColor: colors.cardBackground, padding: 25, borderRadius: 10, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 15, elevation: 5, width: "100%", alignItems: "center" },
    title: { fontSize: 24, fontWeight: "bold", color: colors.text, marginBottom: 10 },
    subtitle: { fontSize: 16, color: colors.subText, marginBottom: 20, textAlign: "center" },
    buttonContainer: { marginTop: 10, width: "100%" },
  });

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.card}>
        <Text style={styles.title}>{i18n.home.title}</Text>
        <Text style={styles.subtitle}>{i18n.home.subtitle}</Text>

        <View style={styles.buttonContainer}>
          <Button
            title={i18n.home.button}
            onPress={() => router.push("/process")} // Corrected path
            color={colors.button}
          />
        </View>

        <View style={styles.buttonContainer}>
          <Button
            title="Disease Predictor"
            onPress={() => router.push("/tabs/disease-predictor")} // Corrected path
            color={colors.button}
          />
        </View>
      </View>
    </ScrollView>
  );
}
