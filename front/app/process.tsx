// app/process.tsx
// Process screen now uses translated strings.
import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Button,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useColorScheme
} from 'react-native';
import { useLanguage } from '../src/LanguageContext';
import { translations } from '../src/i18n';
import { ThemeColors } from '../utils/colors';

export default function ProcessScreen() {
  const colorScheme = useColorScheme() || 'light';
  const colors = ThemeColors[colorScheme];
  const { locale } = useLanguage();
  const i18n = translations[locale];

  const [location, setLocation] = useState('');
  const [status, setStatus] = useState(i18n.process.statusReady);
  const [result, setResult] = useState(i18n.process.resultPrediction);
  const [isLoading, setIsLoading] = useState(false);

  // Use your computer's local IPv4 address.
  const API_BASE_URL = "http://192.168.187.187:8000";

  const processAndPredict = async () => {
    if (!location) {
      Alert.alert(i18n.process.inputRequired, i18n.process.provideLocation);
      return;
    }

    setIsLoading(true);
    setStatus(i18n.process.statusProcessing);
    setResult(i18n.process.resultPrediction);

    try {
      const response = await fetch(`${API_BASE_URL}/process-and-predict?location=${location}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || i18n.common.unknownError);
      }

      const data = await response.json();
      setResult(JSON.stringify(data, null, 2));
      setStatus(i18n.process.statusReady); // Reset to "Ready"
    } catch (error: any) {
      setStatus(`${i18n.common.error}: ${error.message}`);
      setResult(i18n.process.predictionFailed);
      console.error("Prediction Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const styles = StyleSheet.create({
    container: {
      flexGrow: 1,
      padding: 20,
      backgroundColor: colors.background,
      paddingTop: 40, 
    },
    card: {
      backgroundColor: colors.cardBackground,
      padding: 25,
      borderRadius: 10,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 15,
      elevation: 5,
    },
    title: {
      fontSize: 24,
      fontWeight: 'bold',
      color: colors.text,
      textAlign: 'center',
      marginBottom: 10,
    },
    status: {
      fontSize: 16,
      fontStyle: 'italic',
      color: colors.subText,
      textAlign: 'center',
      marginBottom: 20,
    },
    label: {
      fontSize: 16,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 5,
      marginTop: 10,
    },
    textInput: {
      height: 40,
      borderColor: colors.inputBorder,
      borderWidth: 1,
      borderRadius: 5,
      paddingHorizontal: 10,
      fontSize: 16,
      marginBottom: 10,
      color: colors.text
    },
    buttonContainer: {
      marginTop: 20,
    },
    resultTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      marginTop: 20,
      marginBottom: 10,
      color: colors.text,
    },
    resultBox: {
      backgroundColor: colors.cardBackground,
      padding: 15,
      borderRadius: 5,
      maxHeight: 250,
    },
    resultText: {
      fontSize: 14,
      color: colors.subText,
    },
  });

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.background }}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>{i18n.process.title}</Text>
          <Text style={styles.status}>{status}</Text>

          <Text style={styles.label}>{i18n.process.enterLocation}</Text>
          <TextInput
  style={styles.textInput}
  onChangeText={setLocation}
  value={location}
  placeholder={i18n.process.enterLocation}   // ✅ now localized
  placeholderTextColor={colors.subText}
/>

          <View style={styles.buttonContainer}>
            <Button
              title={isLoading ? i18n.process.uploading : i18n.process.processButton}
              onPress={processAndPredict}
              disabled={isLoading}
              color={colors.button}
            />
          </View>
          {isLoading && <ActivityIndicator size="large" color={colors.button} />}
          <Text style={styles.resultTitle}>{i18n.process.resultTitle}</Text>
          <ScrollView style={styles.resultBox}>
            <Text style={styles.resultText}>{result}</Text>
          </ScrollView>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
