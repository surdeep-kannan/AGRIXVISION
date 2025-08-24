import React, { useState } from 'react';
import { Button, Image, View, Text, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = 'http://192.168.187.187:8000/predict-disease';

export default function DiseasePredictor() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Function to pick an image from the device's gallery
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setPrediction(null); // Clear previous prediction
      setConfidence(null);
    }
  };

  // Function to send the image to the backend API
  const uploadImage = async () => {
    if (!image) {
      Alert.alert('No Image Selected', 'Please select an image to predict.');
      return;
    }

    setLoading(true);
    setPrediction('Predicting...');
    
    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('image_file', {
      uri: image,
      name: 'photo.jpg',
      type: 'image/jpeg',
    } as any);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data.predicted_class);
        setConfidence(data.confidence);
      } else {
        Alert.alert('Prediction Failed', data.detail || 'An error occurred on the server.');
        setPrediction(null);
        setConfidence(null);
      }

    } catch (error) {
      Alert.alert('Network Error', 'Could not connect to the backend server. Is it running?');
      setPrediction(null);
      setConfidence(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Button title="Pick an image from the gallery" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={styles.image} />}
      <View style={styles.buttonContainer}>
        <Button title="Predict Disease" onPress={uploadImage} disabled={!image || loading} />
      </View>
      
      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {prediction && !loading && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultText}>Prediction:</Text>
          <Text style={styles.resultValue}>{prediction}</Text>
          {confidence !== null && (
            <Text style={styles.confidenceText}>
              Confidence: {(confidence * 100).toFixed(2)}%
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  buttonContainer: {
    marginVertical: 10,
  },
  image: {
    width: 300,
    height: 300,
    resizeMode: 'contain',
    marginVertical: 20,
  },
  resultContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  resultText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultValue: {
    fontSize: 24,
    color: 'green',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  confidenceText: {
    fontSize: 16,
    color: 'gray',
    marginTop: 5,
  },
});