import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { Text, View, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from '@expo/vector-icons/Ionicons';
import ChatbotScreen from './screens/ChatbotScreen';
import MapScreen from './screens/Mapscreen';
import AgriSubsidyScreen from './screens/AgriSubsidyScreen'; 


const HomeScreen = ({ navigation }: any) => {
  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerText}>AgriXVision</Text>
        </View>


        <View style={styles.welcomeCard}>
          <Text style={styles.welcomeTitle}>Welcome to Your Farm Dashboard</Text>
          <Text style={styles.welcomeText}>
            Easily manage your crops, get expert advice, and explore your fields.
          </Text>
        </View>

        
        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Chatbot')}
        >
          <Text style={styles.cardTitle}>Surya Chatbot</Text>
          <Text style={styles.cardText}>
            Get instant advice on diseases, fertilizers, and farming techniques.
          </Text>
        </TouchableOpacity>

       
        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Map')}
        >
          <Text style={styles.cardTitle}>Map & Analytics</Text>
          <Text style={styles.cardText}>
            View your fields on a map, track crop health, and analyze soil data.
          </Text>
        </TouchableOpacity>

        
        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Subsidiary')}
        >
          <Text style={styles.cardTitle}>Government Subsidies</Text>
          <Text style={styles.cardText}>
            Explore subsidies provided by Central and State Governments of India.
          </Text>
        </TouchableOpacity>
      </ScrollView>
      <StatusBar style="auto" />
    </View>
  );
};

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: '#2E8B57',
          tabBarInactiveTintColor: '#A9A9A9',
          tabBarLabelStyle: { fontSize: 12, fontWeight: 'bold' },
          tabBarStyle: { backgroundColor: '#F0FFF0', borderTopColor: 'transparent', height: 60 },
        }}
      >
        <Tab.Screen
          name="Home"
          component={HomeScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Ionicons name="home" color={color} size={size} />,
          }}
        />
        <Tab.Screen
          name="Map"
          component={MapScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Ionicons name="map" color={color} size={size} />,
          }}
        />
        <Tab.Screen
          name="Chatbot"
          component={ChatbotScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Ionicons name="chatbox-ellipses" color={color} size={size} />,
          }}
        />
        <Tab.Screen
          name="Subsidiary"
          component={AgriSubsidyScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Ionicons name="cube" color={color} size={size} />,
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}


const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#D9F99D' },
  scrollContent: { padding: 16, paddingBottom: 100 },
  header: { flexDirection: 'row', justifyContent: 'center', paddingVertical: 16 },
  headerText: { fontSize: 28, fontWeight: 'bold', color: '#065F46' },
  welcomeCard: { backgroundColor: '#D9F99D', padding: 16, borderRadius: 12, marginVertical: 8 },
  welcomeTitle: { fontSize: 22, fontWeight: 'bold', color: '#065F46' },
  welcomeText: { marginTop: 8, fontSize: 16, color: '#065F46' },
  card: { backgroundColor: '#F0FFF0', padding: 16, borderRadius: 12, marginVertical: 8, borderWidth: 1, borderColor: '#D1D5DB' },
  cardTitle: { fontSize: 18, fontWeight: 'bold', color: '#065F46' },
  cardText: { marginTop: 4, fontSize: 14, color: '#065F46' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});