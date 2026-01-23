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
          <Text style={styles.cardTitle}>AgriX Intelligence</Text>
          <Text style={styles.cardText}>
            Get instant AI advice on diseases, fertilizers, and farming techniques.
          </Text>
        </TouchableOpacity>


        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('Map')}
        >
          <Text style={styles.cardTitle}>Satellite Analytics</Text>
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
          tabBarActiveTintColor: '#059669',
          tabBarInactiveTintColor: '#94a3b8',
          tabBarLabelStyle: { fontSize: 11, fontWeight: '700' },
          tabBarStyle: { backgroundColor: '#ffffff', borderTopColor: '#f1f5f9', height: 65, paddingBottom: 10 },
        }}
      >
        <Tab.Screen
          name="Home"
          component={HomeScreen}
          options={{
            tabBarLabel: 'Dashboard',
            tabBarIcon: ({ color, size }: { color: string; size: number }) => <Ionicons name="grid" color={color} size={22} />,
          }}
        />
        <Tab.Screen
          name="Map"
          component={MapScreen}
          options={{
            tabBarLabel: 'Map',
            tabBarIcon: ({ color, size }: { color: string; size: number }) => <Ionicons name="map" color={color} size={22} />,
          }}
        />
        <Tab.Screen
          name="Chatbot"
          component={ChatbotScreen}
          options={{
            tabBarLabel: 'AgriX AI',
            tabBarIcon: ({ color, size }: { color: string; size: number }) => <Ionicons name="chatbubbles" color={color} size={22} />,
          }}
        />
        <Tab.Screen
          name="Subsidiary"
          component={AgriSubsidyScreen}
          options={{
            tabBarLabel: 'Subsidies',
            tabBarIcon: ({ color, size }: { color: string; size: number }) => <Ionicons name="leaf" color={color} size={22} />,
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}


const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },
  scrollContent: { padding: 20, paddingBottom: 100 },
  header: { paddingVertical: 20, alignItems: 'center' },
  headerText: { fontSize: 32, fontWeight: '900', color: '#10b981', letterSpacing: -1 },
  welcomeCard: { backgroundColor: '#ffffff', padding: 20, borderRadius: 20, marginBottom: 15, borderLeftWidth: 5, borderLeftColor: '#10b981', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 2 },
  welcomeTitle: { fontSize: 24, fontWeight: '800', color: '#1e293b' },
  welcomeText: { marginTop: 8, fontSize: 15, color: '#64748b', lineHeight: 22 },
  card: { backgroundColor: '#ffffff', padding: 20, borderRadius: 20, marginVertical: 8, borderWidth: 1, borderColor: '#f1f5f9', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.03, shadowRadius: 5, elevation: 1 },
  cardTitle: { fontSize: 18, fontWeight: '800', color: '#1e293b' },
  cardText: { marginTop: 4, fontSize: 14, color: '#64748b', lineHeight: 20 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
