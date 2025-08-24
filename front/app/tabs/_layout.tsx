import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";
import { useColorScheme } from "react-native";
import "../../global.css";
import { ThemeColors } from "../../utils/colors";

export default function TabNavigator() {
  const colorScheme = useColorScheme() || "light";
  const colors = ThemeColors[colorScheme as "light" | "dark"];

  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: colors.headerBackground },
        headerTintColor: colors.headerTint,
        tabBarStyle: {
          backgroundColor: colors.headerBackground,
          borderTopColor: "#374151",
        },
        tabBarActiveTintColor: colors.tabBarActive,
        tabBarInactiveTintColor: colors.tabBarInactive,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "home" : "home-outline"}
              size={24}
              color={color}
            />
          ),
        }}
      />

      {/* ✅ Chatbot added back */}
      <Tabs.Screen
        name="chatbot"
        options={{
          title: "Chatbot",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "chatbubbles" : "chatbubbles-outline"}
              size={24}
              color={color}
            />
          ),
        }}
      />

      <Tabs.Screen
        name="disease-predictor"
        options={{
          title: "Predictor",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "leaf" : "leaf-outline"}
              size={24}
              color={color}
            />
          ),
        }}
      />

      <Tabs.Screen
        name="settings"
        options={{
          title: "Settings",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "settings" : "settings-outline"}
              size={24}
              color={color}
            />
          ),
        }}
      />
    </Tabs>
  );
}
