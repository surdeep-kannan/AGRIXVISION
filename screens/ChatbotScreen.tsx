import { useState, useEffect, useRef } from 'react';
import {
  Text,
  View,
  ScrollView,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  Keyboard,
  TouchableOpacity,
  TouchableWithoutFeedback,
  StyleSheet,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: number;
}

const BACKEND_URL = 'http://10.101.59.109:8000';

const OFFLINE_RESPONSES: Record<string, string> = {
  'hello': 'Hello! I am Surya Bot (Offline Mode). How can I help you today?',
  'hi': 'Hi! I am here to help you with farming advice even without internet.',
  'rice': 'For rice, ensure proper standing water in the early stages. Common pests include Brown Plant Hopper.',
  'wheat': 'Wheat requires 4-6 irrigations at critical stages like Crown Root Initiation.',
  'tomato': 'Tomato plants need support and regular watering. Watch out for Early Blight.',
  'fertilizer': 'Organic fertilizers like compost are great! For chemicals, always test your soil first.',
  'weather': 'I cannot get live weather without internet, but generally, check for local signs of rain.',
  'who are you': 'I am AgriXVision Assistant, your agricultural companion.',
  'default': "I'm currently offline, so I can only answer basic questions about common crops like Rice, Wheat, or Tomato. Once you're back online, I can give you much deeper AI-powered advice!"
};

const getOfflineResponse = (msg: string) => {
  const lowered = msg.toLowerCase();
  for (const key in OFFLINE_RESPONSES) {
    if (lowered.includes(key)) return OFFLINE_RESPONSES[key];
  }
  return OFFLINE_RESPONSES['default'];
};

const ChatbotScreen = () => {
  const insets = useSafeAreaInsets();
  const [messages, setMessages] = useState<Message[]>([
    {
      text: 'Hello! I am Surya Bot. Your agricultural assistant.',
      sender: 'bot',
      timestamp: Date.now(),
    },
    {
      text: 'I can help you with crop disease identification, fertilizer advice, and real-time field health analysis.',
      sender: 'bot',
      timestamp: Date.now() + 10,
    },
    {
      text: 'If you are offline, I can still provide basic advice on Rice, Wheat, and Tomato!',
      sender: 'bot',
      timestamp: Date.now() + 20,
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const inputRef = useRef<TextInput | null>(null);
  const scrollViewRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (input.trim()) {
      const userMessage: Message = {
        text: input.trim(),
        sender: 'user',
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput('');
      setIsTyping(true);

      try {
        const response = await fetch(`${BACKEND_URL}/ask-chatbot`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: 'dummy-user-id',
            question: userMessage.text,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const botResponseText = data.answer || 'Sorry, I could not get a response from the bot.';

        const botResponse: Message = {
          text: botResponseText,
          sender: 'bot',
          timestamp: Date.now(),
        };

        setMessages((prev) => [...prev, botResponse]);

      } catch (error) {
        console.error('Error fetching bot response, using fallback:', error);

        await new Promise(resolve => setTimeout(resolve, 1000));

        const botResponse: Message = {
          text: getOfflineResponse(userMessage.text),
          sender: 'bot',
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, botResponse]);

      } finally {
        setIsTyping(false);
      }
    }
  };

  const handleScreenTap = () => {
    Keyboard.dismiss();
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Surya Bot</Text>
      </View>

      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={10}
      >
        <TouchableWithoutFeedback onPress={handleScreenTap} accessible={false}>
          <ScrollView
            ref={scrollViewRef}
            style={styles.messagesContainer}
            contentContainerStyle={styles.messagesList}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            {messages.map((msg, idx) => (
              <View
                key={idx}
                style={[
                  styles.messageBubble,
                  msg.sender === 'user' ? styles.userMessage : styles.botMessage,
                ]}
              >
                <Text style={styles.messageText}>{msg.text}</Text>
              </View>
            ))}

            {isTyping && (
              <View style={[styles.messageBubble, styles.botMessage]}>
                <Text style={styles.typingText}>Surya Bot is typing...</Text>
              </View>
            )}
          </ScrollView>
        </TouchableWithoutFeedback>
        <View
          style={[
            styles.inputContainer,
            { paddingBottom: insets.bottom > 0 ? insets.bottom : 8 }
          ]}
        >
          <TextInput
            ref={inputRef}
            value={input}
            onChangeText={setInput}
            placeholder="Ask a question..."
            onSubmitEditing={handleSend}
            multiline
            style={styles.input}
          />
          <TouchableOpacity onPress={handleSend} style={styles.sendButton}>
            <Ionicons name="send" size={24} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#D9F99D' },
  header: { paddingVertical: 16, paddingHorizontal: 16, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: '#065F46' },
  keyboardView: { flex: 1 },
  messagesContainer: { flex: 1 },
  messagesList: { flexGrow: 1, justifyContent: 'flex-end', paddingHorizontal: 16, paddingVertical: 8 },
  messageBubble: { padding: 12, borderRadius: 16, marginVertical: 4, maxWidth: '80%' },
  userMessage: { alignSelf: 'flex-end', backgroundColor: '#86EFAC' },
  botMessage: { alignSelf: 'flex-start', backgroundColor: '#FFFFFF' },
  messageText: { fontSize: 16, color: '#1F2937' },
  typingText: { fontStyle: 'italic', color: '#6B7280' },
  inputContainer: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 8, paddingTop: 8, backgroundColor: '#D9F99D' },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 100,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 24,
    backgroundColor: '#E6F4D7',
    fontSize: 16,
  },
  sendButton: { marginLeft: 8, padding: 10, borderRadius: 24, backgroundColor: '#10B981' },
});

export default ChatbotScreen;