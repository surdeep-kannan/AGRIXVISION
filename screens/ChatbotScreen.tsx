import React, { useState, useEffect, useRef } from 'react';
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
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { FontAwesome6 } from '@expo/vector-icons';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: number;
}

const BACKEND_URL = 'http://10.101.59.69:8000/ask-chatbot';

const ChatbotScreen = () => {
  const insets = useSafeAreaInsets();
  const [messages, setMessages] = useState<Message[]>([
    {
      text: "Hello! I'm AgriX Intelligence. How can I help with your farming operations today?",
      sender: 'bot',
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollViewRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    scrollViewRef.current?.scrollToEnd({ animated: true });
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMsgText = input.trim();
    const userMessage: Message = {
      text: userMsgText,
      sender: 'user',
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user_local_01',
          question: userMsgText,
        }),
      });

      const data = await response.json();
      const botResponseText = data.answer || data.response || "I'm sorry, I encountered an error processing that.";

      const botResponse: Message = {
        text: botResponseText,
        sender: 'bot',
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      const errorResponse: Message = {
        text: "I'm having trouble reaching the server. Please check your connection and try again.",
        sender: 'bot',
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <View style={styles.outerContainer}>
      <SafeAreaView style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerContent}>
            <View style={styles.botIconCircle}>
              <FontAwesome6 name="leaf" size={20} color="#10b981" />
            </View>
            <View>
              <Text style={styles.headerTitle}>AgriX Intelligence</Text>
              <Text style={styles.headerSubtitle}>AI Farm Assistant</Text>
            </View>
          </View>
        </View>

        <KeyboardAvoidingView
          style={styles.keyboardView}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 20}
        >
          <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
            <View style={{ flex: 1 }}>
              <ScrollView
                ref={scrollViewRef}
                style={styles.messagesContainer}
                contentContainerStyle={styles.messagesList}
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
                    <Text style={[
                      styles.messageText,
                      msg.sender === 'user' ? styles.userText : styles.botText
                    ]}>
                      {msg.text}
                    </Text>
                  </View>
                ))}

                {isTyping && (
                  <View style={[styles.messageBubble, styles.botMessage, styles.typingBubble]}>
                    <ActivityIndicator size="small" color="#10b981" />
                    <Text style={styles.typingText}>Thinking...</Text>
                  </View>
                )}
              </ScrollView>

              {/* Input Area */}
              <View style={[styles.inputWrapper, { marginBottom: insets.bottom + 10 }]}>
                <View style={styles.inputContainer}>
                  <TextInput
                    value={input}
                    onChangeText={setInput}
                    placeholder="Ask AgriX anything..."
                    placeholderTextColor="#94a3b8"
                    style={styles.input}
                    multiline
                  />
                  <TouchableOpacity
                    onPress={handleSend}
                    style={[styles.sendButton, !input.trim() && styles.sendButtonDisabled]}
                    disabled={!input.trim() || isTyping}
                  >
                    <FontAwesome6 name="paper-plane" size={18} color="#FFFFFF" />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  outerContainer: { flex: 1, backgroundColor: '#f8fafc' },
  container: { flex: 1 },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
    backgroundColor: '#ffffff',
  },
  headerContent: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  botIconCircle: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#ecfdf5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: { fontSize: 18, fontWeight: '800', color: '#1e293b' },
  headerSubtitle: { fontSize: 12, color: '#10b981', fontWeight: '600' },
  keyboardView: { flex: 1 },
  messagesContainer: { flex: 1 },
  messagesList: { flexGrow: 1, paddingHorizontal: 20, paddingVertical: 20, gap: 12 },
  messageBubble: {
    padding: 14,
    borderRadius: 18,
    maxWidth: '85%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#10b981',
    borderBottomRightRadius: 4,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#ffffff',
    borderBottomLeftRadius: 4,
  },
  messageText: { fontSize: 15, lineHeight: 22 },
  userText: { color: '#ffffff', fontWeight: '500' },
  botText: { color: '#334155' },
  typingBubble: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 10 },
  typingText: { fontSize: 13, color: '#64748b', fontStyle: 'italic' },
  inputWrapper: { paddingHorizontal: 20, paddingTop: 10 },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderRadius: 25,
    paddingHorizontal: 6,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 3,
  },
  input: {
    flex: 1,
    paddingHorizontal: 15,
    paddingVertical: 8,
    fontSize: 15,
    color: '#1e293b',
    maxHeight: 100,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#10b981',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 4,
  },
  sendButtonDisabled: { backgroundColor: '#cbd5e1' },
});

export default ChatbotScreen;
