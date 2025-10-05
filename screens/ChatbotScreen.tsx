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
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: number;
}

const BACKEND_URL = 'http://10.72.55.187:8000'; 

const ChatbotScreen = () => {
  const insets = useSafeAreaInsets();
  const [messages, setMessages] = useState<Message[]>([
    {
      text: 'Hello! I am Surya Bot. I can help you with your farming needs. What would you like to know?',
      sender: 'bot',
      timestamp: Date.now(),
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
        console.error('Error fetching bot response:', error);
        const errorMessage: Message = {
          text: "I'm sorry, I am unable to connect to the bot right now. Please try again later.",
          sender: 'bot',
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, errorMessage]);

      } finally {
        setIsTyping(false);
      }
    }
  };
  
  const handleScreenTap = () => {
    Keyboard.dismiss();
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#D9F99D' }}>
      <View
        style={{
          paddingVertical: 16,
          paddingHorizontal: 16,
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#065F46' }}>Surya Bot</Text>
      </View>

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={10}
      >
        <TouchableWithoutFeedback onPress={handleScreenTap} accessible={false}>
          
          <ScrollView
            ref={scrollViewRef}
            style={{ flex: 1 }}
            contentContainerStyle={{
              flexGrow: 1,
              justifyContent: 'flex-end',
              paddingHorizontal: 16,
              paddingVertical: 8,
            }}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            {messages.map((msg, idx) => (
              <View
                key={idx}
                style={{
                  alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  backgroundColor: msg.sender === 'user' ? '#86EFAC' : '#FFFFFF',
                  padding: 12,
                  borderRadius: 16,
                  marginVertical: 4,
                  maxWidth: '80%',
                }}
              >
                <Text>{msg.text}</Text>
              </View>
            ))}

            {isTyping && (
              <View
                style={{
                  alignSelf: 'flex-start',
                  backgroundColor: '#FFFFFF',
                  padding: 12,
                  borderRadius: 16,
                  marginVertical: 4,
                  maxWidth: '80%',
                }}
              >
                <Text style={{ fontStyle: 'italic', color: '#6B7280' }}>
                  Surya Bot is typing...
                </Text>
              </View>
            )}
          </ScrollView>
        </TouchableWithoutFeedback>

        
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            paddingHorizontal: 8,
            paddingTop: 8,
            paddingBottom: insets.bottom > 0 ? insets.bottom : 8,
            backgroundColor: '#D9F99D',
          }}
        >
          <TextInput
            ref={inputRef}
            value={input}
            onChangeText={setInput}
            placeholder="Ask a question..."
            onSubmitEditing={handleSend}
            multiline
            style={{
              flex: 1,
              minHeight: 40,
              maxHeight: 100,
              paddingHorizontal: 12,
              paddingVertical: 8,
              borderWidth: 1,
              borderColor: '#D1D5DB',
              borderRadius: 24,
              backgroundColor: '#E6F4D7',
            }}
          />
          <TouchableOpacity
            onPress={handleSend}
            style={{
              marginLeft: 8,
              padding: 10,
              borderRadius: 24,
              backgroundColor: '#10B981',
            }}
          >
            <Ionicons name="send" size={24} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

export default ChatbotScreen;
