import React from 'react';
import { Text, View, ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const AgriSubsidyScreen = () => {
  const centralSubsidies = [
    {
      title: 'PM-KISAN Scheme',
      description: 'Provides direct income support of ₹6,000 per year to eligible farmer families. In 2025, the 19th instalment was released to over 9.8 crore farmers.',
    },
    {
      title: 'Rashtriya Krishi Vikas Yojana (RKVY)',
      description: 'Offers financial assistance and grants to boost agricultural economy. Includes seed subsidies covering 50-75% of costs depending on crop and region.',
    },
    {
      title: 'Agriculture Infrastructure Fund',
      description: 'Provides subsidies for agricultural infrastructure development, with ₹475.23 crore allocated in recent budgets.',
    },
    {
      title: 'Solar Agricultural Pumps Subsidy',
      description: 'Central government offers up to 30% or 50% subsidy on installation costs for solar pumps.',
    },
    {
      title: 'APEDA Export Promotion Scheme',
      description: 'Supports agricultural exports with subsidies and incentives.',
    },
    {
      title: 'Urea Subsidy Scheme',
      description: 'Subsidizes urea fertilizers for farmers.',
    },
    {
      title: 'Soil Health Card Scheme',
      description: 'Provides subsidies for soil testing and health cards.',
    },
  ];

  const stateSubsidies = [
    {
      title: 'State-Specific Seed Subsidies',
      description: 'Many states offer additional seed subsidies (50-75% coverage) under schemes like RKVY, varying by region.',
    },
    {
      title: 'State Agricultural Input Subsidies',
      description: 'States reimburse suppliers for inputs like fertilizers at subsidized rates, complementing central schemes.',
    },
    {
      title: 'MSME and Agri Subsidies',
      description: 'Over ₹2.3 lakh crore allocated for agriculture and MSMEs, with state-level implementations.',
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>Government Subsidies</Text>
      </View>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.sectionTitle}>Central Government Subsidies</Text>
        {centralSubsidies.map((subsidy, index) => (
          <View key={index} style={styles.card}>
            <Text style={styles.cardTitle}>{subsidy.title}</Text>
            <Text style={styles.cardText}>{subsidy.description}</Text>
          </View>
        ))}

        <Text style={styles.sectionTitle}>State Government Subsidies</Text>
        {stateSubsidies.map((subsidy, index) => (
          <View key={index} style={styles.card}>
            <Text style={styles.cardTitle}>{subsidy.title}</Text>
            <Text style={styles.cardText}>{subsidy.description}</Text>
          </View>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#D9F99D' },
  header: { paddingVertical: 16, paddingHorizontal: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  headerText: { fontSize: 24, fontWeight: 'bold', color: '#065F46' },
  scrollContent: { padding: 16, paddingBottom: 100 },
  sectionTitle: { fontSize: 20, fontWeight: 'bold', color: '#065F46', marginVertical: 16 },
  card: { backgroundColor: '#F0FFF0', padding: 16, borderRadius: 12, marginVertical: 8, borderWidth: 1, borderColor: '#D1D5DB' },
  cardTitle: { fontSize: 18, fontWeight: 'bold', color: '#065F46' },
  cardText: { marginTop: 4, fontSize: 14, color: '#065F46' },
});

export default AgriSubsidyScreen;