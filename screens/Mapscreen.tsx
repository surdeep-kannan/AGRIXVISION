import React, { useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Alert, Platform } from 'react-native';
import { WebView, WebViewMessageEvent } from 'react-native-webview';
import * as Location from 'expo-location';

interface WebviewMessage {
  action: 'get_location';
}

const HTML_TEMPLATE = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mobile Farm Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body, html { margin: 0; padding: 0; height: 100%; font-family: 'Inter', sans-serif; overflow: hidden; background-color: #f7f9fc; }
    .container { display: flex; flex-direction: column; height: 100vh; width: 100vw; }
    #map-container { position: relative; flex-grow: 1; height: 60%; }
    #map { height: 100%; width: 100%; }
    #info-panel { flex: none; height: 40%; padding: 1rem; background: #ffffff; overflow-y: auto; border-top-left-radius: 12px; border-top-right-radius: 12px; box-shadow: 0 -4px 12px rgba(0,0,0,0.08); }
    h2, h3 { border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-top: 0; color: #2d3748; }
    .location-input { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; justify-content: center; }
    .location-input input { flex: 1; padding: 0.5rem; border: 1px solid #cbd5e0; border-radius: 8px; min-width: 80px; }
    .location-input button { padding: 0.5rem 1rem; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; color: #fff; background-color: #4a5568; transition: background-color 0.2s ease-in-out; }
    .location-input button:hover { background-color: #2d3748; }
    #status { text-align: center; margin-top: 1rem; color: #718096; font-style: italic; }
    .map-buttons { position: absolute; top: 1rem; right: 1rem; z-index: 1000; display: flex; flex-direction: column; gap: 0.5rem; background: rgba(255, 255, 255, 0.9); padding: 0.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .map-buttons button { padding: 0.5rem 0.75rem; font-size: 0.875rem; cursor: pointer; border: 1px solid #e2e8f0; background: #fff; border-radius: 6px; transition: all 0.2s; }
    .map-buttons button:hover { background-color: #edf2f7; border-color: #a0aec0; }
    .legend { position: absolute; bottom: 1rem; left: 1rem; z-index: 1000; background: rgba(255, 255, 255, 0.85); padding: 0.75rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15); max-width: 150px; }
    .legend h4 { margin: 0 0 0.5rem; text-align: center; border-bottom: none; }
    .legend div { display: flex; align-items: center; margin-bottom: 0.25rem; }
    .legend i { width: 16px; height: 16px; float: left; margin-right: 0.5rem; opacity: 0.9; border: 1px solid #999; border-radius: 4px; }
    #stats p, #soil-stats p { margin: 0.5rem 0; font-size: 0.9rem; color: #4a5568; }
  </style>
</head>
<body>
  <div class="container">
    <div id="map-container">
      <div id="map"></div>
      <div class="map-buttons">
        <button id="show-ndvi">Health</button>
        <button id="show-ndwi">Water</button>
        <button id="show-soil">Soil</button>
        <button id="show-lst">Temp.</button>
      </div>
      <div id="legend" class="legend"></div>
    </div>
    <div id="info-panel">
      <h2>Field Analysis</h2>
      <div class="location-input">
        <input type="text" id="lat-input" placeholder="Latitude">
        <input type="text" id="lon-input" placeholder="Longitude">
        <button id="manual-submit">Analyze</button>
        <button id="gps-submit">GPS</button>
      </div>
      <p id="status">Enter coordinates or use your GPS.</p>
      <div id="stats"></div>
      <h3>Soil Properties</h3>
      <div id="soil-stats"></div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    (function() {
      const map = L.map('map').setView([20.5937, 78.9629], 5);
      const statusElement = document.getElementById('status');
      const statsElement = document.getElementById('stats');
      const soilStatsElement = document.getElementById('soil-stats');
      const legendElement = document.getElementById('legend');
      let ndviLayer, ndwiLayer, soilMoistureLayer, lstLayer;
      let fieldBoundaryLayer = null;

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap'
      }).addTo(map);

      const legends = {
        ndvi: \`<h4>Health (NDVI)</h4><div><i style="background:green"></i><span>Healthy</span></div><div><i style="background:yellow"></i><span>Stressed</span></div><div><i style="background:red"></i><span>Bare Soil</span></div>\`,
        ndwi: \`<h4>Water Stress (NDWI)</h4><div><i style="background:blue"></i><span>High Water</span></div><div><i style="background:white;"></i><span>Low Water</span></div><div><i style="background:yellow"></i><span>Stressed</span></div>\`,
        soil: \`<h4>Soil Moisture</h4><div><i style="background:blue"></i><span>Very Wet</span></div><div><i style="background:cyan"></i><span>Moist</span></div><div><i style="background:yellow"></i><span>Dry</span></div><div><i style="background:red"></i><span>Very Dry</span></div>\`,
        lst: \`<h4>Surface Temp. (°C)</h4><div><i style="background:red"></i><span>High (~40)</span></div><div><i style="background:yellow"></i><span>Moderate</span></div><div><i style="background:green"></i><span>Cool</span></div><div><i style="background:blue"></i><span>Low (~15)</span></div>\`
      };

      function updateLegend(type) {
        legendElement.innerHTML = legends[type] || '';
      }

      function clearLayers() {
        map.eachLayer(l => {
          if (l instanceof L.TileLayer && l.getAttribution() !== '&copy; OpenStreetMap') {
            map.removeLayer(l);
          }
        });
      }

      async function loadSatelliteData(lat, lon) {
        statusElement.innerText = 'Analyzing satellite data... Please wait.';
        statsElement.innerHTML = "";
        soilStatsElement.innerHTML = "";
        clearLayers();
        if (fieldBoundaryLayer) {
          map.removeLayer(fieldBoundaryLayer);
        }

        try {
          const response = await fetch(\`http://10.72.55.187:8000/get_field_health?lat=\${lat}&lon=\${lon}\`);
          const data = await response.json();
          
          if (data.error) {
            statusElement.innerText = \`Error: \${data.error}\`;
            return;
          }

          statusElement.innerText = "Data loaded successfully.";
          statsElement.innerHTML = \`
            <p><strong>Health Status:</strong> \${data.health_status}</p>
            <p><strong>Avg. Surface Temp:</strong> \${data.avg_temp_celsius || 'N/A'} °C</p>
          \`;
          soilStatsElement.innerHTML = \`<p><strong>Est. Soil Organic Carbon:</strong> \${data.soil_organic_carbon || 'N/A'}</p>\`;

          fieldBoundaryLayer = L.polygon(data.field_boundary, {color: '#3498db', fillOpacity: 0.2, weight: 2}).addTo(map);
          map.fitBounds(fieldBoundaryLayer.getBounds(), {padding: [50, 50]});

          ndviLayer = L.tileLayer(data.ndvi_map_url, {attribution: 'Mock Data'});
          ndwiLayer = L.tileLayer(data.ndwi_map_url, {attribution: 'Mock Data'});
          soilMoistureLayer = L.tileLayer(data.soil_moisture_map_url, {attribution: 'Mock Data'});
          lstLayer = L.tileLayer(data.lst_map_url, {attribution: 'Mock Data'});
          
          ndviLayer.addTo(map);
          updateLegend('ndvi');

          document.getElementById('show-ndvi').onclick = () => { clearLayers(); ndviLayer.addTo(map); updateLegend('ndvi'); };
          document.getElementById('show-ndwi').onclick = () => { clearLayers(); ndwiLayer.addTo(map); updateLegend('ndwi'); };
          document.getElementById('show-soil').onclick = () => { clearLayers(); soilMoistureLayer.addTo(map); updateLegend('soil'); };
          document.getElementById('show-lst').onclick = () => { clearLayers(); lstLayer.addTo(map); updateLegend('lst'); };

        } catch (error) {
          statusElement.innerText = "Could not connect to the server or process data.";
        }
      }

      document.getElementById('manual-submit').addEventListener('click', () => {
        const lat = parseFloat(document.getElementById('lat-input').value);
        const lon = parseFloat(document.getElementById('lon-input').value);
        if (isNaN(lat) || isNaN(lon)) {
          statusElement.innerText = "Please enter valid Latitude and Longitude.";
          return;
        }
        loadSatelliteData(lat, lon);
      });
    })();
  </script>
</body>
</html>
`;

export default function MapScreen() {
  const webViewRef = useRef<WebView>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (Platform.OS === 'android' && !Location.hasServicesEnabledAsync()) {
      Alert.alert(
        'Location Services Disabled',
        'Please enable location services to use GPS features.',
        [{ text: 'OK' }],
        { cancelable: false }
      );
    }
  }, []);

  const handleMessage = async (event: WebViewMessageEvent) => {
    try {
      const data: WebviewMessage = JSON.parse(event.nativeEvent.data);
      if (data.action === 'get_location' && webViewRef.current && isLoaded) {
        let { status } = await Location.requestForegroundPermissionsAsync();
        
        if (status !== 'granted') {
          webViewRef.current.injectJavaScript(`
            (function() {
              const statusElement = document.getElementById('status');
              if (statusElement) {
                statusElement.innerText = "Geolocation permission denied. Please enable it in your device settings.";
              }
            })();
          `);
          Alert.alert(
            'Permission Required',
            'This app needs location permission to use GPS. Please grant permission in settings.',
            [{ text: 'OK' }]
          );
          return;
        }

        let location;
        try {
          location = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.High,
            maximumAge: 10000,
          });
        } catch {
          webViewRef.current.injectJavaScript(`
            (function() {
              const statusElement = document.getElementById('status');
              if (statusElement) {
                statusElement.innerText = "Current location is unavailable. Ensure location services are enabled and try again.";
              }
            })();
          `);
          Alert.alert(
            'Location Unavailable',
            'Unable to get current location. Please ensure location services are enabled and you have a GPS signal.',
            [{ text: 'OK' }]
          );
          return;
        }

        webViewRef.current.injectJavaScript(`
          (function() {
            const latInput = document.getElementById('lat-input');
            const lonInput = document.getElementById('lon-input');
            if (latInput && lonInput) {
              latInput.value = '${location.coords.latitude.toFixed(6)}';
              lonInput.value = '${location.coords.longitude.toFixed(6)}';
              const manualSubmitBtn = document.getElementById('manual-submit');
              if (manualSubmitBtn) manualSubmitBtn.click();
            }
          })();
        `);
      }
    } catch {
      Alert.alert('Error', 'An unexpected error occurred. Please try again.');
    }
  };

  const injectedJavaScript = `
    document.getElementById('gps-submit').addEventListener('click', () => {
      window.ReactNativeWebView.postMessage(JSON.stringify({ action: 'get_location' }));
    });
    true;
  `;

  return (
    <View style={styles.container}>
      <WebView
        ref={webViewRef}
        originWhitelist={['*']}
        source={{ html: HTML_TEMPLATE }}
        style={styles.webView}
        onMessage={handleMessage}
        injectedJavaScript={injectedJavaScript}
        onLoadEnd={() => setIsLoaded(true)}
        onError={(syntheticEvent) => {
          const { nativeEvent } = syntheticEvent;
          Alert.alert('WebView Error', nativeEvent.description);
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  webView: {
    flex: 1,
  },
});
