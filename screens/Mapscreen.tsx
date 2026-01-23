import React, { useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { WebView, WebViewMessageEvent } from 'react-native-webview';
import * as Location from 'expo-location';
import * as Network from 'expo-network';

const HTML_TEMPLATE = `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriXVision | Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root {
  --primary: #10b981;
  --primary-dark: #059669;
  --slate-50: #f8fafc;
  --slate-100: #f1f5f9;
  --slate-800: #1e293b;
  --glass: rgba(255, 255, 255, 0.85);
  --glass-border: rgba(255, 255, 255, 0.4);
}

body, html {
  margin: 0; padding: 0; height: 100%; width: 100%;
  font-family: 'Plus Jakarta Sans', sans-serif;
  overflow: hidden;
}

#map { height: 100%; width: 100%; z-index: 1; }

/* Control Overlays */
.overlay {
  position: absolute;
  z-index: 1000;
  pointer-events: auto;
}

/* Header Search Bar */
.top-bar {
  top: 15px; left: 15px; right: 15px;
  display: flex; gap: 8px; align-items: center;
  background: var(--glass);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  padding: 8px 12px;
  border-radius: 14px;
  border: 1px solid var(--glass-border);
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  z-index: 5000;
}

.brand {
  display: flex; align-items: center; gap: 6px;
  color: var(--primary); font-weight: 800; font-size: 16px;
  margin-right: 8px;
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-right: 10px;
}

.logo-img {
  height: 32px;
  width: auto;
}

.input-group { display: flex; flex: 1; gap: 4px; min-width: 0; }
input {
  flex: 1;
  background: var(--slate-100);
  border: 1px solid transparent;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  color: var(--slate-800);
  outline: none;
  min-width: 0;
  transition: border-color 0.2s;
}
input:focus { border-color: var(--primary); }
input::placeholder { color: #94a3b8; }

.btn-main {
  background: var(--primary);
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 700;
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
  transition: transform 0.1s, background 0.2s;
  display: flex; align-items: center; gap: 6px;
}
.btn-main:active { transform: scale(0.96); background: var(--primary-dark); }

.btn-icon {
  background: var(--slate-100);
  color: #475569;
  border: none;
  width: 36px; height: 36px;
  flex-shrink: 0;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  font-size: 16px;
}

input::placeholder { color: #94a3b8; }

.side-controls {
  top: 90px; left: 15px;
  display: flex; flex-direction: column; gap: 8px;
  z-index: 9999 !important;
}

.layer-btn {
  background: white;
  border: 1px solid #e2e8f0;
  padding: 10px 8px;
  border-radius: 12px;
  color: #64748b;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  width: 60px;
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  pointer-events: auto;
}
.layer-btn.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

/* Bottom Stats Card */
.stats-card {
  bottom: 25px; left: 15px; right: 15px;
  background: var(--glass);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: 16px;
  border-radius: 20px;
  border: 1px solid var(--glass-border);
  box-shadow: 0 8px 32px rgba(0,0,0,0.12);
  transform: translateY(130%);
  transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  pointer-events: none;
}
.stats-card.visible { transform: translateY(0); pointer-events: auto; }

.stats-grid { 
  display: grid; 
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); 
  gap: 12px; 
}

@media (max-width: 380px) {
  .stats-grid { grid-template-columns: 1fr; gap: 8px; }
  .stat-item { display: flex; justify-content: space-between; align-items: center; text-align: left; }
  .stat-label { margin-bottom: 0; }
  .top-bar { padding: 6px; gap: 4px; }
  input { font-size: 12px; }
  .btn-main { padding: 8px 10px; font-size: 12px; }
}

.stat-item { text-align: center; }
.stat-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.stat-value { font-size: 15px; font-weight: 700; color: var(--slate-800); }

.status-indicator {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.05);
  font-size: 12px; font-weight: 600; color: var(--primary);
}

#status-text { margin: 0; font-size: 12px; color: #64748b; text-align: center; margin-top: 10px; }

/* Pulse animation for loading */
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
.loading { animation: pulse 1.5s infinite; color: var(--primary); font-weight: 700; }
</style>
</head>
<body>

<div id="map"></div>

<!-- Floating UI Elements -->
<div class="overlay top-bar">
  <div class="brand"><i class="fas fa-leaf"></i> <span>AGRIX</span></div>
  <div class="input-group">
    <input id="lat" placeholder="Lat" type="number" step="any">
    <input id="lon" placeholder="Lon" type="number" step="any">
  </div>
  <button class="btn-icon" onclick="gps()"><i class="fas fa-location-crosshairs"></i></button>
  <button class="btn-main" onclick="analyze()" id="analyze-btn"><i class="fas fa-wand-magic-sparkles"></i> <span>Analyze</span></button>
</div>

<div class="overlay side-controls">
  <button class="layer-btn active" id="btn-ndvi" onclick="switchLayer('ndvi_map_url', this)"><i class="fas fa-heart-pulse"></i>Health</button>
  <button class="layer-btn" id="btn-ndwi" onclick="switchLayer('ndwi_map_url', this)"><i class="fas fa-droplet"></i>Water</button>
  <button class="layer-btn" id="btn-moist" onclick="switchLayer('soil_moisture_map_url', this)"><i class="fas fa-seedling"></i>Soil</button>
  <button class="layer-btn" id="btn-temp" onclick="switchLayer('lst_map_url', this)"><i class="fas fa-temperature-high"></i>Temp</button>
</div>

<div class="overlay stats-card" id="stats-card">
  <div class="stats-grid">
    <div class="stat-item">
      <div class="stat-label">Health Status</div>
      <div class="stat-value" id="val-health">---</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Surface Temp</div>
      <div class="stat-value" id="val-temp">---</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Soil Carbon</div>
      <div class="stat-value" id="val-carbon">---</div>
    </div>
  </div>
  <div class="status-indicator" id="status-indicator">
    <span id="status-icon"><i class="fas fa-circle-notch fa-spin" style="display:none;" id="spinner"></i><i class="fas fa-bolt" id="static-icon"></i></span>
    <span id="status-msg">Ready for Analysis</span>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var BACKEND_URL = "http://10.101.59.69:8000";
console.log("Connecting to:", BACKEND_URL);
var map = L.map('map', { zoomControl: false, tap: false }).setView([20.5937, 78.9629], 5);

// Modern Tile Layer
L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  maxZoom: 19
}).addTo(map);

var currentLayer = null;
var currentTileLayer = null;
var lastData = null;

function analyze() {
  let lat = parseFloat(document.getElementById("lat").value);
  let lon = parseFloat(document.getElementById("lon").value);
  if (isNaN(lat) || isNaN(lon)) {
    updateStatus("Enter coordinates", "fa-circle-exclamation");
    return;
  }
  load(lat, lon);
}

function updateStatus(msg, iconClass, isLoading) {
  const msgEl = document.getElementById("status-msg");
  const spinner = document.getElementById("spinner");
  const staticIcon = document.getElementById("static-icon");
  
  msgEl.innerText = msg;
  msgEl.className = isLoading ? "loading" : "";
  
  if (isLoading) {
    spinner.style.display = "inline-block";
    staticIcon.style.display = "none";
  } else {
    spinner.style.display = "none";
    staticIcon.style.display = "inline-block";
    staticIcon.className = "fas " + iconClass;
  }
}

function switchLayer(type, btn) {
  if (!lastData) return;
  
  if (!btn) btn = document.getElementById("btn-ndvi");
  if (!type) type = "ndvi_map_url";

  document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (currentTileLayer) map.removeLayer(currentTileLayer);
  
  if (lastData[type]) {
    currentTileLayer = L.tileLayer(lastData[type], {
      attribution: 'Google Earth Engine',
      opacity: 0.8
    }).addTo(map);
    updateStatus("Viewing " + btn.innerText, "fa-satellite-dish");
  } else {
    updateStatus("Data missing", "fa-triangle-exclamation");
  }
}

function gps() {
  window.ReactNativeWebView.postMessage(JSON.stringify({ action: "get_location" }));
}

async function load(lat, lon) {
  updateStatus("Scanning field...", "", true);
  document.getElementById("stats-card").classList.remove("visible");
  
  try {
    let url = BACKEND_URL + "/get_field_health?lat=" + lat + "&lon=" + lon;
    console.log("Production Scan:", url);
    
    let res = await fetch(url, {
      method: "GET",
      mode: "cors",
      cache: "no-store"
    });
    
    if (!res.ok) throw "Server reachable but returned " + res.status;
    let data = await res.json();

    if (data.error) throw data.error;
    lastData = data;

    // Update Card Data
    document.getElementById("val-health").innerText = data.health_status || "Analyzed";
    document.getElementById("val-temp").innerText = (data.avg_temp_celsius || "--") + "°C";
    document.getElementById("val-carbon").innerText = data.soil_organic_carbon || "Normal";

    if (currentLayer) map.removeLayer(currentLayer);
    
    // Auto-select NDVI
    switchLayer('ndvi_map_url', document.getElementById("btn-ndvi"));

    if (data.field_boundary && data.field_boundary.length > 0) {
      currentLayer = L.polygon(data.field_boundary, {
        color: "#10b981", 
        weight: 3, 
        fillColor: "#10b981",
        fillOpacity: 0.1
      }).addTo(map);
      map.flyToBounds(currentLayer.getBounds(), { padding: [50, 50], duration: 1 });
    } else {
      console.warn("No boundary found, zooming to point.");
      map.flyTo([lat, lon], 17, { duration: 1 });
    }
    
    setTimeout(() => {
      document.getElementById("stats-card").classList.add("visible");
      updateStatus("Analysis Complete", "fa-circle-check");
    }, 500);

  } catch (e) {
    console.error("Critical Load Error:", e);
    updateStatus("Sync Failed", "fa-circle-xmark");
    alert("Connection Error. Please check if your internet is on and try again.");
  }
}

window.onerror = function(msg, url, line) {
  window.ReactNativeWebView.postMessage(JSON.stringify({ type: "ERROR", msg: msg, line: line }));
  return false;
};

window.addEventListener("message", e => {
  let d = JSON.parse(e.data);
  if (d.type === "SET_OFFLINE") {
    updateStatus(d.value ? "Working Offline" : "Connected", d.value ? "fa-moon" : "fa-sun");
  }
});
</script>
</body>
</html>
`;

export default function MapScreen() {
  const webViewRef = useRef<WebView>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const t = setInterval(async () => {
      const s = await Network.getNetworkStateAsync();
      if (loaded) {
        webViewRef.current?.postMessage(JSON.stringify({ type: "SET_OFFLINE", value: !s.isConnected }));
      }
    }, 4000);
    return () => clearInterval(t);
  }, [loaded]);

  const handleMessage = async (e: WebViewMessageEvent) => {
    const data = JSON.parse(e.nativeEvent.data);
    if (data.action === "get_location") {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") { Alert.alert("Permission needed"); return; }
      const loc = await Location.getCurrentPositionAsync({});
      webViewRef.current?.injectJavaScript(
        "document.getElementById('lat').value='" + loc.coords.latitude.toFixed(6) + "';" +
        "document.getElementById('lon').value='" + loc.coords.longitude.toFixed(6) + "';" +
        "analyze();"
      );
    } else if (data.type === "ERROR") {
      console.warn("WebView Error:", data.msg, "at line", data.line);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <WebView
        ref={webViewRef}
        source={{ html: HTML_TEMPLATE }}
        onMessage={handleMessage}
        onLoadEnd={() => setLoaded(true)}
        javaScriptEnabled
        domStorageEnabled
        mixedContentMode="compatibility"
      />
    </View>
  );
}
