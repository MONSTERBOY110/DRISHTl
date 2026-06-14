# Quick Start Guide - DRISTI CCTV Integration

## 🚀 Get Started in 5 Minutes

### Step 1: Start the Server
```bash
# Navigate to project directory
cd DRISTI

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the server
python main.py
```

You'll see:
```
======================================================================
DRISTI - Lost Person Detection System v2.0
======================================================================
Starting server at http://localhost:8000
======================================================================
```

### Step 2: Open in Browser
- Go to: `http://localhost:8000`
- You'll see the DRISTI interface with:
  - CCTV Footage Preview (top)
  - Upload Lost Person Photo (middle)
  - "Connect with CCTV" button (top-right)

### Step 3: Add CCTV Camera (Optional)
1. Click **"Connect with CCTV"** button
2. Fill in camera details:
   ```
   Camera Name: Main Entrance
   Location: Gate A
   RTSP URL: rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101
   Duration: 15 seconds
   ```
3. Click **"Add Camera"**
4. Click **"Save Configuration"**

### Step 4: Search for Lost Person
1. **Upload Photo**: Click upload area, select person's photo
2. **Click Search**: Button becomes enabled
3. **Wait for Results**: System processes video (10-30 seconds)
4. **View Matches**: See detected matches with confidence scores

---

## 📝 Two Ways to Use

### Using Pre-recorded Videos (No CCTV needed)
✅ Perfect for demo/testing
- Place MP4 files in `CCTVS/` folder
- Upload photo
- Click "Search CCTV Footage"
- Get results

### Using Live CCTV Cameras
✅ Real-time person tracking
- Click "Connect with CCTV"
- Add camera RTSP URLs
- Upload photo
- System captures 15 seconds from each camera
- Get results from live footage

---

## 🎥 Finding Your Camera's RTSP URL

### Quick Method:
1. Know your camera IP (e.g., `192.168.1.100`)
2. Open browser: `http://camera-ip`
3. Login (default: `admin/12345`)
4. Find "RTSP URL" in settings
5. Copy the URL

### By Camera Brand:

**Hikvision**
```
rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101
```

**Dahua**
```
rtsp://admin:admin@192.168.1.100:554/stream/main
```

**Axis**
```
rtsp://admin:password@192.168.1.100/axis-media/media.amp
```

**Reolink**
```
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
```

> See `docs/CCTV_SETUP_GUIDE.md` for detailed instructions for your camera model

---

## 🔍 Understanding Results

Each match shows:
- **Confidence %**: How similar to uploaded photo (0-100%)
- **Camera**: Which camera detected the person
- **Time**: Timestamp in video (MM:SS format)
- **Frame**: Frame number in video
- **Snapshot**: Captured face image

**Example:**
```
Main Entrance | 92.5% Match | 00:41 | Frame 1250
```

---

## ⚠️ Troubleshooting

### CCTV Won't Connect
```
❌ Problem: "Cannot connect to RTSP stream"

✅ Solutions:
1. Check camera IP is correct
2. Verify RTSP URL format
3. Check username/password
4. Ensure camera is on same network
5. Try: ping camera-ip
```

### No Matches Found
```
❌ Problem: Search returns no results

✅ Solutions:
1. Ensure video has clear face images
2. Check lighting/video quality
3. Upload clearer photo of person
4. Try with longer capture duration
```

### Server Won't Start
```
❌ Problem: Error when running python main.py

✅ Solutions:
1. Check Python 3.8+: python --version
2. Install dependencies: pip install -r requirements.txt
3. Check port 8000 not in use: netstat -an | grep 8000
4. Try different port: uvicorn main:app --port 8001
```

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | Dual-core | Quad-core |
| Storage | 2GB | 10GB |
| Network | 100Mbps | 1Gbps |
| Python | 3.8+ | 3.10+ |

---

## 📁 File Locations

- **Videos**: `CCTVS/` folder (place MP4 files here)
- **Uploads**: `uploads/` folder (uploaded photos)
- **Results**: `results/` folder (search results & snapshots)
- **Logs**: `logs/` folder (system logs)
- **Config**: `cctv_config.json` (CCTV settings)

---

## 🎯 Tips for Best Results

1. **Clear Photos**: Upload clear, front-facing photos
2. **Good Lighting**: Ensure video has adequate lighting
3. **Natural Behavior**: Person's pose in photo should match video
4. **Multiple Cameras**: Add multiple cameras for better coverage
5. **Optimal Duration**: Use 15 seconds for balanced speed/accuracy

---

## 🔄 Workflow Example

```
1. Police receives missing person report
   ↓
2. Officer gets photo of lost person
   ↓
3. Officer opens DRISTI at http://localhost:8000
   ↓
4. Officer uploads person's photo
   ↓
5. System searches all CCTV cameras automatically
   ↓
6. Results show last known locations with timestamps
   ↓
7. Officers dispatch to those locations
```

---

## 📞 Support

### Check These First:
1. README.md - Detailed documentation
2. docs/CCTV_SETUP_GUIDE.md - Camera setup guide
3. Browser console (F12 → Console) - Error messages
4. Server logs - Check terminal output

### Common Issues:
- CCTV connection → See CCTV_SETUP_GUIDE.md
- No results → Check video quality
- Slow processing → Reduce video duration
- Port already in use → Change port in main.py

---

## 🚀 Next Steps

1. **Add Multiple Cameras** - Better coverage
2. **Integrate with Notification System** - Alert when person found
3. **Store Search History** - Track all searches
4. **Adjust Confidence Threshold** - Fine-tune sensitivity
5. **Performance Optimization** - Use GPU acceleration

---

## 📚 More Information

- **Full Setup Guide**: `docs/CCTV_SETUP_GUIDE.md`
- **RTSP URL Guide**: Camera brand specific instructions
- **API Documentation**: See `/api/` endpoints in main.py
- **Architecture**: `docs/ARCHITECTURE.md`

---

**Version**: 2.0.0  
**Last Updated**: December 2025

Enjoy using DRISTI! 🎉
