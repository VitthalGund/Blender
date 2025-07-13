# BlenderGenix - The Future of 3D Creation 🎨🚀🌟

**BlenderGenix** is a groundbreaking AI-powered platform that redefines 3D design. Seamlessly integrated with **Blender 3.6.0**, it empowers users—professionals, students, and hobbyists alike—to create stunning 3D scenes using **natural language prompts**, **images**, **audio**, and an intuitive **canvas-based drawing interface**.

Powered by a **FastAPI backend**, Dockerized for portability, and enhanced with a custom **Blender add-on**, BlenderGenix leverages leading AI models like **Whisper**, **BLIP**, **DETR**, **spaCy**, and **LLaMA 3.1:8B** via **Ollama**.

---

## 🌟 Why BlenderGenix?

- **Unmatched Versatility**: Accepts inputs from text, images, audio, or canvas sketches.
- **AI-Powered Innovation**: Simplifies complex Blender tasks with intelligent automation.
- **Scalable Excellence**: Designed for future expansion into VR, mobile, and cloud.
- **Community Love**: A planned showcase site will celebrate your creativity.
- **Proven Reliability**: Dockerized architecture ensures plug-and-play deployment.

---

## 📚 Table of Contents

- [Project Vision, Mission, Impact, and Motivation](#project-vision-mission-impact-and-motivation)
- [Technology Used](#technology-used)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Windows (WSL2 + Docker Desktop)](#windows-setup-docker-desktop-with-wsl2)
  - [Linux](#linux-setup)
- [Running the Project](#running-the-project)
- [Using the Blender Add-on](#using-the-blender-add-on)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)
- [Technical Details](#technical-details)

---

## 📌 Project Vision, Mission, Impact, and Motivation

### Vision

BlenderGenix envisions a world where 3D creation is as simple as **sketching or speaking**. We aim to transform Blender into the ultimate AI-powered creative hub for everyone—from professionals to first-time users.

### Mission

To fuse cutting-edge AI with Blender’s powerful ecosystem, creating a platform that supports:
- Context-aware project editing
- AI-generated 3D elements via speech, text, or drawing
- A community-driven showcase website
- Full scalability from desktop to cloud and VR

### Impact

- 🎉 **Democratizing Creativity**: Accessible 3D modeling with themes and canvas support.
- 💡 **Redefining Industry Standards**: In gaming, film, and education.
- 📚 **Educational Powerhouse**: Ideal for teaching 3D modeling and AI concepts.
- 🌐 **Global Community**: Share your work and get inspired by others.
- 📈 **Future-Proof**: Built for mobile, VR, and beyond.

### Motivation

BlenderGenix emerged from the desire to break down the barriers of 3D modeling. Artists, educators, and students now have a smarter, simpler, and more creative way to bring ideas to life.

---

## ⚙️ Technology Used

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10)
- **Containerization**: Docker + Docker Compose
- **AI Models**:
  - [Ollama](https://ollama.com) + LLaMA 3.1:8B for NLP
  - [spaCy](https://spacy.io) for text parsing
  - [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) for image captioning
  - [DETR](https://github.com/facebookresearch/detectron2) for object detection
  - [Whisper](https://github.com/openai/whisper) for speech-to-text
- **3D Interface**: [Blender 3.6.0](https://www.blender.org/)
- **Other Tools**:
  - OpenCV (canvas drawing analysis)
  - SQLite (scene context persistence)
  - Git (version control)

---

## ✨ Features

- 🎤 **Natural Language Processing**: Describe your scene and see it built.
- 🖼️ **Image-Based Scene Generation**: Analyze and render objects from images.
- 🗣️ **Voice Commands**: Whisper integration for hands-free design.
- 🎨 **Canvas Drawing**: Draw shapes to trigger Blender actions.
- 🧠 **Context Awareness**: Edit and enhance existing .blend projects.
- 🧱 **Custom Themes**: Style scenes using predefined `themes.json` presets.
- 📦 **Dockerized Backend**: Run seamlessly across environments.
- 🔌 **Blender Add-on**: Unified interface for inputs.
- 🌐 **Showcase Website (Planned)**: Share your work with the world.

---

## 📋 Prerequisites

### OS

- **Windows 10/11** with **WSL2** and **Docker Desktop**
- **Linux (Ubuntu recommended)**

### Hardware

- 16+ GB RAM
- 512+ GB SSD
- Multi-core CPU

### Software

- Docker & Docker Compose
- Blender 3.6.0
- [Ollama](https://ollama.com) with `llama3.1:8b`
- Python 3.10 (optional)

---

## 🗂️ Project Structure

```

BlenderGenix/
├── backend.py               # FastAPI backend
├── blender\_addon.py         # Blender integration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── themes.json
├── .env
├── scene\_context.db
├── templates/               # Template storage
├── canvas\_template.png      # Example canvas image
├── logo.png
├── test.png
├── TECHNICAL\_DETAILS.md     # In-depth architecture docs

````

---

## 🖥️ Setup Instructions

### 🪟 Windows Setup (Docker Desktop with WSL2)

1. **Install**:
   - Docker Desktop with WSL2 (Ubuntu)
   - Blender 3.6.0
   - Ollama + pull `llama3.1:8b`

2. **Setup**:
```powershell
cd C:\Users\<YourUsername>\Videos
mkdir BlenderGenix
cd BlenderGenix
New-Item scene_context.db
icacls ".\scene_context.db" /grant Everyone:F
mkdir templates
````

3. **Configure `.env`**:

```env
OLLAMA_URL=http://ollama:11434
BACKEND_URL=http://localhost:8000
```

---

### 🐧 Linux Setup

1. **Install**:

```bash
sudo apt install docker.io docker-compose
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
```

2. **Install Blender 3.6.0**

3. **Setup**:

```bash
mkdir -p ~/BlenderGenix/templates
cd ~/BlenderGenix
touch scene_context.db
chmod 666 scene_context.db
chmod -R 777 templates
```

4. **Configure `.env`**:

```env
OLLAMA_URL=http://ollama:11434
BACKEND_URL=http://localhost:8000
```

---

## 🚀 Running the Project

1. **Build**:

```bash
docker-compose build backend
```

2. **Start Services**:

```bash
docker-compose up -d backend ollama
```

3. **Check Logs**:

```bash
docker logs blender-backend-1
```

---

## 🧩 Using the Blender Add-on

1. **Install Add-on**:

   * Blender → Preferences → Add-ons → Install `blender_addon.py`

2. **Use**:

   * Open a `.blend` file
   * Select object or open canvas
   * Input text/audio/image or draw on canvas
   * Click execute to run

---

## 🔗 API Endpoints

### `POST /process`

#### Request:

```json
{
  "text": "Create a red sphere",
  "theme": "futuristic",
  "image": "<base64>",
  "audio": "<base64>",
  "canvas": "<base64>"
}
```

#### Response:

```json
{
  "script": "<blender_script>",
  "preview": "<base64_image>"
}
```

#### Example:

```bash
curl -X POST http://localhost:8000/process \
-H "Content-Type: application/json" \
-d '{"text": "Create a red sphere", "theme": "futuristic"}'
```

---

## 🛠️ Troubleshooting

* Use `docker logs blender-backend-1` to check backend issues
* Test Ollama: `curl http://localhost:11434`
* Pre-download models for unstable internet environments

---

## ⚠️ Known Issues

* High memory usage with large LLMs
* Canvas drawing accuracy may require fine-tuning

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Follow \[PEP 8]\([https://pe](https://pe)


ps.python.org/pep-0008/)
4\. Submit a pull request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📘 Technical Details

Want to understand the architecture and AI pipelines? Check out [TECHNICAL\_DETAILS.md](TECHNICAL_DETAILS.md).

---

BlenderGenix — Built for Creativity. Powered by AI.
