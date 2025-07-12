import json
import whisper as openai_whisper
from transformers import pipeline
from ollama import Client
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import spacy
import webcolors
import logging
import asyncio
from functools import lru_cache
import sqlite3
from dotenv import load_dotenv
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    filename="/app/blender_control.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    type: str
    params: dict = {}


class ProcessRequest(BaseModel):
    text: str = None
    audio: str = None
    image: str = None
    canvas: str = None
    file: str = None
    theme: str = None
    selected_object: str = None  # New optional field


class EnhancePromptRequest(BaseModel):
    text: str


class BlenderControlBackend:
    def __init__(self):
        load_dotenv()
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        try:
            logger.debug("Loading whisper model")
            self.whisper_model = openai_whisper.load_model("tiny")
            logger.debug("Whisper loaded")

            logger.debug("Loading image captioner")
            self.image_captioner = pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-large"
            )
            logger.debug("Image captioner loaded")

            logger.debug("Loading object detector")
            self.object_detector = pipeline(
                "object-detection", model="facebook/detr-resnet-50"
            )
            logger.debug("Object detector loaded")

            logger.debug("Connecting to Ollama")
            self.ollama_client = Client(host=self.ollama_url)
            logger.debug("Ollama connected")

            self.nlp = spacy.load("en_core_web_sm")
            self.conn = sqlite3.connect("/app/scene_context.db")
            self.cursor = self.conn.cursor()
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, type TEXT, location TEXT, scale TEXT, material TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Initialization failed: {e}")

        self.app = FastAPI()
        self.theme_config = self.load_theme_config()
        self.scene_cache = {}

        @self.app.post("/process")
        async def process_request(request: ProcessRequest):
            logger.info(f"Processing request: {request}")
            return await self.process_request(
                text=request.text,
                audio_base64=request.audio,
                image_base64=request.image,
                canvas_base64=request.canvas,
                file_base64=request.file,
                theme=request.theme,
                selected_object=request.selected_object,
            )

        @self.app.post("/enhance_prompt")
        async def enhance_prompt(request: EnhancePromptRequest):
            logger.info(f"Enhancing prompt: {request.text}")
            return {"enhanced_prompt": self.process_text(request.text, enhance=True)}

        @self.app.post("/mcp")
        async def handle_mcp_request(request: MCPRequest):
            logger.info(f"Handling MCP request: {request.type}")
            return self.process_mcp_request(request.type, request.params)

        @self.app.get("/scene")
        async def get_scene():
            logger.info("Retrieving scene info")
            return await self.get_scene_info()

        @self.app.post("/export")
        async def export_scene():
            logger.info("Exporting scene")
            return await self.export_scene()

        @self.app.get("/templates")
        async def get_templates():
            logger.info("Listing templates")
            return await self.list_templates()

        @self.app.post("/load_template")
        async def load_template_endpoint(template: str):
            logger.info(f"Loading template: {template}")
            return await self.load_template(template)

    def load_theme_config(self):
        default_themes = {
            "modern": {
                "name": "ModernMaterial",
                "color": [0.2, 0.2, 0.2, 1],
                "material_type": "metal",
            },
            "futuristic": {
                "name": "FuturisticMaterial",
                "color": [0.75, 0.75, 0.75, 1],
                "material_type": "metal",
            },
            "rustic": {
                "name": "RusticMaterial",
                "color": [0.6, 0.4, 0.2, 1],
                "material_type": "wood",
            },
            "vintage": {
                "name": "VintageMaterial",
                "color": [0.76, 0.7, 0.5, 1],
                "material_type": "wood",
            },
            "industrial": {
                "name": "IndustrialMaterial",
                "color": [0.3, 0.3, 0.3, 1],
                "material_type": "metal",
            },
            "minimalist": {
                "name": "MinimalistMaterial",
                "color": [1, 1, 1, 1],
                "material_type": "plastic",
            },
            "scifi": {
                "name": "SciFiMaterial",
                "color": [0, 1, 1, 1],
                "material_type": "metal",
            },
            "classic": {
                "name": "ClassicMaterial",
                "color": [0.8, 0.6, 0.4, 1],
                "material_type": "wood",
            },
        }
        try:
            if os.path.exists("/app/themes.json"):
                with open("/app/themes.json", "r") as f:
                    return json.load(f)
            return default_themes
        except Exception as e:
            logger.error(f"Failed to load theme config: {e}")
            return default_themes

    def save_base64_file(self, base64_data, file_path):
        if not base64_data:
            return None
        try:
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(base64_data))
            return file_path
        except Exception as e:
            logger.error(f"Base64 decoding failed for {file_path}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    def process_text(self, text_input, enhance=False):
        if not text_input:
            return ""
        if enhance:
            try:
                doc = self.nlp(text_input)
                detailed_prompt = []
                current_object = None
                attributes = {}

                self.cursor.execute(
                    "SELECT name, type, location FROM scene_objects ORDER BY timestamp DESC"
                )
                scene_context = self.cursor.fetchall()
                context_str = "Current scene objects: " + ", ".join(
                    [f"{row[0]} ({row[1]}) at {row[2]}" for row in scene_context]
                )

                for token in doc:
                    if token.pos_ == "NOUN" and token.dep_ in ("dobj", "nsubj"):
                        current_object = token.text
                        attributes[current_object] = {
                            "color": "default",
                            "material": "default",
                            "size": "default",
                            "parts": {},
                        }
                    elif token.pos_ in ("ADJ", "NOUN") and current_object:
                        if (
                            token.text.lower() in webcolors.CSS3_NAMES_TO_HEX
                            or token.text.startswith("#")
                            or token.text.lower() in ["silver", "gold", "neon"]
                        ):
                            attributes[current_object]["color"] = token.text.lower()
                        elif token.text.lower() in [
                            "wood",
                            "metal",
                            "plastic",
                            "glass",
                            "leather",
                            "fabric",
                        ]:
                            attributes[current_object]["material"] = token.text.lower()
                        elif token.text.lower() in ["large", "small", "medium"]:
                            attributes[current_object]["size"] = token.text.lower()
                        elif token.dep_ == "compound" and current_object:
                            attributes[current_object]["parts"][token.text] = {
                                "color": "default"
                            }
                    elif token.dep_ == "prep" and token.text in [
                        "next",
                        "above",
                        "below",
                    ]:
                        detailed_prompt.append(
                            f"positioned {token.text} to the {current_object}"
                        )

                for ent in doc.ents:
                    if ent.label_ == "QUANTITY" and current_object:
                        attributes[current_object]["size"] = ent.text

                for obj, attrs in attributes.items():
                    size_map = {
                        "large": "2m x 2m x 1m",
                        "medium": "1m x 1m x 0.75m",
                        "small": "0.5m x 0.5m x 0.5m",
                        "default": "1m x 1m x 0.75m",
                    }
                    color_map = {
                        "red": [1, 0, 0, 1],
                        "blue": [0, 0, 1, 1],
                        "green": [0, 1, 0, 1],
                        "brown": [0.6, 0.4, 0.2, 1],
                        "white": [1, 1, 1, 1],
                        "black": [0, 0, 0, 1],
                        "silver": [0.75, 0.75, 0.75, 1],
                        "gold": [1, 0.84, 0, 1],
                        "neon": [0, 1, 1, 1],
                        "default": [0.8, 0.8, 0.8, 1],
                    }
                    material_map = {
                        "wood": "wood",
                        "metal": "metal",
                        "plastic": "plastic",
                        "glass": "glass",
                        "leather": "leather",
                        "fabric": "fabric",
                        "default": "plastic",
                    }

                    if attrs["color"].startswith("#"):
                        try:
                            rgb = webcolors.hex_to_rgb(attrs["color"])
                            color_map[attrs["color"]] = [
                                rgb[0] / 255,
                                rgb[1] / 255,
                                rgb[2] / 255,
                                1,
                            ]
                        except ValueError:
                            logger.warning(
                                f"Invalid hex color {attrs['color']}, using default"
                            )
                            color_map[attrs["color"]] = color_map["default"]
                    elif attrs["color"] in webcolors.CSS3_NAMES_TO_HEX:
                        rgb = webcolors.css3_hex_to_rgb(
                            webcolors.CSS3_NAMES_TO_HEX[attrs["color"]]
                        )
                        color_map[attrs["color"]] = [
                            rgb[0] / 255,
                            rgb[1] / 255,
                            rgb[2] / 255,
                            1,
                        ]
                    elif attrs["color"] not in color_map:
                        logger.warning(
                            f"Unrecognized color {attrs['color']}, using default"
                        )
                        color_map[attrs["color"]] = color_map["default"]

                    obj_desc = f"{obj} (size: {size_map[attrs['size']]}, material: {material_map[attrs['material']]}, color: {attrs['color']})"
                    for part, part_attrs in attrs["parts"].items():
                        if part_attrs["color"] in webcolors.CSS3_NAMES_TO_HEX:
                            rgb = webcolors.css3_hex_to_rgb(
                                webcolors.CSS3_NAMES_TO_HEX[part_attrs["color"]]
                            )
                            color_map[part_attrs["color"]] = [
                                rgb[0] / 255,
                                rgb[1] / 255,
                                rgb[2] / 255,
                                1,
                            ]
                        elif part_attrs["color"].startswith("#"):
                            try:
                                rgb = webcolors.hex_to_rgb(part_attrs["color"])
                                color_map[part_attrs["color"]] = [
                                    rgb[0] / 255,
                                    rgb[1] / 255,
                                    rgb[2] / 255,
                                    1,
                                ]
                            except ValueError:
                                logger.warning(
                                    f"Invalid hex color {part_attrs['color']}, using default"
                                )
                                color_map[part_attrs["color"]] = color_map["default"]
                        elif part_attrs["color"] not in color_map:
                            logger.warning(
                                f"Unrecognized color {part_attrs['color']}, using default"
                            )
                            color_map[part_attrs["color"]] = color_map["default"]
                        obj_desc += f", {part} (color: {part_attrs['color']})"
                    detailed_prompt.append(obj_desc)

                return (
                    f"{context_str}. {text_input}. {' '.join(detailed_prompt)}"
                    if detailed_prompt
                    else f"{context_str}. {text_input}"
                )
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Text processing failed: {e}"
                )
        return text_input

    def process_audio(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path)
            text = result["text"].lower()
            return self.process_text(text)
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image file")
            caption = self.image_captioner(image)[0]["generated_text"]
            detections = self.object_detector(image)
            objects = [f"{det['label']} at {det['box']}" for det in detections]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            shape_desc = (
                "complex shape with detailed edges"
                if edge_density > 0.1
                else "simple shape with smooth edges"
            )
            return f"Image description: {caption}. Detected objects: {', '.join(objects)}. Shape: {shape_desc}."
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    def process_canvas(self, canvas_path):
        return self.process_image(canvas_path)

    def process_file(self, file_path):
        try:
            if file_path.endswith(".blend"):
                script = f"""import bpy
bpy.ops.wm.open_mainfile(filepath="{file_path}")"""
            elif file_path.endswith(".obj"):
                script = f"""import bpy
bpy.ops.import_scene.obj(filepath="{file_path}")"""
            else:
                raise ValueError("Unsupported file format")
            return script
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {e}")

    def combine_inputs(
        self,
        text=None,
        audio_path=None,
        image_path=None,
        canvas_path=None,
        file_path=None,
        selected_object=None,
    ):
        prompt_parts = []
        if text:
            prompt_parts.append(self.process_text(text))
        if audio_path:
            prompt_parts.append(self.process_audio(audio_path))
        if image_path:
            prompt_parts.append(self.process_image(image_path))
        if canvas_path:
            prompt_parts.append(self.process_canvas(canvas_path))
        if selected_object:
            prompt_parts.append(f"Modify object: {selected_object}")
        return " ".join(prompt_parts)

    @lru_cache(maxsize=100)
    def generate_mcp_command(self, prompt, theme=None, selected_object=None):
        try:
            examples = [
                {
                    "instruction": "Create a red sphere",
                    "command": {
                        "type": "create_object",
                        "params": {
                            "object_type": "sphere",
                            "material": {"name": "Red", "color": [1, 0, 0, 1]},
                        },
                    },
                },
                {
                    "instruction": "Make the selected object blue",
                    "command": {
                        "type": "modify_material",
                        "params": {
                            "object_name": "selected",
                            "material": {"name": "BlueMaterial", "color": [0, 0, 1, 1]},
                        },
                    },
                },
                {
                    "instruction": "Move the table created earlier to the left",
                    "command": {
                        "type": "move_object",
                        "params": {"object_name": "table", "location": [-2, 0, 0]},
                    },
                },
            ]
            self.cursor.execute(
                "SELECT name, type, location FROM scene_objects ORDER BY timestamp DESC"
            )
            scene_context = self.cursor.fetchall()
            context_str = "Current scene objects: " + ", ".join(
                [f"{row[0]} ({row[1]}) at {row[2]}" for row in scene_context]
            )
            full_prompt = f"Given the following examples and scene context, generate an MCP command for the instruction:\n{context_str}\n"
            for ex in examples:
                full_prompt += f"Instruction: {ex['instruction']}\nCommand: {json.dumps(ex['command'])}\n\n"
            full_prompt += f"Instruction: {prompt}\n"
            if selected_object:
                full_prompt += f"Target object: {selected_object}\n"
            full_prompt += "Command:"
            if theme:
                full_prompt += f"\nApply the '{theme}' theme to the scene or objects."
            response = self.ollama_client.generate(
                model="llama3.1:8b", prompt=full_prompt
            )
            command = json.loads(response["response"].strip())
            if selected_object and command["type"] in [
                "modify_material",
                "move_object",
            ]:
                command["params"]["object_name"] = selected_object
            return command
        except Exception as e:
            logger.error(f"MCP command generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"MCP command generation failed: {e}"
            )

    def convert_image_to_3d(self, image_path):
        return {"type": "generate_model", "params": {"image_path": image_path}}

    def process_mcp_request(self, command_type, params, theme=None):
        try:
            script = ""
            if command_type == "create_object":
                material = params.get(
                    "material", {"name": "Default", "color": [0.8, 0.8, 0.8, 1]}
                )
                if theme:
                    material = self.apply_theme_material(theme, material)
                scale = params.get("scale", [1, 1, 1])
                part = params.get("part", "")
                object_type = params.get("object_type", "cube")
                script = f"""import bpy
bpy.ops.mesh.primitive_{object_type}_add()
bpy.context.object.scale = {tuple(scale)}
bpy.context.object.active_material = bpy.data.materials.new(name="{material['name']}_{part}")
bpy.context.object.active_material.diffuse_color = {tuple(material['color'])}"""
                if params.get("location") == "relative_to_selected":
                    script += f"\nbpy.context.object.location.x += 2"
                elif params.get("location") == "inside_body":
                    script += f"\nbpy.context.object.location.z += 0.5"
                elif params.get("location") == "below_body":
                    script += f"\nbpy.context.object.location.z -= 0.5"
                self.cursor.execute(
                    "INSERT INTO scene_objects (name, type, location, scale, material) VALUES (?, ?, ?, ?, ?)",
                    (
                        f"{object_type}_{part}",
                        object_type,
                        str(params.get("location", [0, 0, 0])),
                        str(scale),
                        str(material),
                    ),
                )
                self.conn.commit()
            elif command_type == "modify_material":
                object_name = params.get("object_name", "selected")
                material = params.get(
                    "material", {"name": "Default", "color": [0.8, 0.8, 0.8, 1]}
                )
                if theme:
                    material = self.apply_theme_material(theme, material)
                script = f"""import bpy
obj = bpy.context.active_object if "{object_name}" == "selected" else bpy.data.objects["{object_name}"]
material = bpy.data.materials.new(name="{material['name']}")
material.diffuse_color = {tuple(material['color'])}
obj.data.materials.append(material)"""
                self.cursor.execute(
                    "UPDATE scene_objects SET material = ? WHERE name = ?",
                    (str(material), object_name),
                )
                self.conn.commit()
            elif command_type == "move_object":
                object_name = params.get("object_name", "selected")
                location = params.get("location", [0, 0, 0])
                script = f"""import bpy
obj = bpy.context.active_object if "{object_name}" == "selected" else bpy.data.objects["{object_name}"]
obj.location = {tuple(location)}"""
                self.cursor.execute(
                    "UPDATE scene_objects SET location = ? WHERE name = ?",
                    (str(location), object_name),
                )
                self.conn.commit()
            elif command_type == "generate_model":
                script = f"""import bpy
bpy.ops.pixelmodeller.generate_model(image_path="{params['image_path']}")"""
            elif command_type in ["undo", "redo", "save"]:
                script = (
                    f"""import bpy
bpy.ops.ed.{command_type}()"""
                    if command_type in ["undo", "redo"]
                    else f"""import bpy
bpy.ops.wm.save_mainfile()"""
                )
            return {"script": script, "status": "success" if script else "error"}
        except Exception as e:
            logger.error(f"MCP request processing failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "suggestions": self.generate_error_suggestions(str(e)),
            }

    def apply_theme_material(self, theme, material):
        return self.theme_config.get(theme, material)

    def generate_error_suggestions(self, error_message):
        try:
            prompt = f"Given the error '{error_message}', suggest possible fixes for a Blender control system."
            response = self.ollama_client.generate(model="llama3.1:8b", prompt=prompt)
            return json.loads(response["response"].strip()) or []
        except Exception:
            return []

    async def get_scene_info(self):
        if self.blender_host in self.scene_cache:
            return self.scene_cache[self.blender_host]
        try:
            temp_file = "/app/temp_scene.json"
            script = f"""import bpy
import json
response = {{
    'selected_object': bpy.context.active_object.name if bpy.context.active_object else None,
    'objects': [obj.name for obj in bpy.context.scene.objects],
    'file_path': bpy.data.filepath
}}
with open('{temp_file}', 'w') as f:
    json.dump(response, f)
"""
            await self.send_to_blender({"script": script})
            with open(temp_file, "r") as f:
                scene_info = json.load(f)
            os.remove(temp_file)
            self.scene_cache[self.blender_host] = scene_info
            return scene_info
        except Exception as e:
            logger.error(f"Scene info retrieval failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Scene info retrieval failed: {e}"
            )

    async def export_scene(self):
        try:
            temp_file = "/app/output.blend"
            script = f"""import bpy
bpy.ops.wm.save_as_mainfile(filepath="{temp_file}")"""
            await self.send_to_blender({"script": script})
            with open(temp_file, "rb") as f:
                file_data = base64.b64encode(f.read()).decode()
            os.remove(temp_file)
            return {"file": file_data, "scene_info": await self.get_scene_info()}
        except Exception as e:
            logger.error(f"Scene export failed: {e}")
            raise HTTPException(status_code=500, detail=f"Scene export failed: {e}")

    async def generate_preview(self):
        try:
            temp_file = "/app/preview.png"
            script = f"""import bpy
bpy.context.scene.render.filepath = "{temp_file}"
bpy.ops.render.render(write_still=True)"""
            await self.send_to_blender({"script": script})
            with open(temp_file, "rb") as f:
                preview_data = base64.b64encode(f.read()).decode()
            os.remove(temp_file)
            return {"preview": preview_data, "status": "success"}
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Preview generation failed: {e}"
            )

    async def list_templates(self):
        try:
            template_dir = "/app/templates"
            if not os.path.exists(template_dir):
                return {"templates": []}
            templates = [f for f in os.listdir(template_dir) if f.endswith(".blend")]
            return {"templates": templates}
        except Exception as e:
            logger.error(f"Template listing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Template listing failed: {e}")

    async def load_template(self, template_name):
        try:
            template_path = os.path.join("/app/templates", template_name)
            if not os.path.exists(template_path):
                raise ValueError("Template not found")
            script = f"""import bpy
bpy.ops.wm.open_mainfile(filepath="{template_path}")"""
            return await self.send_to_blender({"script": script})
        except Exception as e:
            logger.error(f"Template loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"Template loading failed: {e}")

    async def send_to_blender(self, request):
        try:
            temp_script = "/app/temp_script.py"
            with open(temp_script, "w") as f:
                f.write(request["script"])
            return {"status": "success", "script": request["script"]}
        except Exception as e:
            logger.error(f"Send to Blender failed: {e}")
            raise HTTPException(status_code=500, detail=f"Send to Blender failed: {e}")

    async def process_request(
        self,
        text=None,
        audio_base64=None,
        image_base64=None,
        canvas_base64=None,
        file_base64=None,
        theme=None,
        selected_object=None,
    ):
        try:
            audio_path = (
                self.save_base64_file(audio_base64, "/app/temp_audio.wav")
                if audio_base64
                else None
            )
            image_path = (
                self.save_base64_file(image_base64, "/app/temp_image.png")
                if image_base64
                else None
            )
            canvas_path = (
                self.save_base64_file(canvas_base64, "/app/temp_canvas.png")
                if canvas_base64
                else None
            )
            file_path = (
                self.save_base64_file(file_base64, "/app/temp_file.blend")
                if file_base64
                else None
            )

            file_script = self.process_file(file_path) if file_path else ""
            prompt = self.combine_inputs(
                text,
                audio_path,
                image_path,
                canvas_path,
                selected_object=selected_object,
            )

            commands = self.generate_mcp_command(prompt, theme, selected_object)
            if not isinstance(commands, list):
                commands = [commands]

            script = file_script
            for cmd in commands:
                result = self.process_mcp_request(cmd["type"], cmd["params"], theme)
                if result["status"] == "error":
                    return {
                        "status": "error",
                        "message": result["message"],
                        "suggestions": result["suggestions"],
                    }
                script += "\n" + result["script"]

            if image_path or canvas_path:
                model_path = image_path or canvas_path
                model_command = self.convert_image_to_3d(model_path)
                model_result = self.process_mcp_request(
                    model_command["type"], model_command["params"], theme
                )
                if model_result["status"] == "error":
                    return {
                        "status": "error",
                        "message": model_result["message"],
                        "suggestions": model_result["suggestions"],
                    }
                script += "\n" + model_result["script"]

            preview_result = await self.generate_preview()
            scene_info = await self.get_scene_info()
            export_result = await self.export_scene()

            for path in [audio_path, image_path, canvas_path, file_path]:
                if path and os.path.exists(path):
                    os.remove(path)

            return {
                "script": script,
                "mcp_commands": commands,
                "scene_info": scene_info,
                "updated_file": export_result["file"],
                "preview": preview_result["preview"],
            }
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "suggestions": self.generate_error_suggestions(str(e)),
            }

    def run_server(self):
        import uvicorn

        logger.info("Started server on port 8000")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


backend = BlenderControlBackend()
app = backend.app
