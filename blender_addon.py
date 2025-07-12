import bpy
import requests
import json
import os
import base64
from bpy.types import Operator, Panel
from bpy.props import StringProperty, EnumProperty

bl_info = {
    "name": "AI Blender Control",
    "version": (1, 4),
    "blender": (3, 0, 0),
    "category": "Automation",
}


def get_scene_objects(self, context):
    """Generate a list of scene objects for the dropdown."""
    items = [(obj.name, obj.name, "") for obj in bpy.context.scene.objects]
    items.insert(0, ("NONE", "None", "No object selected"))
    return items


class AIControlOperator(Operator):
    bl_idname = "wm.ai_control"
    bl_label = "AI Control Operator"
    prompt: StringProperty(name="Prompt")
    theme: StringProperty(name="Theme")
    input_type: EnumProperty(
        name="Input Type",
        items=[
            ("TEXT", "Text", ""),
            ("AUDIO", "Audio", ""),
            ("IMAGE", "Image", ""),
            ("CANVAS", "Canvas", ""),
        ],
        default="TEXT",
    )
    file_path: StringProperty(name="File Path", subtype="FILE_PATH")
    selected_object: EnumProperty(
        name="Selected Object",
        items=get_scene_objects,
        description="Select an object to modify",
    )

    def execute(self, context):
        try:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            data = {"theme": self.theme}

            if self.input_type == "TEXT":
                data["text"] = self.prompt
            elif self.input_type == "AUDIO":
                with open(self.file_path, "rb") as f:
                    data["audio"] = base64.b64encode(f.read()).decode()
            elif self.input_type == "IMAGE":
                with open(self.file_path, "rb") as f:
                    data["image"] = base64.b64encode(f.read()).decode()
            elif self.input_type == "CANVAS":
                with open(self.file_path, "rb") as f:
                    data["canvas"] = base64.b64encode(f.read()).decode()

            # Include selected object if not "NONE"
            if self.selected_object != "NONE":
                data["selected_object"] = self.selected_object

            response = requests.post(f"{backend_url}/process", json=data)
            response.raise_for_status()
            result = response.json()

            if result["status"] == "error":
                self.report({"ERROR"}, f"Backend error: {result['message']}")
                return {"CANCELLED"}

            # Execute script
            script = result.get("script", "")
            if script:
                exec(script)

            # Save preview
            preview = result.get("preview", "")
            if preview:
                preview_path = os.path.join(os.path.dirname(__file__), "preview.png")
                with open(preview_path, "wb") as f:
                    f.write(base64.b64decode(preview))
                self.report({"INFO"}, f"Preview saved at {preview_path}")

            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Execution failed: {e}")
            return {"CANCELLED"}


class AIControlPanel(Panel):
    bl_label = "AI Blender Control"
    bl_idname = "PT_AIControlPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Control"

    def draw(self, context):
        layout = self.layout
        op = layout.operator("wm.ai_control", text="Execute Command")
        layout.prop(op, "prompt")
        layout.prop(op, "theme")
        layout.prop(op, "input_type")
        if op.input_type != "TEXT":
            layout.prop(op, "file_path")
        layout.prop(op, "selected_object")


def register():
    bpy.utils.register_class(AIControlOperator)
    bpy.utils.register_class(AIControlPanel)


def unregister():
    bpy.utils.unregister_class(AIControlOperator)
    bpy.utils.unregister_class(AIControlPanel)


if __name__ == "__main__":
    register()
