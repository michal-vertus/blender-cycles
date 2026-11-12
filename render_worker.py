import os
import sys
import json
import base64
import subprocess
import urllib.request
import runpod

BLENDER_SCRIPT = r'''
import bpy
import json
import math
import os
from mathutils import Vector

with open("/tmp/params.json", "r") as f:
    params = json.load(f)

# 1. Load Base 3D Scene
scene_file = params.get("scene_file")
if scene_file and os.path.exists(scene_file):
    ext = os.path.splitext(scene_file)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=scene_file)
    elif ext in (".glb", ".gltf"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=scene_file)
    elif ext == ".obj":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            bpy.ops.wm.obj_import(filepath=scene_file)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=scene_file)
elif not bpy.data.filepath:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1), segments=64, ring_count=32)
    sphere = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("Ball")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1)
        bsdf.inputs["Roughness"].default_value = 0.2
    sphere.data.materials.append(mat)

scene = bpy.context.scene

# 2. Frame Timeline Selection
if "frame" in params and params["frame"] is not None:
    scene.frame_set(int(params["frame"]))

# 3. 32-bit HDRI Environment Setup
hdri_file = params.get("hdri_file")
if hdri_file and os.path.exists(hdri_file):
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputWorld")
    node_bg = nodes.new("ShaderNodeBackground")
    node_env = nodes.new("ShaderNodeTexEnvironment")
    node_map = nodes.new("ShaderNodeMapping")
    node_coord = nodes.new("ShaderNodeTexCoord")

    img = bpy.data.images.load(hdri_file)
    node_env.image = img

    strength = float(params.get("hdri_strength", 1.0))
    rotation_deg = float(params.get("hdri_rotation", 0.0))
    node_bg.inputs["Strength"].default_value = strength
    node_map.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)

    links.new(node_coord.outputs["Generated"], node_map.inputs["Vector"])
    links.new(node_map.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

# 4. Volumetric Atmosphere
vol_config = params.get("volumetrics")
if vol_config:
    density = float(vol_config.get("density", 0.005))
    anisotropy = float(vol_config.get("anisotropy", 0.5))
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    node_out = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    node_vol = nodes.new("ShaderNodeVolumeScatter")
    node_vol.inputs["Density"].default_value = density
    node_vol.inputs["Anisotropy"].default_value = anisotropy
    links.new(node_vol.outputs["Volume"], node_out.inputs["Volume"])

# 5. Camera Optics and Depth of Field
cam_config = params.get("camera")
cam_obj = scene.camera
if not cam_obj:
    bpy.ops.object.camera_add(location=(0, -6, 2.5), rotation=(math.radians(75), 0, 0))
    cam_obj = bpy.context.active_object
    scene.camera = cam_obj

if cam_config:
    if "location" in cam_config:
        cam_obj.location = Vector(cam_config["location"])
    if "look_at" in cam_config:
        target = Vector(cam_config["look_at"])
        direction = target - cam_obj.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()
    elif "rotation" in cam_config:
        cam_obj.rotation_euler = [math.radians(r) for r in cam_config["rotation"]]

    cam_data = cam_obj.data
    if "focal_length" in cam_config:
        cam_data.lens = float(cam_config["focal_length"])
    if "fstop" in cam_config or "focus_distance" in cam_config:
        cam_data.dof.use_dof = True
        if "fstop" in cam_config:
            cam_data.dof.aperture_fstop = float(cam_config["fstop"])
        if "focus_distance" in cam_config:
            cam_data.dof.focus_distance = float(cam_config["focus_distance"])

# 6. Auto-Bevel Sharp Edges
if params.get("auto_bevel", False):
    for obj in scene.objects:
        if obj.type == "MESH" and not obj.modifiers.get("AutoBevel"):
            bev = obj.modifiers.new(name="AutoBevel", type="BEVEL")
            bev.width = 0.004
            bev.segments = 2
            bev.limit_method = "ANGLE"
            bev.angle_limit = math.radians(35)

# 7. Procedural Python Override
custom_script = params.get("script")
if custom_script:
    exec(custom_script, {"bpy": bpy, "math": math, "Vector": Vector, "scene": scene})

# 8. Render Engine and Cycles Device
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
chosen_device = None
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            chosen_device = dev_type
            break
    except Exception:
        continue

for d in prefs.devices:
    d.use = (d.type == chosen_device)

if chosen_device:
    scene.cycles.device = "GPU"
    scene.cycles.denoiser = "OPTIX" if chosen_device == "OPTIX" else "OPENIMAGEDENOISE"
else:
    scene.cycles.device = "CPU"
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

scene.cycles.use_denoising = True
scene.cycles.samples = int(params.get("samples", 64))
scene.view_settings.view_transform = params.get("view_transform", "AgX")
scene.view_settings.look = params.get("look", "None")

scene.render.resolution_x = int(params.get("width", 1280))
scene.render.resolution_y = int(params.get("height", 720))
scene.render.resolution_percentage = 100
scene.render.filepath = "/tmp/output.png"
scene.render.image_settings.file_format = "PNG"

print("DEVICE_USED:", chosen_device)
bpy.ops.render.render(write_still=True)
'''


def download_file(url, target_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(target_path, "wb") as out:
        out.write(resp.read())


def handler(job):
    inp = job.get("input", {})
    params = {
        "samples": int(inp.get("samples", 64)),
        "width": int(inp.get("width", 1280)),
        "height": int(inp.get("height", 720)),
        "view_transform": inp.get("view_transform", "AgX"),
        "look": inp.get("look", "None"),
        "frame": inp.get("frame"),
        "camera": inp.get("camera"),
        "volumetrics": inp.get("volumetrics"),
        "auto_bevel": bool(inp.get("auto_bevel", False)),
        "script": inp.get("script"),
        "hdri_strength": inp.get("hdri_strength", 1.0),
        "hdri_rotation": inp.get("hdri_rotation", 0.0),
    }

    out_path = "/tmp/output.png"
    for p in (out_path, "/tmp/scene_input", "/tmp/environment.exr", "/tmp/params.json", "/tmp/run_render.py"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    scene_url = inp.get("scene_url") or inp.get("blend_url")
    if scene_url:
        clean_url = scene_url.split("?")[0].lower()
        ext = ".blend"
        for candidate in (".blend", ".glb", ".gltf", ".obj"):
            if clean_url.endswith(candidate):
                ext = candidate
                break
        scene_path = f"/tmp/scene_input{ext}"
        try:
            download_file(scene_url, scene_path)
            params["scene_file"] = scene_path
        except Exception as e:
            return {"error": f"Failed to download scene asset: {str(e)}"}

    hdri_url = inp.get("hdri_url")
    if hdri_url:
        hdri_path = "/tmp/environment.exr"
        try:
            download_file(hdri_url, hdri_path)
            params["hdri_file"] = hdri_path
        except Exception as e:
            return {"error": f"Failed to download HDRI environment: {str(e)}"}

    with open("/tmp/params.json", "w") as f:
        json.dump(params, f)
    with open("/tmp/run_render.py", "w") as f:
        f.write(BLENDER_SCRIPT)

    cmd = ["blender", "-b", "-P", "/tmp/run_render.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_tail = result.stdout[-2000:]

    if result.returncode != 0 or not os.path.exists(out_path):
        return {
            "error": "Render execution failed",
            "log": log_tail,
            "stderr": result.stderr[-2000:]
        }

    device_line = [line for line in result.stdout.splitlines() if "DEVICE_USED" in line]
    with open(out_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")

    return {
        "status": "success",
        "device": device_line[-1] if device_line else "unknown",
        "image_base64": encoded
    }


runpod.serverless.start({"handler": handler})
