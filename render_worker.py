import os
import json
import base64
import subprocess
import urllib.request
import runpod

BLENDER_SCRIPT = r'''
import bpy, json, math

params = json.load(open("/tmp/params.json"))

def build_default_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1), segments=64, ring_count=32)
    sphere = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("Ball")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1)
    bsdf.inputs["Roughness"].default_value = 0.2
    sphere.data.materials.append(mat)
    bpy.ops.object.light_add(type="SUN", location=(4, -4, 6))
    bpy.context.active_object.data.energy = 4
    bpy.context.active_object.rotation_euler = (math.radians(50), 0, math.radians(30))
    bpy.ops.object.camera_add(location=(0, -6, 2.5), rotation=(math.radians(75), 0, 0))
    bpy.context.scene.camera = bpy.context.active_object
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.07, 0.1, 1)
    bpy.context.scene.world = world

if not bpy.data.filepath:
    build_default_scene()

scene = bpy.context.scene
scene.render.engine = "CYCLES"

prefs = bpy.context.preferences.addons["cycles"].preferences
chosen = None
for t in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = t
        prefs.get_devices()
        if any(d.type == t for d in prefs.devices):
            chosen = t
            break
    except Exception:
        continue

for d in prefs.devices:
    d.use = (d.type == chosen)

if chosen:
    scene.cycles.device = "GPU"
    scene.cycles.denoiser = "OPTIX" if chosen == "OPTIX" else "OPENIMAGEDENOISE"
else:
    scene.cycles.device = "CPU"
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

scene.cycles.use_denoising = True
scene.cycles.samples = params["samples"]
scene.render.resolution_x = params["width"]
scene.render.resolution_y = params["height"]
scene.render.resolution_percentage = 100
scene.render.filepath = "/tmp/output.png"
scene.render.image_settings.file_format = "PNG"

print("DEVICE_USED:", chosen)
bpy.ops.render.render(write_still=True)
'''


def handler(job):
    inp = job.get("input", {})
    params = {
        "samples": int(inp.get("samples", 64)),
        "width": int(inp.get("width", 960)),
        "height": int(inp.get("height", 540)),
    }
    out_path = "/tmp/output.png"
    blend_path = "/tmp/scene.blend"

    for p in (out_path, blend_path):
        if os.path.exists(p):
            os.remove(p)

    with open("/tmp/params.json", "w") as f:
        json.dump(params, f)
    with open("/tmp/run_render.py", "w") as f:
        f.write(BLENDER_SCRIPT)

    cmd = ["blender", "-b"]
    blend_url = inp.get("blend_url")
    if blend_url:
        urllib.request.urlretrieve(blend_url, blend_path)
        cmd.append(blend_path)
    cmd += ["-P", "/tmp/run_render.py"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    log_tail = result.stdout[-1500:]

    if result.returncode != 0 or not os.path.exists(out_path):
        return {"error": "Render failed", "log": log_tail, "stderr": result.stderr[-1500:]}

    device_line = [l for l in result.stdout.splitlines() if "DEVICE_USED" in l]
    with open(out_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")

    return {
        "status": "success",
        "device": device_line[-1] if device_line else "unknown",
        "image_base64": encoded,
    }


runpod.serverless.start({"handler": handler})
