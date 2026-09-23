"""python build_viewer.py [scene.json] -> scene_viewer.html (three.js page with the scene embedded)"""
import sys, pathlib
scene = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "scene.json").read_text()
tpl = pathlib.Path("scene_viewer_template.html").read_text()
out = tpl.replace("/*SCENE_JSON*/", scene.replace("</", "<\\/"))
pathlib.Path("scene_viewer.html").write_text(out)
print("scene_viewer.html", round(len(out) / 1e6, 2), "MB")
