# save as debug_detect.py at project root, run once
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"],
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

faces = app.get(frame)
print(f"Faces found: {len(faces)}")
for f in faces:
    print(f"  score={f.det_score:.3f}  bbox={f.bbox}  kps={f.kps}")