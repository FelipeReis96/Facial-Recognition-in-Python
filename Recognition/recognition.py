import cv2
import face_recognition
import sqlite3
import numpy as np

# Criar/Conectar ao banco de dados
def init_db():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        encoding BLOB)''')
    conn.commit()
    conn.close()

def save_face(name, encoding):
    encoding = np.array(encoding, dtype=np.float32)[:128]  # Garantir 128 dimensões
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    faces = []
    for row in cursor.fetchall():
        encoding = np.frombuffer(row[1], dtype=np.float32)
        if encoding.shape[0] != 128:
            continue  # Ignorar dados corrompidos
        faces.append((row[0], encoding))
    conn.close()
    return faces

# Inicializa o banco de dados
init_db()

# Carregar rostos salvos antes de iniciar a captura de vídeo
known_faces = load_faces()

# Captura de vídeo
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconhecimento Facial", 240, 180)
cv2.setWindowProperty("Reconhecimento Facial", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)

frame_skip = 3  # Processar 1 a cada 3 frames
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    message = "Nenhum rosto reconhecido"
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_encoding = np.array(face_encoding, dtype=np.float32)[:128]  # Garantir 128 dimensões
        matches = face_recognition.compare_faces([f[1] for f in known_faces], face_encoding, tolerance=0.6)
        name = "Desconhecido"
        
        if True in matches:
            face_distances = face_recognition.face_distance([f[1] for f in known_faces], face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_faces[best_match_index][0]
            message = f"Reconhecido: {name}"
        
        cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (0, 255, 0), 2)
        cv2.putText(frame, name, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Reconhecimento Facial", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        name = input("Digite o nome da pessoa: ")
        
        if face_encodings:
            save_face(name, face_encodings[0])
            known_faces.append((name, face_encodings[0]))
            print(f"Rosto de {name} salvo com sucesso!")
        else:
            print("Nenhum rosto detectado.")

video_capture.release()
cv2.destroyAllWindows()








