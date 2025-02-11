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
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    faces = [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in cursor.fetchall()]
    conn.close()
    return faces

def recognize_face(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([f[1] for f in known_faces], face_encoding)
        name = "Desconhecido"
        
        if True in matches:
            matched_idx = matches.index(True)
            name = known_faces[matched_idx][0]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Inicializa o banco de dados
init_db()

# Captura de vídeo
video_capture = cv2.VideoCapture(0)
known_faces = load_faces()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame = recognize_face(frame, known_faces)
    cv2.imshow("Reconhecimento Facial", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        name = input("Digite o nome da pessoa: ")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if face_encodings:
            save_face(name, face_encodings[0])
            known_faces.append((name, face_encodings[0]))
            print(f"Rosto de {name} salvo com sucesso!")
        else:
            print("Nenhum rosto detectado.")

video_capture.release()
cv2.destroyAllWindows()
