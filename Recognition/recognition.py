import cv2
import os
import numpy as np

# Diretório do banco de dados
db_path = "face_db/"
os.makedirs(db_path, exist_ok=True)

label_ids = {}

# Carregar classificador Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise Exception("Erro ao carregar o classificador Haarcascade.")

# Criar reconhecedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar dados de treinamento
def load_training_data(data_folder):
    global label_ids
    faces, labels = [], []
    label_ids = {}
    current_id = 0

    if not os.listdir(data_folder):
        print(f"A pasta '{data_folder}' está vazia.")
        return faces, labels

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(("png", "jpg")):
                path = os.path.join(root, file)
                label = os.path.basename(root).lower()

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Erro ao carregar {path}")
                    continue

                img = cv2.resize(img, (100, 100))

                faces.append(img)
                labels.append(label_ids[label])

    return faces, labels

# Função para capturar múltiplas imagens para um usuário
def capture_multiple_faces(num_samples=30):
    global cap

    print(f"Capturando {num_samples} imagens. Posicione o rosto na câmera.")

    # Fechar qualquer instância aberta da câmera
    if cap.isOpened():
        cap.release()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    if not cap.isOpened():
        raise Exception("Erro: Não foi possível abrir a câmera.")

    # Criar pasta para o novo usuário
    user_id = len(os.listdir(db_path)) + 1
    user_folder = os.path.join(db_path, f"user{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            # Salvar a imagem na pasta do usuário
            save_path = os.path.join(user_folder, f"{count + 1}.jpg")
            cv2.imwrite(save_path, face_img)
            count += 1

            cv2.putText(frame, f"Capturando {count}/{num_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Capturando Rostos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Captura concluída! Imagens salvas em: {user_folder}")

    cap.release()
    cv2.destroyAllWindows()

# Função para treinar o modelo
def train_model():
    global recognizer, label_ids
    faces, labels = load_training_data(db_path)
    
    if faces:
        recognizer.train(faces, np.array(labels))
        print("Reconhecedor treinado com sucesso!")
    else:
        print("Nenhum dado de treinamento encontrado.")

# Carregar dados de treinamento e treinar o modelo
train_model()

# Captura de vídeo da câmera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    raise Exception("Erro: Não foi possível abrir a câmera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        if label_ids and len(label_ids) > 0:
            label, confidence = recognizer.predict(face_img)
            if confidence < 86 and label in label_ids.values():
                name = list(label_ids.keys())[list(label_ids.values()).index(label)]
                cv2.putText(frame, f"Reconhecido: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Nenhum dado treinado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF

    # Registrar novo rosto (tecla 's')
    if key == ord('s'):
        print("Registrando novo rosto...")
        capture_multiple_faces(num_samples=30)
        train_model()

    # Sair (tecla 'q')
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
