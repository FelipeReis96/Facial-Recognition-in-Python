import cv2
import os
import numpy as np

# Diretório para armazenar rostos conhecidos
db_path = "face_db/"
os.makedirs(db_path, exist_ok=True)

# Carregar o classificador de detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verificar se o classificador foi carregado corretamente
if face_cascade.empty():
    raise Exception("Erro ao carregar o classificador Haarcascade. Verifique se o arquivo está disponível.")

# Criar o reconhecedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens de treinamento
def load_training_data(data_folder):
    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    # Verificar se a pasta está vazia
    if not os.listdir(data_folder):
        print(f"A pasta '{data_folder}' está vazia. Adicione imagens de treinamento.")
        return faces, labels, label_ids

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # Carregar a imagem em tons de cinza
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Erro ao carregar a imagem: {path}")
                    continue

                # Redimensionar a imagem (opcional)
                img = cv2.resize(img, (100, 100))

                faces.append(img)
                labels.append(label_ids[label])

    return faces, labels, label_ids

# Função para registrar um novo rosto
def register_face(face_img):
    # Gerar nome automaticamente baseado no número de subpastas
    user_id = len(os.listdir(db_path)) + 1
    name = f"user{user_id}"

    # Criar uma subpasta para o novo usuário
    user_folder = os.path.join(db_path, name)
    os.makedirs(user_folder, exist_ok=True)

    # Contar quantas imagens já existem para evitar sobrescrever
    num_images = len(os.listdir(user_folder))
    save_path = os.path.join(user_folder, f"{num_images + 1}.jpg")

    # Salvar a imagem do rosto
    cv2.imwrite(save_path, face_img)
    print(f"Rosto de {name} salvo em {save_path}")
    
    return name

# Função para capturar múltiplas imagens de um rosto
def capture_multiple_faces(num_samples=30):
    print(f"Capturando {num_samples} imagens. Posicione o rosto na câmera.")
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            name = register_face(face_img)  # Registra o rosto com o nome gerado automaticamente
            count += 1
            cv2.putText(frame, f"Capturando {count}/{num_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Capturando Rostos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Função para treinar o modelo
def train_model():
    global recognizer, faces, labels, label_ids
    faces, labels, label_ids = load_training_data(db_path)
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        print("Reconhecedor treinado com sucesso!")
    else:
        print("Nenhum dado de treinamento encontrado. Registre novos rostos.")

# Carregar dados de treinamento e treinar o modelo
train_model()

# Captura de vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # Reconhecer o rosto apenas se o modelo foi treinado
        if len(faces) > 0 and len(label_ids) > 0:  # Verifica se há dados de treinamento
            label, confidence = recognizer.predict(face_img)
            if confidence < 85:  # Ajuste o limiar conforme necessário
                name = list(label_ids.keys())[list(label_ids.values()).index(label)]
                cv2.putText(frame, f"Reconhecido: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Face Recognition', frame)

    # Capturar tecla pressionada
    key = cv2.waitKey(1) & 0xFF

    # Registrar novo rosto (tecla 's')
    if key == ord('s'):
        print("Registrando novo rosto...")
        capture_multiple_faces(num_samples=30)  # Captura 30 imagens
        train_model()  # Treina o modelo automaticamente após o registro

    # Sair (tecla 'q')
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
