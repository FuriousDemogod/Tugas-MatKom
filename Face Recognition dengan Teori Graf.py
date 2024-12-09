import cv2
import numpy as np
import dlib
import face_recognition
import matplotlib.pyplot as plt
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Dokumen/Kuliah/MatKom/points-master/shape_predictor_68_face_landmarks.dat')
Daftar_Wajah = []
Daftar_Nama_dari_Wajah = []

def detect_faces_and_landmarks(gambar):
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            cv2.circle(gambar, (x, y), 2, (255, 0, 0), -1)

        def draw_lines(p, is_closed=False, thickness=1):
            for i in range(len(p) - 1):
                cv2.line(gambar, p[i], p[i + 1], (0, 255, 0), thickness)
            if is_closed:
                cv2.line(gambar, p[0], p[-1], (0, 255, 0), thickness)

        jaw = points[0:17]
        left_brow = points[17:22]
        right_brow = points[22:27]
        nose = points[27:36]
        left_eye = points[36:42]
        right_eye = points[42:48]
        mouth_outer = points[48:60]
        mouth_inner = points[60:68]

        draw_lines(jaw)
        draw_lines(left_brow)
        draw_lines(right_brow)
        draw_lines(nose)
        draw_lines(left_eye, is_closed=True)
        draw_lines(right_eye, is_closed=True)
        draw_lines(mouth_outer, is_closed=True)
        draw_lines(mouth_inner, is_closed=True)

def recognize_faces(gambar, Daftar_Wajah, Daftar_Nama_dari_Wajah):
    face_locations = face_recognition.face_locations(gambar)
    face_encodings = face_recognition.face_encodings(gambar, face_locations)

    face_namas = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(Daftar_Wajah, face_encoding)
        nama = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            nama = Daftar_Nama_dari_Wajah[first_match_index]

        face_namas.append(nama)

    return face_locations, face_namas
    
def detect_face_shape(points):
    jaw_width = np.linalg.norm(np.array(points[0]) - np.array(points[16]))
    eye_distance = np.linalg.norm(np.array(points[36]) - np.array(points[45]))
    
def is_valid_file(path_file):
    return os.path.exists(path_file) and path_file.lower().endswith(('.png', '.jpg', '.jpeg'))

def load_image(path_file):
    gambar = cv2.imread(path_file)
    if gambar is None:
        raise ValueError("Gambar tidak dapat dibaca.")
    return gambar

def process_image(gambar, Daftar_Wajah, Daftar_Nama_dari_Wajah):
    detect_faces_and_landmarks(gambar)
    face_locations, face_namas = recognize_faces(gambar, Daftar_Wajah, Daftar_Nama_dari_Wajah)

    for (top, right, bottom, left), nama in zip(face_locations, face_namas):
        cv2.rectangle(gambar, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(gambar, nama, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        points = [(predictor(gambar, dlib.rectangle(left, top, right, bottom)).part(n).x,
                   predictor(gambar, dlib.rectangle(left, top, right, bottom)).part(n).y) for n in range(68)]
        face_shape = detect_face_shape(points)
        cv2.putText(gambar, face_shape, (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return gambar

def display_image(gambar):
    gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
    plt.imshow(gambar_rgb)
    plt.axis('off')
    plt.show()
    
def main(path_file=None, Daftar_Wajah=[], Daftar_Nama_dari_Wajah=[]):
    if path_file is None:
        print("Path file tidak diberikan.")
        return

    if not is_valid_file(path_file):
        print("File tidak ditemukan atau format file tidak didukung.")
        return

    try:
        gambar = load_image(path_file)
        processed_image = process_image(gambar, Daftar_Wajah, Daftar_Nama_dari_Wajah)
        display_image(processed_image)
    except ValueError as e:
        print(e)
        
def wajah_dikenali(path_dari_gambar, Daftar_Wajah, Daftar_Nama_dari_Wajah, nama):
    gambar = face_recognition.load_image_file(path_dari_gambar)
    encoding = face_recognition.face_encodings(gambar)[0]
    Daftar_Wajah.append(encoding)
    Daftar_Nama_dari_Wajah.append(nama)

wajah_dikenali("D:\Dokumen\Kuliah\MatKom\Sampel\Data Wajah\elon_musk1.jpg", Daftar_Wajah, Daftar_Nama_dari_Wajah, "Mas Elon")

path_file = "D:\Dokumen\Kuliah\MatKom\Sampel\Data Wajah\elon_musk2.jpg"

main(path_file, Daftar_Wajah, Daftar_Nama_dari_Wajah)