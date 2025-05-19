from loader import *

def recognize():
    detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)
    
    flag = False
    detected_persons = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if not flag:
            height, width = frame.shape[:2]
            ratio = width / height
            new_width = 540
            new_height = int(new_width / ratio)
            flag = True
            
        frame = cv2.resize(frame, (new_width, new_height))
        
        faces = detector.detect_faces(frame)
        
        if not faces:
            continue
    
        for face in faces:
            x, y, w, h = face['box']
            
            cropped_face = frame[y:y + h, x:x + w ]

            # Get embedding
            embedding = get_embedding(cropped_face)
            
            if embedding is None:
                print("No face detected")

            D, I = index.search(np.array([embedding], dtype=np.float32), 1)
            
            matched_name = "Unknown"
            confidence = 0.0

            if D[0][0] < 0.8:
                matched_name = person_id_list[I[0][0]]
                confidence = 1 - (D[0][0] / 0.8)
                detected_persons.add(matched_name)
                print(f"Match: {matched_name}, Distance: {D[0][0]:.4f}")
            
            if cropped_face is not None: 
                if matched_name != "Unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, str(matched_name), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    cv2.putText(frame, f'{confidence * 100:.2f}%', (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, str(matched_name), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(frame, f'{confidence * 100:.2f}%', (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    
            cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
                
    cap.release()
    cv2.destroyAllWindows()
    
    # Print detected persons
    print("\n✅ Unique persons detected during session:")
    for person in detected_persons:
        print(f"-> {person}")


if __name__ == "__main__":
    # Start the face recognition process
    recognize()