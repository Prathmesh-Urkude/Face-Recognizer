from loader import *


def webcam_data():
    detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Let the camera warm up
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    name = input("Enter name: ")
    img_id = 0
    print("Press spacebar to capture images capture images. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            continue
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Empty frame detected")
            continue
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if not faces:
            continue
        
        for face in faces:
            x,y,w,h = face['box']
            cropped_face = frame[y:y+h, x:x+h]
            
            if cropped_face is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(frame, str(img_id), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            
            cv2.imshow("Face Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == 32:  # Spacebar key
                
                # Get embedding
                embedding = get_embedding(cropped_face) # Should be (512,)
                
                if embedding is None:
                    print("Error: Could not get embedding.")
                    continue
                
                add_face_to_faiss(embedding, name)
                img_id += 1
            
        if cv2.waitKey(10) & 0xFF == ord('q') or img_id >= 5: # press 'q' to exit
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam data collection completed.")


# This function captures webcam data and detects faces using MTCNN
# It draws rectangles around detected faces and displays the video feed   
if __name__ == "__main__":
    webcam_data()