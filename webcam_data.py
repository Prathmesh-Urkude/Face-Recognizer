import cv2
import mtcnn

def webcam_data():
    detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect_faces(frame)
        
        if not faces:
            continue
        
        for face in faces:
            x,y,w,h = face['box']
            cropped_face = frame[y:y+h, x:x+h]
            
            if cropped_face is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            cv2.imshow("Face Detection", frame)
            
        if cv2.waitKey(10) & 0xFF == ord('q'): # press 'q' to exit
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam data collection completed.")

# This function captures webcam data and detects faces using MTCNN
# It draws rectangles around detected faces and displays the video feed   
if __name__ == "__main__":
    webcam_data()