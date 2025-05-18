from loader import *

# Add new person from folder
def add_person(person_name, folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    added = 0
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"[INFO] Skipping non-image file: {img_path}")
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        if image.shape[:2] > (1000, 1000):
            print(f"[INFO] Skipping image {img_path} due to size.")
            continue
        
        detector = mtcnn.MTCNN()
        # Detect faces
        faces = detector.detect_faces(image)
        if not faces:
            return None
        
        x, y, w, h = faces[0]['box']
        face = image[y:y + h, x:x + w]
    
        embedding = get_embedding(face)
        if embedding is None:
            print("Error: Could not get embedding.")
            continue
                
        add_face_to_faiss(embedding, person_name)
        added += 1
        
        if added >= 5:
            break

    print(f"[INFO] Added {added} images for '{person_name}'.")
    

    
if __name__ == "__main__":
    
    person_name = input("Enter the name of the person: ")
    folder_path = input("Enter the folder path containing images: ")
    folder_path = r"{}".format(folder_path)
    
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
    else:
        add_person(person_name, folder_path)