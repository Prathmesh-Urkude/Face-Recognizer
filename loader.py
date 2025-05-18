import cv2
import os
import faiss
import mtcnn
import tensorflow as tf
import numpy as np
import json
import time

#####################################################################
# load FaceNet model
def load_graph(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        
    return graph

# Path to the .pb model file
graph = load_graph("20180402-114759.pb") 
sess = tf.compat.v1.Session(graph=graph)

# Get input/output tensors
images_placeholder = graph.get_tensor_by_name("input:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
embeddings_tensor = graph.get_tensor_by_name("embeddings:0")


#####################################################################
def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Normalize
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    
    # Expand dims for batch input
    face = np.expand_dims(face, axis=0)
    return face 

def get_embedding(face):
    preprocessed_face = preprocess_face(face)
    if preprocessed_face is None:
        return None
    
    feed_dict = {
        images_placeholder: preprocessed_face,
        phase_train_placeholder: False
    }
    
    embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)
    return embedding[0]


#####################################################################
id_file = "person_ids.json"

# Load existing person_id_list
if os.path.exists(id_file):
    with open(id_file, 'r') as f:
        try: 
            person_id_list = json.load(f)
        except json.JSONDecodeError:
            print("Error loading person IDs. Starting with an empty list.")
            person_id_list = []
else:
    person_id_list = []
   
    
# Save person's id after every addition
def save_person_ids():
    with open(id_file, 'w') as f:
        json.dump(person_id_list, f)


#####################################################################
# Load FAISS index (or create a new one)
index_file = "faiss_index_ivfpq.bin"

try:
    index = faiss.read_index(index_file)
    print("FAISS index loaded!")
except:
    d = 512   # FaceNet embeddings are 512-Demensional vectors
    index = faiss.IndexFlatL2(d)  # Use L2 distance for similarity
    print("New FAISS index created.")
    

#####################################################################
# Add face embedding to FAISS index and save person ID        
def add_face_to_faiss(face_embedding, person_id):
    if face_embedding is not None:
        face_embedding = np.array([face_embedding], dtype=np.float32)
        index.add(face_embedding)
        person_id_list.append(person_id)
        print(f"Added {person_id} to FAISS index.")

        # Save both index and ID mapping
        faiss.write_index(index, index_file)
        save_person_ids()