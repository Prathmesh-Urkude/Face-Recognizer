from loader import *

def delete_person_from_index(person_to_delete, index, person_id_list):
    # Reconstruct all embeddings
    all_embeddings = [index.reconstruct(i) for i in range(index.ntotal)]

    # Filter out the embeddings and names of the person to delete
    filtered_embeddings = []
    filtered_names = []

    for emb, name in zip(all_embeddings, person_id_list):
        if name != person_to_delete:
            filtered_embeddings.append(emb)
            filtered_names.append(name)

    # Rebuild the index
    dim = index.d  # 512 for FaceNet
    new_index = faiss.IndexFlatL2(dim)
    if filtered_embeddings:
        new_index.add(np.array(filtered_embeddings, dtype=np.float32))
    else:
        print("No entries left in the index.")
        
    print(f"Removed all entries for: {person_to_delete}")
    
    person_id_list.clear()
    person_id_list.extend(filtered_names)
    
    faiss.write_index(new_index, index_file)
    save_person_ids()



if __name__ == "__main__":
    person_name = input("Enter the name of the person to delete: ")
    
    delete_person_from_index(person_name, index, person_id_list)

