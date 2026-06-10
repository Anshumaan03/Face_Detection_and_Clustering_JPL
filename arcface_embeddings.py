import os
import json
import mysql.connector
from deepface import DeepFace

# 1. Database Configuration
db_config = {
    "host": "127.0.0.1",
    "user": "root",              # Replace with your MySQL username
    "password": "Anshu@2003",  # Replace with your MySQL password
    "database": "face_db"
}

# The absolute path to your flat image folder
DATASET_DIR = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"

def init_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("🚀 Connected to MySQL successfully.")
    return conn, cursor

def extract_identity(file_name):
    """
    Extracts the person's name from the file name.
    Example: 'Elton_John_0001.jpg' -> 'Elton_John'
    Adjust the logic below if your naming convention is different!
    """
    # Remove file extension (.jpg, .png, etc.)
    base_name = os.path.splitext(file_name)[0]
    
    # If files are named 'FirstName_LastName_0001', split by the last underscore
    if "_" in base_name:
        parts = base_name.split("_")
        # If the last part is a number (e.g., '0001'), drop it to get the name
        if parts[-1].isdigit():
            return "_".join(parts[:-1])
        return base_name
    return base_name

def main():
    conn, cursor = init_db()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Directory '{DATASET_DIR}' not found.")
        return

    print("🧠 Initializing ArcFace Model...")
    
    # Track success/failure counts
    saved_count = 0
    failed_count = 0
    
    # Loop directly through all 5000 images in the flat folder
    all_files = os.listdir(DATASET_DIR)
    print(f"📂 Found {len(all_files)} total items in raw folder. Starting processing...")

    for img_name in all_files:
        # Process only valid image formats
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(DATASET_DIR, img_name)
            
            # Automatically figure out who this person is based on the file name
            identity = extract_identity(img_name)
            
            try:
                # Run the ArcFace model pass
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=False  # Keeps processing moving even on tough photos
                )
                
                # Extract the 512D ArcFace vector list
                embedding_vector = embedding_objs[0]["embedding"]
                
                # Convert vector to string for MySQL LONGTEXT column
                embedding_json_str = json.dumps(embedding_vector)
                
                # Insert data into MySQL
                insert_query = """
                    INSERT INTO face_embeddings (file_name, identity_label, embedding_json)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (img_name, identity, embedding_json_str))
                conn.commit()
                
                saved_count += 1
                print(f"   ✅ Saved [{saved_count}]: {img_name} ──► Identity: {identity}")
                
            except Exception as e:
                failed_count += 1
                print(f"   ⚠️ Skipped {img_name}. Error: {e}")

    # Clean up
    cursor.close()
    conn.close()
    print(f"\n🎉 Processing Complete!")
    print(f"📊 Summary: Successfully saved {saved_count} embeddings. Failed/Skipped: {failed_count}")

if __name__ == "__main__":
    main()