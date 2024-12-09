import os
import cv2
import face_recognition

# Full path to the known faces directory
known_faces_dir = r"C:\Users\Arjun YS\Desktop\Python\Projects\Face Recog Project\known_faces"

# Check if the directory exists
if not os.path.isdir(known_faces_dir):
    print(f"Directory does not exist: {known_faces_dir}")
    exit()

# Load known faces and their names



known_face_encodings = []
known_face_names = []

# Load the known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Ensure the image contains at least one face
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use the filename without the extension as the name

# Load an image with unknown faces
unknown_image_path = r'C:\Users\Arjun YS\OneDrive\Pictures\WIN_20240723_09_48_57_Pro.jpg'  # Using raw string for the path
unknown_image = face_recognition.load_image_file(unknown_image_path)

# Find all face locations and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to BGR color for OpenCV
unknown_image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# List to hold names of recognized faces in the unknown image
recognized_names = []

# Loop over each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Add the recognized name to the list
    recognized_names.append(name)

    # Draw a box around the face
    cv2.rectangle(unknown_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw a label with a name below the face
    cv2.rectangle(unknown_image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_image_bgr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Resize the image to fit the screen (if needed)
scale_percent = 50  # percentage of original size
width = int(unknown_image_bgr.shape[1] * scale_percent / 100)
height = int(unknown_image_bgr.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(unknown_image_bgr, dim, interpolation=cv2.INTER_AREA)

# Display the resulting image
cv2.imshow('Image with Faces Highlighted', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of faces detected
print(f"Number of faces detected: {len(face_locations)}")

# Print the names of the faces detected
print("Names of faces detected:", recognized_names)
