This project is a real-time face recognition system that identifies two individuals — myself and my friend — using face embeddings and an SVM classifier. It was developed as a mini-project assigned during my initial phase at CSIR-CMERI before beginning a major project.

⚠️ As I am not continuing further with the CSIR-CMERI organization, I am uploading and documenting this face recognition mini-project here for reference and future use.

📌 Project Description
This system performs:

Face detection using OpenCV.
Face embedding extraction using a pre-trained FaceNet model.
Label encoding and storage.
Face recognition using a Support Vector Machine (SVM) classifier.
Real-time face recognition via webcam.

It supports recognition of two individuals, organized in datasets/me and datasets/friend.
-> Note: Due to a technical issue in VS Code, I couldn’t commit and push the datasets directly. 

CSIR_PROJECT/
│
├── datasets/
│   ├── friend/                 # Friend's face images
│   └── me/                     # My face images
│
├── detect_label.py            # Detects faces and prepares labeled dataset
├── encode_faces.py            # Encodes labels to numerical format
├── extract.py                 # Extracts face embeddings using FaceNet
├── svm.py                     # Trains the SVM classifier
├── recognize.py               # Performs real-time recognition
│
├── facenet_keras.h5           # Pre-trained FaceNet model
├── face_data.npy              # Raw face image data
├── face_embeddings.npy        # Extracted FaceNet embeddings
├── face_labels.npy            # Encoded label array
├── face_classifier.pkl        # Trained SVM model
├── label_encoder.pkl          # Label encoder for decoding predictions
│
├── test.jpg                   # Test input image
├── your_output_path.jpg       # Output image with prediction
└── README.md                  # This documentation

🔁 Execution Flow:
Run the following scripts in order:

detect_label.py – Detect faces and save them with correct labels.
encode_faces.py – Encode face labels into numerical format.
extract.py – Extract FaceNet embeddings from detected faces.
svm.py – Train the SVM classifier on the embeddings.
recognize.py – Run real-time face recognition using webcam.

Output: 
Live face recognition via webcam with names shown.
Predictions on static images (e.g., test.jpg) are saved to your_output_path.jpg.
