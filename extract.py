from facenet_pytorch import InceptionResnetV1
import torch

# Load the pretrained model
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(face_pixels):
    # Ensure input is a float tensor
    face_pixels = face_pixels.astype('float32') / 255.0
    face_tensor = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 160, 160)

    with torch.no_grad():
        embedding = model(face_tensor)
    
    return embedding[0].numpy()  # Convert to NumPy array
