import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Define the model
class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = GenderModel()
model.load_state_dict(torch.load("gender_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Pixel values ko normalize karna (mean aur standard deviation ke according).
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# haarcascade_frontalface_default.xml: Front-facing faces detect karne ke liye configuration file.

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working!")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for the model
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_rgb)
        face_tensor = transform(face_img).unsqueeze(0)  # Add batch dimension

        # Predict gender
        with torch.no_grad():
            output = model(face_tensor)
            prediction = torch.argmax(output, 1).item()
            label = "Male" if prediction == 0 else "Female"

        # Display prediction on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Gender Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
