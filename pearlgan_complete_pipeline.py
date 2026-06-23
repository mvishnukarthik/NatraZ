import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 1. DATASET LOADER
# ==========================================
class PairedIRDataset(Dataset):
    """Loads matching Infrared and RGB images from folders."""
    def __init__(self, ir_dir, rgb_dir, transform=None):
        self.ir_dir = ir_dir
        self.rgb_dir = rgb_dir
        self.transform = transform
        self.image_filenames = os.listdir(ir_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        ir_path = os.path.join(self.ir_dir, img_name)
        rgb_path = os.path.join(self.rgb_dir, img_name)

        # Load images
        ir_image = Image.open(ir_path).convert("L") # L = Grayscale/IR
        rgb_image = Image.open(rgb_path).convert("RGB")

        if self.transform:
            ir_image = self.transform(ir_image)
            rgb_image = self.transform(rgb_image)

        return ir_image, rgb_image

# ==========================================
# 2. PEARLGAN ARCHITECTURE COMPONENTS
# ==========================================
class GradientAlignmentLoss(nn.Module):
    def __init__(self):
        super(GradientAlignmentLoss, self).__init__()
        import numpy as np
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        self.weight_x = nn.Parameter(torch.from_numpy(kernel_x).view(1, 1, 3, 3), requires_grad=False)
        self.weight_y = nn.Parameter(torch.from_numpy(kernel_y).view(1, 1, 3, 3), requires_grad=False)
        self.criterion = nn.L1Loss()

    def forward(self, gen_img, real_ir):
        gen_gray = 0.299 * gen_img[:, 0:1] + 0.587 * gen_img[:, 1:2] + 0.114 * gen_img[:, 2:3]
        gen_grad_x = F.conv2d(gen_gray, self.weight_x, padding=1)
        gen_grad_y = F.conv2d(gen_gray, self.weight_y, padding=1)
        ir_grad_x = F.conv2d(real_ir, self.weight_x, padding=1)
        ir_grad_y = F.conv2d(real_ir, self.weight_y, padding=1)
        return self.criterion(torch.abs(gen_grad_x) + torch.abs(gen_grad_y), torch.abs(ir_grad_x) + torch.abs(ir_grad_y))

class TopDownAttention(nn.Module):
    def __init__(self, in_channels):
        super(TopDownAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.conv(x)

class PearlGenerator(nn.Module):
    def __init__(self):
        super(PearlGenerator, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.attention = TopDownAttention(256)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.final(self.dec2(self.dec1(self.attention(self.enc3(self.enc2(self.enc1(x)))))))

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(epochs=50, batch_size=4, ir_dir='./dataset/ir', rgb_dir='./dataset/rgb'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
    ])
    
    # NOTE: Ensure these folders exist and contain images!
    dataset = PairedIRDataset(ir_dir=ir_dir, rgb_dir=rgb_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model & Optimizer
    generator = PearlGenerator().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss Functions
    l1_loss = nn.L1Loss()
    grad_loss = GradientAlignmentLoss().to(device)

    print("Starting Training...")
    for epoch in range(epochs):
        for i, (ir_imgs, rgb_imgs) in enumerate(dataloader):
            ir_imgs, rgb_imgs = ir_imgs.to(device), rgb_imgs.to(device)

            # --- Train Generator ---
            optimizer.zero_grad()
            
            # Generate fake color images
            generated_rgb = generator(ir_imgs)
            
            # Calculate Losses
            loss_color = l1_loss(generated_rgb, rgb_imgs)
            loss_edge = grad_loss(generated_rgb, ir_imgs)
            
            # Combined Hackathon Loss
            total_loss = loss_color + (0.5 * loss_edge)
            
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss.item():.4f}")

    # Save the trained model weights
    torch.save(generator.state_dict(), "pearlgan_generator.pth")
    print("Training Complete! Model saved as 'pearlgan_generator.pth'")
    return generator

# ==========================================
# 4. INFERENCE & YOLO VALIDATION
# ==========================================
def generate_and_evaluate(generator_model, test_ir_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load and prepare the IR image
    ir_image = Image.open(test_ir_image_path).convert("L")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = transform(ir_image).unsqueeze(0).to(device)

    # 2. Generate the Color Image
    generator_model.eval()
    with torch.no_grad():
        fake_rgb_tensor = generator_model(input_tensor)
    
    # Convert tensor back to image format
    fake_rgb_img = fake_rgb_tensor.squeeze().cpu().numpy()
    fake_rgb_img = ((fake_rgb_img * 0.5 + 0.5) * 255).astype('uint8')
    fake_rgb_img = fake_rgb_img.transpose(1, 2, 0)
    fake_rgb_img = cv2.cvtColor(fake_rgb_img, cv2.COLOR_RGB2BGR)
    
    # Save output
    output_path = "generated_color_output.jpg"
    cv2.imwrite(output_path, fake_rgb_img)
    print(f"Colorized image saved to {output_path}")

    # 3. Run YOLO validation on the output
    print("Running YOLOv8 Object Detection on generated image...")
    yolo_model = YOLO('yolov8n.pt') 
    results = yolo_model(output_path)
    
    annotated_frame = results[0].plot()
    cv2.imwrite("yolo_final_proof.jpg", annotated_frame)
    print("Saved proof with bounding boxes to 'yolo_final_proof.jpg'")

# ==========================================
# EXECUTION COMMANDS
# ==========================================
if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    # Copy the 'path' printed from kagglehub here, and append the subfolders
    # Example: KAGGLE_BASE_PATH = "/root/.cache/kagglehub/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2/versions/1"
    
    KAGGLE_BASE_PATH = "./dataset" # Replace this!
    
    # Look at the os.listdir() output to get the exact subfolder names:
    IR_FOLDER = os.path.join(KAGGLE_BASE_PATH, 'thermal') # e.g., 'train/thermal'
    RGB_FOLDER = os.path.join(KAGGLE_BASE_PATH, 'rgb')    # e.g., 'train/rgb'

    # Check if dataset exists
    if not os.path.exists(IR_FOLDER) or not os.path.exists(RGB_FOLDER):
        print(f"ERROR: Could not find {IR_FOLDER} or {RGB_FOLDER}.")
        print("Please check the KAGGLE_BASE_PATH and folder names!")
    else:
        # STEP 1: Train the model
        trained_gen = train_model(epochs=10, batch_size=4, ir_dir=IR_FOLDER, rgb_dir=RGB_FOLDER)
        
        # STEP 2: Pick one IR image from your dataset to test
        test_image = os.path.join(IR_FOLDER, os.listdir(IR_FOLDER)[0])
        
        # STEP 3: Colorize it and run YOLO on it
        generate_and_evaluate(trained_gen, test_image)