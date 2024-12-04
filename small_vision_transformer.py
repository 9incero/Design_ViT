import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import argparse
import os

# CSV 데이터셋 로더 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, supported_extensions=None):
        self.data = pd.read_csv(csv_file)  # CSV 파일 읽기
        self.transform = transform        # 이미지 전처리 함수
        self.supported_extensions = supported_extensions or [".jpg", ".JPG", ".png", ".PNG"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 경로 및 레이블 가져오기
        base_img_path = self.data.iloc[idx, 0]  # 확장자 없는 경로
        label = self.data.iloc[idx, 1]         # 레이블
        
        # 확장자를 탐지하여 실제 이미지 경로 찾기
        img_path = self.find_image(base_img_path)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for base path: {base_img_path} with supported extensions: {self.supported_extensions}")
        
        # 이미지 로드 및 변환 적용
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, int(label)

    def find_image(self, base_img_path):
        """
        주어진 경로에 대해 지원하는 확장자를 탐지하고 실제 경로를 반환합니다.
        """
        for ext in self.supported_extensions:
            potential_path = base_img_path + ext
            if os.path.exists(potential_path):
                return potential_path
        return None

# 결과 저장 함수
def save_results(filepath, content):
    with open(filepath, "a") as f:
        f.write(content + "\n")

# 모델 저장 함수
def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 모델 로드 함수
def load_model(model, load_path, num_classes):
    model.head = nn.Linear(model.head.in_features, num_classes)  # 출력 레이어 구성
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")
    return model

# 학습 및 검증 함수
def train_and_evaluate(args, mode="train"):
    # 모델 생성
    model = timm.create_model(args.model_name, pretrained=(mode == "train"))
    model.head = nn.Linear(model.head.in_features, args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if mode == "train":
        # 데이터 로더 생성
        train_dataset = CustomDataset(args.train_data, transform=transform)
        val_dataset = CustomDataset(args.val_data, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_accuracy = 0.0
        results_filepath = f"{args.name}_results.txt"

        # 학습 루프
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 검증 루프
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total * 100
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.2f}%")
            save_results(results_filepath, f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.2f}%")

            # 최고 성능 모델 저장
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_model(model, args.save_path)

        print(f"Training complete. Best Accuracy: {best_accuracy:.2f}%")
        save_results(results_filepath, f"Best Accuracy: {best_accuracy:.2f}%")

    elif mode == "test":
        # 테스트 데이터 로더
        test_dataset = CustomDataset(args.test_data, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = load_model(model, args.save_path, args.num_classes)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        results_filepath = f"{args.name}_results.txt"
        save_results(results_filepath, f"Test Accuracy: {accuracy:.2f}%")

# Argument Parser 정의
def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer with different args")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224", help="Model name from timm")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    return parser.parse_args()

# 메인 함수
if __name__ == "__main__":
    args = parse_args()
    args.save_path = f"/checkpoints/{args.name}_fine_tuned_vit.pth"

    # 학습 및 검증 실행
    train_and_evaluate(args, mode="train")
    # 테스트 실행
    train_and_evaluate(args, mode="test")
