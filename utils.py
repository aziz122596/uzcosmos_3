import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Денормализует PyTorch тензор изображения (C, H, W) для визуализации."""
    if not isinstance(tensor, torch.Tensor):
         # Если это NumPy, используем другую функцию или конвертируем
         return denormalize_numpy(tensor, mean, std)

    tensor = tensor.clone().detach().cpu()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    tensor.mul_(std_t).add_(mean_t) # Денормализация
    img_np = tensor.numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def denormalize_numpy(image_np, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    """Денормализует NumPy array изображения."""
    if image_np.max() > 1.1: # Если изображение уже в формате 0-255
         return image_np.astype(np.uint8)
    img = image_np.copy()
    # Убедимся, что mean и std подходят по размерности (для C в конце)
    if img.shape[-1] == 3:
        mean = np.array(mean)
        std = np.array(std)
        img *= std
        img += mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def plot_training_history(history, save_path="training_history_pytorch.png"):
    """Строит графики истории обучения (лосс и метрики)."""
    epochs = range(1, len(history['train_loss']) + 1)
    num_plots = 1 + len([k for k in history if k.startswith('val_') and k != 'val_loss'])
    plt.figure(figsize=(6 * num_plots, 5))

    # График Лосса
    ax_loss = plt.subplot(1, num_plots, 1)
    ax_loss.plot(epochs, history['train_loss'], label='Train Loss')
    ax_loss.plot(epochs, history['val_loss'], label='Validation Loss')
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("История функции потерь")
    ax_loss.legend()
    ax_loss.grid(True)

    # Графики Метрик
    plot_idx = 2
    for key in history.keys():
        if key.startswith('val_') and key != 'val_loss':
            metric_name = key.replace('val_', '')
            ax_met = plt.subplot(1, num_plots, plot_idx)
            ax_met.plot(epochs, history[key], label=f'Validation {metric_name.capitalize()}')
        
            # if f'train_{metric_name}' in history:
            #     ax_met.plot(epochs, history[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
            ax_met.set_xlabel("Эпоха")
            ax_met.set_ylabel("Метрика")
            ax_met.set_title(f"История метрики: {metric_name.capitalize()}")
            ax_met.legend()
            ax_met.grid(True)
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"График истории обучения сохранен в: {save_path}")
    plt.close()


def visualize_predictions(images_tensor, true_masks_tensor, predicted_probs_tensor,
                          num_examples=5, filename="prediction_examples_pytorch.png"):
    """Визуализирует N примеров: Изображение | Истинная маска | Предсказанная маска (PyTorch)."""
    num_examples = min(num_examples, images_tensor.shape[0])
    if num_examples == 0:
        print("Нет примеров для визуализации.")
        return

    plt.figure(figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        img = denormalize_tensor(images_tensor[i]) # Денормализуем тензор
        true_mask = true_masks_tensor[i].cpu().numpy().squeeze()
        pred_mask_prob = predicted_probs_tensor[i].cpu().numpy().squeeze()
        pred_mask_binary = (pred_mask_prob > 0.5).astype(np.uint8) # Порог

        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f"Изображение {i+1}")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title(f"Истинная маска {i+1}")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(pred_mask_binary, cmap='gray')
        plt.title(f"Предсказание {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Визуализация предсказаний сохранена в файл: {filename}")
    plt.close()