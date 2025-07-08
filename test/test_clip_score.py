import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def collate_fn(batch):
    gen_imgs, captions = zip(*batch)
    tensor_gen_imgs = [transforms.ToTensor()(img) for img in gen_imgs]
    return tensor_gen_imgs, list(captions)


def main():
    # 生成一些假图片和caption
    num_samples = 100  # 可以适当调大测试显存
    image_size = (224, 224, 3)
    gen_images = []
    captions = []
    for i in range(num_samples):
        arr = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
        img = Image.fromarray(arr)
        gen_images.append(img)
        captions.append(f"This is a test caption {i}")

    # DataLoader
    data_loader = DataLoader(
        list(zip(gen_images, captions)),
        collate_fn=collate_fn,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    # 初始化CLIPScore
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
    clip_score.to("cuda")

    all_scores = []
    for batch_eval in data_loader:
        images, batch_captions = batch_eval
        images = [img.to("cuda", non_blocking=True) * 255 for img in images]
        images = torch.stack(images, dim=0)  # 合并为一个 batch tensor
        scores = _clip_score_update(
            images, batch_captions, clip_score.model, clip_score.processor
        )[0]
        all_scores.extend(scores.detach().cpu().tolist())
        print(f"Batch scores: {scores}")

    print(f"All scores: {all_scores}")
    print(f"Average score: {sum(all_scores)/len(all_scores)}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
