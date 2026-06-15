import torch

def custom_collate(batch):
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    target_phrases = [item["target_phrase"] for item in batch]
    bboxes = torch.stack([item["bbox"] for item in batch])
    answers = [item["answer"] for item in batch]
    families = [item["family"] for item in batch]

    return {
        "images": images,  # List of PIL images
        "questions": questions,
        "target_phrases": target_phrases,
        "bboxes": bboxes,
        "answers": answers,
        "families": families,
    }
