import torch
from torch import nn
from transformers import AutoModel


class ClassificationModel(nn.Module):
    def __init__(self,
                 learning_rate=5e-5,
                 hidden_dim=256,
                 num_classes=3,
                 dropout_rate=0.1,
                 backbone_model_name="microsoft/deberta-v3-large"):
        super(ClassificationModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(
            backbone_model_name,
            dtype=torch.float16,
            device_map="auto"  # otomatis pilih GPU kalau tersedia
        )
        bert_dim = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # Ambil CLS token (representasi pertama)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        logits = self.classifier(cls_output)
        return logits


if __name__ == "__main__":
    model = ClassificationModel(
        num_classes=3, backbone_model_name="google-bert/bert-base-uncased")
    input_ids = torch.randint(0, 1000, (2, 512))  # Example input
    attention_mask = torch.ones((2, 512))         # Example attention mask
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(outputs.shape)  # torch.Size([2, 3])
