import torch


class EmbeddingsFileDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings_file_path: str):
        super(EmbeddingsFileDataset, self).__init__()

        self.embeddings_file_path = embeddings_file_path
        self.samples = self.build_samples(
            embeddings_file_path=embeddings_file_path
        )

    @staticmethod
    def build_samples(embeddings_file_path: str):
        content = torch.load(embeddings_file_path)
        samples = []
        for _, embedding_npy in content.items():
            samples.append(embedding_npy)
        return samples

    def __getitem__(self, index: int):
        embedding_npy = self.samples[index]

        embedding = torch.from_numpy(embedding_npy)

        return embedding

    def __len__(self):
        return len(self.samples)
