import torch
from utils.helpers import ensure_path_join
import numpy as np

def sample_synthetic_uniform_embeddings(n_contexts):
    embeddings = torch.nn.functional.normalize(torch.randn([n_contexts, 512])).numpy()
    return {str(id_name): id_embedding for id_name, id_embedding in enumerate(embeddings)}


def sample_authentic_embeddings(n_contexts):
    all_authentic_path = "idiff_face_iccv2023_code/data/embeddings_elasticface_ffhq_128.npy"
    all_authentic_contexts = torch.load(all_authentic_path)

    id_names = list(all_authentic_contexts.keys())[:n_contexts]
    return {id_name: all_authentic_contexts[id_name] for id_name in id_names}


def sample_lfw_embeddings(n_contexts):
    all_lfw_path = "idiff_face_iccv2023_code/data/embeddings_elasticface_lfw_128_ffhq_aligned.npy"
    all_lfw_contexts = torch.load(all_lfw_path)

    id_names = list(all_lfw_contexts.keys())[:n_contexts]
    return {id_name: all_lfw_contexts[id_name] for id_name in id_names}


def sample_related_model_embeddings(n_contexts, model_name):
    all_path = f"idiff_face_iccv2023_code/samples/related_models/embeddings/embeddings_elasticface_{model_name}.npy"
    all_contexts = torch.load(all_path)

    id_names = list(all_contexts.keys())

    samples_contexts = {}
    ids = np.array([i.split("_")[0] for i in list(all_contexts.keys())])

    for label in np.unique(ids)[:n_contexts]:
        print(label)
        idxs = np.where(ids == label)
        idx = idxs[0][0]
        samples_contexts[id_names[idx]] = all_contexts[id_names[idx]]

    assert len(samples_contexts) == 5000

    return samples_contexts



def sample_random_input_elasticface_embeddings(n_contexts, device="cuda"):
    from create_nn_visualisation import load_elasticface
    from tqdm import tqdm

    random_input = torch.randn([n_contexts, 3, 112, 112])
    batch_size = 100

    model = load_elasticface(device, "/workspace/igd-slbt-master-thesis/utils/Elastic_R100_295672backbone.pth")
    model.eval()

    dataset = torch.utils.data.TensorDataset(random_input)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    @torch.no_grad()
    def embed_images(img_batch):
        img_batch = img_batch.to(device)
        emb_batch = model(img_batch).detach()
        return torch.nn.functional.normalize(emb_batch)

    embeddings = []
    for random_input_batch in tqdm(loader, total=len(loader)):
        random_input_batch = random_input_batch[0]
        embeddings.append(embed_images(random_input_batch).detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(embeddings.shape)

    return {str(id_name): id_embedding for id_name, id_embedding in enumerate(embeddings)}


if __name__ == '__main__':

    n_contexts = 15_000

    random_uniform_embeddings = sample_synthetic_uniform_embeddings(n_contexts)
    torch.save(random_uniform_embeddings, ensure_path_join(f"samples/contexts/random_synthetic_uniform_{n_contexts}.npy"))
    del random_uniform_embeddings

    #random_authentic_embeddings = sample_authentic_embeddings(n_contexts)
    #torch.save(random_authentic_embeddings, ensure_path_join(f"samples/contexts/random_authentic_{n_contexts}.npy"))
    #del random_authentic_embeddings

    #for model_name in ["sface"]:
    #    random_embeddings = sample_related_model_embeddings(n_contexts, model_name)
    #    torch.save(random_embeddings, ensure_path_join(f"samples/contexts/random_{model_name}_{n_contexts}.npy"))
    #    del random_embeddings

    #random_lfw_embeddings = sample_lfw_embeddings(n_contexts)
    #torch.save(random_lfw_embeddings, ensure_path_join(f"samples/contexts/random_lfw_{n_contexts}.npy"))
    #del random_lfw_embeddings

    #random_input_elasticface_embeddings = sample_random_input_elasticface_embeddings(n_contexts)
    #torch.save(random_input_elasticface_embeddings, ensure_path_join(f"samples/contexts/random_input_elasticface_{n_contexts}.npy"))
    #del random_input_elasticface_embeddings



