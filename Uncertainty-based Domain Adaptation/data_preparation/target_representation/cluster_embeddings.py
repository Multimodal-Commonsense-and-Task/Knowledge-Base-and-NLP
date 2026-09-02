import argparse
import torch
import numpy as np
import faiss
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster document embeddings with Faiss KMeans")
    parser.add_argument("--embedding_file", type=str, required=True,
                        help="File path to the saved document embeddings (PyTorch file containing a dict mapping docid to embedding)")
    parser.add_argument("--assignments_file", type=str, required=True,
                        help="File path to save the clustering assignments (JSONL file)")
    parser.add_argument("--centroids_file", type=str, required=True,
                        help="File path to save the cluster centroids (JSON file)")
    parser.add_argument("--num_clusters", type=int, required=True,
                        help="Number of clusters to form")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="Number of iterations for KMeans training")
    # Uncomment the following argument if you want to use GPU clustering with Faiss:
    # parser.add_argument("--use_gpu", action="store_true", help="Use GPU for clustering if available")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load the document embeddings.
    #    Expected format: a dictionary mapping docid to embedding (torch.Tensor)
    print("Loading embeddings from:", args.embedding_file)
    emb_dict = torch.load(args.embedding_file, map_location="cpu")
    docids = list(emb_dict.keys())
    
    # Create a tensor of shape (N, d) from the embeddings
    embeddings = torch.stack([emb_dict[docid] for docid in docids])
    embeddings_np = embeddings.cpu().numpy()
    n_samples, d = embeddings_np.shape
    print(f"Loaded {n_samples} embeddings of dimension {d}.")

    # 2. Set up and train Faiss KMeans.
    n_clusters = args.num_clusters
    print(f"Clustering into {n_clusters} clusters with {args.n_iter} iterations...")
    
    kmeans = faiss.Kmeans(d, n_clusters, niter=args.n_iter, verbose=True, seed=42)
    # Uncomment the next lines to use GPU if desired and available:
    # res = faiss.StandardGpuResources()
    # kmeans = faiss.index_cpu_to_gpu(res, 0, kmeans)
    
    kmeans.train(embeddings_np)

    # 3. Assign each embedding to its nearest cluster centroid.
    distances, assignments = kmeans.index.search(embeddings_np, 1)
    cluster_labels = assignments.squeeze()  # shape: (N,)

    # 4. Save the clustering assignments.
    # Each line in the output file will be a JSON object with the document id and its cluster label.
    print("Saving clustering assignments to:", args.assignments_file)
    with open(args.assignments_file, "w") as fout:
        for docid, label in zip(docids, cluster_labels):
            record = {"docid": docid, "cluster": int(label)}
            fout.write(json.dumps(record) + "\n")

    # 5. Save the cluster centroids.
    # The centroids are stored as a NumPy array of shape (num_clusters, d)
    centroids = kmeans.centroids  # shape: (num_clusters, d)
    centroids_list = centroids.tolist()  # convert to a regular Python list for JSON serialization

    # Here we save a JSON object mapping each cluster id to its centroid vector.
    centroids_dict = {str(i): centroid for i, centroid in enumerate(centroids_list)}
    print("Saving cluster centroids to:", args.centroids_file)
    with open(args.centroids_file, "w") as fcent:
        json.dump(centroids_dict, fcent, indent=2)

    print("Clustering and saving completed.")

if __name__ == "__main__":
    main()
