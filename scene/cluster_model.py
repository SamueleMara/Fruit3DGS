import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterModel(nn.Module):
    """
    Differentiable clustering of 3D Gaussians for unsupervised object grouping,
    now including integrated render-mask loss computation.
    """
    def __init__(self, num_clusters, num_gaussians, device='cuda'):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_gaussians = num_gaussians
        self.device = device

        # Learnable cluster logits: Gaussian -> Cluster
        self.A_logits = nn.Parameter(torch.randn(num_gaussians, num_clusters, device=device) * 0.01)

        # Cluster statistics
        self.register_buffer('cluster_centroids', torch.zeros(num_clusters, 3, device=device))
        self.register_buffer('cluster_dispersion', torch.zeros(num_clusters, device=device))
        self.register_buffer('cluster_occupancy', torch.zeros(num_clusters, device=device))

    @property
    def A(self):
        """Softmax over clusters to get Gaussian -> Cluster probabilities"""
        return F.softmax(self.A_logits, dim=1)  # (G, K)

    def compute_cluster_stats(self, gaussians):
        """Update centroids, dispersion, and occupancy from Gaussian positions and weights"""
        A = self.A
        w = gaussians.weights  # (G,)
        occupancy = (A * w[:, None]).sum(dim=0) + 1e-8  # (K,)
        self.cluster_occupancy.copy_(occupancy)

        centroids = ((A * w[:, None]).T @ gaussians.x) / occupancy[:, None]
        self.cluster_centroids.copy_(centroids)

        diff = gaussians.x[None, :, :] - centroids[:, None, :]
        sq_dist = (diff ** 2).sum(-1)
        dispersion = ((sq_dist * (A * w[:, None]).T).sum(dim=1)) / occupancy
        self.cluster_dispersion.copy_(dispersion)

    # -------------------------------
    # Core loss functions
    # -------------------------------
    def render_soft_masks(self, render_pkg):
        """
        Compute per-cluster soft masks in image space using Gaussian contributions
        render_pkg['R']: (P,G) pixel->Gaussian opacity contributions
        Returns: r_pc (P,K)
        """
        R = render_pkg['R']  # (P,G)
        A = self.A  # (G,K)
        r_pc = R @ A  # (P,K)
        return r_pc

    def compute_render_mask_loss(self, render_pkg, instance_mask, eps=1e-8):
        """
        Differentiable soft render-mask loss
        instance_mask: (P,) 1 for lemon pixels, 0 for background
        """
        r_pc = self.render_soft_masks(render_pkg)  # (P,K)
        # Sum over clusters for total prediction
        pred_mask = r_pc.sum(dim=1)  # (P,)
        pred_mask = torch.clamp(pred_mask, 0.0, 1.0)
        # BCE loss
        loss = F.binary_cross_entropy(pred_mask, instance_mask.float())
        return loss

    def compute_dispersion_loss(self, gaussians):
        """Encourage clusters to be compact"""
        self.compute_cluster_stats(gaussians)
        return self.cluster_dispersion.sum()

    def compute_separation_loss(self, min_dist=0.1):
        """Encourage clusters to be separated in space"""
        centroids = self.cluster_centroids
        K = self.num_clusters
        loss = 0.0
        for i in range(K):
            for j in range(i+1, K):
                d = torch.norm(centroids[i] - centroids[j])
                loss += F.relu(min_dist - d) ** 2
        return loss

    def compute_covisibility_loss(self, gaussians, top_k=5):
        """Encourage similar cluster assignments for Gaussians across views"""
        A = self.A
        G, K = A.shape
        loss = 0.0
        top_vals, top_idx = torch.topk(A, k=min(top_k, K), dim=1)
        for g in range(G):
            for i in range(top_idx.shape[1]):
                for j in range(i+1, top_idx.shape[1]):
                    ci, cj = top_idx[g,i].item(), top_idx[g,j].item()
                    loss += (A[g,ci] - A[g,cj]) ** 2
        return loss / G

    def compute_prune_loss(self, gaussians):
        """Encourage unused Gaussians to go to zero opacity"""
        w = gaussians.weights  # (G,)
        return w.abs().sum()

    # -------------------------------
    # Unified loss computation
    # -------------------------------
    def compute_losses(self, gaussians, render_pkg, instance_mask,
                       lambda_render=1.0, lambda_disp=1.0, lambda_sep=1.0,
                       lambda_cov=1.0, lambda_prune=1.0):
        """
        Compute all clustering-related losses in one call
        """
        losses = {}
        # Render-mask loss
        losses['render'] = lambda_render * self.compute_render_mask_loss(render_pkg, instance_mask)
        # Dispersion
        losses['disp'] = lambda_disp * self.compute_dispersion_loss(gaussians)
        # Separation
        losses['sep'] = lambda_sep * self.compute_separation_loss()
        # Co-visibility
        losses['cov'] = lambda_cov * self.compute_covisibility_loss(gaussians)
        # Pruning
        losses['prune'] = lambda_prune * self.compute_prune_loss(gaussians)

        # Total
        losses['total'] = sum(losses.values())
        return losses

    # -------------------------------
    # Cluster refinement
    # -------------------------------
    def update_from_gaussians(self, gaussians):
        """Update centroids/dispersion"""
        self.compute_cluster_stats(gaussians)

    def merge_split(self, merge_thresh=0.9, split_thresh=1.0):
        """Simple merge/split heuristics based on cosine similarity & dispersion"""
        K = self.num_clusters
        A = self.A
        merged = set()
        for i in range(K):
            if i in merged: continue
            for j in range(i+1,K):
                if j in merged: continue
                sim = F.cosine_similarity(A[:,i], A[:,j], dim=0)
                if sim > merge_thresh:
                    with torch.no_grad():
                        self.A_logits[:,i] += self.A_logits[:,j]
                        self.A_logits[:,j] = -1e6
                    merged.add(j)
        for c in range(K):
            if self.cluster_dispersion[c] > split_thresh:
                with torch.no_grad():
                    self.A_logits[:,c] += torch.randn_like(self.A_logits[:,c]) * 0.01
