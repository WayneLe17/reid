import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torchvision import transforms
from PIL import Image
from app.core.config import settings


class CentroidsReIDService:
    
    def __init__(
        self, 
        model_path: str = None,
        device: Union[str, int] = None,
        use_centroids: bool = True,
        normalize_features: bool = True,
        distance_func: str = 'cosine'
    ):
        self.device = device or settings.DEVICE
        self.model_path = model_path or settings.REID_MODEL_PATH
        self.use_centroids = use_centroids
        self.normalize_features = normalize_features
        self.distance_func = distance_func
        
        self.model = self._load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self) -> nn.Module:
        model_path = Path(self.model_path)
        
        if model_path.exists():
            print(f"Loading model from: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'pytorch-lightning_version' in checkpoint:
                    print("Detected PyTorch Lightning checkpoint")
                    state_dict = checkpoint['state_dict']
                    model = self._create_ctl_lightning_model(state_dict)
                    model = self._load_lightning_weights(model, state_dict)
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    
                    if any('backbone' in k for k in state_dict.keys()):
                        model = self._create_ctl_compatible_model(state_dict)
                    else:
                        model = self._create_standard_model(state_dict)
                    
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model = self._create_standard_model(checkpoint)
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint
        else:
            print(f"Model path {model_path} not found. Using default ResNet50.")
            model = self._create_default_model()
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _create_ctl_compatible_model(self, state_dict: dict) -> nn.Module:
        import torchvision.models as models
        
        feature_dim = 2048
        for k, v in state_dict.items():
            if 'fc' in k and len(v.shape) > 0:
                feature_dim = v.shape[0]
                break
        
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, feature_dim)
        
        class CTLCompatibleModel(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
                
                x = self.backbone.avgpool(x)
                x = torch.flatten(x, 1)
                
                return x
        
        return CTLCompatibleModel(model)
    
    def _create_ctl_lightning_model(self, state_dict: dict) -> nn.Module:
        import torchvision.models as models
        
        has_bn = any('bn.' in k for k in state_dict.keys())
        has_fc_query = any('fc_query' in k for k in state_dict.keys())
        
        if any('resnet' in k.lower() or 'layer' in k for k in state_dict.keys()):
            backbone = models.resnet50(pretrained=False)
        else:
            backbone = models.resnet50(pretrained=False)
        
        class LightningCTLModel(nn.Module):
            def __init__(self, backbone, feature_dim=2048):
                super().__init__()
                self.backbone = nn.Sequential(*list(backbone.children())[:-2])
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
                if has_bn:
                    self.bn = nn.BatchNorm1d(feature_dim)
                if has_fc_query:
                    self.fc_query = nn.Linear(2048, feature_dim)
                
            def forward(self, x):
                x = self.backbone(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                
                if hasattr(self, 'bn'):
                    x = self.bn(x)
                if hasattr(self, 'fc_query'):
                    x = self.fc_query(x)
                
                return x
        
        return LightningCTLModel(backbone)
    
    def _load_lightning_weights(self, model: nn.Module, state_dict: dict) -> nn.Module:
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k[6:]
            else:
                new_key = k
            
            if new_key.startswith('backbone.'):
                new_key = new_key.replace('backbone.', 'backbone.')
            
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        return model
    
    def _create_standard_model(self, state_dict: dict) -> nn.Module:
        import torchvision.models as models
        
        model = models.resnet50(pretrained=False)
        
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if 'fc.weight' in k:
                    num_features = model.fc.in_features
                    output_dim = v.shape[0]
                    model.fc = nn.Linear(num_features, output_dim)
                    break
        
        return model
    
    def _create_default_model(self) -> nn.Module:
        import torchvision.models as models
        
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2048)
        return model
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                
                if self.normalize_features:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                
                features = features.cpu().numpy().squeeze()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def get_all_images_per_id(self, crops_dir: str) -> Dict[int, List[Path]]:
        crops_path = Path(crops_dir)
        id_images = {}
        
        if not crops_path.exists():
            return id_images
        
        id_folders = [f for f in crops_path.iterdir() if f.is_dir() and f.name.startswith('id_')]
        
        for id_folder in sorted(id_folders):
            tracking_id = int(id_folder.name.split('_')[1])
            
            image_files = list(id_folder.glob("*.jpg")) + list(id_folder.glob("*.png"))
            
            if image_files:
                id_images[tracking_id] = image_files
                
        return id_images
    
    def get_first_image_from_each_id(self, crops_dir: str) -> Dict[int, Path]:
        crops_path = Path(crops_dir)
        id_images = {}
        
        if not crops_path.exists():
            return id_images
        
        id_folders = [f for f in crops_path.iterdir() if f.is_dir() and f.name.startswith('id_')]
        
        for id_folder in sorted(id_folders):
            tracking_id = int(id_folder.name.split('_')[1])
            crop_path = id_folder / "crop.jpg"
            
            if crop_path.exists():
                id_images[tracking_id] = crop_path
                
        return id_images
    
    def compute_centroid(self, features_list: List[np.ndarray]) -> np.ndarray:
        if not features_list:
            return np.array([])
        
        features_array = np.vstack(features_list)
        centroid = np.mean(features_array, axis=0)
        
        if self.normalize_features:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
        return centroid
    
    def extract_features_batch(self, id_images: Dict[int, Union[Path, List[Path]]]) -> Tuple[np.ndarray, List[int]]:
        features_list = []
        id_list = []
        
        for id_num, image_paths in id_images.items():
            if isinstance(image_paths, Path):
                features = self.extract_features(image_paths)
                if features is not None:
                    features_list.append(features)
                    id_list.append(id_num)
            else:
                if self.use_centroids and len(image_paths) > 1:
                    id_features = []
                    for img_path in image_paths:
                        feat = self.extract_features(img_path)
                        if feat is not None:
                            id_features.append(feat)
                    
                    if id_features:
                        centroid = self.compute_centroid(id_features)
                        features_list.append(centroid)
                        id_list.append(id_num)
                else:
                    features = self.extract_features(image_paths[0])
                    if features is not None:
                        features_list.append(features)
                        id_list.append(id_num)
        
        if not features_list:
            return np.array([]), []
        
        features_array = np.vstack(features_list)
        return features_array, id_list
    
    def compute_distance_matrix(self, features: np.ndarray) -> np.ndarray:
        if self.distance_func == 'cosine':
            similarity_matrix = cosine_similarity(features)
            distance_matrix = 1 - similarity_matrix
        elif self.distance_func == 'euclidean':
            distance_matrix = euclidean_distances(features)
            if distance_matrix.max() > 0:
                distance_matrix = distance_matrix / distance_matrix.max()
        else:
            raise ValueError(f"Unknown distance function: {self.distance_func}")
        
        return distance_matrix
    
    def cluster_identities(
        self,
        features: np.ndarray,
        id_list: List[int],
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5
    ) -> Dict:
        if len(features) == 0:
            return {}
        
        distance_matrix = self.compute_distance_matrix(features)
        
        if self.distance_func == 'cosine':
            similarity_matrix = 1 - distance_matrix
        else:
            similarity_matrix = 1 - distance_matrix
        
        if n_clusters is not None:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='precomputed',
                linkage='average'
            )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[int(cluster_id)] = []
            clusters[int(cluster_id)].append(id_list[i])
        
        tracking_to_cluster = {}
        for i, cluster_id in enumerate(cluster_labels):
            tracking_id = id_list[i]
            tracking_to_cluster[tracking_id] = int(cluster_id)
            
        return {
            'cluster_labels': cluster_labels.tolist(),
            'id_list': id_list,
            'similarity_matrix': similarity_matrix.tolist(),
            'distance_matrix': distance_matrix.tolist(),
            'n_clusters_found': len(np.unique(cluster_labels)),
            'clusters': clusters,
            'tracking_to_cluster': tracking_to_cluster,
            'distance_func': self.distance_func,
            'used_centroids': self.use_centroids
        }
    
    def process_clustering(
        self,
        crops_dir: str,
        distance_threshold: float = 0.2,
        n_clusters: Optional[int] = None
    ) -> Optional[Dict]:
        if self.use_centroids:
            id_images = self.get_all_images_per_id(crops_dir)
        else:
            id_images = self.get_first_image_from_each_id(crops_dir)
        
        if not id_images:
            return None
        
        features, id_list = self.extract_features_batch(id_images)
        if len(features) == 0:
            return None
        
        return self.cluster_identities(features, id_list, n_clusters, distance_threshold)