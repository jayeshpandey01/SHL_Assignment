# Memory-Optimized Wav2Vec2 Extraction Fix
# Add this to your Code.ipynb to replace the problematic section

import os, time, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
import joblib
import glob
import librosa
import gc

FEATURE_DIR = "features"
DATA_ROOT = ""
Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

# OPTIMIZED PARAMS FOR 4GB GPU
W2V_MODEL = "facebook/wav2vec2-base-960h"
W2V_SR = 16000
BATCH = 1  # Reduced to 1 for 4GB GPU
PCA_DIM = 128
MAX_AUDIO_LENGTH = 10  # Limit audio to 10 seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Enable memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load CSVs
train_csv = pd.read_csv(os.path.join(DATA_ROOT, "csvs", "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_ROOT, "csvs", "test.csv"))

def ensure_wav(x): 
    x = str(x)
    return x if x.lower().endswith(".wav") else x + ".wav"

train_csv['filename'] = train_csv['filename'].apply(ensure_wav)
test_csv['filename'] = test_csv['filename'].apply(ensure_wav)

def find_audio_path(base_dir, fname):
    p = os.path.join(base_dir, fname)
    if os.path.exists(p): return p
    matches = glob.glob(os.path.join(base_dir, "**", os.path.basename(fname)), recursive=True)
    if matches: return matches[0]
    base = os.path.splitext(fname)[0]
    matches = glob.glob(os.path.join(base_dir, "**", base + ".*"), recursive=True)
    return matches[0] if matches else None

train_audio_dir = os.path.join(DATA_ROOT, "audios", "train")
test_audio_dir = os.path.join(DATA_ROOT, "audios", "test")
train_paths = [find_audio_path(train_audio_dir, f) for f in train_csv['filename'].tolist()]
test_paths = [find_audio_path(test_audio_dir, f) for f in test_csv['filename'].tolist()]

# Load model
print("Loading processor + model...")
processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
model = Wav2Vec2Model.from_pretrained(W2V_MODEL).to(device)
model.eval()

# Use half precision to save memory
if torch.cuda.is_available():
    model = model.half()

def load_audio_for_w2v(path, sr=W2V_SR, max_duration=MAX_AUDIO_LENGTH):
    """Load audio with length limit"""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_duration)
    return y.astype('float32')

def extract_w2v_embeddings(paths, batch=BATCH, save_prefix="w2v_raw"):
    """Memory-optimized extraction"""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    raw_embs = []
    
    for i in tqdm(range(0, len(paths), batch), desc="w2v batches"):
        batch_paths = paths[i:i+batch]
        
        try:
            # Load audio
            waves = [load_audio_for_w2v(p) for p in batch_paths]
            
            # Process
            inputs = processor(waves, sampling_rate=W2V_SR, return_tensors="pt", padding=True)
            input_values = inputs["input_values"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_values, dtype=torch.long))
            
            # Move to device
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            
            # Use half precision
            if torch.cuda.is_available():
                input_values = input_values.half()
            
            # Extract with no_grad
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)
                last = outputs.last_hidden_state
                
                # Compute masked mean pooling
                if attention_mask.shape[1] >= last.size(1):
                    mask = attention_mask[:, :last.size(1)].unsqueeze(-1)
                    summed = (last * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = (summed / counts).cpu().float().numpy()
                else:
                    pooled = last.mean(dim=1).cpu().float().numpy()
                
                raw_embs.append(pooled)
            
            # Clear GPU cache every batch
            del input_values, attention_mask, outputs, last
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save checkpoint every 20 batches
            if (i // batch) % 20 == 0 and len(raw_embs) > 0:
                partial = np.vstack(raw_embs)
                np.savez_compressed(
                    os.path.join(FEATURE_DIR, f"{save_prefix}_partial_{i//batch}.npz"),
                    X=partial
                )
                print(f"  Checkpoint saved at batch {i//batch}")
        
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Add zero embeddings for failed batch
            raw_embs.append(np.zeros((len(batch_paths), model.config.hidden_size), dtype=np.float32))
        
        # Aggressive garbage collection
        if i % 10 == 0:
            gc.collect()
    
    if len(raw_embs) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    
    embs = np.vstack(raw_embs)
    return embs

# Run extraction
print("\nStarting extraction...")
t0 = time.time()

print("Extracting train embeddings...")
w2v_train_raw = extract_w2v_embeddings(train_paths, batch=BATCH, save_prefix="w2v_train_raw")

print("Extracting test embeddings...")
w2v_test_raw = extract_w2v_embeddings(test_paths, batch=BATCH, save_prefix="w2v_test_raw")

print(f"Extraction time: {time.time()-t0:.1f}s")
print(f"Train shape: {w2v_train_raw.shape}, Test shape: {w2v_test_raw.shape}")

# PCA reduction
print(f"\nApplying PCA to {PCA_DIM} dimensions...")
pca = PCA(n_components=PCA_DIM, random_state=42)
w2v_train_p = pca.fit_transform(w2v_train_raw.astype('float32'))
w2v_test_p = pca.transform(w2v_test_raw.astype('float32'))

# Save
np.savez_compressed(
    os.path.join(FEATURE_DIR, f"w2v_pca_{PCA_DIM}.npz"),
    train=w2v_train_p,
    test=w2v_test_p
)
joblib.dump(pca, os.path.join(FEATURE_DIR, f"w2v_pca_{PCA_DIM}_pca.joblib"))

print(f"Saved: {os.path.join(FEATURE_DIR, f'w2v_pca_{PCA_DIM}.npz')}")
print(f"PCA shapes: Train {w2v_train_p.shape}, Test {w2v_test_p.shape}")

# Final cleanup
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n Extraction complete!")
