"""
Memory-Optimized Wav2Vec2 Feature Extraction
Fixes CUDA OOM errors on 4GB GPU
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
import joblib
import glob
import librosa
import gc

# =====================================================
# CONFIGURATION - OPTIMIZED FOR 4GB GPU
# =====================================================
FEATURE_DIR = "features"
DATA_ROOT = ""
Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

W2V_MODEL = "facebook/wav2vec2-base-960h"
W2V_SR = 16000
BATCH = 1  # Reduced to 1 for 4GB GPU (was 8)
PCA_DIM = 128
MAX_AUDIO_LENGTH = 10  # Limit audio to 10 seconds (was unlimited)
SAVE_EVERY = 20  # Save partial results every 20 batches

# Enable memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# =====================================================
# LOAD DATA
# =====================================================
train_csv = pd.read_csv(os.path.join(DATA_ROOT, "csvs", "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_ROOT, "csvs", "test.csv"))

def ensure_wav(x):
    x = str(x)
    return x if x.lower().endswith(".wav") else x + ".wav"

train_csv['filename'] = train_csv['filename'].apply(ensure_wav)
test_csv['filename'] = test_csv['filename'].apply(ensure_wav)

def find_audio_path(base_dir, fname):
    p = os.path.join(base_dir, fname)
    if os.path.exists(p):
        return p
    matches = glob.glob(os.path.join(base_dir, "**", os.path.basename(fname)), recursive=True)
    if matches:
        return matches[0]
    base = os.path.splitext(fname)[0]
    matches = glob.glob(os.path.join(base_dir, "**", base + ".*"), recursive=True)
    return matches[0] if matches else None

train_audio_dir = os.path.join(DATA_ROOT, "audios", "train")
test_audio_dir = os.path.join(DATA_ROOT, "audios", "test")
train_paths = [find_audio_path(train_audio_dir, f) for f in train_csv['filename'].tolist()]
test_paths = [find_audio_path(test_audio_dir, f) for f in test_csv['filename'].tolist()]

print(f"Train files: {len(train_paths)}")
print(f"Test files: {len(test_paths)}")

# =====================================================
# LOAD MODEL WITH MEMORY OPTIMIZATION
# =====================================================
print("\nLoading Wav2Vec2 model...")
processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
model = Wav2Vec2Model.from_pretrained(W2V_MODEL)

# Use half precision to save memory
if torch.cuda.is_available():
    model = model.half()  # FP16
    print("Using FP16 (half precision) to save memory")

model = model.to(device)
model.eval()

# Freeze model to save memory
for param in model.parameters():
    param.requires_grad = False

print("Model loaded successfully")

# =====================================================
# MEMORY-OPTIMIZED AUDIO LOADING
# =====================================================
def load_audio_for_w2v(path, sr=W2V_SR, max_length=MAX_AUDIO_LENGTH):
    """Load audio with length limit to prevent OOM"""
    try:
        # Load only first MAX_AUDIO_LENGTH seconds
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_length)
        return y.astype('float32')
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(sr, dtype='float32')  # Return 1 second of silence

# =====================================================
# MEMORY-OPTIMIZED EXTRACTION
# =====================================================
def extract_w2v_embeddings_optimized(paths, batch=BATCH, save_prefix="w2v_raw"):
    """Extract embeddings with aggressive memory management"""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    raw_embs = []
    
    # Check if partial results exist
    partial_files = sorted(glob.glob(os.path.join(FEATURE_DIR, f"{save_prefix}_partial_*.npz")))
    if partial_files:
        print(f"Found {len(partial_files)} partial files, loading...")
        for pf in partial_files:
            data = np.load(pf)
            raw_embs.append(data['X'])
        start_idx = len(raw_embs) * batch * SAVE_EVERY
        print(f"Resuming from index {start_idx}")
    else:
        start_idx = 0
    
    num_batches = (len(paths) - start_idx + batch - 1) // batch
    
    for batch_idx in tqdm(range(num_batches), desc=f"{save_prefix} extraction"):
        i = start_idx + batch_idx * batch
        batch_paths = paths[i:i+batch]
        
        # Load audio
        waves = [load_audio_for_w2v(p) for p in batch_paths]
        
        # Process with Wav2Vec2
        try:
            inputs = processor(waves, sampling_rate=W2V_SR, return_tensors="pt", padding=True)
            input_values = inputs["input_values"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_values, dtype=torch.long))
            
            # Move to device
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            
            # Use FP16 if available
            if torch.cuda.is_available():
                input_values = input_values.half()
            
            # Extract features
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)
                last = outputs.last_hidden_state  # (B, T', hidden)
                
                # Masked mean pooling
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
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM at batch {batch_idx}, clearing cache and retrying with smaller audio...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Retry with even shorter audio
                waves = [load_audio_for_w2v(p, max_length=5) for p in batch_paths]
                try:
                    inputs = processor(waves, sampling_rate=W2V_SR, return_tensors="pt", padding=True)
                    input_values = inputs["input_values"].to(device)
                    if torch.cuda.is_available():
                        input_values = input_values.half()
                    
                    with torch.no_grad():
                        outputs = model(input_values)
                        pooled = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
                        raw_embs.append(pooled)
                    
                    del input_values, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    print(f"Failed even with shorter audio, using zeros")
                    raw_embs.append(np.zeros((len(batch_paths), model.config.hidden_size), dtype=np.float32))
            else:
                raise e
        
        # Save partial results periodically
        if (batch_idx + 1) % SAVE_EVERY == 0:
            partial = np.vstack(raw_embs)
            partial_path = os.path.join(FEATURE_DIR, f"{save_prefix}_partial_{batch_idx}.npz")
            np.savez_compressed(partial_path, X=partial)
            print(f"\nSaved partial results: {partial.shape}")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if len(raw_embs) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    
    embs = np.vstack(raw_embs)
    return embs

# =====================================================
# MAIN EXTRACTION
# =====================================================
print("\n" + "="*60)
print("STARTING MEMORY-OPTIMIZED EXTRACTION")
print("="*60)

t0 = time.time()

# Extract train embeddings
print("\nExtracting TRAIN embeddings...")
w2v_train_raw = extract_w2v_embeddings_optimized(train_paths, batch=BATCH, save_prefix="w2v_train_raw")
print(f"Train shape: {w2v_train_raw.shape}")
print(f"Time: {time.time()-t0:.1f}s")

# Clear memory before test
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Extract test embeddings
print("\nExtracting TEST embeddings...")
t1 = time.time()
w2v_test_raw = extract_w2v_embeddings_optimized(test_paths, batch=BATCH, save_prefix="w2v_test_raw")
print(f"Test shape: {w2v_test_raw.shape}")
print(f"Time: {time.time()-t1:.1f}s")

print(f"\nTotal extraction time: {time.time()-t0:.1f}s")

# =====================================================
# PCA REDUCTION
# =====================================================
print("\n" + "="*60)
print(f"APPLYING PCA (reducing to {PCA_DIM} dimensions)")
print("="*60)

pca = PCA(n_components=PCA_DIM, random_state=42)
w2v_train_pca = pca.fit_transform(w2v_train_raw.astype('float32'))
w2v_test_pca = pca.transform(w2v_test_raw.astype('float32'))

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Save results
output_path = os.path.join(FEATURE_DIR, f"w2v_pca_{PCA_DIM}.npz")
np.savez_compressed(output_path, train=w2v_train_pca, test=w2v_test_pca)
print(f"\nSaved: {output_path}")

pca_model_path = os.path.join(FEATURE_DIR, f"w2v_pca_{PCA_DIM}_pca.joblib")
joblib.dump(pca, pca_model_path)
print(f"Saved PCA model: {pca_model_path}")

print("\n" + "="*60)
print("✅ EXTRACTION COMPLETE!")
print("="*60)
print(f"Train PCA shape: {w2v_train_pca.shape}")
print(f"Test PCA shape: {w2v_test_pca.shape}")
print(f"Total time: {time.time()-t0:.1f}s")

# Clean up partial files
print("\nCleaning up partial files...")
for pattern in ["w2v_train_raw_partial_*.npz", "w2v_test_raw_partial_*.npz"]:
    for f in glob.glob(os.path.join(FEATURE_DIR, pattern)):
        try:
            os.remove(f)
            print(f"Removed: {os.path.basename(f)}")
        except:
            pass

print("\n✨ Done!")
